"""SilverStore + run_silver against live Elasticsearch (AC-005.5 / AC-005.6).

Uses a throwaway index ``silver-test-<uuid>`` that is deleted afterwards; bronze is faked so no
MySQL rows are touched. Documents are synthetic. Skips cleanly if ES is unreachable.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from elasticsearch import Elasticsearch

from tcs_bot.medallion.models import BronzeDocument
from tcs_bot.medallion.silver.pipeline import build_silver_record, run_silver
from tcs_bot.medallion.silver.store import SilverStore

pytestmark = pytest.mark.integration

ES_URL = "http://127.0.0.1:9200"
FIXTURES = Path(__file__).parents[1] / "fixtures" / "silver"


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> Iterator[SilverStore]:
    monkeypatch.setenv("LLM_PROVIDER", "fake")  # never call OpenAI from tests
    monkeypatch.setenv("OPENAI_API_KEY", "")
    from tcs_bot.config import get_settings

    get_settings.cache_clear()
    es = Elasticsearch(ES_URL, request_timeout=10)
    try:
        if not es.ping():
            pytest.skip("Elasticsearch not reachable")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Elasticsearch not reachable: {exc}")
    st = SilverStore(es=es, index=f"silver-test-{uuid.uuid4().hex[:12]}")
    try:
        yield st
    finally:
        st.delete_index()
        es.close()
        get_settings.cache_clear()


def _doc(doc_id: int, name: str, content_type: str, source: str = "web") -> BronzeDocument:
    return BronzeDocument(
        id=doc_id,
        url=f"https://news.example.com/{name}",
        canonical_url=f"https://news.example.com/{name}",
        depth=1,
        source=source,
        raw_content=(FIXTURES / name).read_text(encoding="utf-8"),
        content_type=content_type,
    )


DOCS = [
    ("article.html", "text/html", "web"),
    ("tax_collected_at_source.html", "text/html", "web"),
    ("tata_motors.html", "text/html", "web"),
    ("reddit_thread.json", "application/json", "reddit"),
    ("hn_thread.json", "application/json", "hackernews"),
]


class FakeRepo:
    def __init__(self, docs: list[BronzeDocument]) -> None:
        self.docs = docs
        self.marked: list[int] = []

    def iter_unprocessed(self, layer: str = "silver", limit: int | None = None):
        yield from self.docs[:limit] if limit else self.docs

    def mark_processed(self, ids: list[int], layer: str = "silver") -> None:
        self.marked.extend(ids)


def test_ensure_upsert_get_iter_stats(store: SilverStore):
    """AC-005.5: explicit mapping, idempotent upsert by doc_id, nested signal aggregations."""
    assert store.ensure_index() is True
    assert store.ensure_index() is False
    mapping = store.es.indices.get_mapping(index=store.index)[store.index]["mappings"]
    assert mapping["properties"]["risk_signals"]["type"] == "nested"
    assert mapping["properties"]["clean_text"]["analyzer"] == "english"

    records = [build_silver_record(_doc(i + 1, n, ct, src)) for i, (n, ct, src) in enumerate(DOCS)]
    assert store.upsert(records, refresh=True) == len(records)
    assert store.upsert(records[:1], refresh=True) == 1  # idempotent re-index
    assert store.count() == len(records)

    got = store.get(1)
    assert got is not None and got.title == records[0].title
    assert got.risk_signals == records[0].risk_signals
    assert store.get(424242) is None

    relevant = {r.doc_id for r in store.iter_relevant()}
    assert relevant == {r.doc_id for r in records if r.relevant}
    assert 2 not in relevant and 3 not in relevant  # tax + Tata Motors false positives
    assert {r.doc_id for r in store.iter_relevant(min_relevance=0.99)} <= relevant

    stats = store.stats()
    assert stats["total"] == len(records)
    assert stats["relevant"] == len(relevant)
    assert stats["by_source"]["web"] == 3
    assert "data_privacy_cyber" in stats["by_category"]
    assert sum(stats["by_severity"].values()) == sum(len(r.risk_signals) for r in records)

    hits = store.es.search(
        index=store.index,
        query={
            "nested": {
                "path": "risk_signals",
                "query": {"term": {"risk_signals.category": "labor_hr"}},
            }
        },
    )["hits"]["hits"]
    assert [h["_id"] for h in hits] == ["4"]


def test_run_silver_end_to_end_with_fake_bronze(store: SilverStore):
    """AC-005.5: run_silver indexes every doc and marks them processed after the upsert."""
    docs = [_doc(i + 10, n, ct, src) for i, (n, ct, src) in enumerate(DOCS)]
    repo = FakeRepo(docs)
    stats = run_silver(repo=repo, store=store)
    assert stats["processed"] == stats["indexed"] == len(docs)
    assert stats["failed"] == 0
    assert sorted(repo.marked) == [d.id for d in docs]
    store.es.indices.refresh(index=store.index)
    assert store.count() == len(docs)
    assert store.count(relevant_only=True) == stats["relevant"]
