"""Gold against the live Qdrant (AC-005.7, AC-005.9, AC-005.11).

Uses a throwaway collection prefix and drops its collections afterwards; the corpus is the
SYNTHETIC 3-topic fixture. Skips when Qdrant is not reachable."""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

from tcs_bot.config import Settings, get_settings
from tcs_bot.medallion.gold import search
from tcs_bot.medallion.gold.pipeline import run_gold
from tcs_bot.medallion.gold.vector_store import GoldVectorStore
from tests.unit.test_gold_synthetic import FakeSilverStore, synthetic_records

pytestmark = pytest.mark.integration


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Settings]:
    monkeypatch.setenv("LLM_PROVIDER", "fake")  # never call OpenAI from tests
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("GOLD_DIR", str(tmp_path / "gold"))
    get_settings.cache_clear()
    yield get_settings()
    get_settings.cache_clear()


@pytest.fixture
def live_vs(settings: Settings) -> Iterator[GoldVectorStore]:
    from qdrant_client import QdrantClient

    client = QdrantClient(url=settings.qdrant_url, timeout=5)
    try:
        client.get_collections()
    except Exception as exc:
        pytest.skip(f"Qdrant not reachable at {settings.qdrant_url}: {exc}")
    vs = GoldVectorStore(client=client, prefix=f"test_gold_{uuid.uuid4().hex[:8]}")
    search.set_vector_store(vs)
    try:
        yield vs
    finally:
        search.set_vector_store(None)
        vs.with_model(f"lsa_{settings.lsa_dim}").drop()
        leftovers = [c.name for c in client.get_collections().collections if vs.prefix in c.name]
        for name in leftovers:
            client.delete_collection(name)
        client.close()


def test_gold_run_and_search_on_live_qdrant(settings: Settings, live_vs: GoldVectorStore) -> None:
    """AC-005.7/9: run_gold upserts into real Qdrant collections; filtered search works."""
    res = run_gold(FakeSilverStore(synthetic_records()), live_vs, settings)
    assert res["qdrant"] == "ok", res["qdrant"]
    vs = live_vs.with_model(f"lsa_{settings.lsa_dim}")
    assert vs.docs_collection.startswith(live_vs.prefix)
    assert vs.count("docs") == 30 and vs.count("chunks") == res["chunks"]

    hits = search.search_risks("ransomware breach exfiltrated passwords", k=5)
    assert len(hits) == 5 and all("/cyber/" in h.url for h in hits)
    legal = search.search_risks("court verdict", k=5, categories=["legal_litigation"], min_risk=60)
    assert legal and all(h.risk_score >= 60 for h in legal)
    assert all("legal_litigation" in h.risk_categories for h in legal)

    summary = json.loads(Path(res["paths"]["summary_json"]).read_text())
    assert summary["meta"]["qdrant_collections"] == [vs.docs_collection, vs.chunks_collection]

    # idempotent full rebuild
    run_gold(FakeSilverStore(synthetic_records()), live_vs, settings)
    assert vs.count("docs") == 30
