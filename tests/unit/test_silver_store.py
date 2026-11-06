"""SilverStore against a fake ES client (AC-005.5): mapping, bulk upsert, failures."""

from __future__ import annotations

from typing import Any

from elastic_transport import ConnectionError as TransportConnectionError

from tcs_bot.medallion.models import SilverRecord
from tcs_bot.medallion.silver import store as store_mod
from tcs_bot.medallion.silver.store import SILVER_MAPPING, SILVER_SETTINGS, SilverStore


class FakeIndices:
    def __init__(self) -> None:
        self.created: dict[str, dict[str, Any]] = {}

    def exists(self, index: str) -> bool:
        return index in self.created

    def create(self, index: str, mappings: dict, settings: dict) -> None:
        self.created[index] = {"mappings": mappings, "settings": settings}


class FakeES:
    def __init__(self) -> None:
        self.indices = FakeIndices()


def _rec(doc_id: int) -> SilverRecord:
    return SilverRecord(
        doc_id=doc_id,
        url=f"https://example.com/{doc_id}",
        domain="example.com",
        source="web",
        source_type="news",
        depth=1,
        title="Fictional",
        clean_text="SYNTHETIC FIXTURE text.",
    )


def test_mapping_fields():
    """AC-005.5: explicit mapping for silver-tcs-risk."""
    p = SILVER_MAPPING["properties"]
    assert p["doc_id"]["type"] == "long"
    for f in ("url", "domain", "source", "source_type", "language", "risk_categories", "extractor"):
        assert p[f]["type"] == "keyword"
    assert p["title"]["type"] == "text" and p["title"]["fields"]["keyword"]["type"] == "keyword"
    assert p["clean_text"] == {"type": "text", "analyzer": "english"}
    assert p["summary"]["type"] == "text"
    assert p["published_at"]["type"] == p["processed_at"]["type"] == "date"
    assert p["depth"]["type"] == "integer"
    for f in ("tcs_relevance", "sentiment", "risk_score"):
        assert p[f]["type"] == "float"
    assert p["relevant"]["type"] == "boolean"
    assert p["entities"]["type"] == "object"
    assert all(v["type"] == "keyword" for v in p["entities"]["properties"].values())
    sig = p["risk_signals"]
    assert sig["type"] == "nested"
    assert sig["properties"]["evidence"]["type"] == "text"
    assert SILVER_SETTINGS == {"number_of_shards": 1, "number_of_replicas": 0}
    # every SilverRecord field is mapped explicitly
    assert set(SilverRecord.model_fields) <= set(p)


def test_ensure_index_creates_once_default_index_name():
    es = FakeES()
    st = SilverStore(es=es)  # type: ignore[arg-type]
    assert st.index == "silver-tcs-risk"
    assert st.ensure_index() is True
    assert st.ensure_index() is False
    assert es.indices.created["silver-tcs-risk"]["mappings"] is SILVER_MAPPING


def test_upsert_bulk_ids_and_partial_failures(monkeypatch):
    seen: dict[str, Any] = {}

    def fake_bulk(es, actions, **kw):
        seen["actions"] = list(actions)
        seen["kw"] = kw
        return 1, [{"index": {"_id": "2", "status": 400, "error": {"type": "mapper_parsing"}}}]

    monkeypatch.setattr(store_mod.helpers, "bulk", fake_bulk)
    st = SilverStore(es=FakeES(), index="t")  # type: ignore[arg-type]
    assert st.upsert([_rec(1), _rec(2)], refresh=True) == 1
    assert [a["_id"] for a in seen["actions"]] == ["1", "2"]
    assert seen["actions"][0]["_index"] == "t"
    assert seen["actions"][0]["_source"]["doc_id"] == 1
    assert seen["kw"]["refresh"] == "wait_for"
    assert st.last_failed_ids == {2}
    assert st.upsert([]) == 0 and st.last_failed_ids == set()


def test_upsert_es_down_returns_zero_and_marks_all_failed(monkeypatch):
    def down(*a, **kw):
        raise TransportConnectionError("connection refused")

    monkeypatch.setattr(store_mod.helpers, "bulk", down)
    st = SilverStore(es=FakeES(), index="t")  # type: ignore[arg-type]
    assert st.upsert([_rec(1), _rec(3)]) == 0
    assert st.last_failed_ids == {1, 3}
