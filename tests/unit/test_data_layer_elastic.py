"""Offline unit tests for search.elastic helpers (AC-001.7). ES client is faked."""

from __future__ import annotations

from elastic_transport import ConnectionError as TransportConnectionError
from structlog.testing import capture_logs

from vendor_risk_ai.schemas.domain import ChunkRecord
from vendor_risk_ai.search import elastic


class FakeIndices:
    def __init__(self, existing: set[str]) -> None:
        self.existing = existing
        self.created: dict[str, dict] = {}

    def exists(self, index: str) -> bool:
        return index in self.existing

    def create(self, index: str, mappings: dict, settings: dict) -> None:
        self.created[index] = mappings
        self.existing.add(index)


class FakeES:
    def __init__(self, existing: set[str] | None = None) -> None:
        self.indices = FakeIndices(existing or set())

    def search(self, **kw):
        raise TransportConnectionError("connection refused")

    def ping(self):
        raise TransportConnectionError("connection refused")


def test_ensure_indices_creates_missing_with_mappings(monkeypatch):
    """AC-001.7: indices created with explicit mappings only if absent."""
    fake = FakeES(existing={"market-quotes"})
    monkeypatch.setattr(elastic, "get_es", lambda: fake)
    with capture_logs():
        elastic.ensure_indices()
    assert list(fake.indices.created) == ["vendor-chunks"]
    emb = fake.indices.created["vendor-chunks"]["properties"]["embedding"]
    assert emb == {"type": "dense_vector", "dims": 1536, "index": True, "similarity": "cosine"}


def test_quotes_mapping_fields():
    props = elastic.QUOTES_MAPPING["properties"]
    assert props["ticker"]["type"] == "keyword"
    assert props["name"]["fields"]["keyword"]["type"] == "keyword"
    assert props["scraped_at"]["type"] == "date"
    assert all(props[f]["type"] == "float" for f in ("price", "change_pct", "pe_ratio"))


def test_quote_query_accepts_pair_or_symbol():
    q = elastic._quote_query("IBM:NYSE")
    assert {"term": {"ticker": "IBM:NYSE"}} in q["bool"]["should"]
    pair = {"bool": {"filter": [{"term": {"ticker": "IBM"}}, {"term": {"exchange": "NYSE"}}]}}
    assert pair in q["bool"]["should"]
    assert elastic._quote_query("IBM")["bool"]["should"] == [{"term": {"ticker": "IBM"}}]


def test_latest_quote_and_ping_never_raise(monkeypatch):
    monkeypatch.setattr(elastic, "get_es", lambda: FakeES())
    with capture_logs():
        assert elastic.latest_quote("IBM:NYSE") is None
        assert elastic.ping() is False
    assert elastic.latest_quote("") is None


def test_chunk_doc_id():
    c = ChunkRecord(document_id=7, chunk_index=2, text="x")
    assert elastic.chunk_doc_id(3, c) == "3:7:2"
