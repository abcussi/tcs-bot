"""Integration tests for Redis L2 cache (AC-001.6) and Elasticsearch indices (AC-001.7).

Uses Redis db 15 with a unique key prefix and test-only index names, so shared data is untouched.
Each service skips cleanly if unreachable.
"""

from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from structlog.testing import capture_logs

from tcs_bot.cache.cache import TwoTierCache
from tcs_bot.config import get_settings
from tcs_bot.schemas.domain import ChunkRecord
from tcs_bot.search import elastic

pytestmark = pytest.mark.integration

REDIS_TEST_URL = "redis://127.0.0.1:6379/15"
QUOTES_IDX = "dltest-market-quotes"
CHUNKS_IDX = "dltest-vendor-chunks"


# --------------------------------------------------------------------------- Redis
@pytest.fixture
def redis_client():
    import redis

    r = redis.Redis.from_url(REDIS_TEST_URL, socket_connect_timeout=1)
    try:
        r.ping()
    except Exception as exc:
        pytest.skip(f"Redis not reachable: {exc}")
    prefix = f"dltest:{uuid.uuid4().hex[:8]}:"
    yield r, prefix
    for k in r.scan_iter(f"{prefix}*"):
        r.delete(k)
    r.close()


def test_l2_redis_round_trip_across_instances(redis_client):
    """AC-001.6: value written by one process-level cache is served from Redis to another."""
    r, prefix = redis_client
    a = TwoTierCache(REDIS_TEST_URL, ttl=30, maxsize=8, key_prefix=prefix)
    b = TwoTierCache(REDIS_TEST_URL, ttl=30, maxsize=8, key_prefix=prefix)
    assert a.ping()
    a.set("report:1", {"score": 61.5, "flags": ["leverage"]})
    assert 0 < r.ttl(f"{prefix}report:1") <= 30
    assert b.get("report:1") == {"score": 61.5, "flags": ["leverage"]}
    calls = []
    assert b.get_or_set("report:1", lambda: calls.append(1) or {"x": 1}) == {
        "score": 61.5,
        "flags": ["leverage"],
    }
    assert calls == []
    a.delete("report:1")
    b.clear_local()
    assert b.get("report:1") is None


def test_redis_down_url_degrades_to_l1():
    """AC-001.6: unreachable Redis never raises."""
    c = TwoTierCache("redis://127.0.0.1:6390/0", ttl=30, maxsize=8)
    with capture_logs() as logs:
        c.set("k", {"v": 1})
        assert c.get("k") == {"v": 1}
    assert any(e["event"] == "cache.redis_unavailable" for e in logs)


# --------------------------------------------------------------------------- Elasticsearch
@pytest.fixture
def es_test_indices():
    old = {k: os.environ.get(k) for k in ("ELASTIC_QUOTES_INDEX", "ELASTIC_CHUNKS_INDEX")}
    os.environ["ELASTIC_QUOTES_INDEX"] = QUOTES_IDX
    os.environ["ELASTIC_CHUNKS_INDEX"] = CHUNKS_IDX
    get_settings.cache_clear()
    elastic.reset_es()
    if not elastic.ping():
        _restore(old)
        pytest.skip("Elasticsearch not reachable")
    es = elastic.get_es()
    es.indices.delete(index=[QUOTES_IDX, CHUNKS_IDX], ignore_unavailable=True)
    try:
        yield es
    finally:
        es.indices.delete(index=[QUOTES_IDX, CHUNKS_IDX], ignore_unavailable=True)
        _restore(old)


def _restore(old: dict[str, str | None]) -> None:
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    get_settings.cache_clear()
    elastic.reset_es()


def test_ensure_indices_creates_explicit_mappings_idempotently(es_test_indices):
    """AC-001.7: both indices created with explicit mappings if absent; re-run is a no-op."""
    es = es_test_indices
    with capture_logs():
        elastic.ensure_indices()
        elastic.ensure_indices()
    q = es.indices.get_mapping(index=QUOTES_IDX)[QUOTES_IDX]["mappings"]["properties"]
    assert q["ticker"]["type"] == "keyword"
    assert q["name"]["fields"]["keyword"]["type"] == "keyword"
    assert q["price"]["type"] == "float"
    assert q["scraped_at"]["type"] == "date"
    c = es.indices.get_mapping(index=CHUNKS_IDX)[CHUNKS_IDX]["mappings"]["properties"]
    assert c["embedding"]["type"] == "dense_vector"
    assert c["embedding"]["dims"] == 1536
    assert c["embedding"]["similarity"] == "cosine"
    assert c["vendor_id"]["type"] == "long"


def test_index_chunks_bulk_and_ids(es_test_indices):
    es = es_test_indices
    with capture_logs():
        elastic.ensure_indices()
        vec = [0.0] * 1536
        vec[0] = 1.0
        chunks = [
            ChunkRecord(
                document_id=11,
                chunk_index=0,
                text="going concern doubt",
                embedding=vec,
                embedding_model="text-embedding-3-small",
                doc_type="financial_report",
                title="FS",
            ),
            ChunkRecord(
                document_id=11,
                chunk_index=1,
                text="short vector skipped",
                embedding=[0.1, 0.2],
                embedding_model="hash",
                doc_type="news",
                title="N",
            ),
        ]
        assert elastic.index_chunks(42, chunks, refresh=True) == 2
        assert elastic.index_chunks(42, chunks, refresh=True) == 2  # idempotent ids
    assert es.count(index=CHUNKS_IDX)["count"] == 2
    d0 = es.get(index=CHUNKS_IDX, id="42:11:0")["_source"]
    assert d0["vendor_id"] == 42 and len(d0["embedding"]) == 1536
    d1 = es.get(index=CHUNKS_IDX, id="42:11:1")["_source"]
    assert "embedding" not in d1 and d1["text"] == "short vector skipped"
    assert elastic.index_chunks(42, []) == 0


def test_latest_quote_returns_newest(es_test_indices):
    es = es_test_indices
    with capture_logs():
        elastic.ensure_indices()
    now = datetime.now(UTC)
    for i, price in enumerate([100.0, 101.5, 99.25]):
        es.index(
            index=QUOTES_IDX,
            document={
                "ticker": "IBM",
                "exchange": "NYSE",
                "name": "International Business Machines Corp",
                "currency": "USD",
                "price": price,
                "scraped_at": (now - timedelta(minutes=10 * (2 - i))).isoformat(),
            },
        )
    es.index(
        index=QUOTES_IDX,
        document={
            "ticker": "IBM",
            "exchange": "XETRA",
            "price": 1.0,
            "scraped_at": (now + timedelta(minutes=1)).isoformat(),
        },
    )
    es.indices.refresh(index=QUOTES_IDX)
    assert elastic.latest_quote("IBM:NYSE")["price"] == 99.25
    assert elastic.latest_quote("IBM")["exchange"] == "XETRA"  # newest across exchanges
    assert elastic.latest_quote("NOPE:NYSE") is None


def test_latest_quote_missing_index_and_down_es(monkeypatch):
    """latest_quote never raises: missing index or unreachable ES -> None."""
    monkeypatch.setenv("ELASTIC_QUOTES_INDEX", "dltest-does-not-exist")
    get_settings.cache_clear()
    elastic.reset_es()
    try:
        with capture_logs():
            if elastic.ping():
                assert elastic.latest_quote("IBM:NYSE") is None
            monkeypatch.setenv("ELASTIC_URL", "http://127.0.0.1:9299")
            get_settings.cache_clear()
            elastic.reset_es()
            assert elastic.latest_quote("IBM:NYSE") is None
            assert elastic.ping() is False
    finally:
        monkeypatch.undo()
        get_settings.cache_clear()
        elastic.reset_es()
