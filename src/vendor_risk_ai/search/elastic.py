"""Elasticsearch client and index management (AC-001.7).

Indices (names from settings): ``market-quotes`` (written by the gfinance crawler) and
``vendor-chunks`` (document chunks + dense embeddings). Read helpers degrade gracefully:
``latest_quote`` and ``ping`` never raise; ``index_chunks`` returns 0 on transport errors.
``ensure_indices`` raises if ES is unreachable (the orchestrator needs to know).
"""

from __future__ import annotations

import threading
from contextlib import suppress
from typing import Any

from elasticsearch import Elasticsearch, helpers
from elasticsearch import exceptions as es_exc

from vendor_risk_ai.config import get_settings
from vendor_risk_ai.logging_setup import get_logger
from vendor_risk_ai.schemas.domain import ChunkRecord

log = get_logger(__name__)

EMBEDDING_DIMS = 1536  # text-embedding-3-small

QUOTES_MAPPING: dict[str, Any] = {
    "properties": {
        "ticker": {"type": "keyword"},
        "exchange": {"type": "keyword"},
        "currency": {"type": "keyword"},
        "name": {"type": "text", "fields": {"keyword": {"type": "keyword", "ignore_above": 256}}},
        "price": {"type": "float"},
        "change": {"type": "float"},
        "change_pct": {"type": "float"},
        "previous_close": {"type": "float"},
        "market_cap": {"type": "float"},
        "pe_ratio": {"type": "float"},
        "dividend_yield": {"type": "float"},
        "day_range": {"type": "keyword"},
        "year_range": {"type": "keyword"},
        "scraped_at": {"type": "date"},
        "source_url": {"type": "keyword"},
    }
}

CHUNKS_MAPPING: dict[str, Any] = {
    "properties": {
        "vendor_id": {"type": "long"},
        "document_id": {"type": "long"},
        "chunk_index": {"type": "integer"},
        "title": {"type": "text"},
        "doc_type": {"type": "keyword"},
        "text": {"type": "text"},
        "embedding": {
            "type": "dense_vector",
            "dims": EMBEDDING_DIMS,
            "index": True,
            "similarity": "cosine",
        },
        "embedding_model": {"type": "keyword"},
    }
}

_INDEX_SETTINGS: dict[str, Any] = {"number_of_shards": 1, "number_of_replicas": 0}

_es: Elasticsearch | None = None
_es_lock = threading.Lock()


def get_es() -> Elasticsearch:
    """Process-wide client (lazy, thread-safe)."""
    global _es
    if _es is None:
        with _es_lock:
            if _es is None:
                _es = Elasticsearch(
                    get_settings().elastic_url,
                    request_timeout=5,
                    max_retries=1,
                    retry_on_timeout=False,
                )
    return _es


def reset_es() -> None:
    """Drop the singleton client (tests / config reload)."""
    global _es
    with _es_lock:
        client, _es = _es, None
    if client is not None:
        with suppress(Exception):
            client.close()


def _create_if_absent(es: Elasticsearch, name: str, mappings: dict[str, Any]) -> bool:
    if es.indices.exists(index=name):
        return False
    try:
        es.indices.create(index=name, mappings=mappings, settings=_INDEX_SETTINGS)
    except es_exc.BadRequestError as exc:
        if getattr(exc, "error", "") == "resource_already_exists_exception":
            return False  # lost a creation race (e.g. with the crawler) -> fine
        raise
    log.info("es.index.created", index=name)
    return True


def ensure_indices() -> None:
    """Create ``market-quotes`` and ``vendor-chunks`` with explicit mappings if absent."""
    s = get_settings()
    es = get_es()
    _create_if_absent(es, s.elastic_quotes_index, QUOTES_MAPPING)
    _create_if_absent(es, s.elastic_chunks_index, CHUNKS_MAPPING)


def chunk_doc_id(vendor_id: int, chunk: ChunkRecord) -> str:
    return f"{vendor_id}:{chunk.document_id}:{chunk.chunk_index}"


def index_chunks(vendor_id: int, chunks: list[ChunkRecord], *, refresh: bool = False) -> int:
    """Bulk-index chunks (idempotent ids); returns the number indexed (0 if ES is unavailable).

    Embeddings whose length differs from :data:`EMBEDDING_DIMS` are omitted (text still indexed)
    because ``dense_vector`` would reject the whole document.
    """
    if not chunks:
        return 0
    index = get_settings().elastic_chunks_index
    actions: list[dict[str, Any]] = []
    skipped_vectors = 0
    for c in chunks:
        src: dict[str, Any] = {
            "vendor_id": vendor_id,
            "document_id": c.document_id,
            "chunk_index": c.chunk_index,
            "title": c.title,
            "doc_type": c.doc_type,
            "text": c.text,
            "embedding_model": c.embedding_model,
        }
        if c.embedding is not None:
            if len(c.embedding) == EMBEDDING_DIMS:
                src["embedding"] = c.embedding
            else:
                skipped_vectors += 1
        actions.append({"_index": index, "_id": chunk_doc_id(vendor_id, c), "_source": src})
    if skipped_vectors:
        log.warning(
            "es.index_chunks.embedding_dims_mismatch",
            skipped=skipped_vectors,
            expected=EMBEDDING_DIMS,
        )
    try:
        ok, errors = helpers.bulk(
            get_es(), actions, raise_on_error=False, refresh="wait_for" if refresh else False
        )
    except (es_exc.TransportError, es_exc.ApiError) as exc:
        log.warning("es.index_chunks.failed", vendor_id=vendor_id, error=str(exc)[:200])
        return 0
    if errors:
        log.warning("es.index_chunks.partial", vendor_id=vendor_id, errors=len(errors))
    return int(ok)


def _quote_query(ticker: str) -> dict[str, Any]:
    t = ticker.strip()
    should: list[dict[str, Any]] = [{"term": {"ticker": t}}]
    if ":" in t:
        sym, exch = t.split(":", 1)
        should.append(
            {"bool": {"filter": [{"term": {"ticker": sym}}, {"term": {"exchange": exch}}]}}
        )
    return {"bool": {"should": should, "minimum_should_match": 1}}


def latest_quote(ticker: str) -> dict | None:
    """Newest ``market-quotes`` document for ``ticker`` ("IBM:NYSE" or "IBM"); never raises."""
    if not ticker or not ticker.strip():
        return None
    try:
        resp = get_es().search(
            index=get_settings().elastic_quotes_index,
            query=_quote_query(ticker),
            sort=[{"scraped_at": {"order": "desc", "unmapped_type": "date"}}],
            size=1,
        )
        hits = resp["hits"]["hits"]
        return dict(hits[0]["_source"]) if hits else None
    except es_exc.NotFoundError:
        return None
    except Exception as exc:
        log.warning("es.latest_quote.failed", ticker=ticker, error=str(exc)[:200])
        return None


def ping() -> bool:
    try:
        return bool(get_es().ping())
    except Exception:
        return False
