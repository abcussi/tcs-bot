"""Elasticsearch gold store (AC-005.7): document + chunk indices, one pair per embedding model.

Indices: ``{prefix}-docs--{model_slug}`` and ``{prefix}-chunks--{model_slug}`` (default prefix
``gold-tcs-risk``), each with a ``dense_vector`` field (``index=true``, cosine) searched through
the ES ``knn`` API, so vectors from different embedding models never share a space. Document ids
are deterministic (``doc_id`` for docs, ``doc_id:chunk_index`` for chunks): re-running gold
overwrites instead of duplicating, and every write is followed by a refresh so a search right
after an upsert sees it.

Elasticsearch failures surface as :class:`~tcs_bot.errors.DependencyUnavailable`
(``vector_store_unavailable``); per-document bulk failures are returned as a smaller count and
listed in :attr:`GoldVectorStore.last_failed_ids`.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from elasticsearch import exceptions as es_exc

from tcs_bot.errors import DependencyUnavailable, NotFoundError
from tcs_bot.logging_setup import get_logger
from tcs_bot.medallion.models import GoldHit, SilverRecord

if TYPE_CHECKING:
    from elasticsearch import Elasticsearch

log = get_logger(__name__)

DEFAULT_PREFIX = "gold-tcs-risk"
VECTOR_FIELD = "embedding"
MAX_DIMS = 4096  # ES 8.15 limit for an indexed dense_vector (1024 before 8.11)
PAYLOAD_TEXT_MAX = 1000
MAX_K = 1000
_BATCH = 256
_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]*$")

Level = Literal["docs", "chunks"]

_KW = {"type": "keyword"}
GOLD_SETTINGS: dict[str, Any] = {"number_of_shards": 1, "number_of_replicas": 0}


def gold_mapping(dim: int) -> dict[str, Any]:
    """Index mapping for ``dim``-dimensional vectors (``dynamic: false`` like silver)."""
    return {
        "dynamic": False,  # unknown payload fields stay in _source but are not indexed
        "properties": {
            "doc_id": {"type": "long"},
            "chunk_index": {"type": "integer"},
            "title": {
                "type": "text",
                "fields": {"keyword": {"type": "keyword", "ignore_above": 512}},
            },
            "url": {"type": "keyword", "ignore_above": 2048},
            "domain": _KW,
            "source": _KW,
            "source_type": _KW,
            "published_at": {"type": "date"},
            "risk_categories": _KW,
            "risk_score": {"type": "float"},
            "sentiment": {"type": "float"},
            "cluster": {"type": "integer"},
            "community": {"type": "integer"},
            "text": {"type": "text"},
            VECTOR_FIELD: {
                "type": "dense_vector",
                "dims": dim,
                "index": True,
                "similarity": "cosine",
                # full-precision HNSW: ES 8.15 defaults to int8 quantization, which reorders
                # near-ties; the gold corpus is small, so exact scores beat the memory saving
                "index_options": {"type": "hnsw"},
            },
        },
    }


@dataclass(frozen=True)
class ChunkRef:
    """One chunk of a silver document (what ``upsert_chunks`` stores)."""

    record: SilverRecord
    chunk_index: int
    text: str


def point_id(kind: Level, doc_id: int, chunk_index: int | None = None) -> str:
    """Deterministic ES ``_id``: ``doc_id`` for docs, ``doc_id:chunk_index`` for chunks."""
    return str(doc_id) if kind == "docs" else f"{doc_id}:{chunk_index or 0}"


def _doc_id_of(es_id: str) -> int:
    try:
        return int(es_id.split(":", 1)[0])
    except ValueError:
        return -1


def _payload(
    rec: SilverRecord, chunk_index: int | None, text: str, extra: Mapping[str, Any] | None
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "doc_id": rec.doc_id,
        "chunk_index": chunk_index,
        "title": rec.title,
        "url": rec.url,
        "domain": rec.domain,
        "source": rec.source,
        "source_type": rec.source_type,
        "published_at": rec.published_at.isoformat() if rec.published_at else None,
        "risk_categories": list(rec.risk_categories),
        "risk_score": float(rec.risk_score),
        "sentiment": float(rec.sentiment),
        "cluster": None,
        "community": None,
        "text": text[:PAYLOAD_TEXT_MAX],
    }
    if extra:
        payload.update(extra)
    return payload


def _unavailable(op: str, exc: Exception) -> DependencyUnavailable:
    log.error("gold.vector_store_failed", op=op, error_type=type(exc).__name__)
    return DependencyUnavailable(
        f"Elasticsearch vector store {op} failed ({type(exc).__name__}); "
        "is Elasticsearch up and gold built? run `tcs-bot medallion run`",
        code="vector_store_unavailable",
    )


def _guard[T](op: str, fn: Callable[[], T]) -> T:
    """Run ``fn``; map Elasticsearch transport/API errors to ``DependencyUnavailable``."""
    try:
        return fn()
    except (es_exc.TransportError, es_exc.ApiError) as exc:
        raise _unavailable(op, exc) from exc


class GoldVectorStore:
    """Dense-vector store on Elasticsearch (the only gold vector backend).

    ``es`` defaults to the process-wide client (:func:`tcs_bot.search.elastic.get_es`).
    """

    def __init__(
        self,
        es: Elasticsearch | None = None,
        model_slug: str = "lsa_256",
        prefix: str | None = None,
    ) -> None:
        if prefix is None:
            from tcs_bot.config import get_settings

            prefix = get_settings().gold_index_prefix
        if not _NAME.match(prefix):
            raise ValueError(f"invalid gold index prefix {prefix!r}: lowercase letters/digits/-_.")
        if es is None:
            from tcs_bot.search.elastic import get_es

            es = get_es()
        self.es = es
        self.model_slug = model_slug
        self.prefix = prefix
        self.last_failed_ids: set[int] = set()

    def with_model(self, model_slug: str) -> GoldVectorStore:
        """Same client / prefix, indices of another embedding model."""
        if model_slug == self.model_slug:
            return self
        return GoldVectorStore(es=self.es, model_slug=model_slug, prefix=self.prefix)

    # ------------------------------------------------------------------ indices
    def collection(self, level: Level) -> str:
        return f"{self.prefix}-{level}--{self.model_slug}"

    @property
    def docs_collection(self) -> str:
        return self.collection("docs")

    @property
    def chunks_collection(self) -> str:
        return self.collection("chunks")

    def ensure(self, dim: int, recreate: bool = False) -> None:
        """Create both indices for ``dim``-dimensional vectors (``recreate`` drops them first).

        Raises ``ValueError`` for an unsupported ``dim`` or when an existing index has another
        dimension, and ``DependencyUnavailable`` when Elasticsearch fails.
        """
        if not 1 <= dim <= MAX_DIMS:
            raise ValueError(
                f"dim {dim} outside the Elasticsearch dense_vector range 1..{MAX_DIMS}"
            )
        for name in (self.docs_collection, self.chunks_collection):
            _guard("ensure", lambda name=name: self._ensure_one(name, dim, recreate))

    def _ensure_one(self, name: str, dim: int, recreate: bool) -> None:
        exists = bool(self.es.indices.exists(index=name))
        if exists and recreate:
            self.es.indices.delete(index=name, ignore_unavailable=True)
            exists = False
        if exists:
            props = self.es.indices.get_mapping(index=name)[name]["mappings"]["properties"]
            have = props.get(VECTOR_FIELD, {}).get("dims")
            if have != dim:
                raise ValueError(f"index {name} has dim {have}, expected {dim}")
            return
        try:
            self.es.indices.create(index=name, mappings=gold_mapping(dim), settings=GOLD_SETTINGS)
        except es_exc.BadRequestError as exc:
            if getattr(exc, "error", "") != "resource_already_exists_exception":
                raise
            return
        log.info("gold.index_created", index=name, dim=dim)

    def drop(self) -> None:
        """Delete both indices of this model (missing ones are ignored)."""
        for name in (self.docs_collection, self.chunks_collection):
            _guard(
                "drop",
                lambda name=name: self.es.indices.delete(index=name, ignore_unavailable=True),
            )

    def delete_docs(self, doc_ids: Sequence[int]) -> int:
        """Remove every doc and chunk of ``doc_ids`` (e.g. documents dropped by dedupe)."""
        ids = [int(i) for i in doc_ids]
        if not ids:
            return 0
        deleted = 0
        for name in (self.docs_collection, self.chunks_collection):

            def run(name: str = name) -> int:
                if not self.es.indices.exists(index=name):
                    return 0
                res = self.es.delete_by_query(
                    index=name, query={"terms": {"doc_id": ids}}, refresh=True
                )
                return int(res.get("deleted", 0))

            deleted += _guard("delete", run)
        return deleted

    # ------------------------------------------------------------------ writes
    def _upsert(
        self,
        index: str,
        ids: list[str],
        doc_ids: list[int],
        vectors: np.ndarray,
        payloads: list[dict[str, Any]],
    ) -> int:
        vectors = np.asarray(vectors, dtype=np.float32)
        if len(ids) != len(vectors):
            raise ValueError(f"{len(ids)} documents but {len(vectors)} vectors")
        if not ids:
            return 0
        failed: set[int] = set()
        sendable: list[int] = []
        norms = np.linalg.norm(vectors.reshape(len(ids), -1), axis=1)
        for i, norm in enumerate(norms):  # cosine dense_vector rejects zero-magnitude vectors
            if norm > 0 and np.isfinite(norm):
                sendable.append(i)
            else:
                failed.add(doc_ids[i])

        def run() -> int:
            written = 0
            for start in range(0, len(sendable), _BATCH):
                ops: list[dict[str, Any]] = []
                for i in sendable[start : start + _BATCH]:
                    ops.append({"index": {"_index": index, "_id": ids[i]}})
                    ops.append({**payloads[i], VECTOR_FIELD: vectors[i].tolist()})
                res = self.es.bulk(operations=ops)
                for item in res.get("items", []):
                    info = item.get("index", {})
                    if info.get("error") or int(info.get("status", 200)) >= 300:
                        failed.add(_doc_id_of(str(info.get("_id", ""))))
                        log.warning(
                            "gold.upsert.item_failed",
                            index=index,
                            id=info.get("_id"),
                            error=str(info.get("error"))[:200],
                        )
                    else:
                        written += 1
            self.es.indices.refresh(index=index)  # read-your-write for the next search
            return written

        count = _guard("upsert", run)
        self.last_failed_ids |= failed
        if failed:
            log.warning("gold.upsert.partial", index=index, failed=len(failed), written=count)
        return count

    def upsert_docs(
        self,
        records: Sequence[SilverRecord],
        vectors: np.ndarray,
        extra_payload: Mapping[int, Mapping[str, Any]] | None = None,
    ) -> int:
        """Bulk upsert one vector per document; returns the number written."""
        self.last_failed_ids = set()
        extra_payload = extra_payload or {}
        return self._upsert(
            self.docs_collection,
            [point_id("docs", r.doc_id) for r in records],
            [r.doc_id for r in records],
            vectors,
            [
                _payload(r, None, r.summary or r.clean_text, extra_payload.get(r.doc_id))
                for r in records
            ],
        )

    def upsert_chunks(
        self,
        chunks: Sequence[ChunkRef],
        vectors: np.ndarray,
        extra_payload: Mapping[int, Mapping[str, Any]] | None = None,
    ) -> int:
        """Bulk upsert one vector per chunk; returns the number written."""
        self.last_failed_ids = set()
        extra_payload = extra_payload or {}
        return self._upsert(
            self.chunks_collection,
            [point_id("chunks", c.record.doc_id, c.chunk_index) for c in chunks],
            [c.record.doc_id for c in chunks],
            vectors,
            [
                _payload(c.record, c.chunk_index, c.text, extra_payload.get(c.record.doc_id))
                for c in chunks
            ],
        )

    # ------------------------------------------------------------------ reads
    def search(
        self,
        vector: np.ndarray | Sequence[float],
        k: int = 10,
        categories: Sequence[str] | None = None,
        min_risk: float | None = None,
        level: Level = "chunks",
        *,
        sources: Sequence[str] | None = None,
        source_types: Sequence[str] | None = None,
        cluster: int | None = None,
        community: int | None = None,
    ) -> list[GoldHit]:
        """Top-``k`` nearest hits by cosine (``GoldHit.score`` is the cosine, not the ES score).

        Filters are applied inside the kNN search (exact filtering, not post-filtering):
        ``categories``/``sources``/``source_types`` match any value, ``min_risk`` is a lower bound
        on ``risk_score``, ``cluster``/``community`` match one id.
        """
        vec = np.asarray(vector, dtype=np.float32).ravel()
        if k <= 0 or not np.any(vec):  # zero vector (e.g. out-of-vocabulary query)
            return []
        k = min(int(k), MAX_K)
        filters: list[dict[str, Any]] = []
        for field, values in (
            ("risk_categories", categories),
            ("source", sources),
            ("source_type", source_types),
        ):
            if values:
                filters.append({"terms": {field: list(values)}})
        if min_risk is not None:
            filters.append({"range": {"risk_score": {"gte": min_risk}}})
        if cluster is not None:
            filters.append({"term": {"cluster": int(cluster)}})
        if community is not None:
            filters.append({"term": {"community": int(community)}})
        knn: dict[str, Any] = {
            "field": VECTOR_FIELD,
            "query_vector": vec.tolist(),
            "k": k,
            "num_candidates": min(max(k * 10, 100), 10000),
        }
        if filters:
            knn["filter"] = {"bool": {"filter": filters}}
        name = self.collection(level)
        try:
            res = self.es.search(index=name, knn=knn, size=k, source_excludes=[VECTOR_FIELD])
        except es_exc.NotFoundError as exc:
            raise NotFoundError(
                f"gold index {name} not found - run `tcs-bot medallion run`", code="gold_not_built"
            ) from exc
        except (es_exc.TransportError, es_exc.ApiError) as exc:
            raise _unavailable("search", exc) from exc
        hits: list[GoldHit] = []
        for h in res["hits"]["hits"]:
            pl = h.get("_source") or {}
            hits.append(
                GoldHit(
                    doc_id=int(pl["doc_id"]),
                    chunk_index=pl.get("chunk_index"),
                    score=2.0 * float(h["_score"]) - 1.0,  # ES cosine score is (1 + cos) / 2
                    title=pl.get("title") or "",
                    url=pl.get("url") or "",
                    text=pl.get("text") or "",
                    risk_categories=list(pl.get("risk_categories") or []),
                    risk_score=float(pl.get("risk_score") or 0),
                    cluster=pl.get("cluster"),
                )
            )
        return hits

    def count(self, level: Level = "chunks") -> int:
        """Number of documents in the index of ``level`` (0 when it does not exist)."""
        name = self.collection(level)

        def run() -> int:
            if not self.es.indices.exists(index=name):
                return 0
            return int(self.es.count(index=name)["count"])

        return _guard("count", run)

    def stats(self) -> dict[str, Any]:
        """Index names and document counts for this embedding model."""
        return {
            "backend": "elasticsearch",
            "model_slug": self.model_slug,
            "docs_index": self.docs_collection,
            "chunks_index": self.chunks_collection,
            "docs": self.count("docs"),
            "chunks": self.count("chunks"),
        }
