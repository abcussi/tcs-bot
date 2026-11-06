"""Qdrant gold store (AC-005.7): document + chunk collections, one pair per embedding model.

Collections: ``{prefix}_docs__{model_slug}`` and ``{prefix}_chunks__{model_slug}`` (cosine), so
vectors from different embedding models never share a space. Point ids are deterministic UUID5s
of ``(kind, doc_id, chunk_index)``: re-running gold overwrites instead of duplicating.
"""

from __future__ import annotations

import uuid
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from tcs_bot.logging_setup import get_logger
from tcs_bot.medallion.models import GoldHit, SilverRecord

if TYPE_CHECKING:
    from qdrant_client import QdrantClient

log = get_logger(__name__)

DEFAULT_PREFIX = "gold_tcs_risk"
PAYLOAD_TEXT_MAX = 1000
_BATCH = 256
_NS = uuid.UUID("6f1c2a8e-9b51-4b2e-8d0c-3a7f5e0d2c11")  # fixed namespace for point ids

Level = Literal["docs", "chunks"]


@dataclass(frozen=True)
class ChunkRef:
    """One chunk of a silver document (what ``upsert_chunks`` stores)."""

    record: SilverRecord
    chunk_index: int
    text: str


def point_id(kind: Level, doc_id: int, chunk_index: int | None = None) -> str:
    return str(uuid.uuid5(_NS, f"{kind}:{doc_id}:{'' if chunk_index is None else chunk_index}"))


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


class GoldVectorStore:
    def __init__(
        self,
        client: QdrantClient | None = None,
        url: str | None = None,
        model_slug: str = "lsa_256",
        prefix: str = DEFAULT_PREFIX,
        timeout: int = 10,
    ) -> None:
        if client is None:
            from qdrant_client import QdrantClient

            if url is None:
                from tcs_bot.config import get_settings

                url = get_settings().qdrant_url
            client = QdrantClient(url=url, timeout=timeout, check_compatibility=False)
        self.client = client
        self.model_slug = model_slug
        self.prefix = prefix

    def with_model(self, model_slug: str) -> GoldVectorStore:
        """Same client / prefix, collections of another embedding model."""
        if model_slug == self.model_slug:
            return self
        return GoldVectorStore(client=self.client, model_slug=model_slug, prefix=self.prefix)

    # ------------------------------------------------------------------ collections
    def collection(self, level: Level) -> str:
        return f"{self.prefix}_{level}__{self.model_slug}"

    @property
    def docs_collection(self) -> str:
        return self.collection("docs")

    @property
    def chunks_collection(self) -> str:
        return self.collection("chunks")

    def ensure(self, dim: int, recreate: bool = False) -> None:
        from qdrant_client import models

        for name in (self.docs_collection, self.chunks_collection):
            exists = self.client.collection_exists(name)
            if exists and recreate:
                self.client.delete_collection(name)
                exists = False
            if exists:
                size = self.client.get_collection(name).config.params.vectors.size  # type: ignore[union-attr]
                if size != dim:
                    raise ValueError(f"collection {name} has dim {size}, expected {dim}")
                continue
            self.client.create_collection(
                name, vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE)
            )
            for field, schema in (
                ("doc_id", models.PayloadSchemaType.INTEGER),
                ("risk_categories", models.PayloadSchemaType.KEYWORD),
                ("risk_score", models.PayloadSchemaType.FLOAT),
                ("source_type", models.PayloadSchemaType.KEYWORD),
            ):
                try:  # local (:memory:) mode has no payload indexes; harmless there
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning)
                        self.client.create_payload_index(name, field, field_schema=schema)
                except Exception as exc:  # pragma: no cover - server-side best effort
                    log.debug("gold.qdrant.payload_index_skipped", field=field, error=str(exc))
            log.info("gold.qdrant.collection_created", collection=name, dim=dim)

    def drop(self) -> None:
        for name in (self.docs_collection, self.chunks_collection):
            if self.client.collection_exists(name):
                self.client.delete_collection(name)

    # ------------------------------------------------------------------ writes
    def _upsert(self, collection: str, ids: list[str], vectors: np.ndarray, payloads: list) -> int:
        from qdrant_client import models

        vectors = np.asarray(vectors, dtype=np.float32)
        if len(ids) != len(vectors):
            raise ValueError(f"{len(ids)} points but {len(vectors)} vectors")
        for start in range(0, len(ids), _BATCH):
            points = [
                models.PointStruct(id=pid, vector=vec.tolist(), payload=pl)
                for pid, vec, pl in zip(
                    ids[start : start + _BATCH],
                    vectors[start : start + _BATCH],
                    payloads[start : start + _BATCH],
                    strict=True,
                )
            ]
            self.client.upsert(collection, points=points, wait=True)
        return len(ids)

    def upsert_docs(
        self,
        records: Sequence[SilverRecord],
        vectors: np.ndarray,
        extra_payload: Mapping[int, Mapping[str, Any]] | None = None,
    ) -> int:
        extra_payload = extra_payload or {}
        ids = [point_id("docs", r.doc_id) for r in records]
        payloads = [
            _payload(r, None, r.summary or r.clean_text, extra_payload.get(r.doc_id))
            for r in records
        ]
        return self._upsert(self.docs_collection, ids, vectors, payloads)

    def upsert_chunks(
        self,
        chunks: Sequence[ChunkRef],
        vectors: np.ndarray,
        extra_payload: Mapping[int, Mapping[str, Any]] | None = None,
    ) -> int:
        extra_payload = extra_payload or {}
        ids = [point_id("chunks", c.record.doc_id, c.chunk_index) for c in chunks]
        payloads = [
            _payload(c.record, c.chunk_index, c.text, extra_payload.get(c.record.doc_id))
            for c in chunks
        ]
        return self._upsert(self.chunks_collection, ids, vectors, payloads)

    # ------------------------------------------------------------------ reads
    def search(
        self,
        vector: np.ndarray | Sequence[float],
        k: int = 10,
        categories: Sequence[str] | None = None,
        min_risk: float | None = None,
        level: Level = "chunks",
    ) -> list[GoldHit]:
        from qdrant_client import models

        vec = np.asarray(vector, dtype=np.float32).ravel()
        if k <= 0 or not np.any(vec):  # zero vector (e.g. out-of-vocabulary query)
            return []
        must: list[Any] = []
        if categories:
            must.append(
                models.FieldCondition(
                    key="risk_categories", match=models.MatchAny(any=list(categories))
                )
            )
        if min_risk is not None:
            must.append(models.FieldCondition(key="risk_score", range=models.Range(gte=min_risk)))
        res = self.client.query_points(
            self.collection(level),
            query=vec.tolist(),
            limit=k,
            query_filter=models.Filter(must=must) if must else None,
            with_payload=True,
        )
        hits: list[GoldHit] = []
        for p in res.points:
            pl = p.payload or {}
            hits.append(
                GoldHit(
                    doc_id=int(pl["doc_id"]),
                    chunk_index=pl.get("chunk_index"),
                    score=float(p.score),
                    title=pl.get("title", ""),
                    url=pl.get("url", ""),
                    text=pl.get("text", ""),
                    risk_categories=list(pl.get("risk_categories") or []),
                    risk_score=float(pl.get("risk_score") or 0),
                    cluster=pl.get("cluster"),
                )
            )
        return hits

    def count(self, level: Level = "chunks") -> int:
        name = self.collection(level)
        if not self.client.collection_exists(name):
            return 0
        return int(self.client.count(name, exact=True).count)
