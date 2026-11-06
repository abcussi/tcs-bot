"""Gold Qdrant store with an in-process ``QdrantClient(":memory:")`` (AC-005.7, offline)."""

from __future__ import annotations

import numpy as np
import pytest
from qdrant_client import QdrantClient

from tcs_bot.medallion.gold.embedder import LsaEmbedder, model_slug
from tcs_bot.medallion.gold.pipeline import build_chunks, doc_text
from tcs_bot.medallion.gold.vector_store import GoldVectorStore, point_id
from tcs_bot.medallion.models import SilverRecord

from .test_gold_synthetic import TOPIC_NAMES, synthetic_records, topic_of


@pytest.fixture
def records() -> list[SilverRecord]:
    return synthetic_records()


@pytest.fixture
def setup(records, settings):
    texts = [doc_text(r) for r in records]
    emb = LsaEmbedder(dim=32).fit(texts)
    vs = GoldVectorStore(client=QdrantClient(":memory:"), model_slug=model_slug(emb))
    vs.ensure(emb.dim)
    chunks = build_chunks(records, settings)
    extra = {r.doc_id: {"cluster": TOPIC_NAMES.index(topic_of(r))} for r in records}
    vs.upsert_docs(records, emb.embed(texts), extra)
    vs.upsert_chunks(chunks, emb.embed([c.text for c in chunks]), extra)
    return vs, emb, chunks


def test_collection_names_include_model(records) -> None:
    """AC-005.7: one collection pair per embedding model (no mixed vector spaces)."""
    vs = GoldVectorStore(client=QdrantClient(":memory:"), model_slug="lsa_256")
    assert vs.docs_collection == "gold_tcs_risk_docs__lsa_256"
    assert vs.chunks_collection == "gold_tcs_risk_chunks__lsa_256"
    other = vs.with_model("text_embedding_3_small")
    assert other.client is vs.client
    assert other.chunks_collection == "gold_tcs_risk_chunks__text_embedding_3_small"


def test_upsert_count_and_idempotent_ids(setup, records) -> None:
    """AC-005.7: deterministic UUID5 point ids -> re-upserting does not duplicate."""
    vs, emb, chunks = setup
    assert vs.count("docs") == 30 and vs.count("chunks") == len(chunks) >= 30
    vs.upsert_docs(records, emb.embed([doc_text(r) for r in records]))
    assert vs.count("docs") == 30
    assert point_id("docs", 1) == point_id("docs", 1) != point_id("chunks", 1, 0)


def test_search_returns_topic_hits_with_payload(setup) -> None:
    """AC-005.7/9: chunk search ranks the right topic first; payload maps to GoldHit."""
    vs, emb, _ = setup
    hits = vs.search(emb.embed(["jury verdict trade secrets damages"])[0], k=5)
    assert len(hits) == 5
    assert all("/legal/" in h.url for h in hits)
    h = hits[0]
    assert h.chunk_index is not None and h.text and h.title.startswith("[SYNTHETIC]")
    assert "legal_litigation" in h.risk_categories and h.cluster is not None
    assert hits == sorted(hits, key=lambda x: -x.score)
    docs = vs.search(emb.embed(["jury verdict"])[0], k=3, level="docs")
    assert all(d.chunk_index is None for d in docs)


def test_search_filters(setup) -> None:
    """AC-005.9: category and min-risk payload filters."""
    vs, emb, _ = setup
    q = emb.embed(["employees layoffs salary"])[0]
    cyber = vs.search(q, k=50, categories=["data_privacy_cyber"])
    assert cyber and all("data_privacy_cyber" in h.risk_categories for h in cyber)
    risky = vs.search(q, k=50, min_risk=70)
    assert risky and all(h.risk_score >= 70 for h in risky)
    assert vs.search(q, k=50, categories=["esg"]) == []


def test_zero_vector_and_dim_mismatch(setup) -> None:
    vs, emb, _ = setup
    assert vs.search(np.zeros(emb.dim), k=5) == []
    with pytest.raises(ValueError, match="dim"):
        vs.ensure(emb.dim + 1)
    vs.ensure(emb.dim + 1, recreate=True)
    assert vs.count("docs") == 0
    vs.drop()
    assert vs.count("docs") == 0


def test_vector_count_mismatch_raises(records) -> None:
    vs = GoldVectorStore(client=QdrantClient(":memory:"))
    vs.ensure(4)
    with pytest.raises(ValueError):
        vs.upsert_docs(records[:2], np.ones((3, 4)))
