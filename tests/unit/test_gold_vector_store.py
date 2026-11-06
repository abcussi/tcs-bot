"""Gold Elasticsearch kNN store on an in-memory ES fake (AC-005.7, offline)."""

from __future__ import annotations

import numpy as np
import pytest

from tcs_bot.errors import AppError, DependencyUnavailable, NotFoundError
from tcs_bot.medallion.gold.embedder import LsaEmbedder, model_slug
from tcs_bot.medallion.gold.pipeline import build_chunks, doc_text
from tcs_bot.medallion.gold.vector_store import (
    MAX_DIMS,
    VECTOR_FIELD,
    GoldVectorStore,
    gold_mapping,
    point_id,
)
from tcs_bot.medallion.models import SilverRecord

from .fake_es import DownES, FakeES
from .test_gold_synthetic import TOPIC_NAMES, synthetic_records, topic_of


@pytest.fixture
def records() -> list[SilverRecord]:
    return synthetic_records()


@pytest.fixture
def setup(records, settings):
    texts = [doc_text(r) for r in records]
    emb = LsaEmbedder(dim=32).fit(texts)
    vs = GoldVectorStore(es=FakeES(), model_slug=model_slug(emb), prefix="gold-tcs-risk")
    vs.ensure(emb.dim)
    chunks = build_chunks(records, settings)
    extra = {r.doc_id: {"cluster": TOPIC_NAMES.index(topic_of(r))} for r in records}
    vs.upsert_docs(records, emb.embed(texts), extra)
    vs.upsert_chunks(chunks, emb.embed([c.text for c in chunks]), extra)
    return vs, emb, chunks


def test_index_names_include_model() -> None:
    """AC-005.7: one index pair per embedding model (no mixed vector spaces), valid ES names."""
    vs = GoldVectorStore(es=FakeES(), model_slug="lsa_256", prefix="gold-tcs-risk")
    assert vs.docs_collection == "gold-tcs-risk-docs--lsa_256"
    assert vs.chunks_collection == "gold-tcs-risk-chunks--lsa_256"
    other = vs.with_model("text_embedding_3_small")
    assert other.es is vs.es
    assert other.chunks_collection == "gold-tcs-risk-chunks--text_embedding_3_small"
    assert vs.with_model("lsa_256") is vs
    with pytest.raises(ValueError, match="prefix"):
        GoldVectorStore(es=FakeES(), prefix="Bad Prefix")


def test_mapping_is_dense_vector_cosine_and_strict() -> None:
    """AC-005.7: indexed cosine dense_vector, dynamic false, typed filter fields."""
    m = gold_mapping(256)
    assert m["dynamic"] is False
    assert m["properties"][VECTOR_FIELD] == {
        "type": "dense_vector",
        "dims": 256,
        "index": True,
        "similarity": "cosine",
        "index_options": {"type": "hnsw"},
    }
    props = m["properties"]
    assert props["risk_categories"]["type"] == props["source"]["type"] == "keyword"
    assert props["published_at"]["type"] == "date" and props["title"]["type"] == "text"


def test_ensure_creates_dim_checked_indices_and_is_idempotent() -> None:
    vs = GoldVectorStore(es=FakeES(), prefix="gold-tcs-risk")
    vs.ensure(8)
    vs.ensure(8)  # second call is a no-op
    mapping = vs.es.indices.get_mapping(vs.docs_collection)[vs.docs_collection]["mappings"]
    assert mapping["properties"][VECTOR_FIELD]["dims"] == 8
    with pytest.raises(ValueError, match="dim"):
        vs.ensure(9)
    for bad in (0, MAX_DIMS + 1):
        with pytest.raises(ValueError, match="range"):
            vs.ensure(bad)
    vs.ensure(MAX_DIMS, recreate=True)  # the ES 8.15 maximum is accepted (OpenAI 1536 fits)


def test_upsert_count_and_idempotent_ids(setup, records) -> None:
    """AC-005.7: deterministic ids -> re-upserting does not duplicate; writes are refreshed."""
    vs, emb, chunks = setup
    assert vs.count("docs") == 30 and vs.count("chunks") == len(chunks) >= 30
    assert vs.es.calls.count("indices.refresh") >= 2  # one per upsert -> read-your-write
    vs.upsert_docs(records, emb.embed([doc_text(r) for r in records]))
    assert vs.count("docs") == 30
    assert point_id("docs", 1) == point_id("docs", 1) == "1"
    assert point_id("chunks", 1, 0) == "1:0" != point_id("docs", 1)
    s = vs.stats()
    assert s["backend"] == "elasticsearch" and s["docs"] == 30 and s["chunks"] == len(chunks)


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
    assert all(-1.0 <= x.score <= 1.0 for x in hits)
    docs = vs.search(emb.embed(["jury verdict"])[0], k=3, level="docs")
    assert all(d.chunk_index is None for d in docs)


def test_search_score_is_cosine() -> None:
    """AC-005.7: GoldHit.score is the cosine similarity recovered from the ES score."""
    recs = synthetic_records()[:3]
    vs = GoldVectorStore(es=FakeES(), prefix="gold-tcs-risk")
    vs.ensure(2)
    vs.upsert_docs(recs, np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]))
    hits = vs.search([1.0, 0.0], k=3, level="docs")
    assert [h.doc_id for h in hits] == [r.doc_id for r in recs]
    assert [round(h.score, 4) for h in hits] == [1.0, round(2**-0.5, 4), 0.0]


def test_search_filters(setup) -> None:
    """AC-005.9: category, min-risk, source, cluster filters run inside the kNN."""
    vs, emb, _ = setup
    q = emb.embed(["employees layoffs salary"])[0]
    cyber = vs.search(q, k=50, categories=["data_privacy_cyber"])
    assert cyber and all("data_privacy_cyber" in h.risk_categories for h in cyber)
    assert vs.es.last_knn["filter"] == {
        "bool": {"filter": [{"terms": {"risk_categories": ["data_privacy_cyber"]}}]}
    }
    risky = vs.search(q, k=50, min_risk=70)
    assert risky and all(h.risk_score >= 70 for h in risky)
    assert vs.search(q, k=50, categories=["esg"]) == []
    in_cluster = vs.search(q, k=50, cluster=1)
    assert in_cluster and all(h.cluster == 1 for h in in_cluster)
    assert vs.search(q, k=5, sources=["no-such-source"]) == []
    combined = vs.search(q, k=50, categories=["data_privacy_cyber"], min_risk=70, cluster=1)
    assert all(h.risk_score >= 70 and h.cluster == 1 for h in combined)


def test_knn_request_shape(setup) -> None:
    vs, emb, _ = setup
    vs.search(emb.embed(["breach"])[0], k=7)
    knn = vs.es.last_knn
    assert knn["field"] == VECTOR_FIELD and knn["k"] == 7 and knn["num_candidates"] >= 7
    assert "filter" not in knn
    vs.search(emb.embed(["breach"])[0], k=100_000)  # clamped, never an invalid request
    assert vs.es.last_knn["k"] <= vs.es.last_knn["num_candidates"] <= 10000


def test_zero_vector_dim_mismatch_and_drop(setup) -> None:
    vs, emb, _ = setup
    assert vs.search(np.zeros(emb.dim), k=5) == []
    assert vs.search(emb.embed(["x"])[0], k=0) == []
    with pytest.raises(ValueError, match="dim"):
        vs.ensure(emb.dim + 1)
    vs.ensure(emb.dim + 1, recreate=True)
    assert vs.count("docs") == 0
    vs.drop()
    assert vs.count("docs") == 0


def test_vector_count_mismatch_raises(records) -> None:
    vs = GoldVectorStore(es=FakeES(), prefix="gold-tcs-risk")
    vs.ensure(4)
    with pytest.raises(ValueError):
        vs.upsert_docs(records[:2], np.ones((3, 4)))


def test_per_document_failures_are_reported(records) -> None:
    """AC-005.7: zero vectors / ES item errors are per-doc failures, the rest is written."""
    vs = GoldVectorStore(es=FakeES(), prefix="gold-tcs-risk")
    vs.ensure(3)
    recs = records[:4]
    vecs = np.array([[1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    vs.es.fail_ids = {point_id("docs", recs[3].doc_id)}
    assert vs.upsert_docs(recs, vecs) == 2 and vs.count("docs") == 2
    assert vs.last_failed_ids == {recs[1].doc_id, recs[3].doc_id}
    vs.es.fail_ids = set()
    assert vs.upsert_docs(recs[:1], vecs[:1]) == 1 and vs.last_failed_ids == set()
    assert vs.upsert_docs(recs[:2], np.ones((2, 5))) == 0  # ES rejects the vector length per item
    assert vs.last_failed_ids == {recs[0].doc_id, recs[1].doc_id}


def test_delete_docs_removes_docs_and_chunks(setup) -> None:
    vs, _, chunks = setup
    first_two = sorted({c.record.doc_id for c in chunks})[:2]
    n_chunks = sum(1 for c in chunks if c.record.doc_id in first_two)
    assert vs.delete_docs(first_two) == 2 + n_chunks
    assert vs.count("docs") == 28 and vs.count("chunks") == len(chunks) - n_chunks
    assert vs.delete_docs([]) == 0


def test_missing_index_semantics() -> None:
    """Search before gold is built is a NotFoundError (404); count is 0."""
    vs = GoldVectorStore(es=FakeES(), prefix="gold-tcs-risk")
    assert vs.count("docs") == 0 and vs.delete_docs([1]) == 0
    with pytest.raises(NotFoundError) as exc:
        vs.search([1.0, 0.0], k=3)
    assert exc.value.code == "gold_not_built"


def test_elasticsearch_down_fails_loudly_with_app_error(records) -> None:
    """Constitution §6/§8: ES down -> DependencyUnavailable (503 envelope), never silent."""
    vs = GoldVectorStore(es=DownES(), prefix="gold-tcs-risk")  # type: ignore[arg-type]
    calls = [
        lambda: vs.ensure(4),
        lambda: vs.upsert_docs(records[:1], np.ones((1, 4))),
        lambda: vs.search([1.0, 0.0, 0.0, 0.0], k=3),
        lambda: vs.count(),
        lambda: vs.drop(),
        lambda: vs.delete_docs([1]),
    ]
    for call in calls:
        with pytest.raises(DependencyUnavailable) as exc:
            call()
        assert isinstance(exc.value, AppError)
        assert exc.value.code == "vector_store_unavailable" and exc.value.status_code == 503
