"""Gold embedder + analytics (AC-005.7, AC-005.8) on the SYNTHETIC 3-topic corpus."""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from tcs_bot.config import Settings
from tcs_bot.medallion.gold import analytics as A
from tcs_bot.medallion.gold.embedder import LsaEmbedder, get_gold_embedder, model_slug
from tcs_bot.medallion.gold.pipeline import doc_text
from tcs_bot.medallion.models import SilverRecord

from .test_gold_synthetic import TOPIC_NAMES, synthetic_records, topic_of


@pytest.fixture(scope="module")
def records() -> list[SilverRecord]:
    return synthetic_records()


@pytest.fixture(scope="module")
def texts(records: list[SilverRecord]) -> list[str]:
    return [doc_text(r) for r in records]


@pytest.fixture(scope="module")
def X(texts: list[str]) -> np.ndarray:
    return LsaEmbedder(dim=64).fit(texts).embed(texts)


# ---------------------------------------------------------------------- embedder
def test_lsa_is_deterministic_normalized_and_padded(texts: list[str]) -> None:
    """AC-005.7: LSA vectors are float32, L2-normalized, zero-padded to dim, reproducible."""
    a = LsaEmbedder(dim=64).fit(texts)
    b = LsaEmbedder(dim=64).fit(texts)
    Xa, Xb = a.embed(texts), b.embed(texts)
    assert Xa.dtype == np.float32 and Xa.shape == (30, 64)
    np.testing.assert_array_equal(Xa, Xb)
    np.testing.assert_allclose(np.linalg.norm(Xa, axis=1), 1.0, atol=1e-5)
    assert a.n_components == 29  # min(dim, n_features - 1, n_docs - 1)
    assert not Xa[:, a.n_components :].any()
    assert a.model == "lsa-64" and model_slug(a) == "lsa_64"


def test_lsa_query_embedding_and_out_of_vocabulary(texts: list[str]) -> None:
    """AC-005.7: queries land near their topic; unknown words give a zero vector."""
    emb = LsaEmbedder(dim=64).fit(texts)
    q = emb.embed(["ransomware breach exfiltrated passwords"])[0]
    sims = emb.embed(texts) @ q
    best = np.argsort(-sims)[:5]
    assert all("cyber" in texts[i] for i in best)
    assert not emb.embed(["zzzqx"]).any()
    assert "ransomware" in emb.top_terms("ransomware attack zzzqx", 5)
    assert emb.component_terms(0, 5)


def test_lsa_save_load_roundtrip(tmp_path: Path, texts: list[str]) -> None:
    """AC-005.7: the persisted LSA model embeds queries identically (used by search)."""
    emb = LsaEmbedder(dim=64).fit(texts)
    loaded = LsaEmbedder.load(emb.save(tmp_path / "lsa.pkl"))
    np.testing.assert_array_equal(emb.embed(texts[:3]), loaded.embed(texts[:3]))
    assert loaded.model == emb.model and loaded.dim == 64


@pytest.mark.parametrize("n", [0, 1, 2])
def test_lsa_tiny_corpora(n: int, texts: list[str]) -> None:
    """AC-005.8: 0/1/2 documents never crash; vectors keep the configured dim."""
    emb = LsaEmbedder(dim=16).fit(texts[:n])
    out = emb.embed(texts[:3])
    assert out.shape == (3, 16)
    assert emb.embed([]).shape == (0, 16)


def test_lsa_empty_vocabulary_falls_back_to_hash() -> None:
    emb = LsaEmbedder(dim=16).fit(["the and of", "a an the"])
    assert emb.embed(["anything"]).shape == (1, 16)


def test_lsa_embed_before_fit_raises() -> None:
    with pytest.raises(RuntimeError):
        LsaEmbedder(8).embed(["x"])


def test_get_gold_embedder_offline_is_lsa(settings: Settings, texts: list[str]) -> None:
    """AC-005.7: no OpenAI key -> offline LSA with settings.lsa_dim."""
    emb = get_gold_embedder(settings, texts)
    assert isinstance(emb, LsaEmbedder) and emb.dim == settings.lsa_dim


def test_get_gold_embedder_openai_when_configured(texts: list[str]) -> None:
    """AC-005.7: OpenAI text-embedding-3-small when configured (constructed only, no call)."""
    from tcs_bot.rag.embeddings import OpenAIEmbedder

    s = Settings(llm_provider="openai", openai_api_key="sk-test-not-real")
    emb = get_gold_embedder(s, texts)
    assert isinstance(emb, OpenAIEmbedder)
    assert emb.model == "text-embedding-3-small" and emb.dim == 1536
    assert model_slug(emb) == "text_embedding_3_small"


# ---------------------------------------------------------------------- analytics
def test_cosine_matrix_symmetric_unit_diagonal(X: np.ndarray) -> None:
    """AC-005.8: numpy cosine matrix is symmetric with a unit diagonal, values in [-1, 1]."""
    S = A.cosine_matrix(X)
    assert S.shape == (30, 30)
    np.testing.assert_allclose(S, S.T)
    np.testing.assert_allclose(np.diag(S), 1.0)
    assert S.max() <= 1.0 and S.min() >= -1.0
    assert A.cosine_matrix(np.zeros((0, 4))).shape == (0, 0)
    Z = A.cosine_matrix(np.array([[0.0, 0.0], [1.0, 0.0]]))
    assert Z[0, 1] == 0.0 and Z[0, 0] == 1.0


@pytest.mark.parametrize(("k", "thr"), [(3, 0.0), (5, 0.3), (8, 0.5)])
def test_knn_graph_respects_k_and_threshold(
    X: np.ndarray, records: list[SilverRecord], k: int, thr: float
) -> None:
    """AC-005.8: every edge has cosine >= threshold and is in one endpoint's top-k."""
    ids = [r.doc_id for r in records]
    S = A.cosine_matrix(X)
    G = A.knn_graph(X, ids, k, thr)
    assert set(G.nodes) == set(ids)
    assert nx.number_of_selfloops(G) == 0
    top = A.top_neighbors(S, k)
    topk = {ids[i]: {ids[j] for j in row} for i, row in enumerate(top)}
    for u, v, d in G.edges(data=True):
        assert d["cosine"] >= thr
        assert v in topk[u] or u in topk[v]
        assert d["mutual"] == (v in topk[u] and u in topk[v])
    # each node *selects* at most k neighbours
    for u in ids:
        assert len([v for v in G[u] if v in topk[u]]) <= k
    if thr == 0.0:  # no threshold pruning: every node has >= 1 edge
        assert min(dict(G.degree).values()) >= 1


def test_knn_graph_edges_stay_within_topics(X: np.ndarray, records: list[SilverRecord]) -> None:
    G = A.knn_graph(X, [r.doc_id for r in records], 5, 0.3)
    topic = {r.doc_id: topic_of(r) for r in records}
    assert G.number_of_edges() > 0
    assert all(topic[u] == topic[v] for u, v in G.edges)


def test_kmeans_and_louvain_recover_the_three_topics(
    X: np.ndarray, records: list[SilverRecord]
) -> None:
    """AC-005.8: KMeans (k by silhouette) and Louvain both recover the 3 synthetic topics."""
    truth = [topic_of(r) for r in records]
    labels, info = A.kmeans_clusters(X)
    assert info["k"] == 3 and info["silhouette"] > 0.2
    assert adjusted_rand_score(truth, labels) >= 0.8
    assert np.bincount(labels).tolist() == sorted(np.bincount(labels).tolist(), reverse=True)
    ids = [r.doc_id for r in records]
    comms = A.louvain_communities(A.knn_graph(X, ids, 8, 0.3))
    assert adjusted_rand_score(truth, [comms[i] for i in ids]) >= 0.8


@pytest.mark.parametrize("n", [0, 1, 2])
def test_kmeans_small_n_single_cluster(X: np.ndarray, n: int) -> None:
    labels, info = A.kmeans_clusters(X[:n])
    assert labels.tolist() == [0] * n and info["silhouette"] is None


def test_kmeans_identical_vectors() -> None:
    labels, info = A.kmeans_clusters(np.ones((5, 4)))
    assert set(labels.tolist()) == {0} and info["k"] == 1


def test_centrality_and_projection(X: np.ndarray, records: list[SilverRecord]) -> None:
    """AC-005.8: degree/betweenness/eigenvector per node; PCA 2-D within [-1, 1]."""
    ids = [r.doc_id for r in records]
    cent = A.centrality(A.knn_graph(X, ids, 8, 0.3))
    assert set(cent) == set(ids)
    assert all(set(v) == {"degree", "betweenness", "eigenvector"} for v in cent.values())
    lonely = nx.Graph()
    lonely.add_nodes_from([1, 2])
    assert A.centrality(lonely)[1]["eigenvector"] == 0.0  # fallback, no exception
    P = A.project_2d(X)
    assert P.shape == (30, 2) and np.abs(P).max() == pytest.approx(1.0)
    assert A.project_2d(X[:1]).shape == (1, 2)
    assert A.project_2d(X[:2]).shape == (2, 2)


def test_cluster_summaries(X: np.ndarray, records: list[SilverRecord], texts: list[str]) -> None:
    """AC-005.8: top terms, dominant categories, mean risk, intra cosine, representatives."""
    labels, _ = A.kmeans_clusters(X)
    sums = A.cluster_summaries(X, labels, records, texts)
    assert [s.cluster for s in sums] == [0, 1, 2]
    assert sum(s.size for s in sums) == 30
    by_topic = {}
    for s in sums:
        assert len(s.top_terms) >= 3 and s.label == " / ".join(s.top_terms[:3])
        assert 0 < s.mean_intra_cosine <= 1 and 0 <= s.mean_risk_score <= 100
        assert len(s.representative_doc_ids) == 3
        rep_topics = {topic_of(r) for r in records if r.doc_id in s.representative_doc_ids}
        assert len(rep_topics) == 1
        by_topic[rep_topics.pop()] = s
    assert set(by_topic) == set(TOPIC_NAMES)
    assert next(iter(by_topic["cyber"].dominant_categories)) == "data_privacy_cyber"
    assert next(iter(by_topic["legal"].dominant_categories)) == "legal_litigation"
    assert by_topic["cyber"].mean_risk_score > by_topic["labour"].mean_risk_score
    assert not {"tcs", "synthetic"} & set(by_topic["labour"].top_terms)


def test_dashboard_stats(X: np.ndarray, records: list[SilverRecord]) -> None:
    """AC-005.8: histogram, co-occurrence, sources, timeline, top risky, cross-source pairs."""
    S = A.cosine_matrix(X)
    hist = A.cosine_histogram(S, bins=10)
    assert sum(hist["counts"]) == 30 * 29 // 2 and len(hist["edges"]) == 11
    co = A.category_cooccurrence(records)
    i, j = co["categories"].index("labor_hr"), co["categories"].index("reputational")
    assert co["matrix"][i][j] == co["matrix"][j][i] == 10
    assert A.category_counts(records)["data_privacy_cyber"] == 10
    src = A.source_breakdown(records)
    assert sum(src["by_source"].values()) == 30 and set(src["by_source_type"]) == {
        "news",
        "community",
    }
    tl = A.monthly_timeline(records)
    assert sum(m["count"] for m in tl) == 30 and tl == sorted(tl, key=lambda m: m["month"])
    risky = A.top_risky(records)
    assert len(risky) == 15
    assert [r["risk_score"] for r in risky] == sorted((r["risk_score"] for r in risky), reverse=True)
    pairs = A.top_cross_source_pairs(S, records)
    assert len(pairs) == 20
    assert all(p["a_domain"] != p["b_domain"] for p in pairs)
    assert [p["cosine"] for p in pairs] == sorted((p["cosine"] for p in pairs), reverse=True)
    assert A.top_cross_source_pairs(S[:1, :1], records[:1]) == []
