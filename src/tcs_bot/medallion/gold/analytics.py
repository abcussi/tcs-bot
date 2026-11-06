"""Gold analytics (AC-005.8): cosine matrix, k-NN similarity graph, clusters, centrality, PCA,
per-cluster summaries and dashboard statistics. Pure numpy / scikit-learn / networkx."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Hashable, Sequence
from typing import Any

import networkx as nx
import numpy as np

from tcs_bot.logging_setup import get_logger
from tcs_bot.medallion.models import RISK_CATEGORIES, ClusterSummary, SilverRecord

log = get_logger(__name__)

# Terms every TCS document shares: useless as cluster labels.
DOMAIN_STOP_WORDS = frozenset(
    {"tcs", "tata", "consultancy", "services", "said", "says", "company", "ltd", "limited"}
)


# ---------------------------------------------------------------------- similarity
def l2_normalize(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return np.divide(X, norms, out=np.zeros_like(X), where=norms > 0)


def cosine_matrix(X: np.ndarray) -> np.ndarray:
    """Symmetric ``(n, n)`` cosine similarity, diagonal 1 (zero vectors: 0 off-diagonal)."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] == 0:
        return np.zeros((0, 0))
    Xn = l2_normalize(X)
    S = np.clip(Xn @ Xn.T, -1.0, 1.0)
    S = (S + S.T) / 2.0
    np.fill_diagonal(S, 1.0)
    return S


def top_neighbors(S: np.ndarray, k: int) -> np.ndarray:
    """Indices ``(n, min(k, n-1))`` of each row's most similar *other* rows.

    Sorted by cosine desc, ties by index asc (deterministic)."""
    n = S.shape[0]
    k = max(0, min(k, n - 1))
    if k == 0:
        return np.zeros((n, 0), dtype=int)
    masked = S.copy()
    np.fill_diagonal(masked, -np.inf)
    order = np.lexsort((np.broadcast_to(np.arange(n), (n, n)), -masked), axis=1)
    return order[:, :k]


def knn_graph(
    X: np.ndarray, ids: Sequence[Hashable], k: int, min_cosine: float, S: np.ndarray | None = None
) -> nx.Graph:
    """Symmetric k-NN similarity graph.

    Edge ``u-v`` iff ``cosine(u, v) >= min_cosine`` and ``v`` is among ``u``'s top-``k``
    neighbours or vice versa. Edge attrs: ``cosine``, ``weight`` (cosine clamped > 0, for
    Louvain), ``mutual`` (each is in the other's top-k). Every id is a node; no self loops.
    """
    if S is None:
        S = cosine_matrix(X)
    G = nx.Graph()
    G.add_nodes_from(ids)
    nbrs = top_neighbors(S, k)
    sets = [set(row.tolist()) for row in nbrs]
    for i, row in enumerate(nbrs):
        for j in row.tolist():
            cos = float(S[i, j])
            if cos < min_cosine or i == j:
                continue
            u, v = ids[i], ids[j]
            if G.has_edge(u, v):
                continue
            G.add_edge(u, v, cosine=cos, weight=max(cos, 1e-6), mutual=i in sets[j])
    return G


# ---------------------------------------------------------------------- clustering
def louvain_communities(G: nx.Graph, seed: int = 42) -> dict[Hashable, int]:
    """Node -> community id (0 = largest community)."""
    if G.number_of_nodes() == 0:
        return {}
    if G.number_of_edges() == 0:
        comms: list[set] = [{n} for n in G.nodes]
    else:
        comms = nx.community.louvain_communities(G, weight="weight", seed=seed)
    comms = sorted(comms, key=lambda c: (-len(c), min(str(x) for x in c)))
    return {node: cid for cid, comm in enumerate(comms) for node in comm}


def _relabel_by_size(labels: np.ndarray) -> np.ndarray:
    counts = Counter(labels.tolist())
    # largest first; ties by first appearance
    first = {lab: i for i, lab in reversed(list(enumerate(labels.tolist())))}
    order = sorted(counts, key=lambda lab: (-counts[lab], first[lab]))
    mapping = {old: new for new, old in enumerate(order)}
    return np.array([mapping[lab] for lab in labels.tolist()], dtype=int)


def kmeans_clusters(
    X: np.ndarray, k_range: Sequence[int] | None = None, seed: int = 42
) -> tuple[np.ndarray, dict[str, Any]]:
    """KMeans on L2-normalized vectors, ``k`` chosen by the best cosine silhouette.

    Returns ``(labels, info)`` with ``info = {"k", "silhouette", "scores": {k: score}}``.
    Labels are renumbered so cluster 0 is the largest. ``n < 3`` -> a single cluster.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    X = l2_normalize(np.asarray(X, dtype=np.float64))
    n = X.shape[0]
    single = (np.zeros(n, dtype=int), {"k": 1 if n else 0, "silhouette": None, "scores": {}})
    if n < 3:
        return single
    distinct = len({row.tobytes() for row in np.round(X, 10)})
    ks = [k for k in (k_range or range(2, 13)) if 2 <= k <= min(n - 1, distinct)]
    best: tuple[float, int, np.ndarray] | None = None
    scores: dict[str, float] = {}
    for k in ks:
        labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(X)
        if len(set(labels.tolist())) < 2:
            continue
        score = float(silhouette_score(X, labels, metric="cosine"))
        scores[str(k)] = round(score, 4)
        if best is None or score > best[0] + 1e-9:
            best = (score, k, labels)
    if best is None:
        return single
    score, k, labels = best
    return _relabel_by_size(labels), {"k": k, "silhouette": round(score, 4), "scores": scores}


# ---------------------------------------------------------------------- graph metrics
def centrality(G: nx.Graph) -> dict[Hashable, dict[str, float]]:
    """Degree, betweenness (cosine distance) and eigenvector centrality per node."""
    if G.number_of_nodes() == 0:
        return {}
    degree = nx.degree_centrality(G) if G.number_of_nodes() > 1 else dict.fromkeys(G.nodes, 0.0)
    for _, _, d in G.edges(data=True):
        d["distance"] = 1.0 - d.get("cosine", 0.0)
    n = G.number_of_nodes()
    betweenness = nx.betweenness_centrality(
        G, weight="distance", k=min(n, 300) if n > 300 else None, seed=42
    )
    try:
        if G.number_of_edges() == 0:
            raise nx.NetworkXException("no edges")
        eigen = nx.eigenvector_centrality(G, max_iter=2000, tol=1e-6, weight="weight")
    except (nx.NetworkXException, nx.PowerIterationFailedConvergence):
        eigen = dict(degree)
    return {
        node: {
            "degree": float(degree.get(node, 0.0)),
            "betweenness": float(betweenness.get(node, 0.0)),
            "eigenvector": float(eigen.get(node, 0.0)),
        }
        for node in G.nodes
    }


def project_2d(X: np.ndarray) -> np.ndarray:
    """PCA to 2-D, scaled into [-1, 1] (deterministic, zero-padded for degenerate input)."""
    from sklearn.decomposition import PCA

    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    out = np.zeros((n, 2))
    if n < 2:
        return out
    n_comp = min(2, n, X.shape[1])
    proj = PCA(n_components=n_comp, random_state=42).fit_transform(X)
    out[:, :n_comp] = proj
    scale = np.abs(out).max()
    return out / scale if scale > 0 else out


# ---------------------------------------------------------------------- summaries
def _tfidf(texts: Sequence[str]) -> tuple[Any, np.ndarray]:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

    vec = TfidfVectorizer(
        sublinear_tf=True,
        ngram_range=(1, 2),
        stop_words=list(ENGLISH_STOP_WORDS | DOMAIN_STOP_WORDS),
        max_df=0.9 if len(texts) >= 3 else 1.0,
        min_df=1,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9]+\b",
    )
    try:
        M = vec.fit_transform(list(texts))
    except ValueError:  # empty vocabulary
        return None, np.array([])
    return M, vec.get_feature_names_out()


def cluster_top_terms(texts: Sequence[str], labels: Sequence[int], n: int = 8) -> dict[int, list]:
    """Distinctive TF-IDF terms per cluster (cluster mean minus half the global mean)."""
    M, names = _tfidf(texts)
    labels_arr = np.asarray(labels)
    if M is None or len(names) == 0:
        return {int(c): [] for c in set(labels_arr.tolist())}
    global_mean = np.asarray(M.mean(axis=0)).ravel()
    out: dict[int, list[str]] = {}
    for c in sorted(set(labels_arr.tolist())):
        rows = np.flatnonzero(labels_arr == c)
        mean = np.asarray(M[rows].mean(axis=0)).ravel()
        score = mean - 0.5 * global_mean if len(set(labels_arr.tolist())) > 1 else mean
        picked: list[str] = []
        for i in np.argsort(-score):
            if mean[i] <= 0 or len(picked) >= n:
                break
            term = str(names[i])
            # skip a unigram already covered by a picked bigram and vice versa
            if any(term in p.split() or p in term.split() for p in picked):
                continue
            picked.append(term)
        out[int(c)] = picked
    return out


def cluster_summaries(
    X: np.ndarray,
    labels: Sequence[int],
    records: Sequence[SilverRecord],
    texts: Sequence[str],
    S: np.ndarray | None = None,
    n_terms: int = 8,
    n_representative: int = 3,
) -> list[ClusterSummary]:
    labels_arr = np.asarray(labels, dtype=int)
    if len(labels_arr) == 0:
        return []
    Xn = l2_normalize(X)
    if S is None:
        S = cosine_matrix(X)
    terms = cluster_top_terms(texts, labels_arr, n_terms)
    out: list[ClusterSummary] = []
    for c in sorted(set(labels_arr.tolist())):
        rows = np.flatnonzero(labels_arr == c)
        size = len(rows)
        cats: Counter[str] = Counter(cat for i in rows for cat in records[i].risk_categories)
        if size > 1:
            sub = S[np.ix_(rows, rows)]
            intra = float((sub.sum() - np.trace(sub)) / (size * (size - 1)))
        else:
            intra = 1.0
        centroid = Xn[rows].mean(axis=0)
        sims = Xn[rows] @ centroid
        rep = [records[rows[i]].doc_id for i in np.argsort(-sims, kind="stable")[:n_representative]]
        top = terms.get(int(c), [])
        out.append(
            ClusterSummary(
                cluster=int(c),
                size=size,
                label=" / ".join(top[:3]) or f"cluster {c}",
                top_terms=top,
                dominant_categories=dict(cats.most_common()),
                mean_risk_score=round(float(np.mean([records[i].risk_score for i in rows])), 2),
                mean_intra_cosine=round(intra, 4),
                representative_doc_ids=rep,
            )
        )
    return out


# ---------------------------------------------------------------------- dashboard stats
def cosine_histogram(S: np.ndarray, bins: int = 20) -> dict[str, list]:
    """Histogram of pairwise (upper-triangle) cosines on ``[min(0, min cos), 1]``."""
    n = S.shape[0]
    vals = S[np.triu_indices(n, k=1)] if n > 1 else np.array([])
    lo = float(min(0.0, vals.min())) if vals.size else 0.0
    counts, edges = np.histogram(vals, bins=bins, range=(lo, 1.0))
    return {"edges": [round(float(e), 4) for e in edges], "counts": counts.tolist()}


def category_counts(records: Sequence[SilverRecord]) -> dict[str, int]:
    c = Counter(cat for r in records for cat in r.risk_categories)
    return {cat: c.get(cat, 0) for cat in RISK_CATEGORIES}


def category_cooccurrence(records: Sequence[SilverRecord]) -> dict[str, Any]:
    idx = {cat: i for i, cat in enumerate(RISK_CATEGORIES)}
    M = np.zeros((len(idx), len(idx)), dtype=int)
    for r in records:
        cats = sorted({idx[c] for c in r.risk_categories if c in idx})
        for a in cats:
            for b in cats:
                M[a, b] += 1
    return {"categories": list(RISK_CATEGORIES), "matrix": M.tolist()}


def source_breakdown(records: Sequence[SilverRecord]) -> dict[str, dict[str, int]]:
    return {
        "by_source": dict(Counter(r.source for r in records).most_common()),
        "by_source_type": dict(Counter(r.source_type for r in records).most_common()),
        "by_domain": dict(Counter(r.domain for r in records).most_common(15)),
    }


def monthly_timeline(records: Sequence[SilverRecord]) -> list[dict[str, Any]]:
    by_month: dict[str, list[SilverRecord]] = defaultdict(list)
    for r in records:
        if r.published_at is not None:
            by_month[r.published_at.strftime("%Y-%m")].append(r)
    return [
        {
            "month": m,
            "count": len(rs),
            "mean_risk": round(float(np.mean([r.risk_score for r in rs])), 2),
            "by_category": dict(Counter(c for r in rs for c in r.risk_categories).most_common()),
        }
        for m, rs in sorted(by_month.items())
    ]


def top_risky(records: Sequence[SilverRecord], n: int = 15) -> list[dict[str, Any]]:
    ranked = sorted(records, key=lambda r: (-r.risk_score, r.doc_id))[:n]
    return [
        {
            "doc_id": r.doc_id,
            "title": r.title,
            "url": r.url,
            "domain": r.domain,
            "risk_score": r.risk_score,
            "risk_categories": list(r.risk_categories),
            "published_at": r.published_at.isoformat() if r.published_at else None,
        }
        for r in ranked
    ]


def top_cross_source_pairs(
    S: np.ndarray, records: Sequence[SilverRecord], n: int = 20
) -> list[dict[str, Any]]:
    """Most similar document pairs coming from different domains (same story, other outlet)."""
    m = S.shape[0]
    if m < 2:
        return []
    iu, ju = np.triu_indices(m, k=1)
    vals = S[iu, ju]
    out: list[dict[str, Any]] = []
    for p in np.argsort(-vals, kind="stable"):
        a, b = records[iu[p]], records[ju[p]]
        if a.domain == b.domain and a.source == b.source:
            continue
        out.append(
            {
                "a": a.doc_id,
                "b": b.doc_id,
                "a_title": a.title,
                "b_title": b.title,
                "a_domain": a.domain,
                "b_domain": b.domain,
                "cosine": round(float(vals[p]), 4),
            }
        )
        if len(out) >= n:
            break
    return out
