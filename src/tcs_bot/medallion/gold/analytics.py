"""Gold analytics (AC-005.8): cosine matrix, k-NN similarity graph, clusters, centrality, PCA,
per-cluster summaries and dashboard statistics. Pure numpy / scikit-learn / networkx."""

from __future__ import annotations

import re
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
# News / forum boilerplate and contraction fragments that float to the top of TF-IDF labels.
# fmt: off
BOILERPLATE_STOP_WORDS = frozenset((
        "according", "added", "also", "announced", "april", "article", "articles", "ask", "august",
        "best", "better", "big", "billion", "cent", "click", "com", "comment", "comments",
        "continue", "copyright", "could", "crore", "december", "decided", "decides", "decision",
        "did", "didn", "does", "doesn", "don", "edit", "expected", "february", "firm", "firms",
        "first", "following", "friday", "get", "gets", "going", "good", "group", "hacker", "hn",
        "http", "https", "image", "important", "inc", "including", "india", "indian", "isn",
        "january", "july", "june", "just", "know", "known", "lakh", "largest", "like", "likely",
        "link", "links", "made", "make", "makes", "many", "march", "may", "maybe", "means",
        "million", "monday", "month", "months", "more", "most", "much", "need", "net", "news",
        "newsletter", "november", "now", "number", "october", "often", "one", "only", "org",
        "other", "others", "page", "pages", "part", "people", "per", "percent", "photo", "point",
        "points", "post", "posted", "rapidly", "read", "really", "recent", "recently", "record",
        "report", "reported", "reporter", "reports", "role", "rs", "said", "saturday", "say",
        "says", "see", "seen", "september", "share", "shares", "since", "site", "slightly", "still",
        "subscribe", "sunday", "sure", "take", "taken", "takes", "thing", "things", "think",
        "though", "thousand", "three", "thursday", "time", "times", "today", "told", "took",
        "track", "tuesday", "two", "used", "using", "video", "view", "want", "wasn", "way",
        "wednesday", "week", "weeks", "well", "went", "were", "weren", "whether", "while",
        "wikipedia", "won", "would", "www", "year", "years", "yet",
))
# fmt: on
# Human-readable names used to prefix a cluster label with its dominant risk category.
CATEGORY_NAMES = {
    "ethics": "Ethics",
    "compliance_regulatory": "Compliance",
    "legal_litigation": "Legal",
    "data_privacy_cyber": "Cyber/privacy",
    "labor_hr": "Labour/HR",
    "financial": "Financial",
    "operational_delivery": "Operations",
    "reputational": "Reputation",
    "esg": "ESG",
    "geopolitical_immigration": "Immigration",
}
CATEGORY_LABEL_MIN_SHARE = 0.5
MIN_CLUSTER_SIZE = 2
MAX_K = 12
SILHOUETTE_TOLERANCE = 0.03


def suggest_k(n: int) -> int:
    """Number of clusters that scales with the corpus: ``clamp(round(sqrt(n / 2)), 2, 12)``."""
    if n < 3:
        return 1 if n else 0
    return int(min(MAX_K, max(2, round((n / 2) ** 0.5))))


def graph_params(n: int, k: int, min_cosine: float) -> tuple[int, float]:
    """Effective ``(k, min_cosine)`` for a corpus of ``n`` documents.

    ``k`` (the configured *maximum*) is capped at ``round(sqrt(n))`` (at least 2) so a 25-document
    corpus gets a sparse, readable graph rather than a near-clique. The threshold is used as is.
    """
    return max(1, min(k, max(2, round(n**0.5)), max(n - 1, 1))), min_cosine


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
def _renumber(comms: list[set]) -> dict[Hashable, int]:
    comms = sorted((c for c in comms if c), key=lambda c: (-len(c), min(str(x) for x in c)))
    return {node: cid for cid, comm in enumerate(comms) for node in comm}


def merge_small_communities(
    G: nx.Graph, comms: list[set], min_size: int = MIN_CLUSTER_SIZE
) -> list[set]:
    """Fold communities smaller than ``min_size`` into the neighbouring community they are
    most strongly linked to (sum of edge weights). Nodes with no edge at all stay alone."""
    comms = [set(c) for c in comms]
    where = {n: i for i, c in enumerate(comms) for n in c}
    for i in sorted(range(len(comms)), key=lambda i: len(comms[i])):
        if not comms[i] or len(comms[i]) >= min_size:
            continue
        pull: Counter[int] = Counter()
        for u in comms[i]:
            for v, d in G[u].items():
                if where[v] != i:
                    pull[where[v]] += d.get("weight", 1.0)
        if not pull:
            continue
        tgt = max(pull, key=lambda c: (pull[c], -c))
        for u in comms[i]:
            where[u] = tgt
        comms[tgt] |= comms[i]
        comms[i] = set()
    return [c for c in comms if c]


def louvain_communities(
    G: nx.Graph, seed: int = 42, resolution: float = 1.0, min_size: int = MIN_CLUSTER_SIZE
) -> dict[Hashable, int]:
    """Node -> community id (0 = largest community).

    ``resolution`` < 1 favours fewer, larger communities. Communities smaller than ``min_size``
    are merged into their most strongly connected neighbour (isolated nodes stay singletons).
    """
    if G.number_of_nodes() == 0:
        return {}
    if G.number_of_edges() == 0:
        comms: list[set] = [{n} for n in G.nodes]
    else:
        comms = [
            set(c)
            for c in nx.community.louvain_communities(
                G, weight="weight", seed=seed, resolution=resolution
            )
        ]
        comms = merge_small_communities(G, comms, min_size)
    return _renumber(comms)


def _relabel_by_size(labels: np.ndarray) -> np.ndarray:
    counts = Counter(labels.tolist())
    # largest first; ties by first appearance
    first = {lab: i for i, lab in reversed(list(enumerate(labels.tolist())))}
    order = sorted(counts, key=lambda lab: (-counts[lab], first[lab]))
    mapping = {old: new for new, old in enumerate(order)}
    return np.array([mapping[lab] for lab in labels.tolist()], dtype=int)


def merge_small_clusters(
    X: np.ndarray, labels: np.ndarray, min_size: int = MIN_CLUSTER_SIZE
) -> np.ndarray:
    """Reassign members of clusters smaller than ``min_size`` to the nearest (cosine of
    centroids) cluster that is big enough. No-op when no cluster is big enough."""
    labels = np.asarray(labels, dtype=int).copy()
    counts = Counter(labels.tolist())
    big = [c for c, m in counts.items() if m >= min_size]
    small = [c for c, m in counts.items() if m < min_size]
    if not big or not small:
        return _relabel_by_size(labels)
    Xn = l2_normalize(X)
    cents = {c: l2_normalize(Xn[labels == c].mean(axis=0, keepdims=True))[0] for c in big}
    for c in small:
        for i in np.flatnonzero(labels == c):
            labels[i] = max(big, key=lambda b: (float(Xn[i] @ cents[b]), -b))
    return _relabel_by_size(labels)


def kmeans_clusters(
    X: np.ndarray,
    k_range: Sequence[int] | None = None,
    seed: int = 42,
    min_size: int = MIN_CLUSTER_SIZE,
) -> tuple[np.ndarray, dict[str, Any]]:
    """KMeans on L2-normalized vectors, ``k`` chosen by the best cosine silhouette.

    Without ``k_range`` the candidates run from ``suggest_k(n) - 1`` to ``round(sqrt(n)) + 2``
    (capped at 12), so ``k`` scales with the corpus. Silhouette alone keeps preferring more
    clusters, so the *smallest* ``k`` within ``SILHOUETTE_TOLERANCE`` of the best score wins.
    Clusters smaller than ``min_size`` are folded into the nearest cluster; the reported
    silhouette is that of the final labels. Returns ``(labels, info)`` with
    ``info = {"k", "silhouette", "scores": {k: score}, "k_target", "n_merged"}``. Labels are
    renumbered so cluster 0 is the largest. ``n < 3`` -> a single cluster.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    X = l2_normalize(np.asarray(X, dtype=np.float64))
    n = X.shape[0]
    single = (
        np.zeros(n, dtype=int),
        {"k": 1 if n else 0, "silhouette": None, "scores": {}, "k_target": suggest_k(n)},
    )
    if n < 3:
        return single
    target = suggest_k(n)
    if k_range is None:
        k_range = range(max(2, target - 1), min(MAX_K, round(n**0.5) + 2) + 1)
    distinct = len({row.tobytes() for row in np.round(X, 10)})
    ks = [k for k in k_range if 2 <= k <= min(n - 1, distinct)]
    cands: list[tuple[float, int, np.ndarray]] = []
    scores: dict[str, float] = {}
    for k in ks:
        labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(X)
        if len(set(labels.tolist())) < 2:
            continue
        merged = merge_small_clusters(X, labels, min_size)
        if len(set(merged.tolist())) < 2:
            continue
        score = float(silhouette_score(X, merged, metric="cosine"))
        scores[str(k)] = round(score, 4)
        cands.append((score, k, labels))
    if not cands:
        return single
    top = max(c[0] for c in cands)
    score, k, raw = min(
        (c for c in cands if c[0] >= top - SILHOUETTE_TOLERANCE), key=lambda c: c[1]
    )
    labels = merge_small_clusters(X, raw, min_size)
    info = {
        "k": len(set(labels.tolist())),
        "k_requested": k,
        "silhouette": round(score, 4),
        "scores": scores,
        "k_target": target,
        "n_merged": sum(m for m in Counter(raw.tolist()).values() if m < min_size),
    }
    return labels, info


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
_URL = re.compile(r"https?://\S+|www\.\S+")
_H1B = re.compile(r"\bh[- ]?1[- ]?b\b", re.I)
_US = re.compile(r"\bu\.s\.?(?=\W|$)", re.I)


def _prep_text(text: str) -> str:
    """Lower-case and glue tokens the default tokenizer would split (``H-1B``, ``U.S.``)."""
    text = _URL.sub(" ", text or "")
    text = _H1B.sub("h1b", text)
    return _US.sub("us", text).lower()


def _stem(term: str) -> str:
    return " ".join(t[:-1] if len(t) > 3 and t.endswith("s") else t for t in term.split())


def _term_matrix(texts: Sequence[str]) -> tuple[Any, np.ndarray]:
    """Per-document term weights: sublinear tf, L1-normalised so long pages don't dominate."""
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

    n = len(texts)
    vec = CountVectorizer(
        ngram_range=(1, 2),
        stop_words=list(ENGLISH_STOP_WORDS | DOMAIN_STOP_WORDS | BOILERPLATE_STOP_WORDS),
        max_df=0.7 if n >= 5 else 1.0,
        min_df=1,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9]{2,}\b",
        preprocessor=_prep_text,
    )
    try:
        C = vec.fit_transform(list(texts)).astype(np.float64)
    except ValueError:  # empty vocabulary
        return None, np.array([])
    C.data = 1.0 + np.log(C.data)
    row_sum = np.asarray(C.sum(axis=1)).ravel()
    row_sum[row_sum == 0] = 1.0
    from scipy.sparse import diags

    return diags(1.0 / row_sum) @ C, vec.get_feature_names_out()


def _pick_terms(
    names: np.ndarray, score: np.ndarray, picked: list[str], limit: int, *, unigrams_only: bool
) -> None:
    """Append the best-scoring terms to ``picked`` (in place) up to ``limit``, skipping a term
    that overlaps one already picked (a unigram inside a bigram, ``visa`` / ``visas``)."""
    stems = [set(_stem(t).split()) for t in picked]
    for i in np.argsort(-score, kind="stable"):
        if score[i] <= 0 or len(picked) >= limit:
            return
        term = str(names[i])
        if term in picked or (unigrams_only and " " in term):
            continue
        toks = set(_stem(term).split())
        if any(toks <= s or s <= toks for s in stems):
            continue
        picked.append(term)
        stems.append(toks)


def cluster_top_terms(texts: Sequence[str], labels: Sequence[int], n: int = 8) -> dict[int, list]:
    """Terms that best *discriminate* each cluster (class-based TF-IDF).

    A term scores ``mean tf in the cluster * idf * share of its mass held by this cluster``. It
    must occur in at least ``min(2, cluster size)`` documents of the cluster, which drops the
    one-off bigrams a single long page produces; stop words cover news boilerplate.
    """
    M, names = _term_matrix(texts)
    labels_arr = np.asarray(labels)
    cluster_ids = sorted(set(labels_arr.tolist()))
    if M is None or len(names) == 0:
        return {int(c): [] for c in cluster_ids}
    N = M.shape[0]
    df = np.asarray((M > 0).sum(axis=0)).ravel()
    idf = np.log((N + 1.0) / (df + 1.0)) + 1.0
    means = {
        c: np.asarray(M[np.flatnonzero(labels_arr == c)].mean(axis=0)).ravel() for c in cluster_ids
    }
    total = np.sum(list(means.values()), axis=0)
    out: dict[int, list[str]] = {}
    for c in cluster_ids:
        rows = np.flatnonzero(labels_arr == c)
        support = np.asarray((M[rows] > 0).sum(axis=0)).ravel()
        share = np.divide(means[c], total, out=np.ones_like(total), where=total > 0)
        score = means[c] * idf * share
        strict = score.copy()
        strict[support < min(2, len(rows))] = 0.0
        picked: list[str] = []
        _pick_terms(names, strict, picked, n, unigrams_only=False)
        if len(picked) < min(3, n):  # sparse cluster: top up with single-document unigrams
            _pick_terms(names, score, picked, min(3, n), unigrams_only=True)
        out[int(c)] = picked
    return out


def dominant_category(r: SilverRecord) -> str | None:
    """Risk category carrying the most severity weight in the record's signals."""
    if not r.risk_signals:
        return r.risk_categories[0] if r.risk_categories else None
    weight = {"low": 1, "medium": 2, "high": 3, "critical": 4}
    score: dict[str, int] = {}
    for s in r.risk_signals:
        score[s.category] = score.get(s.category, 0) + weight[s.severity]
    return max(score, key=lambda c: (score[c], -list(score).index(c)))


def cluster_label(top_terms: Sequence[str], category: str | None, n_terms: int = 3) -> str:
    """``"Legal: trade secrets / jury"`` when a risk category dominates, else top terms."""
    if category and top_terms:
        return f"{CATEGORY_NAMES.get(category, category)}: " + " / ".join(top_terms[:2])
    return " / ".join(top_terms[:n_terms])


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
        dom: Counter[str] = Counter(
            d for i in rows if (d := dominant_category(records[i])) is not None
        )
        top_cat = None
        if dom:
            cat, count = dom.most_common(1)[0]
            if count / size >= CATEGORY_LABEL_MIN_SHARE and (size > 1 or len(dom) == 1):
                top_cat = cat
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
                label=cluster_label(top, top_cat) or f"cluster {c}",
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
