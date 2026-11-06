"""Gold result quality (AC-005.8): near-duplicate collapsing, corpus-scaled K / graph, no singleton
clusters, discriminative cluster labels. SYNTHETIC data only: every fixture text carries the marker
``SYNTHETIC FIXTURE`` and nothing describes a real event. No network, no infra."""

from __future__ import annotations

import json
import random
from datetime import UTC, datetime, timedelta
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
from qdrant_client import QdrantClient
from sklearn.metrics import adjusted_rand_score

from tcs_bot.config import Settings
from tcs_bot.medallion.gold import analytics as A
from tcs_bot.medallion.gold.dedupe import dedupe_records, normalize_title
from tcs_bot.medallion.gold.embedder import LsaEmbedder, target_components
from tcs_bot.medallion.gold.pipeline import EMBED_TITLE_WEIGHT, embedding_text, run_gold
from tcs_bot.medallion.gold.vector_store import GoldVectorStore
from tcs_bot.medallion.models import RiskSignal, SilverRecord

from .test_gold_synthetic import FakeSilverStore

MARKER = "SYNTHETIC FIXTURE"

# Invented vocabularies (no real entities): 4 well separated topics + words every document shares.
# fmt: off
POOLS: dict[str, list[str]] = {
    "orchard": ["apple", "orchard", "harvest", "frost", "grafting", "cider", "pruning",
                "blossom", "trellis", "pollination"],
    "harbor": ["tugboat", "harbor", "dredging", "pilot", "berth", "quayside", "ballast",
               "mooring", "tidal", "breakwater"],
    "foundry": ["foundry", "furnace", "casting", "alloy", "crucible", "smelting", "ingot",
                "forging", "slag", "annealing"],
    "observatory": ["telescope", "observatory", "nebula", "parallax", "spectrograph", "eclipse",
                    "aperture", "comet", "zenith"],
}
# fmt: on
CATEGORY = {
    "orchard": "ethics",
    "harbor": "operational_delivery",
    "foundry": "financial",
    "observatory": "esg",
}
COMMON = ["report", "said", "company", "people", "year", "week", "announced", "market"]


def _rec(
    doc_id: int,
    title: str,
    text: str,
    *,
    depth: int = 0,
    category: str = "ethics",
    published: datetime | None = None,
    source: str = "synthetic_news",
    domain: str = "synthetic-news.example",
) -> SilverRecord:
    body = f"{MARKER}. {text}"
    return SilverRecord(
        doc_id=doc_id,
        url=f"https://{domain}/{doc_id}",
        domain=domain,
        source=source,
        source_type="news",
        depth=depth,
        title=title,
        clean_text=body,
        summary=f"{MARKER} summary of {title}",
        word_count=len(body.split()),
        published_at=published,
        tcs_relevance=0.9,
        relevant=True,
        risk_signals=[
            RiskSignal(category=category, severity="high", keyword="k", evidence=f"{MARKER}.")
        ],
        risk_categories=[category],
        risk_score=50.0,
        extractor="synthetic-test-fixture",
    )


def topic_corpus(per_topic: int = 6, seed: int = 3) -> tuple[list[SilverRecord], list[str]]:
    """``len(POOLS) * per_topic`` distinct stories; returns records and their topic names."""
    rng = random.Random(seed)
    recs: list[SilverRecord] = []
    topics: list[str] = []
    doc_id = 100
    for i in range(per_topic):
        for name, pool in POOLS.items():
            doc_id += 1
            title_words = rng.sample(pool, 3)
            body = rng.sample(pool, 7) + rng.sample(COMMON, 3) + rng.sample(pool, 5)
            recs.append(
                _rec(
                    doc_id,
                    f"{' '.join(title_words)} variant{i}{name[:2]}",
                    " ".join(body),
                    category=CATEGORY[name],
                    domain=f"synthetic-{doc_id % 5}.example",
                )
            )
            topics.append(name)
    return recs, topics


# ---------------------------------------------------------------------- dedupe
def test_fixtures_are_marked_synthetic() -> None:
    """AC-005.8: every fixture in this module carries the SYNTHETIC FIXTURE marker."""
    recs, _ = topic_corpus()
    assert all(MARKER in r.clean_text and MARKER in r.summary for r in recs)
    assert all(r.url.split("/")[2].endswith(".example") for r in recs)


def test_normalize_title_strips_outlet_punctuation_and_broken_escapes() -> None:
    """AC-005.8: titles that differ only by outlet suffix / punctuation / encoding collapse."""
    base = normalize_title("Jury cuts damages in synthetic harbor dispute")
    assert normalize_title("Jury cuts damages in synthetic harbor dispute - Example Wire") == base
    assert normalize_title("JURY CUTS DAMAGES, in synthetic harbor dispute!") == base
    assert normalize_title("Owner says it�s a harbor") == normalize_title(
        "Owner says it\\u2019s a harbor"
    )
    # a short headline is kept whole: the part after the dash is content, not an outlet
    assert normalize_title("Harbor - closed") == "harbor closed"


def test_dedupe_keeps_richest_and_records_merged_ids() -> None:
    """AC-005.8: snippet (depth 0) + full page (depth 1), one title -> the full page survives."""
    title = "Furnace casting dispute settles after synthetic arbitration"
    snippet = _rec(1, title, "furnace casting dispute settles " * 3, depth=0)
    page = _rec(2, title, "furnace casting alloy dispute arbitration settles " * 40, depth=1)
    other = _rec(3, "Telescope aperture nebula survey", "telescope aperture nebula survey " * 10)
    res = dedupe_records([snippet, page, other])
    assert [r.doc_id for r in res.records] == [2, 3]  # input order of the kept docs
    assert {k: [r.doc_id for r in v] for k, v in res.merged.items()} == {2: [1]}
    assert res.n_removed == 1
    # silver objects are untouched
    assert snippet.relevant and snippet.doc_id == 1


def test_dedupe_is_transitive_and_deterministic() -> None:
    """AC-005.8: three copies collapse to one; equal richness -> lowest doc_id wins, any order."""
    title = "Harbor dredging contract awarded in synthetic tender"
    copies = [
        _rec(i, f"{title} - Wire {i}", "harbor dredging contract awarded " * 5) for i in (7, 5, 6)
    ]
    a = dedupe_records(copies)
    b = dedupe_records(list(reversed(copies)))
    assert len(a.records) == len(b.records) == 1
    assert a.records[0].doc_id == b.records[0].doc_id == 5
    assert sorted(r.doc_id for r in a.merged[5]) == [6, 7]


def test_dedupe_keeps_recurring_threads_and_distinct_stories() -> None:
    """AC-005.8: same title weeks apart (a recurring thread) or different stories are not merged."""
    t0 = datetime(2026, 3, 1, tzinfo=UTC)
    monthly_a = _rec(
        1, "Who is hiring this month", "harbor pilot berth dredging " * 8, published=t0
    )
    monthly_b = _rec(
        2,
        "Who is hiring this month",
        "telescope zenith aperture comet " * 8 + "harbor",
        published=t0 + timedelta(days=31),
    )
    a = _rec(3, "Orchard frost damages blossom crop", "orchard frost blossom crop damage " * 6)
    b = _rec(
        4, "Orchard harvest festival draws crowds", "orchard harvest festival cider crowds " * 6
    )
    res = dedupe_records([monthly_a, monthly_b, a, b])
    assert [r.doc_id for r in res.records] == [1, 2, 3, 4] and not res.merged


def test_dedupe_topic_corpus_is_unchanged() -> None:
    """AC-005.8: a corpus of distinct stories loses nothing."""
    recs, _ = topic_corpus()
    assert dedupe_records(recs).n_removed == 0


def test_dedupe_edge_cases() -> None:
    """AC-005.8: empty / single input."""
    assert dedupe_records([]).records == []
    one = _rec(1, "Only story", "orchard apple " * 5)
    assert dedupe_records([one]).records == [one]


# ---------------------------------------------------------------------- scaling
@pytest.mark.parametrize(
    ("n", "k"), [(0, 0), (1, 1), (2, 1), (3, 2), (8, 2), (24, 3), (30, 4), (100, 7), (5000, 12)]
)
def test_suggest_k_scales_with_corpus(n: int, k: int) -> None:
    """AC-005.8: k ~ sqrt(n/2) clamped to [2, 12]."""
    assert A.suggest_k(n) == k


def test_graph_params_cap_k_by_corpus_size() -> None:
    """AC-005.8: the configured k is a maximum; small corpora get a sparser graph."""
    assert A.graph_params(25, 8, 0.3) == (5, 0.3)
    assert A.graph_params(400, 8, 0.3) == (8, 0.3)
    assert A.graph_params(3, 8, 0.3)[0] == 2
    assert A.graph_params(1, 8, 0.3)[0] == 1


def test_lsa_rank_scales_with_corpus() -> None:
    """AC-005.8: a small corpus gets a low SVD rank (nearly full rank makes cosines vanish)."""
    assert target_components(10) == 6 and target_components(30) == 9
    assert target_components(1000) == 300
    recs, _ = topic_corpus()
    emb = LsaEmbedder(dim=64).fit([embedding_text(r) for r in recs])
    assert emb.n_components == min(target_components(len(recs)), 64)


def test_embedding_text_is_title_weighted_and_truncated() -> None:
    """AC-005.8: document vectors use title + summary + lead words, not a whole long page."""
    long_body = " ".join(f"w{i}" for i in range(1000))
    r = _rec(1, "Headline words here", long_body)
    text = embedding_text(r, max_words=20)
    assert text.startswith(("Headline words here" + chr(10)) * EMBED_TITLE_WEIGHT)
    assert "w10" in text and "w500" not in text
    assert MARKER in text


# ---------------------------------------------------------------------- clustering
@pytest.fixture(scope="module")
def corpus() -> tuple[list[SilverRecord], list[str], np.ndarray, list[str]]:
    recs, topics = topic_corpus()
    texts = [embedding_text(r) for r in recs]
    X = LsaEmbedder(dim=64).fit(texts).embed(texts)
    return recs, topics, X, texts


def test_kmeans_scales_k_and_leaves_no_singleton(corpus) -> None:
    """AC-005.8: k follows the corpus size, an outlier is folded in, silhouette is reported."""
    recs, topics, X, _ = corpus
    outlier = np.zeros((1, X.shape[1]))
    outlier[0, -1] = 1.0  # orthogonal to everything
    labels, info = A.kmeans_clusters(np.vstack([X, outlier]))
    sizes = np.bincount(labels)
    assert sizes.min() >= A.MIN_CLUSTER_SIZE and labels.shape[0] == len(recs) + 1
    assert info["k_target"] == A.suggest_k(len(recs) + 1)
    assert 2 <= info["k"] <= A.MAX_K and info["silhouette"] > 0.3
    labels, info = A.kmeans_clusters(X)
    assert adjusted_rand_score(topics, labels) > 0.8


def test_merge_small_clusters_moves_members_to_nearest() -> None:
    """AC-005.8: singleton clusters join the closest big cluster; all-small input is a no-op."""
    X = np.array([[1, 0], [0.9, 0.1], [0.8, 0.2], [0, 1], [0.1, 0.9], [0.2, 0.8], [0.95, 0.3]])
    merged = A.merge_small_clusters(X, np.array([0, 0, 0, 1, 1, 1, 2]))
    assert merged[6] == merged[0] and len(set(merged.tolist())) == 2
    all_small = A.merge_small_clusters(X[:3], np.array([0, 1, 2]), min_size=2)
    assert sorted(all_small.tolist()) == [0, 1, 2]


def test_louvain_merges_singleton_communities_but_not_isolates() -> None:
    """AC-005.8: a node hanging off one clique joins it; a node with no edge stays alone."""
    G = nx.Graph()
    for clique in (["a1", "a2", "a3", "a4"], ["b1", "b2", "b3", "b4"]):
        for i, u in enumerate(clique):
            for v in clique[i + 1 :]:
                G.add_edge(u, v, weight=0.9)
    G.add_edge("leaf", "a1", weight=0.4)
    G.add_edge("bridge", "a2", weight=0.2)
    G.add_edge("bridge", "b2", weight=0.2)
    G.add_node("alone")
    comms = A.louvain_communities(G)
    sizes = np.bincount(list(comms.values()))
    assert comms["leaf"] == comms["a1"]
    assert sorted(sizes.tolist(), reverse=True)[-1] == 1 and comms["alone"] not in (
        comms["a1"],
        comms["b1"],
    )
    assert (sizes == 1).sum() == 1  # only the isolate
    assert A.louvain_communities(nx.Graph()) == {}


# ---------------------------------------------------------------------- labels
def test_labels_are_discriminative_and_free_of_boilerplate(corpus) -> None:
    """AC-005.8: top terms come from the cluster's own vocabulary, never news boilerplate."""
    recs, topics, X, texts = corpus
    noisy = [
        f"{t}\nwikipedia known isn maybe decides rapidly role according report said" for t in texts
    ]
    labels, _ = A.kmeans_clusters(X)
    terms = A.cluster_top_terms(noisy, labels, n=6)
    banned = {"wikipedia", "known", "isn", "maybe", "decides", "rapidly", "role", "according"}
    for cluster, picked in terms.items():
        flat = {w for t in picked for w in t.split()}
        assert not flat & banned and not flat & {"synthetic", "fixture", "tcs"}
        members = {topics[i] for i in np.flatnonzero(labels == cluster)}
        assert len(members) == 1
        pool = set(POOLS[members.pop()])
        assert len(picked) >= 3 and sum(w in pool for w in picked[:3]) >= 2


def test_cluster_term_never_needs_two_docs_for_singletons() -> None:
    """AC-005.8: a one-document cluster still gets terms; empty text gives an empty label."""
    texts = ["harbor tugboat dredging berth", "telescope nebula parallax comet"]
    assert A.cluster_top_terms(texts, [0, 1])[0]
    assert A.cluster_top_terms(["", ""], [0, 1]) == {0: [], 1: []}


def test_label_prefixes_dominant_category_only_when_strong(corpus) -> None:
    """AC-005.8: 'Ethics: apple / orchard' for a category-pure cluster, plain terms otherwise."""
    recs, topics, X, texts = corpus
    labels, _ = A.kmeans_clusters(X)
    sums = A.cluster_summaries(X, labels, recs, texts)
    for s in sums:
        members = [topics[i] for i in np.flatnonzero(labels == s.cluster)]
        cat = CATEGORY[members[0]]
        assert s.label.startswith(f"{A.CATEGORY_NAMES[cat]}: ")
    mixed = [
        r.model_copy(
            update={
                "risk_categories": [list(CATEGORY.values())[i % 4]],
                "risk_signals": [
                    RiskSignal(
                        category=list(CATEGORY.values())[i % 4],
                        severity="high",
                        keyword="k",
                        evidence=MARKER,
                    )
                ],
            }
        )
        for i, r in enumerate(recs)
    ]
    plain = A.cluster_summaries(X, np.zeros(len(recs), dtype=int), mixed, texts)
    assert ": " not in plain[0].label and plain[0].label
    assert A.cluster_label(["a", "b", "c"], None) == "a / b / c"
    assert A.cluster_label(["a", "b", "c"], "esg") == "ESG: a / b"


# ---------------------------------------------------------------------- pipeline
@pytest.fixture
def vs() -> GoldVectorStore:
    return GoldVectorStore(client=QdrantClient(":memory:"))


def test_run_gold_dedupes_groups_and_reports_merges(
    tmp_path: Path, vs: GoldVectorStore, settings: Settings
) -> None:
    """AC-005.8: duplicates merge at build time, groups are connected, silver list is untouched."""
    recs, topics = topic_corpus()
    dup_src = recs[0]
    snippet = dup_src.model_copy(
        update={
            "doc_id": 900,
            "depth": 0,
            "clean_text": f"{MARKER}. " + " ".join(dup_src.clean_text.split()[2:12]),
            "word_count": 10,
            "source": "synthetic_snippet",
        }
    )
    full_list = [*recs, snippet]
    cfg = settings.model_copy(update={"gold_graph_k": 5, "gold_graph_min_cosine": 0.3})
    res = run_gold(FakeSilverStore(full_list), vs, cfg, out_dir=tmp_path)
    assert len(full_list) == len(recs) + 1 and snippet.doc_id == 900  # silver input unchanged
    assert res["docs"] == len(recs) and res["duplicates_merged"] == 1
    graph = json.loads((tmp_path / "graph.json").read_text())
    meta = graph["meta"]
    assert meta["n_docs"] == len(recs) and meta["n_duplicates_merged"] == 1
    assert meta["graph_k"] <= 5 and meta["n_isolated"] == 0
    node = next(n for n in graph["nodes"] if n["id"] == dup_src.doc_id)
    assert node["merged_doc_ids"] == [900] and node["also_seen_in"] == 1
    assert node["merged_sources"][0]["source"] == "synthetic_snippet"
    assert all(n["also_seen_in"] == 0 for n in graph["nodes"] if n["id"] != dup_src.doc_id)
    assert 900 not in {n["id"] for n in graph["nodes"]}
    # merged ids are not indexed in Qdrant either
    live = vs.with_model(f"lsa_{settings.lsa_dim}")
    assert live.count("docs") == len(recs)
    # groups: every edge stays within one topic, no singleton cluster / community
    topic_by_id = {r.doc_id: t for r, t in zip(recs, topics, strict=True)}
    assert all(topic_by_id[e["source"]] == topic_by_id[e["target"]] for e in graph["links"])
    assert len(graph["clusters"]) == len(POOLS) and len(graph["communities"]) == len(POOLS)
    assert min(c["size"] for c in graph["clusters"] + graph["communities"]) >= 2
    assert (tmp_path / "graph.graphml").is_file()
    html = (tmp_path / "tcs_risk_graph.html").read_text(encoding="utf-8")
    assert "also seen in" in html


def test_run_gold_small_corpus_has_few_meaningful_groups(
    tmp_path: Path, vs: GoldVectorStore, settings: Settings
) -> None:
    """AC-005.8: 24 synthetic docs -> a handful of clusters, not one per document."""
    recs, _ = topic_corpus(per_topic=6)
    res = run_gold(FakeSilverStore(recs), vs, settings, out_dir=tmp_path)
    assert res["docs"] == 24
    assert 2 <= res["clusters"] <= 8 and 2 <= res["communities"] <= 8
