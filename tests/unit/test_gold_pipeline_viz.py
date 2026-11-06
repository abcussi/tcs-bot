"""Gold pipeline end to end + the self-contained HTML (AC-005.7, AC-005.8), all offline:
fake silver store, in-process Qdrant, artifacts into ``tmp_path``."""

from __future__ import annotations

import json
import re
from pathlib import Path

import networkx as nx
import pytest
from qdrant_client import QdrantClient

from tcs_bot.config import Settings
from tcs_bot.medallion.gold.pipeline import run_gold
from tcs_bot.medallion.gold.vector_store import GoldVectorStore
from tcs_bot.medallion.gold.viz import embed_json, render_html

from .test_gold_synthetic import FakeSilverStore, synthetic_records, topic_of

ARTIFACTS = ("graph.json", "summary.json", "graph.graphml", "tcs_risk_graph.html", "lsa.pkl")


def _embedded(html: str) -> dict:
    m = re.search(r'<script type="application/json" id="gold-data">(.*?)</script>', html, re.S)
    assert m, "embedded JSON block missing"
    return json.loads(m.group(1))


@pytest.fixture
def vs() -> GoldVectorStore:
    return GoldVectorStore(client=QdrantClient(":memory:"))


def test_run_gold_end_to_end(tmp_path: Path, vs: GoldVectorStore, settings: Settings) -> None:
    """AC-005.7/8: silver -> LSA -> Qdrant (docs + chunks) + graph/summary/graphml/html."""
    records = synthetic_records()
    res = run_gold(FakeSilverStore(records), vs, settings, out_dir=tmp_path)
    assert res["docs"] == 30 and res["chunks"] >= 30
    assert res["qdrant"] == "ok" and res["clusters"] == 3
    assert res["embedding_model"] == f"lsa-{settings.lsa_dim}"
    for name in ARTIFACTS:
        assert (tmp_path / name).is_file(), name
    live = vs.with_model(f"lsa_{settings.lsa_dim}")
    assert live.count("docs") == 30 and live.count("chunks") == res["chunks"]

    graph = json.loads((tmp_path / "graph.json").read_text())
    assert len(graph["nodes"]) == 30 and len(graph["links"]) == res["edges"] > 0
    node = graph["nodes"][0]
    for key in ("x", "y", "cluster", "community", "degree", "betweenness", "eigenvector"):
        assert key in node
    assert node["neighbors"] and node["signals"][0]["evidence"]
    assert graph["meta"]["synthetic"] is True and "SYNTHETIC" in graph["meta"]["data_source"]
    # clusters in the Qdrant payload match graph.json
    topic_cluster = {
        (topic_of(r), n["cluster"]) for r, n in zip(records, graph["nodes"], strict=True)
    }
    assert len(topic_cluster) == 3

    summary = json.loads((tmp_path / "summary.json").read_text())
    assert "nodes" not in summary and len(summary["clusters"]) == 3
    assert summary["stats"]["top_risky"] and summary["communities"]

    G = nx.read_graphml(tmp_path / "graph.graphml")
    assert G.number_of_nodes() == 30 and G.number_of_edges() == res["edges"]
    assert json.loads(G.nodes[str(node["id"])]["risk_categories"])

    html = (tmp_path / "tcs_risk_graph.html").read_text()
    assert _embedded(html)["meta"]["n_docs"] == 30

    # re-run is a full rebuild: same counts, no duplicate points
    run_gold(FakeSilverStore(records), vs, settings, out_dir=tmp_path)
    assert live.count("docs") == 30


@pytest.mark.parametrize("n", [0, 1, 2])
def test_run_gold_tiny_corpora(tmp_path: Path, vs: GoldVectorStore, settings, n: int) -> None:
    """AC-005.8: 0/1/2 documents produce valid (small) artifacts, no crash."""
    res = run_gold(FakeSilverStore(synthetic_records()[:n]), vs, settings, out_dir=tmp_path)
    assert res["docs"] == n
    graph = json.loads((tmp_path / "graph.json").read_text())
    assert len(graph["nodes"]) == n
    assert (tmp_path / "tcs_risk_graph.html").is_file()
    assert res["qdrant"] == ("skipped: no documents" if n == 0 else "ok")


def test_run_gold_skips_irrelevant_and_duplicates(tmp_path, vs, settings) -> None:
    """AC-005.6: only relevant silver docs reach gold; duplicate doc_ids collapse."""
    recs = synthetic_records()[:6]
    recs[0] = recs[0].model_copy(update={"relevant": False})
    res = run_gold(FakeSilverStore([*recs, recs[1]]), vs, settings, out_dir=tmp_path)
    assert res["docs"] == 5


def test_run_gold_degrades_when_qdrant_is_down(tmp_path, settings) -> None:
    """Constitution §6: Qdrant down -> artifacts still written, status explains."""

    class Down:
        def __getattr__(self, name):
            raise ConnectionError("qdrant down")

    res = run_gold(
        FakeSilverStore(synthetic_records()[:9]),
        GoldVectorStore(client=Down()),  # type: ignore[arg-type]
        settings,
        out_dir=tmp_path,
    )
    assert res["qdrant"].startswith("unavailable: ConnectionError")
    assert json.loads((tmp_path / "summary.json").read_text())["meta"]["qdrant"] == res["qdrant"]
    assert (tmp_path / "tcs_risk_graph.html").is_file()


# ---------------------------------------------------------------------- viz
def test_html_is_self_contained_and_escapes_data(tmp_path: Path) -> None:
    """AC-005.8: single offline HTML, no external scripts/styles; crawled text cannot break out."""
    evil = '</script><script src="https://evil.example/x.js"></script>&<b>'
    data = {
        "meta": {"n_docs": 1, "embedding_model": "lsa-8", "data_source": "SYNTHETIC test"},
        "nodes": [{"id": 1, "title": evil, "x": 0, "y": 0, "risk_score": 50, "cluster": 0}],
        "links": [],
        "clusters": [],
        "communities": [],
        "stats": {},
    }
    html = render_html(data, tmp_path / "g.html").read_text()
    assert html.startswith("<!doctype html>")
    assert not re.search(r"<script[^>]*\bsrc=", html.replace(embed_json(data), ""))
    assert not re.search(r"<link[^>]+stylesheet", html)
    assert "evil.example" in html and '<script src="https://evil' not in html
    assert _embedded(html)["nodes"][0]["title"] == evil
    for feature in ("min-cos", "seg-layout", "seg-color", "prefers-color-scheme", 'id="q"'):
        assert feature in html
    assert html.count("<script") == 2  # the JSON block + the inline app


def test_html_empty_graph(tmp_path: Path) -> None:
    empty = {"meta": {"n_docs": 0}, "nodes": [], "links": [], "clusters": [], "stats": {}}
    html = render_html(empty, tmp_path / "e.html").read_text()
    assert _embedded(html)["nodes"] == []
    assert "tcs-bot medallion run" in html
