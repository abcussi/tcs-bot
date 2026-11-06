"""Typed read layer against the LIVE stack, strictly READ-ONLY (AC-005.12, AC-005.11).

Reads ``bronze_documents`` (MySQL), ``silver-tcs-risk`` and the gold ES indices plus ``data/gold``
files that the pipeline already produced. It never writes, deletes or recreates anything and never
asserts on article text. Skips cleanly when MySQL / Elasticsearch / silver / gold are not there.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from tcs_bot.config import get_settings
from tcs_bot.medallion.read import DocumentFilters, Page, TcsData, set_tcs_data
from tcs_bot.medallion.read import gold_graph as gg

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def live() -> Iterator[TcsData]:
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("LLM_PROVIDER", "fake")  # never call OpenAI from tests
        mp.setenv("OPENAI_API_KEY", "")
        mp.delenv("MYSQL_DATABASE", raising=False)
        mp.delenv("GOLD_DIR", raising=False)
        get_settings.cache_clear()
        from tcs_bot.db.mysql import ping as mysql_ping
        from tcs_bot.db.mysql import reset_pool
        from tcs_bot.search.elastic import get_es, reset_es

        reset_pool()
        reset_es()
        set_tcs_data(None)
        settings = get_settings()
        if not mysql_ping():
            pytest.skip("MySQL not reachable")
        es = get_es()
        try:
            if not es.ping():
                pytest.skip("Elasticsearch not reachable")
            if not es.indices.exists(index=settings.silver_index):
                pytest.skip("silver index not built")
            if es.count(index=settings.silver_index)["count"] == 0:
                pytest.skip("silver index is empty")
        except Exception as exc:
            pytest.skip(f"Elasticsearch not usable: {exc}")
        if not (Path(settings.gold_dir) / "graph.json").is_file():
            pytest.skip("gold not built (data/gold/graph.json missing)")
        gg.clear_cache()
        try:
            yield TcsData()
        finally:
            set_tcs_data(None)
            gg.clear_cache()
            reset_pool()
            reset_es()
    get_settings.cache_clear()


def test_list_documents_returns_relevant_docs(live: TcsData) -> None:
    """Default list = the relevant silver docs; total matches an independent ES count."""
    store = live.silver
    expected = store.count(relevant_only=True)
    page = live.list_documents(page=Page(limit=100))
    assert page.total == expected > 0
    assert len(page.items) == min(expected, 100) and all(d.relevant for d in page.items)
    scores = [d.risk_score for d in page.items]
    assert scores == sorted(scores, reverse=True)
    assert live.list_documents(DocumentFilters(relevant=None)).total == store.count()
    first = live.list_documents(page=Page(limit=3, offset=0)).items
    second = live.list_documents(page=Page(limit=3, offset=3)).items
    assert not {d.doc_id for d in first} & {d.doc_id for d in second}
    high = live.list_documents(DocumentFilters(min_severity="high", min_risk_score=30))
    assert high.total <= expected and all(d.risk_score >= 30 for d in high.items)


def test_get_document_joins_bronze_silver_gold(live: TcsData) -> None:
    """One id resolves to silver + MySQL bronze lineage (no body) + gold node."""
    graph = gg.load_gold_graph()
    doc_id = min(graph.nodes)
    d = live.get_document(doc_id)
    assert d.silver.doc_id == d.bronze.id == doc_id  # type: ignore[union-attr]
    assert d.bronze is not None and d.bronze.url == d.silver.url
    assert d.bronze.source == d.silver.source and d.bronze.depth == d.silver.depth
    assert d.bronze.raw_content is None and (d.bronze.raw_chars or 0) > 0
    assert d.bronze.fetched_at is not None and d.bronze.http_status > 0
    assert d.silver.clean_text is None and d.silver.risk_signals
    assert d.gold is not None and d.gold.in_gold
    assert d.gold.cluster == graph.nodes[doc_id]["cluster"]
    assert d.gold.community == graph.nodes[doc_id]["community"]
    raw = live.get_document(doc_id, include_raw=True, include_text=True)
    assert raw.bronze is not None and len(raw.bronze.raw_content or "") == d.bronze.raw_chars
    assert raw.silver.clean_text


def test_category_breakdown_sums_match_a_silver_scan(live: TcsData) -> None:
    bd = live.category_breakdown()
    docs = list(live.silver.iter_relevant())
    assert bd.total_documents == len(docs)
    assert sum(c.count for c in bd.categories) == sum(len(r.risk_categories) for r in docs)
    for c in bd.categories:
        assert c.count == sum(c.key in r.risk_categories for r in docs)
    ov = live.risk_overview(top_n=3)
    assert ov.total_documents == len(docs) and len(ov.top_documents) == 3
    assert sum(b.count for b in ov.by_source) == len(docs)
    assert sum(b.count for b in ov.by_month) <= len(docs)
    assert ov.max_risk_score == pytest.approx(max(r.risk_score for r in docs), abs=0.01)


def _cos(a: list[float], b: list[float]) -> float:
    x, y = np.asarray(a), np.asarray(b)
    return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))


def test_similar_documents_are_ordered_by_cosine(live: TcsData) -> None:
    graph = gg.load_gold_graph()
    doc_id = min(graph.nodes)
    res = live.similar_documents(doc_id, k=5)
    assert res.resolved_doc_id == doc_id and 0 < len(res.neighbours) <= 5
    scores = [h.score for h in res.neighbours]
    assert scores == sorted(scores, reverse=True)
    assert doc_id not in {h.doc_id for h in res.neighbours}
    vs = live.vector_store.with_model(res.model)
    mine = live._gold_vector(vs, doc_id)  # noqa: SLF001 - read-only vector fetch
    assert mine is not None
    for h in res.neighbours:
        other = live._gold_vector(vs, h.doc_id)  # noqa: SLF001
        assert other is not None
        assert h.score == pytest.approx(_cos(mine, other), abs=1e-3)


def test_clusters_match_gold_summary(live: TcsData) -> None:
    clusters = live.list_clusters()
    assert len(clusters) == gg.load_gold_graph().meta["n_clusters"] > 0
    detail = live.get_cluster(clusters[0].cluster)
    assert len(detail.members) == clusters[0].size
    in_cluster = live.list_documents(DocumentFilters(cluster=clusters[0].cluster))
    assert {d.doc_id for d in in_cluster.items} == {m.doc_id for m in detail.members}


def test_api_documents_endpoints_on_live_data(live: TcsData) -> None:
    """GET /tcs/risks/documents (+ detail, similar, categories, clusters) through FastAPI."""
    from tcs_bot.api.app import create_app

    set_tcs_data(live)
    healthy = {"mysql": lambda: True, "redis": lambda: True, "elasticsearch": lambda: True}
    try:
        with TestClient(create_app(health_checks=healthy)) as client:
            r = client.get("/tcs/risks/documents", params={"limit": 10})
            assert r.status_code == 200
            body = r.json()
            assert body["total"] == live.silver.count(relevant_only=True) and body["items"]
            doc_id = body["items"][0]["doc_id"]
            detail = client.get(f"/tcs/risks/documents/{doc_id}")
            assert detail.status_code == 200 and detail.json()["bronze"]["raw_content"] is None
            gold_id = min(gg.load_gold_graph().nodes)
            sim = client.get(f"/tcs/risks/documents/{gold_id}/similar", params={"k": 3})
            assert sim.status_code == 200 and len(sim.json()["neighbours"]) == 3
            cats = client.get("/tcs/risks/categories").json()
            assert cats["total_documents"] == body["total"]
            assert client.get("/tcs/risks/clusters").json()
            assert client.get("/tcs/risks/documents/999999999").status_code == 404
    finally:
        set_tcs_data(None)
