"""Gold read side: `/tcs/risks/*` routes and the MCP tools (AC-005.9), offline.

Gold is built from the SYNTHETIC corpus into ``tmp_path`` (``GOLD_DIR``) with an in-memory
fake ES store injected via ``search.set_vector_store``."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tcs_bot.api.app import create_app
from tcs_bot.config import get_settings
from tcs_bot.mcp_server import server
from tcs_bot.medallion.gold import search
from tcs_bot.medallion.gold.pipeline import run_gold
from tcs_bot.medallion.gold.vector_store import GoldVectorStore
from tcs_bot.medallion.models import GoldHit
from tcs_bot.services.report_service import ReportService

from .fake_es import DownES, FakeES
from .test_gold_synthetic import FakeSilverStore, synthetic_records

HEALTHY = {"mysql": lambda: True, "redis": lambda: True, "elasticsearch": lambda: True}


@pytest.fixture
def gold_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[GoldVectorStore]:
    monkeypatch.setenv("GOLD_DIR", str(tmp_path / "gold"))
    get_settings.cache_clear()
    vs = GoldVectorStore(es=FakeES(), prefix="gold-tcs-risk")
    search.set_vector_store(vs)
    yield vs
    search.set_vector_store(None)
    get_settings.cache_clear()


@pytest.fixture
def built(gold_env: GoldVectorStore) -> dict:
    return run_gold(FakeSilverStore(synthetic_records()), gold_env)


@pytest.fixture
def client(service: ReportService) -> Iterator[TestClient]:
    with TestClient(create_app(service, health_checks=HEALTHY)) as c:
        yield c


def test_routes_404_before_gold_is_built(gold_env, client: TestClient) -> None:
    """AC-005.9: before the first gold run every endpoint is a 404 AppError with a hint."""
    for path in ("/tcs/risks/search?q=breach", "/tcs/risks/summary", "/tcs/risks/graph",
                 "/tcs/risks/graph.html"):  # fmt: skip
        r = client.get(path)
        assert r.status_code == 404, path
        err = r.json()["error"]
        assert err["code"] == "gold_not_built" and "tcs-bot medallion run" in err["message"]


def test_routes_after_gold_run(built: dict, client: TestClient) -> None:
    """AC-005.9: search / summary / graph / graph.html once gold exists."""
    r = client.get("/tcs/risks/search", params={"q": "ransomware breach passwords", "k": 4})
    assert r.status_code == 200
    hits = [GoldHit.model_validate(h) for h in r.json()]
    assert len(hits) == 4 and len({h.doc_id for h in hits}) == 4  # one hit per document
    assert all("/cyber/" in h.url for h in hits)

    r = client.get(
        "/tcs/risks/search",
        params={"q": "ransomware breach", "category": "legal_litigation", "min_risk": 0},
    )
    assert r.status_code == 200
    assert all("legal_litigation" in h["risk_categories"] for h in r.json())

    summary = client.get("/tcs/risks/summary").json()
    assert summary["meta"]["n_docs"] == 30 and len(summary["clusters"]) == 3
    graph = client.get("/tcs/risks/graph").json()
    assert len(graph["nodes"]) == 30
    html = client.get("/tcs/risks/graph.html")
    assert html.status_code == 200 and html.headers["content-type"].startswith("text/html")
    assert "gold-data" in html.text


def test_search_validation(built: dict, client: TestClient) -> None:
    assert client.get("/tcs/risks/search", params={"q": ""}).status_code == 422
    assert client.get("/tcs/risks/search", params={"q": "x", "k": 0}).status_code == 422
    r = client.get("/tcs/risks/search", params={"q": "x", "category": "aliens"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "validation_error"
    assert client.get("/tcs/risks/search", params={"q": "zzzqx"}).json() == []


def test_search_maps_elasticsearch_failure_to_503(built: dict, client: TestClient) -> None:
    search.set_vector_store(GoldVectorStore(es=DownES(), prefix="gold-tcs-risk"))  # type: ignore[arg-type]
    r = client.get("/tcs/risks/search", params={"q": "breach"})
    assert r.status_code == 503 and r.json()["error"]["code"] == "vector_store_unavailable"


def test_mcp_tools(built: dict) -> None:
    """AC-005.9: MCP search_tcs_risks(query, k) and tcs_risk_summary()."""
    hits = server.search_tcs_risks("jury verdict damages", k=3)
    assert len(hits) == 3 and all(isinstance(h, GoldHit) and "/legal/" in h.url for h in hits)
    only = server.search_tcs_risks("jury verdict", k=5, category="labor_hr")
    assert all("labor_hr" in h.risk_categories for h in only)
    assert server.tcs_risk_summary()["meta"]["n_docs"] == 30
    with pytest.raises(server.ToolError, match="validation_error"):
        server.search_tcs_risks("x", category="aliens")


async def test_mcp_tools_registered_and_errors(gold_env) -> None:
    tools = {t.name: t for t in await server.mcp.list_tools()}
    assert {"search_tcs_risks", "tcs_risk_summary"} <= set(tools)
    assert {"query", "k", "category"} <= set(tools["search_tcs_risks"].input_schema["properties"])
    with pytest.raises(server.ToolError, match="gold_not_built"):
        server.tcs_risk_summary()
    with pytest.raises(server.ToolError, match="gold_not_built"):
        server.search_tcs_risks("breach")
