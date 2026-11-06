"""Dashboard MCP console bridge (AC-003.11): real in-memory MCP round trips over HTTP."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from tcs_bot.api.app import create_app
from tcs_bot.mcp_server import server
from tcs_bot.schemas.domain import DocumentCreate
from tcs_bot.schemas.report import RiskReport
from tcs_bot.services.report_service import ReportService

from .conftest import GOING_CONCERN_DOC

HEALTHY = {"mysql": lambda: True, "redis": lambda: True, "elasticsearch": lambda: True}


@pytest.fixture
def client(service: ReportService) -> Iterator[TestClient]:
    server.set_service(service)
    with TestClient(create_app(service, health_checks=HEALTHY), raise_server_exceptions=False) as c:
        yield c
    server.set_service(None)


def test_lists_the_mcp_servers_tools_with_input_schemas(client: TestClient) -> None:
    r = client.get("/mcp-console/tools")
    assert r.status_code == 200
    tools = {t["name"]: t for t in r.json()}
    assert {"generate_vendor_risk_report", "list_vendors", "search_tcs_risks"} <= set(tools)
    props = tools["generate_vendor_risk_report"]["input_schema"]["properties"]
    assert {"vendor_id", "vendor_name", "focus_areas"} <= set(props)


def test_call_generate_report_returns_the_structured_risk_report(
    client: TestClient, service: ReportService
) -> None:
    service.ingest_document(
        1, DocumentCreate(doc_type="regulatory_filing", title="10-K", content=GOING_CONCERN_DOC)
    )
    r = client.post(
        "/mcp-console/call",
        json={"tool": "generate_vendor_risk_report", "arguments": {"vendor_id": 1}},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["is_error"] is False
    assert body["request"] == {
        "method": "tools/call",
        "params": {"name": "generate_vendor_risk_report", "arguments": {"vendor_id": 1}},
    }
    assert RiskReport.model_validate(body["structured_content"]).vendor_id == 1


def test_tool_error_is_reported_not_raised(client: TestClient) -> None:
    r = client.post(
        "/mcp-console/call",
        json={"tool": "generate_vendor_risk_report", "arguments": {"vendor_id": 404}},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["is_error"] is True and "not_found" in body["text"]


def test_unknown_tool_is_404_envelope(client: TestClient) -> None:
    r = client.post("/mcp-console/call", json={"tool": "no_such_tool", "arguments": {}})
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "not_found"


def test_dashboard_has_the_mcp_tab(client: TestClient) -> None:
    html = client.get("/dashboard").text
    assert "MCP console" in html and "/mcp-console/call" in html
