"""MCP server tools (AC-003.9)."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from tcs_bot.mcp_server import server
from tcs_bot.schemas.domain import DocumentCreate
from tcs_bot.schemas.report import RiskReport
from tcs_bot.services.report_service import ReportService

from .conftest import GOING_CONCERN_DOC


@pytest.fixture(autouse=True)
def _injected(service: ReportService) -> Iterator[None]:
    server.set_service(service)
    yield
    server.set_service(None)


def test_generate_tool_returns_valid_report(service: ReportService) -> None:
    """AC-003.9: generate_vendor_risk_report returns a RiskReport (by id and by name)."""
    service.ingest_document(
        1, DocumentCreate(doc_type="regulatory_filing", title="10-K", content=GOING_CONCERN_DOC)
    )
    report = server.generate_vendor_risk_report(vendor_id=1, focus_areas=["liquidity"])
    assert isinstance(report, RiskReport)
    assert report.vendor_id == 1 and report.risk_level == "critical"
    by_name = server.generate_vendor_risk_report(vendor_name="Northwind Data Feeds")
    assert by_name.vendor_id == 2


async def test_generate_tool_output_schema_is_risk_report(service: ReportService) -> None:
    """AC-003.9: the tool's outputSchema is the RiskReport schema; structuredContent is its JSON."""
    tools = {t.name: t for t in await server.mcp.list_tools()}
    schema = tools["generate_vendor_risk_report"].output_schema
    assert schema is not None
    expected = RiskReport.model_json_schema()
    assert schema.get("title") == "RiskReport"
    assert set(schema["properties"]) == set(expected["properties"])
    # list_vendors keeps working with its (wrapped list) schema
    assert tools["list_vendors"].output_schema is not None

    result = await server.mcp.call_tool("generate_vendor_risk_report", {"vendor_id": 2})
    data = result.structured_content
    assert isinstance(data, dict) and isinstance(data["generated_at"], str)
    assert RiskReport.model_validate(data).vendor_id == 2


def test_tool_errors_are_tool_errors() -> None:
    """AC-003.9: missing selector / unknown vendor -> ToolError with the error code."""
    with pytest.raises(server.ToolError, match="validation_error"):
        server.generate_vendor_risk_report()
    with pytest.raises(server.ToolError, match="not_found"):
        server.generate_vendor_risk_report(vendor_id=404)


def test_list_vendors_tool() -> None:
    """AC-003.9: list_vendors returns JSON-able vendor dicts, limit respected."""
    out = server.list_vendors(limit=1)
    assert out == [
        {
            "id": 1,
            "external_id": "V-001",
            "name": "Acme Clearing Services",
            "sector": "post-trade",
            "country": "US",
            "ticker": "ACME:NYSE",
            "criticality": "critical",
            "services": "trade matching and clearing connectivity",
        }
    ]
    assert len(server.list_vendors()) == 2


async def test_tools_registered_on_server() -> None:
    """AC-003.9: both tools are exposed by the MCP server with input schemas."""
    tools = {t.name: t for t in await server.mcp.list_tools()}
    assert {"generate_vendor_risk_report", "list_vendors"} <= set(tools)
    props = tools["generate_vendor_risk_report"].input_schema["properties"]
    assert {"vendor_id", "vendor_name", "focus_areas"} <= set(props)


def test_main_parses_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    """AC-003.9: CLI supports stdio (default) and streamable-http with --port."""
    calls: list[tuple] = []
    monkeypatch.setattr(server.mcp, "run", lambda *a, **k: calls.append((a, k)))
    server.main([])
    server.main(["--transport", "streamable-http", "--port", "9123"])
    assert calls[0] == (("stdio",), {})
    assert calls[1] == (("streamable-http",), {"host": "127.0.0.1", "port": 9123})
