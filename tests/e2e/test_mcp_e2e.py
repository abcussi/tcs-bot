"""E2E: launch `uv run tcs-bot-mcp` over stdio with the MCP client SDK and call its tools."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import pytest

from tcs_bot.schemas.api import VendorOut
from tcs_bot.schemas.report import RiskReport, score_to_level

pytestmark = pytest.mark.e2e

ROOT = Path(__file__).resolve().parents[2]

try:  # mcp 2.x client API (FastMCP was renamed MCPServer server-side; client is unchanged)
    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client
except ImportError as exc:  # pragma: no cover - depends on installed SDK
    pytest.skip(f"mcp client API unavailable: {exc}", allow_module_level=True)


def _payload(result: Any) -> Any:
    """Tool result -> python value: prefer structured content, fall back to JSON text blocks."""
    structured = getattr(result, "structured_content", None)
    if structured is None:
        structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict) and set(structured) == {"result"}:
        return structured["result"]
    if structured is not None:
        return structured
    texts = [json.loads(c.text) for c in result.content if getattr(c, "type", "") == "text"]
    return texts[0] if len(texts) == 1 else texts


def _is_error(result: Any) -> bool:
    return bool(getattr(result, "is_error", getattr(result, "isError", False)))


@pytest.fixture
def server_params(api) -> StdioServerParameters:
    """Depends on ``api`` so the test skips when the stack (MySQL/ES/Redis) is not up."""
    if shutil.which("uv") is None:
        pytest.skip("uv not on PATH")
    env = {**os.environ, "LLM_PROVIDER": "fake", "OPENAI_API_KEY": "", "LOG_LEVEL": "WARNING"}
    return StdioServerParameters(
        command="uv", args=["run", "--no-sync", "tcs-bot-mcp"], cwd=str(ROOT), env=env
    )


async def test_mcp_stdio_tools(server_params, created):
    """AC-003.9: stdio MCP server lists both tools; list_vendors and
    generate_vendor_risk_report (by name and by id) return valid structured data; unknown
    vendors come back as tool errors."""
    async with stdio_client(server_params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()

        tools = {t.name for t in (await session.list_tools()).tools}
        assert {"generate_vendor_risk_report", "list_vendors"} <= tools

        res = await session.call_tool("list_vendors", {"limit": 50})
        assert not _is_error(res)
        vendors = [VendorOut.model_validate(v) for v in _payload(res)]
        assert len(vendors) >= 1
        target = vendors[-1]

        res = await session.call_tool(
            "generate_vendor_risk_report",
            {"vendor_name": target.name.lower(), "focus_areas": ["operational resilience"]},
        )
        assert not _is_error(res), res
        report = RiskReport.model_validate(_payload(res))
        assert report.vendor_id == target.id and report.vendor_name == target.name
        assert report.risk_level == score_to_level(report.overall_risk_score)

        res = await session.call_tool("generate_vendor_risk_report", {"vendor_id": target.id})
        assert not _is_error(res)
        assert RiskReport.model_validate(_payload(res)).vendor_id == target.id

        res = await session.call_tool("generate_vendor_risk_report", {"vendor_id": 987_654_321})
        assert _is_error(res)
        assert "not_found" in " ".join(c.text for c in res.content)

        res = await session.call_tool("generate_vendor_risk_report", {})
        assert _is_error(res)
        assert "validation_error" in " ".join(c.text for c in res.content)
