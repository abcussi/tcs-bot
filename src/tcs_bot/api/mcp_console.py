"""Browser bridge to the MCP server (AC-003.11).

A browser cannot speak stdio MCP, so the dashboard calls these two endpoints. Each call opens an
in-memory MCP client session against ``tcs_bot.mcp_server.server.mcp`` and performs the real
protocol round trip (``initialize``, ``tools/list`` or ``tools/call``), so the answer is exactly
what an MCP client such as Claude Desktop receives.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from tcs_bot.errors import NotFoundError
from tcs_bot.schemas.api import ErrorEnvelope

router = APIRouter(prefix="/mcp-console", tags=["mcp"])

_ERRORS: dict[int | str, dict[str, Any]] = {404: {"model": ErrorEnvelope}}


class McpToolOut(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any]


class McpCallIn(BaseModel):
    tool: str = Field(min_length=1, max_length=100)
    arguments: dict[str, Any] = Field(default_factory=dict)


class McpCallOut(BaseModel):
    request: dict[str, Any] = Field(description="The MCP tools/call request that was sent")
    is_error: bool
    text: str = Field(description="The text content blocks of the result, joined")
    structured_content: dict[str, Any] | None = None
    elapsed_ms: int


def _client() -> Any:
    from mcp import Client

    from tcs_bot.mcp_server.server import mcp

    return Client(mcp)


async def _list_tools() -> list[McpToolOut]:
    async with _client() as client:
        result = await client.list_tools()
    return [
        McpToolOut(name=t.name, description=t.description or "", input_schema=t.input_schema)
        for t in result.tools
    ]


async def _call(tool: str, arguments: dict[str, Any]) -> McpCallOut:
    started = time.perf_counter()
    async with _client() as client:
        names = {t.name for t in (await client.list_tools()).tools}
        result = await client.call_tool(tool, arguments) if tool in names else None
    if result is None:  # raised outside the session: errors inside it get wrapped in a group
        raise NotFoundError(f"Unknown MCP tool {tool!r}", {"tools": sorted(names)})
    text = "\n".join(getattr(block, "text", "") for block in result.content)
    return McpCallOut(
        request={"method": "tools/call", "params": {"name": tool, "arguments": arguments}},
        is_error=bool(result.is_error),
        text=text,
        structured_content=result.structured_content,
        elapsed_ms=round((time.perf_counter() - started) * 1000),
    )


# Sync routes run in FastAPI's thread pool; the tools themselves are blocking (DB, LLM), so each
# request gets its own short-lived event loop instead of blocking the server's loop.
@router.get("/tools", response_model=list[McpToolOut])
def list_mcp_tools() -> list[McpToolOut]:
    return asyncio.run(_list_tools())


@router.post("/call", response_model=McpCallOut, responses=_ERRORS)
def call_mcp_tool(body: McpCallIn) -> McpCallOut:
    return asyncio.run(_call(body.tool, body.arguments))
