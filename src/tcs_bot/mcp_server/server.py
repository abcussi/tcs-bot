"""MCP server exposing the risk report generator (AC-003.9).

Run: ``uv run tcs-bot-mcp`` (stdio) or
``uv run tcs-bot-mcp --transport streamable-http --port 8001``.
"""

from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

try:  # mcp < 2
    from mcp.server.fastmcp import FastMCP
    from mcp.server.fastmcp.exceptions import ToolError
except ImportError:  # mcp >= 2 renamed FastMCP -> MCPServer (same decorator API)
    from mcp.server.mcpserver import MCPServer as FastMCP
    from mcp.server.mcpserver.exceptions import ToolError

from tcs_bot.errors import AppError
from tcs_bot.medallion.models import GoldHit
from tcs_bot.medallion.read import DocumentDetail, DocumentPage, SimilarDocuments
from tcs_bot.schemas.api import ReportRequest, VendorOut
from tcs_bot.schemas.report import RiskReport

if TYPE_CHECKING:
    from tcs_bot.services.report_service import ReportService

mcp = FastMCP(
    "vendor-risk",
    instructions=(
        "Generate structured third-party (vendor) risk reports for capital-markets firms. "
        "Call list_vendors to find a vendor, then generate_vendor_risk_report. "
        "For TCS (Tata Consultancy Services) risk intelligence from news and communities use "
        "search_tcs_risks and tcs_risk_summary; browse them with list_tcs_risks, "
        "get_tcs_risk_document, similar_tcs_risks and tcs_risk_clusters."
    ),
)

_service: ReportService | None = None


def get_service() -> ReportService:
    """Lazily build the real service (tests call ``set_service`` with a fake one)."""
    global _service
    if _service is None:
        from tcs_bot.services.report_service import build_default_service

        _service = build_default_service()
    return _service


def set_service(service: ReportService | None) -> None:
    global _service
    _service = service


def _tool_error(exc: AppError) -> ToolError:
    return ToolError(f"{exc.code}: {exc.message}")


@mcp.tool()
def generate_vendor_risk_report(
    vendor_id: int | None = None,
    vendor_name: str | None = None,
    focus_areas: list[str] | None = None,
) -> RiskReport:
    """Generate a structured vendor risk report (score 0-100, risk level, red flags, financial
    health, compliance recommendations, sources, data gaps). Give vendor_id or vendor_name.

    Returns a ``RiskReport``: the tool's MCP ``outputSchema`` is the RiskReport JSON schema and
    the result's ``structuredContent`` is its JSON dump."""
    try:
        req = ReportRequest(
            vendor_id=vendor_id, vendor_name=vendor_name, focus_areas=focus_areas or []
        )
    except ValidationError as exc:
        raise ToolError(f"validation_error: {exc.errors()[0]['msg']}") from exc
    try:
        return get_service().generate_report(req)
    except AppError as exc:
        raise _tool_error(exc) from exc


@mcp.tool()
def list_vendors(limit: int = 50) -> list[dict[str, Any]]:
    """List vendors (id, external_id, name, sector, country, ticker, criticality, services)."""
    limit = max(1, min(int(limit), 500))
    try:
        vendors = get_service().list_vendors(limit=limit, offset=0)
    except AppError as exc:
        raise _tool_error(exc) from exc
    return [
        VendorOut.model_validate(v, from_attributes=True).model_dump(mode="json") for v in vendors
    ]


@mcp.tool()
def search_tcs_risks(query: str, k: int = 10, category: str | None = None) -> list[GoldHit]:
    """Semantic search over the TCS risk gold layer (news + community posts about Tata
    Consultancy Services). Returns the best-matching document chunks with risk categories,
    risk score (0-100) and cluster. Optional ``category``: ethics, compliance_regulatory,
    legal_litigation, data_privacy_cyber, labor_hr, financial, operational_delivery,
    reputational, esg, geopolitical_immigration."""
    from tcs_bot.medallion.gold.search import search_risks
    from tcs_bot.medallion.models import RISK_CATEGORIES

    if category is not None and category not in RISK_CATEGORIES:
        raise ToolError(f"validation_error: unknown risk category {category!r}")
    try:
        return search_risks(
            query, k=max(1, min(int(k), 100)), categories=[category] if category else None
        )
    except AppError as exc:
        raise _tool_error(exc) from exc


@mcp.tool()
def tcs_risk_summary() -> dict[str, Any]:
    """Summary of the TCS risk gold layer: build metadata, KMeans clusters (label, top terms,
    dominant risk categories, mean risk), Louvain communities and dashboard statistics."""
    from tcs_bot.medallion.gold.search import load_summary

    try:
        return load_summary()
    except AppError as exc:
        raise _tool_error(exc) from exc


_SORTS = ("risk_score", "published_at", "tcs_relevance")


def _read_input[M](model: type[M], **fields: Any) -> M:
    """Build a read-layer input model; invalid values become a ``validation_error`` ToolError."""
    try:
        return model(**fields)
    except ValidationError as exc:
        err = exc.errors(include_context=False, include_url=False, include_input=False)[0]
        where = ".".join(str(p) for p in err["loc"])
        raise ToolError(f"validation_error: {where}: {err['msg']}") from exc


@mcp.tool()
def list_tcs_risks(
    category: str | None = None,
    min_severity: str | None = None,
    min_risk_score: float | None = None,
    source: str | None = None,
    source_type: str | None = None,
    query: str | None = None,
    published_from: str | None = None,
    published_to: str | None = None,
    cluster: int | None = None,
    community: int | None = None,
    sort: str = "risk_score",
    order: str = "desc",
    limit: int = 10,
    offset: int = 0,
    include_irrelevant: bool = False,
) -> DocumentPage:
    """List TCS risk documents (silver layer) with filters, sorting and pagination.

    Filters (all optional, ANDed): ``category`` (ethics, compliance_regulatory, legal_litigation,
    data_privacy_cyber, labor_hr, financial, operational_delivery, reputational, esg,
    geopolitical_immigration), ``min_severity`` (low|medium|high|critical),
    ``min_risk_score`` (0-100), ``source`` (e.g. bing_news, hackernews), ``source_type``
    (news|community|corporate|other), ``query`` (free text on title/text), ``published_from`` /
    ``published_to`` (ISO dates), ``cluster`` / ``community`` (gold ids). ``sort`` is
    risk_score|published_at|tcs_relevance, ``order`` asc|desc, ``limit`` 1-100. Returns the page
    ``total`` and one summary row per document (use get_tcs_risk_document for the details)."""
    from tcs_bot.medallion.read import DocumentFilters, Page, get_tcs_data

    if sort not in _SORTS:
        raise ToolError(f"validation_error: sort must be one of {', '.join(_SORTS)}")
    filters = _read_input(
        DocumentFilters,
        category=[category] if category else [],
        min_severity=min_severity,
        min_risk_score=min_risk_score,
        source=[source] if source else [],
        source_type=[source_type] if source_type else [],
        relevant=None if include_irrelevant else True,
        published_from=published_from,
        published_to=published_to,
        cluster=cluster,
        community=community,
        q=query,
    )
    page = _read_input(Page, limit=limit, offset=offset, sort=sort, order=order)
    try:
        return get_tcs_data().list_documents(filters, page)
    except AppError as exc:
        raise _tool_error(exc) from exc


@mcp.tool()
def get_tcs_risk_document(doc_id: int, include_text: bool = False) -> DocumentDetail:
    """One TCS risk document by id: silver fields (risk signals with evidence, entities, scores),
    bronze lineage (url, crawl depth, parent url, query, run id, fetch time, HTTP status; never the
    raw body) and the gold node (cluster, community, centrality, merged duplicates).
    ``include_text`` adds the cleaned article text. ``doc_id`` is the bronze/silver/gold id."""
    from tcs_bot.medallion.read import get_tcs_data

    try:
        return get_tcs_data().get_document(doc_id, include_text=include_text)
    except AppError as exc:
        raise _tool_error(exc) from exc


@mcp.tool()
def similar_tcs_risks(doc_id: int, k: int = 5) -> SimilarDocuments:
    """Documents most similar to ``doc_id`` (cosine over the gold embedding, best first, 1-50)."""
    from tcs_bot.medallion.read import get_tcs_data

    try:
        return get_tcs_data().similar_documents(doc_id, k=max(1, min(int(k), 50)))
    except AppError as exc:
        raise _tool_error(exc) from exc


@mcp.tool()
def tcs_risk_clusters(cluster_id: int | None = None, kind: str = "cluster") -> dict[str, Any]:
    """Gold clusters of TCS risk documents. Without ``cluster_id``: every cluster (label, size,
    top terms, dominant risk categories, mean risk). With it: that cluster plus its member
    documents. ``kind`` is ``cluster`` (KMeans) or ``community`` (Louvain graph communities)."""
    from tcs_bot.medallion.read import get_tcs_data

    if kind not in ("cluster", "community"):
        raise ToolError("validation_error: kind must be 'cluster' or 'community'")
    try:
        data = get_tcs_data()
        if cluster_id is None:
            clusters = data.list_clusters(kind)  # type: ignore[arg-type]
            return {"kind": kind, "clusters": [c.model_dump(mode="json") for c in clusters]}
        return data.get_cluster(cluster_id, kind).model_dump(mode="json")  # type: ignore[arg-type]
    except AppError as exc:
        raise _tool_error(exc) from exc


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="tcs-bot-mcp", description=__doc__)
    parser.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args(argv)

    import structlog

    from tcs_bot.config import get_settings
    from tcs_bot.logging_setup import configure_logging

    configure_logging(get_settings().log_level)
    # stdout is the MCP stdio channel: logs must go to stderr.
    structlog.configure(logger_factory=structlog.PrintLoggerFactory(file=sys.stderr))

    if args.transport == "stdio":
        mcp.run("stdio")
    else:
        mcp.run("streamable-http", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
