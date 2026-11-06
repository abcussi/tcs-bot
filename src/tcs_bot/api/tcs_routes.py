"""TCS risk endpoints: gold search / summary / graph (AC-005.9) and the typed read layer over
bronze + silver + gold (AC-005.12): documents, similar documents, categories, overview, clusters."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, ValidationError

from tcs_bot.errors import ValidationAppError
from tcs_bot.medallion.gold import search as gold
from tcs_bot.medallion.models import (
    RISK_CATEGORIES,
    ClusterSummary,
    GoldHit,
    RiskCategory,
    Severity,
    SourceType,
)
from tcs_bot.medallion.read import (
    CategoryBreakdown,
    ClusterDetail,
    ClusterKind,
    DocumentDetail,
    DocumentFilters,
    DocumentPage,
    Page,
    RiskOverview,
    SimilarDocuments,
    TcsData,
    get_tcs_data,
)
from tcs_bot.medallion.read.models import MAX_PAGE_SIZE, MAX_WINDOW, SortField, SortOrder
from tcs_bot.schemas.api import ErrorEnvelope

_ERRORS: dict[int | str, dict[str, Any]] = {
    404: {
        "model": ErrorEnvelope,
        "description": "Gold not built yet (run `tcs-bot medallion run`)",
    },
    422: {"model": ErrorEnvelope, "description": "Validation error"},
    503: {"model": ErrorEnvelope, "description": "Elasticsearch / embedder unavailable"},
}

router = APIRouter(prefix="/tcs/risks", tags=["tcs-risk"], responses=_ERRORS)


@router.get("/search", response_model=list[GoldHit])
def search_risks(
    q: Annotated[str, Query(min_length=1, max_length=500, description="Free-text risk query")],
    k: Annotated[int, Query(ge=1, le=100)] = 10,
    category: Annotated[
        list[str] | None,
        Query(description=f"Filter by risk category (repeatable): {', '.join(RISK_CATEGORIES)}"),
    ] = None,
    min_risk: Annotated[float | None, Query(ge=0, le=100)] = None,
) -> list[GoldHit]:
    """Semantic search over gold chunks (best chunk per document), optionally filtered."""
    from tcs_bot.errors import ValidationAppError

    bad = sorted(set(category or []) - set(RISK_CATEGORIES))
    if bad:
        raise ValidationAppError(f"unknown risk category: {', '.join(bad)}", details=bad)
    return gold.search_risks(q, k=k, categories=category, min_risk=min_risk)


@router.get("/summary")
def risk_summary() -> dict[str, Any]:
    """Gold ``summary.json``: meta, KMeans clusters, Louvain communities and statistics."""
    return gold.load_summary()


@router.get("/graph")
def risk_graph() -> dict[str, Any]:
    """Gold ``graph.json``: nodes (docs), k-NN cosine links, clusters and statistics."""
    return gold.load_graph()


@router.get("/graph.html", response_class=FileResponse)
def risk_graph_html() -> FileResponse:
    """Self-contained interactive graph visualization."""
    return FileResponse(gold.graph_html_path(), media_type="text/html")


# ------------------------------------------------------------ typed read layer (AC-005.12)
Data = Annotated[TcsData, Depends(get_tcs_data)]


def _build[M: BaseModel](model: type[M], **fields: Any) -> M:
    """Construct an input model; cross-field rule violations become a 422 error envelope."""
    try:
        return model(**fields)
    except ValidationError as exc:
        errors = exc.errors(include_context=False, include_url=False, include_input=False)
        msg = "; ".join(str(e["msg"]).removeprefix("Value error, ") for e in errors)
        details = [{"loc": [str(p) for p in e["loc"]]} for e in errors]
        raise ValidationAppError(msg, details=details) from exc


@router.get("/documents", response_model=DocumentPage, summary="List silver documents")
def list_documents(
    data: Data,
    category: Annotated[
        list[RiskCategory] | None, Query(description="Risk category (repeatable, any of)")
    ] = None,
    min_severity: Annotated[
        Severity | None, Query(description="At least one risk signal of this severity or higher")
    ] = None,
    min_risk_score: Annotated[float | None, Query(ge=0, le=100)] = None,
    source: Annotated[list[str] | None, Query(description="Crawler source (repeatable)")] = None,
    source_type: Annotated[
        list[SourceType] | None, Query(description="news / community / ...")
    ] = None,
    relevant: Annotated[
        Literal["true", "false", "all"],
        Query(description="true: TCS-relevant only (default); false: audit-only; all"),
    ] = "true",
    published_from: Annotated[datetime | None, Query(description="ISO date / datetime")] = None,
    published_to: Annotated[datetime | None, Query(description="ISO date / datetime")] = None,
    cluster: Annotated[int | None, Query(ge=0, description="Gold KMeans cluster id")] = None,
    community: Annotated[int | None, Query(ge=0, description="Gold Louvain community id")] = None,
    q: Annotated[str | None, Query(max_length=500, description="Free text on title / text")] = None,
    sort: SortField = "risk_score",
    order: SortOrder = "desc",
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = 20,
    offset: Annotated[int, Query(ge=0, le=MAX_WINDOW)] = 0,
) -> DocumentPage:
    """Silver-backed list with filters, sorting and pagination (``total`` = all matches)."""
    filters = _build(
        DocumentFilters,
        category=category or [],
        min_severity=min_severity,
        min_risk_score=min_risk_score,
        source=source or [],
        source_type=source_type or [],
        relevant=None if relevant == "all" else relevant == "true",
        published_from=published_from,
        published_to=published_to,
        cluster=cluster,
        community=community,
        q=q,
    )
    page = _build(Page, limit=limit, offset=offset, sort=sort, order=order)
    return data.list_documents(filters, page)


@router.get(
    "/documents/{doc_id}", response_model=DocumentDetail, summary="Document: silver + bronze + gold"
)
def get_document(
    doc_id: int,
    data: Data,
    include_text: Annotated[bool, Query(description="Include silver clean_text")] = False,
    include_raw: Annotated[
        bool, Query(description="Include bronze raw_content (large; off by default)")
    ] = False,
) -> DocumentDetail:
    """Join of bronze lineage (``id``), silver record (risk signals + evidence) and gold node."""
    return data.get_document(doc_id, include_raw=include_raw, include_text=include_text)


@router.get(
    "/documents/{doc_id}/similar",
    response_model=SimilarDocuments,
    summary="Nearest documents by cosine",
)
def similar_documents(
    doc_id: int, data: Data, k: Annotated[int, Query(ge=1, le=50)] = 10
) -> SimilarDocuments:
    """kNN over the stored gold vector of ``doc_id`` (best first, the document itself excluded)."""
    return data.similar_documents(doc_id, k=k)


@router.get("/categories", response_model=CategoryBreakdown, summary="Risk category breakdown")
def category_breakdown(data: Data, relevant_only: bool = True) -> CategoryBreakdown:
    """Documents, mean and max risk per category (all 10 categories, zero counts included)."""
    return data.category_breakdown(relevant_only=relevant_only)


@router.get("/overview", response_model=RiskOverview, summary="Risk overview")
def risk_overview(
    data: Data,
    relevant_only: bool = True,
    top_n: Annotated[int, Query(ge=1, le=20, description="Top documents to return")] = 5,
) -> RiskOverview:
    """Counts and mean / max risk by category, severity, source, source type and month."""
    return data.risk_overview(relevant_only=relevant_only, top_n=top_n)


@router.get("/clusters", response_model=list[ClusterSummary], summary="Gold clusters")
def list_clusters(data: Data, kind: ClusterKind = "cluster") -> list[ClusterSummary]:
    """KMeans clusters (``kind=cluster``) or Louvain communities (``kind=community``)."""
    return data.list_clusters(kind)


@router.get("/clusters/{cluster_id}", response_model=ClusterDetail, summary="One gold cluster")
def get_cluster(cluster_id: int, data: Data, kind: ClusterKind = "cluster") -> ClusterDetail:
    """Cluster summary plus its member documents."""
    return data.get_cluster(cluster_id, kind)
