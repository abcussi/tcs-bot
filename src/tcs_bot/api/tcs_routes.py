"""TCS risk gold endpoints (AC-005.9): semantic search, summary, graph JSON and graph HTML."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Query
from fastapi.responses import FileResponse

from tcs_bot.medallion.gold import search as gold
from tcs_bot.medallion.models import RISK_CATEGORIES, GoldHit
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
