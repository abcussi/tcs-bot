"""Typed read models for the medallion read layer (spec 005, AC-005.12).

Inputs: :class:`DocumentFilters`, :class:`Page`. Outputs: :class:`DocumentSummary` /
:class:`DocumentPage` (list), :class:`DocumentDetail` (silver + bronze lineage + gold node),
:class:`SimilarDocuments`, :class:`CategoryBreakdown`, :class:`RiskOverview`,
:class:`ClusterDetail`.
Existing contracts are reused (:class:`SilverRecord`, :class:`RiskSignal`, :class:`GoldHit`,
:class:`ClusterSummary`); nothing here is written back to a layer.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from tcs_bot.medallion.models import (
    ClusterSummary,
    GoldHit,
    RiskCategory,
    Severity,
    SilverRecord,
    SourceType,
)

SEVERITIES: tuple[str, ...] = ("low", "medium", "high", "critical")
SortField = Literal["risk_score", "published_at", "tcs_relevance"]
SortOrder = Literal["asc", "desc"]
ClusterKind = Literal["cluster", "community"]
MAX_PAGE_SIZE = 100
MAX_WINDOW = 10_000  # Elasticsearch ``index.max_result_window``


# ------------------------------------------------------------------ inputs
class DocumentFilters(BaseModel):
    """All filters are optional and ANDed; list filters match any listed value."""

    category: list[RiskCategory] = Field(default_factory=list)
    min_severity: Severity | None = Field(
        default=None, description="Keep documents with a risk signal of at least this severity"
    )
    min_risk_score: float | None = Field(default=None, ge=0, le=100)
    source: list[str] = Field(default_factory=list)
    source_type: list[SourceType] = Field(default_factory=list)
    relevant: bool | None = Field(
        default=True, description="True: relevant only (default), False: audit-only, None: all"
    )
    published_from: datetime | None = None
    published_to: datetime | None = None
    cluster: int | None = Field(default=None, ge=0, description="Gold KMeans cluster id")
    community: int | None = Field(default=None, ge=0, description="Gold Louvain community id")
    q: str | None = Field(default=None, max_length=500, description="Free text on title / text")

    @field_validator("q")
    @classmethod
    def _strip_q(cls, v: str | None) -> str | None:
        v = (v or "").strip()
        return v or None

    @model_validator(mode="after")
    def _date_order(self) -> DocumentFilters:
        if self.published_from and self.published_to:
            a, b = self.published_from, self.published_to
            if (a.tzinfo is None) != (b.tzinfo is None):
                a, b = a.replace(tzinfo=None), b.replace(tzinfo=None)
            if a > b:
                raise ValueError("published_from must be <= published_to")
        return self


class Page(BaseModel):
    limit: int = Field(default=20, ge=1, le=MAX_PAGE_SIZE)
    offset: int = Field(default=0, ge=0)
    sort: SortField = "risk_score"
    order: SortOrder = "desc"

    @model_validator(mode="after")
    def _window(self) -> Page:
        if self.offset + self.limit > MAX_WINDOW:
            raise ValueError(f"offset + limit must be <= {MAX_WINDOW}")
        return self


# ------------------------------------------------------------------ list
class DocumentSummary(BaseModel):
    """One row of a document list (no body text)."""

    doc_id: int
    title: str
    url: str
    domain: str = ""
    source: str
    source_type: SourceType
    published_at: datetime | None = None
    summary: str = ""
    tcs_relevance: float = 0
    relevant: bool = True
    risk_score: float = 0
    risk_categories: list[str] = Field(default_factory=list)
    sentiment: float = 0
    n_signals: int = 0
    max_severity: Severity | None = None
    language: str | None = None
    word_count: int | None = None


class DocumentPage(BaseModel):
    total: int = Field(description="Documents matching the filters (not just this page)")
    limit: int
    offset: int
    items: list[DocumentSummary]


# ------------------------------------------------------------------ detail
class SilverView(SilverRecord):
    """Silver record; ``clean_text`` is ``None`` unless the caller asked for it."""

    clean_text: str | None = None  # type: ignore[assignment]


class BronzeLineage(BaseModel):
    """Bronze row without its body. ``raw_content`` only with ``include_raw=True``."""

    id: int
    run_id: int | None = None
    url: str
    canonical_url: str
    parent_url: str | None = None
    depth: int
    source: str
    source_type: SourceType
    query: str = ""
    title: str | None = None
    content_type: str = ""
    http_status: int = 0
    published_at: datetime | None = None
    fetched_at: datetime | None = None
    silver_processed_at: datetime | None = None
    raw_chars: int | None = Field(default=None, description="Length of the stored raw content")
    raw_content: str | None = None


class GoldNode(BaseModel):
    """What gold knows about a document (``graph.json`` node)."""

    in_gold: bool = Field(description="False when dedupe folded the document into another one")
    merged_into: int | None = Field(
        default=None, description="Kept doc id when this document was merged as a duplicate"
    )
    cluster: int | None = None
    cluster_label: str | None = None
    community: int | None = None
    degree: float | None = None
    betweenness: float | None = None
    eigenvector: float | None = None
    x: float | None = None
    y: float | None = None
    merged_doc_ids: list[int] = Field(default_factory=list)
    also_seen_in: int = 0
    merged_sources: list[dict[str, Any]] = Field(default_factory=list)
    neighbors: list[tuple[int, float]] = Field(
        default_factory=list, description="[doc_id, cosine] nearest documents in the graph"
    )


class DocumentDetail(BaseModel):
    doc_id: int
    silver: SilverView
    bronze: BronzeLineage | None = Field(
        default=None, description="None when the bronze reader is disabled"
    )
    gold: GoldNode | None = Field(default=None, description="None when gold is not built")


class SimilarDocuments(BaseModel):
    doc_id: int
    resolved_doc_id: int = Field(
        description="Document whose vector was used (differs when doc_id was merged in gold)"
    )
    model: str = Field(description="Embedding model slug of the gold index")
    neighbours: list[GoldHit] = Field(description="Ordered by cosine, best first; excludes itself")


# ------------------------------------------------------------------ aggregations
class CountBucket(BaseModel):
    key: str
    count: int


class RiskBucket(CountBucket):
    avg_risk_score: float = 0
    max_risk_score: float = 0


class CategoryBreakdown(BaseModel):
    relevant_only: bool
    total_documents: int
    categories: list[RiskBucket] = Field(
        description="All 10 categories (zero counts included), most documents first. A document "
        "with several categories counts in each, so counts can exceed total_documents."
    )


class RiskOverview(BaseModel):
    relevant_only: bool
    total_documents: int
    mean_risk_score: float = 0
    max_risk_score: float = 0
    by_category: list[RiskBucket]
    by_severity: list[CountBucket] = Field(description="Risk SIGNAL counts per severity")
    by_source: list[RiskBucket]
    by_source_type: list[RiskBucket]
    by_month: list[RiskBucket] = Field(description="key = YYYY-MM of published_at, ascending")
    top_documents: list[DocumentSummary] = Field(description="Highest risk_score first")


# ------------------------------------------------------------------ clusters
class ClusterDetail(BaseModel):
    kind: ClusterKind
    summary: ClusterSummary
    members: list[DocumentSummary] = Field(description="Gold graph nodes of this cluster")


__all__ = [
    "MAX_PAGE_SIZE",
    "MAX_WINDOW",
    "SEVERITIES",
    "BronzeLineage",
    "CategoryBreakdown",
    "ClusterDetail",
    "ClusterKind",
    "CountBucket",
    "DocumentDetail",
    "DocumentFilters",
    "DocumentPage",
    "DocumentSummary",
    "GoldNode",
    "Page",
    "RiskBucket",
    "RiskOverview",
    "SilverView",
    "SimilarDocuments",
    "SortField",
    "SortOrder",
]
