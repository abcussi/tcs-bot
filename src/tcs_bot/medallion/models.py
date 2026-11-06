"""Medallion contracts shared by bronze / silver / gold (spec 005). Change only with the plan."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

RiskCategory = Literal[
    "ethics",
    "compliance_regulatory",
    "legal_litigation",
    "data_privacy_cyber",
    "labor_hr",
    "financial",
    "operational_delivery",
    "reputational",
    "esg",
    "geopolitical_immigration",
]
RISK_CATEGORIES: tuple[str, ...] = RiskCategory.__args__  # type: ignore[attr-defined]
Severity = Literal["low", "medium", "high", "critical"]
SourceType = Literal["news", "community", "corporate", "other"]


def _now() -> datetime:
    return datetime.now(UTC)


# ------------------------------------------------------------------ bronze
class BronzeDocument(BaseModel):
    """Raw, unstructured document exactly as fetched (stored in MySQL)."""

    id: int | None = None
    run_id: int | None = None
    url: str
    canonical_url: str
    parent_url: str | None = None
    depth: int = Field(ge=0, le=2)
    source: str  # google_news | bing_news | gdelt | hackernews | reddit | local_inbox | web
    source_type: SourceType = "news"
    query: str = ""
    title: str | None = None
    raw_content: str
    content_type: str = "text/html"
    http_status: int = 200
    published_at: datetime | None = None
    fetched_at: datetime = Field(default_factory=_now)


class CrawlStats(BaseModel):
    run_id: int | None = None
    discovered: int = 0
    fetched: int = 0
    stored: int = 0
    duplicates: int = 0
    skipped_irrelevant: int = 0
    blocked: int = 0  # robots.txt / proxy / 4xx
    failed: int = 0
    by_depth: dict[str, int] = Field(default_factory=dict)
    by_source: dict[str, int] = Field(default_factory=dict)


# ------------------------------------------------------------------ silver
class RiskSignal(BaseModel):
    category: RiskCategory
    severity: Severity
    keyword: str = Field(description="Rule or concept that fired")
    evidence: str = Field(description="Sentence from the document supporting the signal")


class Entities(BaseModel):
    organizations: list[str] = Field(default_factory=list)
    people: list[str] = Field(default_factory=list)
    regulators: list[str] = Field(default_factory=list)
    places: list[str] = Field(default_factory=list)
    money: list[str] = Field(default_factory=list)


class SilverRecord(BaseModel):
    """Structured, cleaned, classified document (stored in Elasticsearch)."""

    doc_id: int  # bronze id
    url: str
    domain: str
    source: str
    source_type: SourceType
    depth: int
    parent_url: str | None = None
    query: str = ""
    title: str
    clean_text: str
    summary: str = ""
    language: str = "en"
    word_count: int = 0
    published_at: datetime | None = None
    entities: Entities = Field(default_factory=Entities)
    tcs_relevance: float = Field(ge=0, le=1, default=0)
    relevant: bool = False
    risk_signals: list[RiskSignal] = Field(default_factory=list)
    risk_categories: list[RiskCategory] = Field(default_factory=list)
    sentiment: float = Field(ge=-1, le=1, default=0)
    risk_score: float = Field(ge=0, le=100, default=0)
    extractor: str = "rules-v1"  # or the LLM model id
    processed_at: datetime = Field(default_factory=_now)


# ------------------------------------------------------------------ gold
class GoldHit(BaseModel):
    doc_id: int
    chunk_index: int | None = None
    score: float
    title: str
    url: str
    text: str = ""
    risk_categories: list[str] = Field(default_factory=list)
    risk_score: float = 0
    cluster: int | None = None


class ClusterSummary(BaseModel):
    cluster: int
    size: int
    label: str
    top_terms: list[str] = Field(default_factory=list)
    dominant_categories: dict[str, int] = Field(default_factory=dict)
    mean_risk_score: float = 0
    mean_intra_cosine: float = 0
    representative_doc_ids: list[int] = Field(default_factory=list)
