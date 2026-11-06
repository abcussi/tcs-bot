"""Structured LLM output contract (AC-003.6). The LLM must return exactly this shape."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

RiskLevel = Literal["low", "medium", "high", "critical"]
Severity = Literal["low", "medium", "high", "critical"]
Priority = Literal["low", "medium", "high"]


def score_to_level(score: float) -> RiskLevel:
    if score >= 75:
        return "critical"
    if score >= 50:
        return "high"
    if score >= 25:
        return "medium"
    return "low"


class RedFlag(BaseModel):
    category: Literal["financial", "operational", "compliance", "reputational", "cyber", "concentration", "other"]
    severity: Severity
    description: str = Field(description="One-sentence description of the red flag")
    evidence: str = Field(default="", description="Short quote or paraphrase from the source chunks")


class FinancialHealth(BaseModel):
    summary: str
    liquidity: str = Field(default="unknown", description="Assessment of liquidity / cash position")
    profitability: str = Field(default="unknown")
    leverage: str = Field(default="unknown")
    market_signal: str = Field(default="unknown", description="What market data (price, change, P/E) suggests")


class ComplianceRecommendation(BaseModel):
    priority: Priority
    recommendation: str
    regulation_reference: str = Field(default="", description="e.g. DORA, SEC Reg SCI, FINRA 3110, SOX")


class SourceRef(BaseModel):
    document_id: int | None = None
    chunk_index: int
    title: str | None = None
    score: float = Field(ge=-1.0, le=1.0)


class RiskReport(BaseModel):
    vendor_id: int
    vendor_name: str
    overall_risk_score: float = Field(ge=0, le=100)
    risk_level: RiskLevel
    red_flags: list[RedFlag] = Field(default_factory=list)
    financial_health: FinancialHealth
    compliance_recommendations: list[ComplianceRecommendation] = Field(default_factory=list)
    risk_narrative: str
    sources: list[SourceRef] = Field(default_factory=list)
    data_gaps: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    model: str = "unknown"


class StoredReport(BaseModel):
    id: int
    vendor_id: int
    report: RiskReport
    model: str
    created_at: datetime | None = None
