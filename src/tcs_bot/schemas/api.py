"""HTTP / MCP I/O models (AC-003.1, AC-003.7). Contract: specs/003-risk-report-generator/plan.md."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tcs_bot.schemas.domain import Criticality, DocumentCreate


class ReportRequest(BaseModel):
    """Ask for a risk report. Exactly one of ``vendor_id`` / ``vendor_name`` is needed (id wins)."""

    vendor_id: int | None = Field(default=None, ge=1)
    vendor_name: str | None = Field(default=None, min_length=1, max_length=255)
    focus_areas: list[str] = Field(
        default_factory=list,
        max_length=20,
        description="Optional emphasis, e.g. ['cyber', 'liquidity', 'DORA']",
    )
    top_k: int | None = Field(default=None, ge=1, le=20, description="Chunks to retrieve")
    use_cache: bool = True

    @model_validator(mode="after")
    def _one_vendor_selector(self) -> ReportRequest:
        if self.vendor_id is None and not (self.vendor_name and self.vendor_name.strip()):
            raise ValueError("one of vendor_id or vendor_name is required")
        self.focus_areas = [f.strip() for f in self.focus_areas if f and f.strip()]
        return self


class DocumentIn(DocumentCreate):
    """JSON body of ``POST /vendors/{id}/documents`` (same fields as ``DocumentCreate``)."""


class DocumentOut(BaseModel):
    id: int
    vendor_id: int
    chunks: int = Field(ge=0, description="Chunks stored for this document")
    indexed: bool = Field(description="Whether chunks were also indexed in Elasticsearch")


class VendorOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    external_id: str
    name: str
    sector: str
    country: str
    ticker: str | None = None
    criticality: Criticality
    services: str = ""


DependencyStatus = Literal["ok", "down"]


class HealthOut(BaseModel):
    status: Literal["ok", "degraded"]
    mysql: DependencyStatus
    redis: DependencyStatus
    elasticsearch: DependencyStatus
    llm_provider: Literal["openai", "fake"]


class ErrorBody(BaseModel):
    code: str
    message: str
    details: Any = None


class ErrorEnvelope(BaseModel):
    error: ErrorBody
