"""Domain models shared by db/, search/, rag/ and services/ (contract: specs/001-data-layer/plan.md)."""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Criticality = Literal["low", "medium", "high", "critical"]
DocType = Literal["financial_report", "news", "regulatory_filing", "other"]


class VendorCreate(BaseModel):
    external_id: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=255)
    sector: str = "unknown"
    country: str = "unknown"
    ticker: str | None = Field(default=None, description="Google Finance pair, e.g. 'IBM:NYSE'")
    criticality: Criticality = "medium"
    services: str = ""


class Vendor(VendorCreate):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime | None = None
    updated_at: datetime | None = None


class DocumentCreate(BaseModel):
    doc_type: DocType = "other"
    title: str = Field(min_length=1, max_length=255)
    source: str = "manual"
    published_at: date | None = None
    content: str = Field(min_length=1)


class VendorDocument(DocumentCreate):
    id: int
    vendor_id: int
    content_sha256: str
    created_at: datetime | None = None


class ChunkRecord(BaseModel):
    document_id: int | None = None
    chunk_index: int
    text: str
    embedding: list[float] | None = None
    embedding_model: str | None = None
    doc_type: str | None = None
    title: str | None = None
