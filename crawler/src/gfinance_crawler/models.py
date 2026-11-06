"""Quote model (AC-002.3)."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Quote(BaseModel):
    ticker: str
    exchange: str
    name: str | None = None
    price: float | None = None
    currency: str | None = None
    change: float | None = None
    change_pct: float | None = None
    previous_close: float | None = None
    day_range: str | None = None
    year_range: str | None = None
    market_cap: float | None = None
    pe_ratio: float | None = None
    dividend_yield: float | None = None
    scraped_at: datetime = Field(default_factory=_utcnow)
    source_url: str

    def doc_id(self) -> str:
        """Idempotent per minute (AC-002.4)."""
        return f"{self.ticker}:{self.exchange}:{self.scraped_at:%Y%m%d%H%M}"

    def to_document(self) -> dict:
        return self.model_dump(mode="json")
