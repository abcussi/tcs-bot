"""Crawler-side models. ``RawDocument`` mirrors ``tcs_bot.medallion.models.BronzeDocument`` minus
``id``/``run_id`` (the crawler never imports tcs_bot); bronze maps it 1:1.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

SourceType = Literal["news", "community", "corporate", "other"]


def _now() -> datetime:
    return datetime.now(UTC)


class RawDocument(BaseModel):
    """A document exactly as fetched (depth 0 = search/listing item, 1 = article/thread, 2 = link).

    ``follow_url`` is crawler-internal (where depth+1 lives) and is NOT persisted by bronze.
    """

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
    follow_url: str | None = Field(default=None, exclude=True)


class FetchResult(BaseModel):
    """Outcome of one polite HTTP GET. ``PoliteFetcher.fetch`` never raises; it returns this."""

    url: str
    final_url: str
    status: int = 0  # 0 = no HTTP response (transport error / robots / proxy)
    content_type: str = ""
    text: str = ""
    error: str | None = None
    blocked: bool = False  # robots.txt disallow, 401/403/451, proxy CONNECT refusal

    @property
    def ok(self) -> bool:
        return self.error is None and 200 <= self.status < 300
