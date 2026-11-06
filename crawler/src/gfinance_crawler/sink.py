"""Elasticsearch sink for quotes (AC-002.4)."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

from elasticsearch import BadRequestError, Elasticsearch, helpers

from gfinance_crawler.config import CrawlerSettings
from gfinance_crawler.models import Quote

log = logging.getLogger(__name__)

QUOTES_MAPPING: dict[str, Any] = {
    "properties": {
        "ticker": {"type": "keyword"},
        "exchange": {"type": "keyword"},
        "currency": {"type": "keyword"},
        "name": {"type": "text", "fields": {"keyword": {"type": "keyword", "ignore_above": 256}}},
        "price": {"type": "float"},
        "change": {"type": "float"},
        "change_pct": {"type": "float"},
        "previous_close": {"type": "float"},
        "market_cap": {"type": "float"},
        "pe_ratio": {"type": "float"},
        "dividend_yield": {"type": "float"},
        "day_range": {"type": "keyword"},
        "year_range": {"type": "keyword"},
        "scraped_at": {"type": "date"},
        "source_url": {"type": "keyword"},
    }
}

# Must match ``vendor_risk_ai.search.elastic._INDEX_SETTINGS``: whichever process creates
# ``market-quotes`` first wins (single-node ES -> 0 replicas, otherwise the index is yellow).
QUOTES_INDEX_SETTINGS: dict[str, Any] = {"number_of_shards": 1, "number_of_replicas": 0}


class ElasticSink:
    def __init__(self, settings: CrawlerSettings | None = None, client: Any | None = None) -> None:
        self.settings = settings or CrawlerSettings()
        self.index_name = self.settings.elastic_quotes_index
        self.client = client or Elasticsearch(self.settings.elastic_url, request_timeout=10)

    def ensure_index(self) -> None:
        """Create the quotes index with an explicit mapping if it does not exist (idempotent)."""
        if self.client.indices.exists(index=self.index_name):
            return
        try:
            self.client.indices.create(
                index=self.index_name, mappings=QUOTES_MAPPING, settings=QUOTES_INDEX_SETTINGS
            )
            log.info("created index %s", self.index_name)
        except BadRequestError as exc:  # created concurrently by another process
            if "resource_already_exists" not in str(exc.message):
                raise

    def _actions(self, quotes: Iterable[Quote]) -> Iterable[dict[str, Any]]:
        for quote in quotes:
            yield {
                "_op_type": "index",
                "_index": self.index_name,
                "_id": quote.doc_id(),
                "_source": quote.to_document(),
            }

    def index(self, quotes: Iterable[Quote]) -> int:
        """Bulk index quotes with ``_id = quote.doc_id()``; returns the number indexed."""
        quotes = list(quotes)
        if not quotes:
            return 0
        success, errors = helpers.bulk(
            self.client, self._actions(quotes), raise_on_error=False, stats_only=False
        )
        if errors:
            log.warning("%d quote(s) failed to index: %s", len(errors), errors[:3])
        return int(success)
