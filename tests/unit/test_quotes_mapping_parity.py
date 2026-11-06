"""Contract: crawler and app create ``market-quotes`` identically (spec 002 plan, spec 001 plan).

Whichever process creates the index first wins, so mapping and index settings must be equal.
"""

from __future__ import annotations

from gfinance_crawler import sink as crawler_sink
from gfinance_crawler.models import Quote

from tcs_bot.search import elastic


def test_quotes_mapping_identical_in_crawler_and_app() -> None:
    """AC-002.4 / AC-001.7: crawler QUOTES_MAPPING == app QUOTES_MAPPING."""
    assert crawler_sink.QUOTES_MAPPING == elastic.QUOTES_MAPPING


def test_quotes_mapping_covers_quote_model() -> None:
    """AC-002.3 / AC-002.4: every Quote field is mapped explicitly, nothing extra."""
    assert set(Quote.model_fields) == set(elastic.QUOTES_MAPPING["properties"])


def test_quotes_index_settings_identical() -> None:
    """AC-002.4 / AC-001.7: both sides create market-quotes with 1 shard / 0 replicas."""
    assert crawler_sink.QUOTES_INDEX_SETTINGS == elastic._INDEX_SETTINGS
    assert elastic._INDEX_SETTINGS == {"number_of_shards": 1, "number_of_replicas": 0}
