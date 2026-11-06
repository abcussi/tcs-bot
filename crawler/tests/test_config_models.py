from __future__ import annotations

from datetime import UTC, datetime

from gfinance_crawler.config import CrawlerSettings, parse_tickers
from gfinance_crawler.models import Quote


def test_parse_tickers_handles_spaces_case_and_junk() -> None:
    """AC-002.2 CRAWLER_TICKERS comma list -> (ticker, exchange) pairs."""
    raw = " aapl:nasdaq , IBM:NYSE,,bogus, IBM:NYSE ,:NYSE"
    assert parse_tickers(raw) == [("AAPL", "NASDAQ"), ("IBM", "NYSE")]


def test_settings_from_env(monkeypatch) -> None:
    """AC-002.2 settings are env driven; delay defaults to 1s."""
    monkeypatch.setenv("CRAWLER_TICKERS", "JPM:NYSE,MSFT:NASDAQ")
    monkeypatch.setenv("ELASTIC_QUOTES_INDEX", "q-idx")
    monkeypatch.delenv("CRAWLER_DELAY_SECONDS", raising=False)
    s = CrawlerSettings(_env_file=None)
    assert s.ticker_pairs == [("JPM", "NYSE"), ("MSFT", "NASDAQ")]
    assert s.elastic_quotes_index == "q-idx"
    assert s.crawler_delay_seconds == 1.0
    assert s.quote_url("JPM", "NYSE").endswith("/finance/quote/JPM:NYSE")


def test_quote_doc_id_is_per_minute() -> None:
    """AC-002.4 doc id = {ticker}:{exchange}:{scraped_at minute}."""
    q1 = Quote(
        ticker="IBM",
        exchange="NYSE",
        source_url="u",
        scraped_at=datetime(2026, 11, 6, 14, 5, 1, tzinfo=UTC),
    )
    q2 = q1.model_copy(update={"scraped_at": datetime(2026, 11, 6, 14, 5, 59, tzinfo=UTC)})
    assert q1.doc_id() == "IBM:NYSE:202611061405" == q2.doc_id()


def test_crawler_does_not_import_main_app() -> None:
    """AC-002.1 gfinance_crawler is standalone (never imports vendor_risk_ai)."""
    import pathlib
    import re

    import gfinance_crawler

    pkg = pathlib.Path(gfinance_crawler.__file__).parent
    pattern = re.compile(r"^\s*(from|import)\s+vendor_risk_ai", re.MULTILINE)
    for source in pkg.glob("*.py"):
        assert not pattern.search(source.read_text()), source.name
