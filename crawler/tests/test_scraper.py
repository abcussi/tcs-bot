from __future__ import annotations

import httpx
import pytest
from gfinance_crawler.config import CrawlerSettings
from gfinance_crawler.scraper import GoogleFinanceScraper


def _scraper(settings: CrawlerSettings, handler, sleeps: list[float]) -> GoogleFinanceScraper:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return GoogleFinanceScraper(settings, client, sleep=sleeps.append)


def test_fetch_all_collects_successes_and_counts_failures(
    settings: CrawlerSettings, ibm_html: str
) -> None:
    """AC-002.2 retries 5xx (3 attempts) and a failing ticker does not stop the run."""
    calls: dict[str, int] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        key = request.url.path.rsplit("/", 1)[-1]
        calls[key] = calls.get(key, 0) + 1
        assert request.url.params["hl"] == "en"
        if key == "IBM:NYSE":
            return httpx.Response(200, text=ibm_html)
        return httpx.Response(500, text="boom")

    sleeps: list[float] = []
    scraper = _scraper(settings, handler, sleeps)
    quotes = scraper.fetch_all([("BAD", "NYSE"), ("IBM", "NYSE")])

    assert [q.ticker for q in quotes] == ["IBM"]
    assert quotes[0].price == pytest.approx(187.45)
    assert quotes[0].source_url == "https://finance.test/quote/IBM:NYSE"
    assert calls == {"BAD:NYSE": 3, "IBM:NYSE": 1}
    assert scraper.fetched == 1 and scraper.failed == 1
    assert "BAD:NYSE" in scraper.errors
    assert len(sleeps) == 2  # two backoff waits; polite delay is 0 in these settings


def test_retry_then_success_on_429_and_transport_error(
    settings: CrawlerSettings, ibm_html: str
) -> None:
    """AC-002.2 transient 429 / transport errors are retried."""
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise httpx.ConnectError("reset", request=request)
        if attempts["n"] == 2:
            return httpx.Response(429)
        return httpx.Response(200, text=ibm_html)

    scraper = _scraper(settings, handler, [])
    quote = scraper.fetch("IBM", "NYSE")
    assert quote.price == pytest.approx(187.45)
    assert attempts["n"] == 3


def test_404_is_not_retried(settings: CrawlerSettings) -> None:
    """AC-002.2 client errors (other than 429) fail fast."""
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return httpx.Response(404)

    scraper = _scraper(settings, handler, [])
    assert scraper.fetch_all([("NOPE", "NYSE")]) == []
    assert attempts["n"] == 1 and scraper.failed == 1


def test_polite_delay_between_tickers(settings: CrawlerSettings, ibm_html: str) -> None:
    """AC-002.2 polite delay is applied between tickers (not before the first)."""
    polite = settings.model_copy(update={"crawler_delay_seconds": 0.25})
    sleeps: list[float] = []
    scraper = _scraper(polite, lambda r: httpx.Response(200, text=ibm_html), sleeps)
    scraper.fetch_all([("IBM", "NYSE"), ("IBM", "NYSE"), ("IBM", "NYSE")])
    assert sleeps == [0.25, 0.25]


def test_default_client_uses_settings(settings: CrawlerSettings) -> None:
    """AC-002.2 timeout and user agent come from settings."""
    with GoogleFinanceScraper(settings) as scraper:
        assert scraper.client.headers["User-Agent"] == settings.crawler_user_agent
        assert scraper.client.timeout.read == settings.crawler_timeout_seconds
