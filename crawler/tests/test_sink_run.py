from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from elasticsearch import BadRequestError
from gfinance_crawler import run as run_mod
from gfinance_crawler import sink as sink_mod
from gfinance_crawler.config import CrawlerSettings
from gfinance_crawler.models import Quote
from gfinance_crawler.sink import QUOTES_MAPPING, ElasticSink


def _quote(ticker: str = "IBM", price: float | None = 187.45) -> Quote:
    return Quote(
        ticker=ticker,
        exchange="NYSE",
        price=price,
        source_url="u",
        scraped_at=datetime(2026, 11, 6, 14, 5, tzinfo=UTC),
    )


class FakeIndices:
    def __init__(self, exists: bool = False) -> None:
        self._exists = exists
        self.created: list[dict[str, Any]] = []

    def exists(self, index: str) -> bool:
        return self._exists

    def create(
        self, index: str, mappings: dict[str, Any], settings: dict[str, Any] | None = None
    ) -> None:
        self.created.append({"index": index, "mappings": mappings, "settings": settings})
        self._exists = True


class FakeES:
    def __init__(self, exists: bool = False) -> None:
        self.indices = FakeIndices(exists)


def test_ensure_index_creates_explicit_mapping(settings: CrawlerSettings) -> None:
    """AC-002.4 market-quotes mapping: keywords, text+keyword name, floats, date."""
    es = FakeES()
    sink = ElasticSink(settings, client=es)
    sink.ensure_index()
    sink.ensure_index()  # idempotent
    assert len(es.indices.created) == 1
    props = es.indices.created[0]["mappings"]["properties"]
    assert es.indices.created[0]["index"] == "market-quotes-test"
    for f in ("ticker", "exchange", "currency", "day_range", "year_range", "source_url"):
        assert props[f]["type"] == "keyword"
    assert (
        props["name"]["type"] == "text" and props["name"]["fields"]["keyword"]["type"] == "keyword"
    )
    for f in (
        "price",
        "change",
        "change_pct",
        "previous_close",
        "market_cap",
        "pe_ratio",
        "dividend_yield",
    ):
        assert props[f]["type"] == "float"
    assert props["scraped_at"]["type"] == "date"
    assert set(props) == set(Quote.model_fields)


def test_ensure_index_single_node_settings(settings: CrawlerSettings) -> None:
    """AC-002.4 / spec 002 plan (mapping parity): 1 shard, 0 replicas, like the main app."""
    es = FakeES()
    ElasticSink(settings, client=es).ensure_index()
    assert es.indices.created[0]["settings"] == {"number_of_shards": 1, "number_of_replicas": 0}


def test_ensure_index_tolerates_race(settings: CrawlerSettings) -> None:
    """AC-002.4 concurrent creation (resource_already_exists) is not an error."""
    es = FakeES()

    def create(index: str, mappings: dict[str, Any], **_: Any) -> None:
        raise BadRequestError("resource_already_exists_exception", meta=None, body={})  # type: ignore[arg-type]

    es.indices.create = create  # type: ignore[method-assign]
    ElasticSink(settings, client=es).ensure_index()


def test_index_uses_bulk_with_doc_id(settings: CrawlerSettings, monkeypatch) -> None:
    """AC-002.4 bulk index with _id = doc_id() and return the count."""
    captured: dict[str, Any] = {}

    def fake_bulk(client: Any, actions: Any, **kwargs: Any) -> tuple[int, list[Any]]:
        captured["actions"] = list(actions)
        captured["kwargs"] = kwargs
        return len(captured["actions"]), []

    monkeypatch.setattr(sink_mod.helpers, "bulk", fake_bulk)
    sink = ElasticSink(settings, client=FakeES(exists=True))
    assert sink.index([_quote("IBM"), _quote("MSFT")]) == 2
    first = captured["actions"][0]
    assert first["_id"] == "IBM:NYSE:202611061405"
    assert first["_index"] == "market-quotes-test"
    assert first["_source"]["price"] == 187.45
    assert first["_source"]["scraped_at"].startswith("2026-11-06T14:05")
    assert sink.index([]) == 0


class FakeScraper:
    def __init__(self, quotes: list[Quote]) -> None:
        self.quotes = quotes
        self.pairs: list[tuple[str, str]] | None = None

    def fetch_all(self, pairs: list[tuple[str, str]]) -> list[Quote]:
        self.pairs = pairs
        return self.quotes

    def close(self) -> None:  # pragma: no cover - not called for injected scrapers
        raise AssertionError("injected scraper must not be closed by run_once")


class FakeSink:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.ensured = False
        self.indexed: list[Quote] = []

    def ensure_index(self) -> None:
        if self.fail:
            raise ConnectionError("es down")
        self.ensured = True

    def index(self, quotes: list[Quote]) -> int:
        self.indexed = list(quotes)
        return len(self.indexed)


def test_run_once_summary(settings: CrawlerSettings) -> None:
    """AC-002.6 summary {fetched, parsed, indexed, failed}; price-less quotes not indexed."""
    three = settings.model_copy(update={"crawler_tickers": "IBM:NYSE,EMPTY:NYSE,BAD:NYSE"})
    scraper = FakeScraper([_quote("IBM"), _quote("EMPTY", price=None)])  # BAD failed to fetch
    sink = FakeSink()
    summary = run_mod.run_once(three, scraper=scraper, sink=sink)  # type: ignore[arg-type]
    assert scraper.pairs == [("IBM", "NYSE"), ("EMPTY", "NYSE"), ("BAD", "NYSE")]
    assert sink.ensured and [q.ticker for q in sink.indexed] == ["IBM"]
    assert {k: summary[k] for k in ("fetched", "parsed", "indexed", "failed")} == {
        "fetched": 2,
        "parsed": 1,
        "indexed": 1,
        "failed": 2,
    }
    assert summary["es_error"] is None


def test_run_once_es_down_does_not_raise(settings: CrawlerSettings) -> None:
    """AC-002.6 ES unreachable -> indexed=0 and es_error set, run still returns."""
    summary = run_mod.run_once(
        settings,
        scraper=FakeScraper([_quote()]),
        sink=FakeSink(fail=True),  # type: ignore[arg-type]
    )
    assert summary["indexed"] == 0 and "es down" in summary["es_error"]


@pytest.mark.parametrize(
    ("es_error", "indexed", "code"),
    [
        (None, 3, 0),
        (None, 0, 0),
        ("ConnectionError", 0, 1),
    ],
)
def test_main_once_exit_codes(monkeypatch, es_error: str | None, indexed: int, code: int) -> None:
    """AC-002.5 `--once` exits 0 despite ticker failures; 1 only if ES down and 0 indexed."""
    summary = {"fetched": 3, "parsed": 3, "indexed": indexed, "failed": 5, "es_error": es_error}
    monkeypatch.setattr(run_mod, "run_once", lambda settings=None: summary)
    assert run_mod.main(["--once"]) == code
    assert run_mod.main([]) == code  # --once is the default


def test_main_schedule_uses_cron(monkeypatch) -> None:
    """AC-002.5 `--schedule` runs once at startup and registers CRAWLER_CRON with APScheduler."""
    from apscheduler.schedulers.blocking import BlockingScheduler

    ran: list[Any] = []
    jobs: list[Any] = []
    monkeypatch.setenv("CRAWLER_CRON", "*/5 * * * *")
    monkeypatch.setattr(run_mod, "run_once", lambda settings=None: ran.append(settings) or {})
    monkeypatch.setattr(BlockingScheduler, "start", lambda self: jobs.extend(self.get_jobs()))
    assert run_mod.main(["--schedule"]) == 0
    assert len(ran) == 1
    assert len(jobs) == 1 and "minute='*/5'" in str(jobs[0].trigger)


def test_mapping_constant_is_used_by_integration_readers() -> None:
    """AC-002.4 index name defaults to market-quotes."""
    assert CrawlerSettings(_env_file=None).elastic_quotes_index == "market-quotes"
    assert "properties" in QUOTES_MAPPING
