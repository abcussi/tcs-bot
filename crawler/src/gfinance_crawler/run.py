"""Entrypoint: ``gfinance-crawl --once`` | ``gfinance-crawl --schedule`` (AC-002.5, AC-002.6)."""

from __future__ import annotations

import argparse
import logging
import os
from typing import Any

from gfinance_crawler.config import CrawlerSettings
from gfinance_crawler.scraper import GoogleFinanceScraper
from gfinance_crawler.sink import ElasticSink

log = logging.getLogger("gfinance_crawler")


def run_once(
    settings: CrawlerSettings | None = None,
    scraper: GoogleFinanceScraper | None = None,
    sink: ElasticSink | None = None,
) -> dict[str, Any]:
    """Crawl every configured ticker once and index the results.

    Returns ``{fetched, parsed, indexed, failed}`` plus ``es_error`` (``None`` when ES was fine):
    - fetched: pages retrieved over HTTP,
    - parsed:  quotes with at least a price (only these are indexed),
    - indexed: documents written to Elasticsearch,
    - failed:  tickers that produced no usable quote (fetch error or unparseable page).
    """
    settings = settings or CrawlerSettings()
    pairs = settings.ticker_pairs
    own_scraper = scraper is None
    scraper = scraper or GoogleFinanceScraper(settings)
    try:
        quotes = scraper.fetch_all(pairs)
    finally:
        if own_scraper:
            scraper.close()

    usable = [q for q in quotes if q.price is not None]
    for q in quotes:
        if q.price is None:
            log.warning(
                "no price parsed for %s:%s (layout change or consent page?)", q.ticker, q.exchange
            )

    indexed = 0
    es_error: str | None = None
    if usable:
        try:
            sink = sink or ElasticSink(settings)
            sink.ensure_index()
            indexed = sink.index(usable)
        except Exception as exc:  # noqa: BLE001 - ES down must not crash the scheduler
            es_error = f"{type(exc).__name__}: {exc}"
            log.error("elasticsearch unavailable: %s", es_error)

    summary: dict[str, Any] = {
        "fetched": len(quotes),
        "parsed": len(usable),
        "indexed": indexed,
        "failed": len(pairs) - len(usable),
        "es_error": es_error,
    }
    log.info(
        "crawl summary fetched=%d parsed=%d indexed=%d failed=%d",
        summary["fetched"],
        summary["parsed"],
        summary["indexed"],
        summary["failed"],
    )
    return summary


def _schedule(settings: CrawlerSettings) -> int:
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger

    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_job(
        run_once,
        CronTrigger.from_crontab(settings.crawler_cron, timezone="UTC"),
        kwargs={"settings": settings},
        id="gfinance-crawl",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=300,
    )
    log.info("scheduler started with cron %r", settings.crawler_cron)
    run_once(settings)
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        log.info("scheduler stopped")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gfinance-crawl", description="Google Finance -> Elasticsearch crawler"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="crawl once and exit (default)")
    mode.add_argument("--schedule", action="store_true", help="run on CRAWLER_CRON (APScheduler)")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    settings = CrawlerSettings()
    if args.schedule:
        return _schedule(settings)

    summary = run_once(settings)
    # Partial ticker failures are fine; only "ES unreachable and nothing indexed" is an error.
    return 1 if summary["es_error"] and summary["indexed"] == 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
