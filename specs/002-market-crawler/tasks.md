# 002 — Tasks

- [x] `config.py` CrawlerSettings (+ `ticker_pairs`, `crawler_delay_seconds`=1.0, `quote_url()`)
- [x] `models.py` Quote (AC-002.3) + `doc_id()` = `{ticker}:{exchange}:{YYYYmmddHHMM}` (AC-002.4)
- [x] `scraper.py` tolerant `parse_quote_html` (data-* attrs -> CSS classes -> label text) + number helpers
- [x] `scraper.py` GoogleFinanceScraper: tenacity retry (3x, exp backoff; transport/5xx/429), polite delay,
      `fetch_all` records failures in `.errors` and never raises (AC-002.2)
- [x] `sink.py` ElasticSink: explicit `market-quotes` mapping, `helpers.bulk` with `_id=doc_id()` (AC-002.4)
- [x] `run.py` `run_once` -> `{fetched, parsed, indexed, failed, es_error}`; `gfinance-crawl --once|--schedule`
      (APScheduler CronTrigger.from_crontab(CRAWLER_CRON), runs once at startup) (AC-002.5, AC-002.6)
- [x] `cron/crontab` (`*/15 * * * *`, in sync with CRAWLER_CRON) + `Dockerfile` (uv + supercronic)
- [x] Tests: fixtures `ibm_nyse.html` / `degraded.html`, parser, helpers, MockTransport scraper, sink, run,
      CLI; live-ES test marked `integration` (AC-002.7)

Notes
- Only quotes with a price are indexed; `failed` = configured tickers without a usable quote.
- Exit code 0 even with failed tickers; 1 only when ES is unreachable and nothing was indexed.
- `es_error` is an extra summary key (superset of AC-002.6) used for the exit code.
