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

## Open (architect review 2026-11-06)
- [x] `sink.py:48` create the index with `settings={"number_of_shards": 1, "number_of_replicas": 0}`
      (today a crawler-first create gives 1 replica -> yellow on single-node ES).
- [x] Contract test in `tests/unit` asserting `gfinance_crawler.sink.QUOTES_MAPPING ==
      tcs_bot.search.elastic.QUOTES_MAPPING` and `set(Quote.model_fields) == mapping props`.
- [x] `crawler/Dockerfile:22-23`: per-arch supercronic checksum (amd64/arm64) instead of a single
      SHA1 that breaks BuildKit builds on arm64 (TARGETARCH=arm64 downloads the arm64 binary).

Resolution notes (developer, 2026-11-06)
- `sink.QUOTES_INDEX_SETTINGS` = 1 shard / 0 replicas; parity test `tests/unit/test_quotes_mapping_parity.py`.
- Dockerfile: `SUPERCRONIC_SHA1SUM_AMD64` (pinned) / `SUPERCRONIC_SHA1SUM_ARM64` (empty = not
  verified, warning printed) selected by `TARGETARCH`; pin the arm64 SHA1 from the release page.
