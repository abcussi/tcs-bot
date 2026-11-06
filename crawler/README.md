# gfinance-crawler

Scrapes Google Finance quote pages (`{CRAWLER_BASE_URL}/{TICKER}:{EXCHANGE}`) for every
`CRAWLER_TICKERS` entry and bulk-indexes them into Elasticsearch (`ELASTIC_QUOTES_INDEX`,
default `market-quotes`). Spec: `specs/002-market-crawler/`. Never imports `vendor_risk_ai`.

```bash
uv run gfinance-crawl --once        # crawl once, log + return {fetched, parsed, indexed, failed}
uv run gfinance-crawl --schedule    # APScheduler on CRAWLER_CRON (also runs once at startup)
uv run pytest crawler/tests -q      # offline unit tests (HTML fixtures)
uv run pytest crawler/tests -m integration   # indexes a fixture quote into a live ES
docker build -f crawler/Dockerfile -t gfinance-crawler .   # supercronic + cron/crontab
```

Config (env / `.env`): `CRAWLER_TICKERS`, `CRAWLER_CRON`, `CRAWLER_BASE_URL`,
`CRAWLER_TIMEOUT_SECONDS`, `CRAWLER_USER_AGENT`, `CRAWLER_DELAY_SECONDS` (default 1.0),
`ELASTIC_URL`, `ELASTIC_QUOTES_INDEX`. Keep `cron/crontab` in sync with `CRAWLER_CRON`.

- Parser (`scraper.parse_quote_html`) prefers `data-last-price`/`data-currency-code`, then CSS
  classes, then key-stat label text; anything missing is `None` and it never raises.
- Fetch: tenacity retries (3 attempts, exponential backoff) on transport errors, 5xx and 429;
  a failing ticker is counted in `failed` and never stops the run. Only quotes with a price are
  indexed. Doc id `{ticker}:{exchange}:{YYYYmmddHHMM}` makes re-runs within a minute idempotent.
- Exit code is 0 even with failed tickers; 1 only when Elasticsearch is unreachable and nothing
  was indexed. Honors `HTTPS_PROXY` / `SSL_CERT_FILE` (httpx `trust_env`).
