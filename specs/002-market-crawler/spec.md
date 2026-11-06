# 002 — Google Finance market crawler

## Why
Risk narratives need market context (price, change, market cap, P/E, 52w range) for listed vendors.

## Acceptance criteria
- AC-002.1 Lives in `crawler/` as uv workspace member `gfinance-crawler`, importable as `gfinance_crawler`.
- AC-002.2 Fetches `https://www.google.com/finance/quote/{TICKER}:{EXCHANGE}` for each `CRAWLER_TICKERS` entry with timeout, retries (tenacity, exp backoff) and a polite delay.
- AC-002.3 Parses into a `Quote` model: ticker, exchange, name, price, currency, change, change_pct, previous_close, day_range, year_range, market_cap, pe_ratio, dividend_yield, scraped_at, source_url.
  Parser is tolerant: missing fields => None, never crash the run.
- AC-002.4 Bulk-indexes quotes into ES index `market-quotes` (doc id = `{ticker}:{exchange}:{scraped_at minute}` => idempotent per minute).
- AC-002.5 Scheduling: `gfinance-crawl --once` runs once; `gfinance-crawl --schedule` runs APScheduler with `CRAWLER_CRON`; `crawler/cron/crontab` + `crawler/Dockerfile` (supercronic) for containers.
- AC-002.6 Run summary `{fetched, parsed, indexed, failed}` logged and returned.
- AC-002.7 Unit tests parse a saved HTML fixture (no network).
