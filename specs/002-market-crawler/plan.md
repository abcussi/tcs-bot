# 002 — Plan / contracts
```python
gfinance_crawler/config.py   CrawlerSettings(BaseSettings): crawler_tickers, crawler_cron, crawler_base_url,
                             crawler_timeout_seconds, crawler_user_agent, elastic_url, elastic_quotes_index
gfinance_crawler/models.py   class Quote(BaseModel)
gfinance_crawler/scraper.py  def parse_quote_html(html: str, ticker: str, exchange: str, url: str) -> Quote
                             class GoogleFinanceScraper: fetch(ticker, exchange) -> Quote ; fetch_all(pairs) -> list[Quote]
gfinance_crawler/sink.py     class ElasticSink: ensure_index() ; index(quotes) -> int
gfinance_crawler/run.py      def run_once(settings=None, scraper=None, sink=None) -> dict ; def main(argv=None) -> int
```
ES mapping for `market-quotes`: ticker/exchange/currency keyword, name text+keyword, numeric fields float,
scraped_at date. The main app reads it via `tcs_bot.search.elastic.latest_quote(ticker)`.
Mapping parity (architect review 2026-11-06): `sink.QUOTES_MAPPING` must equal
`tcs_bot.search.elastic.QUOTES_MAPPING` and the index must be created with
`settings={"number_of_shards": 1, "number_of_replicas": 0}` like the app does — whichever process
creates `market-quotes` first wins. `ticker`/`exchange` are stored upper-case; the app looks quotes up
by vendor ticker `"SYM:EXCH"` (term on `ticker`=SYM + `exchange`=EXCH), so vendor tickers must be
upper-case too.
