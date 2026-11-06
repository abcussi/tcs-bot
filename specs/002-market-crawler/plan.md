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
scraped_at date. The main app reads it via `vendor_risk_ai.search.elastic.latest_quote(ticker)`.
