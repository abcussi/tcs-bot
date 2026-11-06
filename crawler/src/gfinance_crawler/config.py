"""Crawler settings (env / .env driven). Never imports vendor_risk_ai."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_TICKERS = "AAPL:NASDAQ,MSFT:NASDAQ,JPM:NYSE,GS:NYSE,MS:NYSE,IBM:NYSE,ORCL:NYSE,ICE:NYSE"


def parse_tickers(raw: str) -> list[tuple[str, str]]:
    """Parse ``"AAPL:NASDAQ, IBM:NYSE"`` into ``[("AAPL", "NASDAQ"), ("IBM", "NYSE")]``.

    Blank and malformed entries (no ``:``) are skipped; values are upper-cased and de-duplicated.
    """
    pairs: list[tuple[str, str]] = []
    for item in raw.split(","):
        ticker, sep, exchange = item.strip().partition(":")
        ticker, exchange = ticker.strip().upper(), exchange.strip().upper()
        if not sep or not ticker or not exchange:
            continue
        if (ticker, exchange) not in pairs:
            pairs.append((ticker, exchange))
    return pairs


class CrawlerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    crawler_tickers: str = DEFAULT_TICKERS
    crawler_cron: str = "*/15 * * * *"
    crawler_base_url: str = "https://www.google.com/finance/quote"
    crawler_timeout_seconds: float = 15.0
    crawler_user_agent: str = (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/130 Safari/537.36"
    )
    crawler_delay_seconds: float = 1.0
    elastic_url: str = "http://127.0.0.1:9200"
    elastic_quotes_index: str = "market-quotes"

    @property
    def ticker_pairs(self) -> list[tuple[str, str]]:
        return parse_tickers(self.crawler_tickers)

    def quote_url(self, ticker: str, exchange: str) -> str:
        return f"{self.crawler_base_url.rstrip('/')}/{ticker}:{exchange}"
