from __future__ import annotations

from pathlib import Path

import pytest
from gfinance_crawler.config import CrawlerSettings

FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


@pytest.fixture
def ibm_html() -> str:
    return load_fixture("ibm_nyse.html")


@pytest.fixture
def degraded_html() -> str:
    return load_fixture("degraded.html")


@pytest.fixture
def settings() -> CrawlerSettings:
    return CrawlerSettings(
        _env_file=None,
        crawler_tickers="IBM:NYSE,BAD:NYSE",
        crawler_base_url="https://finance.test/quote",
        crawler_delay_seconds=0,
        elastic_url="http://es.test:9200",
        elastic_quotes_index="market-quotes-test",
    )
