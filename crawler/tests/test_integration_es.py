from __future__ import annotations

import uuid

import pytest
from elasticsearch import Elasticsearch
from gfinance_crawler.config import CrawlerSettings
from gfinance_crawler.scraper import parse_quote_html
from gfinance_crawler.sink import ElasticSink

ES_URL = "http://127.0.0.1:9200"
pytestmark = pytest.mark.integration


@pytest.fixture
def es() -> Elasticsearch:
    client = Elasticsearch(ES_URL, request_timeout=5)
    try:
        if not client.ping():
            pytest.skip("Elasticsearch not reachable")
    except Exception:  # noqa: BLE001
        pytest.skip("Elasticsearch not reachable")
    return client


def test_index_fixture_quote_into_live_es(es: Elasticsearch, ibm_html: str) -> None:
    """AC-002.4 a fixture-parsed quote is indexed idempotently into live ES and read back."""
    index = f"market-quotes-it-{uuid.uuid4().hex[:8]}"
    settings = CrawlerSettings(_env_file=None, elastic_url=ES_URL, elastic_quotes_index=index)
    sink = ElasticSink(settings, client=es)
    quote = parse_quote_html(
        ibm_html, "IBM", "NYSE", "https://www.google.com/finance/quote/IBM:NYSE"
    )
    try:
        sink.ensure_index()
        assert sink.index([quote]) == 1
        assert sink.index([quote]) == 1  # same minute -> same _id, overwritten not duplicated
        es.indices.refresh(index=index)
        assert es.count(index=index)["count"] == 1
        doc = es.get(index=index, id=quote.doc_id())["_source"]
        assert doc["ticker"] == "IBM" and doc["price"] == pytest.approx(187.45)
        mapping = es.indices.get_mapping(index=index)[index]["mappings"]["properties"]
        assert mapping["ticker"]["type"] == "keyword" and mapping["scraped_at"]["type"] == "date"
        hits = es.search(
            index=index, query={"term": {"ticker": "IBM"}}, sort=[{"scraped_at": "desc"}], size=1
        )["hits"]["hits"]
        assert hits[0]["_source"]["market_cap"] == pytest.approx(172.3e9, rel=1e-6)
    finally:
        es.indices.delete(index=index, ignore_unavailable=True)
