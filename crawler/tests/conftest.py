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


# ------------------------------------------------------------------ spec 005 news crawler
NEWS_FIXTURES = FIXTURES / "news"


def news_fixture(name: str) -> str:
    return (NEWS_FIXTURES / name).read_text(encoding="utf-8")


class FakeWeb:
    """Offline web for the news crawler: routes ``host + path`` (query ignored) to canned
    responses via ``httpx.MockTransport``. Unknown URLs (incl. robots.txt) -> 404."""

    def __init__(self) -> None:
        self.routes: dict[str, object] = {}
        self.requests: list[str] = []

    def add(self, url: str, body: object, status: int = 200, ctype: str = "text/html") -> None:
        import httpx

        parts = httpx.URL(url)
        self.routes[f"{parts.host}{parts.path}"] = (status, ctype, body)

    def handler(self, request):  # noqa: ANN001 - httpx.Request
        import httpx

        self.requests.append(str(request.url))
        route = self.routes.get(f"{request.url.host}{request.url.path}")
        if route is None:
            return httpx.Response(404, text="not found")
        status, ctype, body = route
        if isinstance(body, Exception):
            raise body
        if callable(body):
            return body(request)
        content = body.encode() if isinstance(body, str) else body
        return httpx.Response(status, content=content, headers={"content-type": ctype})

    def client(self):
        import httpx

        return httpx.Client(transport=httpx.MockTransport(self.handler), follow_redirects=True)

    def fetched(self, needle: str) -> list[str]:
        return [u for u in self.requests if needle in u]


def full_news_web() -> FakeWeb:
    """Every synthetic fixture wired at the URLs the sources/articles point to."""
    web = FakeWeb()
    web.add(
        "https://news.google.com/rss/search",
        news_fixture("google_news.xml"),
        ctype="application/rss+xml",
    )
    web.add(
        "https://www.bing.com/news/search",
        news_fixture("bing_news.xml"),
        ctype="application/rss+xml",
    )
    web.add(
        "https://api.gdeltproject.org/api/v2/doc/doc",
        news_fixture("gdelt.json"),
        ctype="application/json",
    )
    web.add(
        "https://hn.algolia.com/api/v1/search",
        news_fixture("hn_search.json"),
        ctype="application/json",
    )
    web.add(
        "https://hn.algolia.com/api/v1/items/90000001",
        news_fixture("hn_item.json"),
        ctype="application/json",
    )
    web.add(
        "https://www.reddit.com/search.json",
        news_fixture("reddit_search.json"),
        ctype="application/json",
    )
    web.add(
        "https://www.reddit.com/r/developersIndia/comments/syn001/synthetic_tcs_appraisal.json",
        news_fixture("reddit_thread.json"),
        ctype="application/json",
    )
    lawsuit = news_fixture("article_lawsuit.html")
    web.add(
        "https://economictimes.indiatimes.com/tech/tcs-synthetic-lawsuit/articleshow/1001.cms",
        lawsuit,
    )
    web.add(
        "https://economictimes.indiatimes.com/sports/cricket-synthetic-score",
        news_fixture("irrelevant.html"),
    )
    web.add(
        "https://www.livemint.com/companies/tcs-synthetic-whistleblower-11.html",
        news_fixture("article_whistleblower.html"),
    )
    web.add("https://www.livemint.com/companies/tcs-synthetic-depth3.html", lawsuit)
    web.add(
        "https://www.reuters.com/technology/tcs-synthetic-data-breach-2026-11-01/",
        news_fixture("article_breach.html"),
    )
    web.add(
        "https://www.tcs.com/news/synthetic-statement", news_fixture("corporate_statement.html")
    )
    web.add(
        "https://www.financialexpress.com/money/tcs-update-synthetic/",
        news_fixture("tax_collected_at_source.html"),
    )
    web.add(
        "https://www.business-standard.com/companies/tcs-synthetic-security-review.html",
        news_fixture("article_breach.html").replace("data breach", "security review"),
    )

    # Google News article links "redirect" to the publisher.
    def _google_redirect(request):  # noqa: ANN001
        import httpx

        target = {
            "/rss/articles/CBMiSYNTHETIC001": "https://economictimes.indiatimes.com/tech/tcs-synthetic-lawsuit/articleshow/1001.cms",
            "/rss/articles/CBMiSYNTHETIC003": "https://www.livemint.com/companies/tcs-synthetic-whistleblower-11.html",
        }.get(request.url.path)
        if target is None:
            return httpx.Response(404)
        return httpx.Response(302, headers={"location": target})

    for gid in ("CBMiSYNTHETIC001", "CBMiSYNTHETIC003"):
        web.add(f"https://news.google.com/rss/articles/{gid}", _google_redirect)
    return web


@pytest.fixture
def news_web() -> FakeWeb:
    return full_news_web()
