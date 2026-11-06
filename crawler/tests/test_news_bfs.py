"""Spec 005 bronze crawler: breadth-first crawl, caps, dedupe, relevance, blocked sources
(AC-005.2, AC-005.3). Offline: every URL is served by ``FakeWeb`` from synthetic fixtures."""

from __future__ import annotations

import httpx
import pytest
from conftest import FakeWeb, full_news_web, news_fixture
from gfinance_crawler.news.bfs import HARD_MAX_DEPTH, HARD_MAX_DOCS, RiskCrawler
from gfinance_crawler.news.fetch import PoliteFetcher
from gfinance_crawler.news.links import canonical_url
from gfinance_crawler.news.models import RawDocument
from gfinance_crawler.news.sources import (
    BingNewsRSS,
    GdeltDoc,
    GoogleNewsRSS,
    HackerNews,
    LocalInbox,
    RedditSearch,
)

QUERIES = ["TCS lawsuit", "TCS data breach"]


def _crawler(web: FakeWeb, **kw) -> RiskCrawler:
    f = PoliteFetcher(client=web.client(), delay=0, sleep=lambda s: None)
    sources = kw.pop(
        "sources",
        None,
    ) or [
        GoogleNewsRSS(f),
        BingNewsRSS(f),
        GdeltDoc(f),
        HackerNews(f),
        RedditSearch(f, subreddits=["developersIndia"]),
    ]
    kw.setdefault("per_query_limit", 10)
    return RiskCrawler(sources, f, **kw)


@pytest.fixture
def web() -> FakeWeb:
    return full_news_web()


def test_full_crawl_is_breadth_first_and_depth_capped(web):
    """AC-005.2: depths are non-decreasing in yield order and never exceed 2."""
    crawler = _crawler(web)
    docs = list(crawler.crawl(QUERIES))
    depths = [d.depth for d in docs]
    assert depths == sorted(depths)
    assert set(depths) == {0, 1, 2}
    assert max(depths) <= HARD_MAX_DEPTH
    s = crawler.stats
    assert s["stored"] == len(docs)
    assert s["by_depth"] == {str(k): depths.count(k) for k in (0, 1, 2)}
    assert sum(s["by_source"].values()) == len(docs)
    assert s["blocked"] == 0 and s["failed"] == 0


def test_lineage_parent_query_and_source(web):
    docs = list(_crawler(web).crawl(["TCS lawsuit"]))
    by_url = {d.url: d for d in docs}
    by_canon = {d.canonical_url for d in docs}
    for d in docs:
        if d.depth == 0:
            assert d.parent_url is None
        else:
            assert d.parent_url in by_url
            parent = by_url[d.parent_url]
            assert parent.depth == d.depth - 1 and d.query == parent.query
        assert isinstance(d, RawDocument) and d.canonical_url == canonical_url(d.url)
    et = next(d for d in docs if "economictimes" in d.url and d.depth == 1)
    assert et.source == "google_news"  # depth 1 inherits the item's source
    assert et.title.startswith("SYNTHETIC: TCS lawsuit") and et.published_at is not None
    assert et.http_status == 200 and et.content_type == "text/html"
    corp = next(d for d in docs if "tcs.com" in d.url)
    assert corp.depth == 2 and corp.source == "web" and corp.source_type == "corporate"
    bs = next(d for d in docs if "business-standard" in d.url)
    assert bs.depth == 2 and bs.source_type == "news"  # link found in an HN thread
    reddit_thread = next(d for d in docs if d.url.endswith(".json") and "reddit" in d.url)
    assert reddit_thread.depth == 1 and reddit_thread.source_type == "community"
    assert "https://www.livemint.com/companies/tcs-synthetic-whistleblower-11.html" in by_canon


def test_dedupe_on_canonical_url(web):
    """AC-005.2: the same article via Google redirect, Bing (utm param) and GDELT is stored once
    and fetched once; repeated query results are duplicates."""
    crawler = _crawler(web)
    docs = list(crawler.crawl(QUERIES + ["TCS lawsuit"]))
    keys = [d.canonical_url for d in docs]
    assert len(keys) == len(set(keys))
    et = "economictimes.indiatimes.com/tech/tcs-synthetic-lawsuit"
    assert sum(1 for k in keys if et in k) == 1
    assert len(web.fetched(et)) == 1
    assert crawler.stats["duplicates"] > 0


def test_relevance_filters_tax_collected_at_source_and_off_topic(web):
    """AC-005.3: TCS-as-tax items and pages are not stored; irrelevant / non-allowed links are
    not followed."""
    crawler = _crawler(web)
    docs = list(crawler.crawl(QUERIES))
    urls = " ".join(d.url for d in docs)
    assert "SYNTHETIC002" not in urls  # Google "TCS on foreign remittance" item
    assert "tcs-on-lrs-synthetic" not in urls  # GDELT tax item, never followed
    assert "synthetic_tcs_lrs" not in urls  # Reddit LRS post
    assert "tcs-update-synthetic" not in urls  # fetched (relevant anchor) but page is tax
    assert len(web.fetched("tcs-update-synthetic")) == 1
    assert web.fetched("cricket-synthetic-score") == []  # off-topic anchor
    assert web.fetched("lrs-rules") == []  # "TCS on LRS remittance rules" anchor
    assert web.fetched("blog.example.org") == [] and web.fetched("example.org/tcs-blog") == []
    assert web.fetched("tcs-on-lrs-synthetic") == []
    assert crawler.stats["skipped_irrelevant"] >= 4
    assert crawler.stats["links_skipped"] > 0


def test_depth2_pages_are_never_expanded(web):
    docs = list(_crawler(web).crawl(QUERIES))
    assert any("tcs.com/news/synthetic-statement" in d.url for d in docs)
    assert web.fetched("tcs.com/news/synthetic-depth3") == []


@pytest.mark.parametrize("max_depth", [0, 1])
def test_max_depth_respected(web, max_depth):
    docs = list(_crawler(web, max_depth=max_depth).crawl(QUERIES))
    assert docs and max(d.depth for d in docs) == max_depth
    if max_depth == 0:
        assert web.fetched("economictimes") == [] and web.fetched("api/v1/items") == []


@pytest.mark.parametrize("max_docs", [1, 3, 10, 12])
def test_max_docs_cap(web, max_docs):
    """AC-005.2: never more than max_docs documents per run; nothing fetched after the cap."""
    crawler = _crawler(web, max_docs=max_docs)
    docs = list(crawler.crawl(QUERIES))
    assert len(docs) == max_docs == crawler.stats["stored"]


def test_cap_reached_stops_fetching(web):
    crawler = _crawler(web, max_docs=2)
    list(crawler.crawl(QUERIES))
    assert web.fetched("economictimes") == []  # depth 1 never started


def test_hard_caps_clamp_configuration(web):
    crawler = _crawler(web, max_docs=10_000, max_depth=7)
    assert crawler.max_docs == HARD_MAX_DOCS == 500
    assert crawler.max_depth == HARD_MAX_DEPTH == 2
    docs = list(crawler.crawl(QUERIES))
    assert max(d.depth for d in docs) <= 2
    assert _crawler(web, max_docs=0).max_docs == 0
    assert list(_crawler(web, max_docs=0).crawl(QUERIES)) == []


def test_blocked_source_does_not_stop_others(web):
    """AC-005.2: robots-blocked Reddit and a proxy-refused Google are recorded; the rest crawl."""
    web.add(
        "https://www.reddit.com/robots.txt", news_fixture("robots_disallow.txt"), ctype="text/plain"
    )
    web.add("https://news.google.com/rss/search", httpx.ProxyError("CONNECT 403 Forbidden"))
    crawler = _crawler(web)
    docs = list(crawler.crawl(QUERIES))
    sources = {d.source for d in docs}
    assert "reddit" not in sources and "google_news" not in sources
    assert {"bing_news", "gdelt", "hackernews"} <= sources
    s = crawler.stats
    assert s["blocked"] == 2  # each blocked source counted once, then disabled
    assert len(web.fetched("rss/search")) == 1  # not retried for the second query
    assert any("reddit" in e for e in s["errors"])


def test_failed_fetches_are_counted_not_raised(web):
    web.add(
        "https://www.reuters.com/technology/tcs-synthetic-data-breach-2026-11-01/", "x", status=500
    )
    web.add("https://hn.algolia.com/api/v1/items/90000001", "no", status=403)
    crawler = _crawler(web)
    docs = list(crawler.crawl(QUERIES))
    assert docs
    assert crawler.stats["failed"] >= 1 and crawler.stats["blocked"] >= 1
    assert not any("reuters" in d.url for d in docs if d.depth > 0)


def test_broken_source_is_isolated(web):
    class Boom:
        name = "boom"
        source_type = "news"
        calls = 0

        def search(self, query, limit):
            Boom.calls += 1
            raise ValueError("parser exploded")

    f = PoliteFetcher(client=web.client(), delay=0, sleep=lambda s: None)
    crawler = RiskCrawler([Boom(), GdeltDoc(f)], f, per_query_limit=5)
    docs = list(crawler.crawl(["q1", "q2", "q3", "q4"]))
    assert any(d.source == "gdelt" for d in docs)
    assert crawler.stats["failed"] == 3 and Boom.calls == 3  # disabled after 3 failures


def test_local_inbox_is_query_independent_and_trusted(web, tmp_path):
    (tmp_path / "memo.txt").write_text("SYNTHETIC FIXTURE memo with no company mention", "utf-8")
    f = PoliteFetcher(client=web.client(), delay=0, sleep=lambda s: None)
    crawler = RiskCrawler([LocalInbox(tmp_path), GdeltDoc(f)], f, per_query_limit=1)
    docs = list(crawler.crawl(QUERIES))
    inbox = [d for d in docs if d.source == "local_inbox"]
    assert len(inbox) == 1 and inbox[0].depth == 0 and docs[0].source == "local_inbox"
    assert crawler.stats["duplicates"] == 0 or all(d.source != "local_inbox" for d in docs[1:])


def test_default_per_query_limit_reserves_budget(web):
    crawler = _crawler(web, per_query_limit=None)
    assert crawler._limit_for(20) == 2
    big = _crawler(web, per_query_limit=None, max_docs=500)
    assert big._limit_for(1) == 500 // (3 * 5)
