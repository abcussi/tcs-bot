"""Spec 005 bronze crawler: level-0 sources parse synthetic fixtures (AC-005.1)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from conftest import FakeWeb, full_news_web, news_fixture
from gfinance_crawler.news.fetch import PoliteFetcher
from gfinance_crawler.news.sources import (
    BingNewsRSS,
    GdeltDoc,
    GoogleNewsRSS,
    HackerNews,
    LocalInbox,
    RedditSearch,
    SearchError,
    Source,
    unwrap_bing,
)


def _fetcher(web: FakeWeb) -> PoliteFetcher:
    return PoliteFetcher(client=web.client(), delay=0, sleep=lambda s: None)


@pytest.fixture
def fetcher() -> PoliteFetcher:
    return _fetcher(full_news_web())


def test_all_sources_satisfy_protocol(fetcher, tmp_path):
    for s in (
        GoogleNewsRSS(fetcher),
        BingNewsRSS(fetcher),
        GdeltDoc(fetcher),
        HackerNews(fetcher),
        RedditSearch(fetcher),
        LocalInbox(tmp_path),
    ):
        assert isinstance(s, Source)
        assert s.source_type in {"news", "community", "corporate", "other"}


def test_google_news_rss(news_web):
    src = GoogleNewsRSS(_fetcher(news_web))
    docs = src.search("TCS lawsuit", 10)
    assert len(docs) == 3
    d = docs[0]
    assert d.url == "https://news.google.com/rss/articles/CBMiSYNTHETIC001?oc=5"
    assert d.follow_url == d.url  # redirect link kept
    assert d.depth == 0 and d.source == "google_news" and d.source_type == "news"
    assert d.query == "TCS lawsuit" and d.title.startswith("SYNTHETIC: TCS faces")
    assert d.raw_content.startswith("<item>") and "SYNTHETIC" in d.raw_content
    assert '<source url="https://economictimes.indiatimes.com">' in d.raw_content
    assert d.published_at == datetime(2026, 11, 2, 8, 30, tzinfo=UTC)
    assert d.content_type == "application/rss+xml"
    assert len(src.search("TCS lawsuit", 1)) == 1
    q = parse_qs(urlsplit(news_web.fetched("rss/search")[0]).query)
    assert q == {"q": ["TCS lawsuit"], "hl": ["en-IN"], "gl": ["IN"], "ceid": ["IN:en"]}


def test_bing_news_unwraps_apiclick(news_web):
    docs = BingNewsRSS(_fetcher(news_web)).search("TCS lawsuit", 10)
    assert len(docs) == 2
    assert docs[0].url.startswith("http://www.bing.com/news/apiclick.aspx")
    assert docs[0].follow_url == (
        "https://economictimes.indiatimes.com/tech/tcs-synthetic-lawsuit/articleshow/1001.cms"
        "?utm_source=bing"
    )
    assert "format=rss" in news_web.fetched("bing.com/news/search")[0]
    assert unwrap_bing("https://example.com/x") == "https://example.com/x"


def test_gdelt_doc(news_web):
    docs = GdeltDoc(_fetcher(news_web)).search("TCS data breach", 5)
    assert [d.follow_url for d in docs][0] == (
        "https://www.reuters.com/technology/tcs-synthetic-data-breach-2026-11-01/"
    )
    d = docs[0]
    assert d.url.startswith("https://api.gdeltproject.org/api/v2/doc/doc?") and "item=" in d.url
    assert d.url != d.follow_url and json.loads(d.raw_content)["domain"] == "reuters.com"
    assert d.published_at == datetime(2026, 11, 1, 6, 15, tzinfo=UTC)
    q = parse_qs(urlsplit(news_web.fetched("doc/doc?")[0]).query)
    assert q["mode"] == ["artlist"] and q["format"] == ["json"] and q["maxrecords"] == ["5"]
    assert q["sort"] == ["datedesc"]


def test_hackernews(news_web):
    docs = HackerNews(_fetcher(news_web)).search("TCS", 5)
    assert [d.url for d in docs] == [
        "https://news.ycombinator.com/item?id=90000001",
        "https://news.ycombinator.com/item?id=90000005",
    ]
    assert {d.follow_url for d in docs} == {"https://hn.algolia.com/api/v1/items/90000001"}
    assert docs[1].title == "SYNTHETIC: TCS data breach discussion"  # from story_title
    assert all(d.source_type == "community" for d in docs)
    assert "tags=%28story%2Ccomment%29" in news_web.fetched("hn.algolia.com/api/v1/search")[0]


def test_reddit_global_and_subreddits(news_web):
    news_web.add(
        "https://www.reddit.com/r/developersIndia/search.json",
        news_fixture("reddit_search.json"),
        ctype="application/json",
    )
    src = RedditSearch(_fetcher(news_web), subreddits=["developersIndia", "TCS"])
    docs = src.search("TCS layoffs", 5)
    assert [d.url for d in docs] == [
        "https://www.reddit.com/r/developersIndia/comments/syn001/synthetic_tcs_appraisal/",
        "https://www.reddit.com/r/india/comments/syn002/synthetic_tcs_lrs/",
    ]  # deduped across endpoints
    assert docs[0].follow_url == (
        "https://www.reddit.com/r/developersIndia/comments/syn001/synthetic_tcs_appraisal.json"
    )
    calls = news_web.fetched("search.json")
    assert len(calls) == 3
    assert "restrict_sr=1" in calls[1] and "t=year" in calls[0] and "sort=relevance" in calls[0]


def test_reddit_blocked_by_robots_raises_blocked(news_web):
    news_web.add(
        "https://www.reddit.com/robots.txt", news_fixture("robots_disallow.txt"), ctype="text/plain"
    )
    with pytest.raises(SearchError) as exc:
        RedditSearch(_fetcher(news_web)).search("TCS", 5)
    assert exc.value.blocked
    assert news_web.fetched("search.json") == []


def test_search_error_on_http_failure_and_bad_json():
    web = FakeWeb()
    web.add("https://hn.algolia.com/api/v1/search", "<html>not json</html>")
    with pytest.raises(SearchError) as exc:
        HackerNews(_fetcher(web)).search("TCS", 5)
    assert not exc.value.blocked
    web.add("https://news.google.com/rss/search", httpx.ProxyError("CONNECT 403"))
    with pytest.raises(SearchError) as exc:
        GoogleNewsRSS(_fetcher(web)).search("TCS", 5)
    assert exc.value.blocked


def _minimal_pdf(text: str) -> bytes:
    """A tiny valid one-page PDF with ``text`` (no external tooling)."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)
    return bytes(out)


def test_local_inbox_reads_supported_files(tmp_path):
    """AC-005.1: offline/manual docs in data/bronze/inbox become depth-0 documents."""
    (tmp_path / "a_note.txt").write_text("SYNTHETIC FIXTURE note about TCS audit.", "utf-8")
    (tmp_path / "b_page.html").write_text(news_fixture("article_breach.html"), "utf-8")
    (tmp_path / "c_item.json").write_text(
        json.dumps(
            {
                "title": "SYNTHETIC json item",
                "url": "https://www.tcs.com/news/synthetic-statement",
                "published_at": "2026-10-01T00:00:00Z",
                "content": "SYNTHETIC FIXTURE",
            }
        ),
        "utf-8",
    )
    (tmp_path / "d_memo.md").write_text("# SYNTHETIC memo\nTCS", "utf-8")
    (tmp_path / "e_report.pdf").write_bytes(_minimal_pdf("SYNTHETIC TCS PDF"))
    (tmp_path / "f_empty.txt").write_text("  ", "utf-8")
    (tmp_path / "g_image.png").write_bytes(b"\x89PNG")
    (tmp_path / "README.md").write_text("drop files here", "utf-8")
    (tmp_path / ".gitkeep").write_text("", "utf-8")
    inbox = LocalInbox(tmp_path)
    docs = inbox.search("ignored", None)
    names = [d.url.rsplit("/", 1)[-1] for d in docs]
    assert names == [
        "a_note.txt",
        "b_page.html",
        "synthetic-statement",
        "d_memo.md",
        "e_report.pdf",
    ]
    by = {n: d for n, d in zip(names, docs, strict=True)}
    assert by["a_note.txt"].url.startswith("file://") and by["a_note.txt"].source == "local_inbox"
    assert by["a_note.txt"].source_type == "other" and by["a_note.txt"].depth == 0
    assert by["b_page.html"].title.startswith("SYNTHETIC: TCS probes")
    assert by["synthetic-statement"].source_type == "corporate"
    assert by["synthetic-statement"].published_at == datetime(2026, 10, 1, tzinfo=UTC)
    assert by["d_memo.md"].content_type == "text/markdown"
    assert "SYNTHETIC TCS PDF" in by["e_report.pdf"].raw_content
    assert by["e_report.pdf"].content_type == "text/plain"
    assert any("f_empty" in s for s in inbox.skipped)
    assert len(inbox.search("", 2)) == 2
    assert LocalInbox(tmp_path / "missing").search("", 10) == []


# ------------------------------------------------ real response shapes (synthetic mirrors)
def test_google_news_rss_is_blocked_when_robots_disallow_rss():
    """Real news.google.com/robots.txt disallows /rss/ for generic agents: blocked, not bypassed."""
    web = FakeWeb()
    web.add(
        "https://news.google.com/robots.txt",
        news_fixture("robots_google_news_like.txt"),
        ctype="text/plain",
    )
    web.add("https://news.google.com/rss/search", news_fixture("google_news.xml"))
    with pytest.raises(SearchError) as exc:
        GoogleNewsRSS(_fetcher(web)).search("TCS lawsuit", 5)
    assert exc.value.blocked and "robots" in str(exc.value)
    assert web.fetched("rss/search") == []


def test_bing_empty_feed_returns_no_documents():
    web = FakeWeb()
    web.add(
        "https://www.bing.com/news/search",
        news_fixture("bing_news_empty.xml"),
        ctype="application/xml",
    )
    assert BingNewsRSS(_fetcher(web)).search("TCS lawsuit", 5) == []


def test_gdelt_plain_text_error_is_a_non_blocked_search_error_with_hint():
    """GDELT answers bad/odd queries with HTTP 200 + a plain-text body."""
    web = FakeWeb()
    web.add(
        "https://api.gdeltproject.org/api/v2/doc/doc",
        news_fixture("gdelt_throttled.txt"),
        ctype="text/html",
    )
    with pytest.raises(SearchError) as exc:
        GdeltDoc(_fetcher(web)).search("TCS", 5)
    assert not exc.value.blocked and "slow down" in str(exc.value)


def test_gdelt_429_is_reported_blocked():
    web = FakeWeb()
    web.add(
        "https://api.gdeltproject.org/api/v2/doc/doc",
        news_fixture("gdelt_throttled.txt"),
        status=429,
        ctype="text/plain",
    )
    with pytest.raises(SearchError) as exc:
        GdeltDoc(_fetcher(web)).search("TCS", 5)
    assert exc.value.blocked and "429" in str(exc.value)
    assert len(web.fetched("doc/doc")) == 1  # no retry


def test_reddit_429_and_403_are_reported_blocked():
    for status in (403, 429):
        web = FakeWeb()
        web.add("https://www.reddit.com/search.json", "nope", status=status, ctype="text/plain")
        with pytest.raises(SearchError) as exc:
            RedditSearch(_fetcher(web), subreddits=[]).search("TCS", 5)
        assert exc.value.blocked


def test_hackernews_real_shape_comment_hits():
    """Comment hits have no ``title`` (story_title is used) and carry ``_highlightResult``."""
    web = FakeWeb()
    web.add(
        "https://hn.algolia.com/api/v1/search",
        news_fixture("hn_search_highlight.json"),
        ctype="application/json",
    )
    docs = HackerNews(_fetcher(web)).search("TCS lawsuit", 5)
    assert [d.url for d in docs] == [
        "https://news.ycombinator.com/item?id=90000102",
        "https://news.ycombinator.com/item?id=90000202",
    ]
    assert docs[0].title == "SYNTHETIC: fictional store policy thread"
    assert docs[1].follow_url == "https://hn.algolia.com/api/v1/items/90000201"
    assert docs[1].published_at == datetime(2026, 10, 3, 10, 0, tzinfo=UTC)
    assert docs[0].published_at == datetime(2026, 10, 1, 10, 0, tzinfo=UTC)
    assert "_highlightResult" in docs[0].raw_content  # bronze keeps the hit as fetched
