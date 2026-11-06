"""Spec 005 bronze crawler: PoliteFetcher robots/delay/retries/blocked handling (AC-005.2)."""

from __future__ import annotations

import httpx
from conftest import FakeWeb, news_fixture
from gfinance_crawler.news.fetch import PoliteFetcher

URL = "https://news.test/a"


class Clock:
    def __init__(self) -> None:
        self.t = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.sleeps.append(s)
        self.t += s


def _fetcher(web: FakeWeb, **kw) -> PoliteFetcher:
    clock = Clock()
    kw.setdefault("delay", 0)
    return PoliteFetcher(client=web.client(), sleep=clock.sleep, clock=clock, **kw)


def test_ok_fetch_returns_text_and_metadata():
    web = FakeWeb()
    web.add(URL, "<html>TCS</html>", ctype="text/html; charset=utf-8")
    res = _fetcher(web).fetch(URL)
    assert res.ok and res.status == 200 and res.content_type == "text/html"
    assert res.text == "<html>TCS</html>" and res.final_url == URL and not res.blocked


def test_robots_disallow_blocks_without_fetching_page():
    """AC-005.2: robots.txt respected; disallow => blocked, page never requested."""
    web = FakeWeb()
    web.add("https://news.test/robots.txt", news_fixture("robots_disallow.txt"), ctype="text/plain")
    web.add(URL, "<html>TCS</html>")
    f = _fetcher(web)
    res = f.fetch(URL)
    assert res.blocked and res.error == "robots.txt disallow" and not res.ok
    assert web.fetched("/a") == []
    f.fetch("https://news.test/b")
    assert len(web.fetched("robots.txt")) == 1  # cached per origin


def test_robots_missing_or_unreachable_allows():
    web = FakeWeb()  # robots.txt -> 404
    web.add(URL, "ok", ctype="text/plain")
    assert _fetcher(web).fetch(URL).ok
    web2 = FakeWeb()
    web2.add("https://news.test/robots.txt", httpx.ConnectError("down"))
    web2.add(URL, "ok", ctype="text/plain")
    assert _fetcher(web2).fetch(URL).ok


def test_robots_can_be_disabled():
    web = FakeWeb()
    web.add("https://news.test/robots.txt", news_fixture("robots_disallow.txt"), ctype="text/plain")
    web.add(URL, "ok", ctype="text/plain")
    assert _fetcher(web, robots=False).fetch(URL).ok
    assert web.fetched("robots.txt") == []


def test_blocked_statuses():
    for status in (401, 403, 429, 451):
        web = FakeWeb()
        web.add(URL, "no", status=status)
        res = _fetcher(web).fetch(URL)
        assert res.blocked and res.status == status and not res.ok


def test_proxy_connect_refusal_is_blocked_and_not_retried():
    web = FakeWeb()
    web.add(URL, httpx.ProxyError("403 Forbidden"))
    res = _fetcher(web, max_attempts=3).fetch(URL)
    assert res.blocked and res.status == 0 and "proxy" in res.error
    assert len(web.fetched("/a")) == 1


def test_retries_5xx_then_succeeds():
    web = FakeWeb()
    responses = iter([httpx.Response(503), httpx.Response(502), httpx.Response(200, text="ok")])
    web.add(URL, lambda request: next(responses))
    res = _fetcher(web, max_attempts=3).fetch(URL)
    assert res.ok and res.text == "ok"
    assert len(web.fetched("/a")) == 3


def test_429_is_blocked_and_never_retried():
    """Real GDELT/Reddit behaviour: throttled => report blocked, do not hammer or bypass."""
    web = FakeWeb()
    web.add(URL, news_fixture("gdelt_throttled.txt"), status=429, ctype="text/plain")
    res = _fetcher(web, max_attempts=3).fetch(URL)
    assert res.blocked and not res.ok and res.status == 429
    assert "rate limited" in res.error and "slow down" in res.error  # body hint kept
    assert len(web.fetched("/a")) == 1


def test_gdelt_host_gets_a_minimum_5s_spacing():
    web = FakeWeb()
    gdelt = "https://api.gdeltproject.org/api/v2/doc/doc"
    web.add(gdelt, "{}", ctype="application/json")
    web.add("https://news.test/a", "ok", ctype="text/plain")
    clock = Clock()
    f = PoliteFetcher(client=web.client(), delay=1.0, sleep=clock.sleep, clock=clock, robots=False)
    f.fetch(gdelt + "?q=1")
    f.fetch(gdelt + "?q=2")
    assert clock.sleeps == [5.5]
    f.fetch("https://news.test/a")
    f.fetch("https://news.test/a")
    assert clock.sleeps == [5.5, 1.0]  # ordinary hosts keep the default delay


def test_persistent_5xx_returns_error_never_raises():
    web = FakeWeb()
    web.add(URL, "boom", status=502)
    res = _fetcher(web, max_attempts=2).fetch(URL)
    assert not res.ok and not res.blocked and res.status == 502
    assert "after retries" in res.error


def test_transport_error_and_bad_urls_never_raise():
    web = FakeWeb()
    web.add(URL, httpx.ConnectTimeout("slow"))
    res = _fetcher(web, max_attempts=2).fetch(URL)
    assert not res.ok and not res.blocked and "ConnectTimeout" in res.error
    assert _fetcher(web).fetch("ftp://x/y").error.startswith("unsupported scheme")
    assert not _fetcher(web).fetch("http://[bad").ok


def test_404_and_binary_are_errors_not_blocked():
    web = FakeWeb()
    web.add("https://news.test/img", b"\x89PNG", ctype="image/png")
    f = _fetcher(web)
    missing = f.fetch("https://news.test/missing")
    assert missing.status == 404 and not missing.blocked and not missing.ok
    assert "unsupported content-type" in f.fetch("https://news.test/img").error


def test_per_domain_delay():
    """AC-005.2: consecutive requests to one host wait ``delay``; other hosts do not."""
    web = FakeWeb()
    for u in ("https://a.test/1", "https://a.test/2", "https://b.test/1"):
        web.add(u, "x", ctype="text/plain")
    clock = Clock()
    f = PoliteFetcher(client=web.client(), delay=2.0, sleep=clock.sleep, clock=clock)
    f.fetch("https://a.test/1")  # robots + page => one wait
    f.fetch("https://a.test/2")
    f.fetch("https://b.test/1")  # robots + page => one wait
    assert clock.sleeps == [2.0, 2.0, 2.0]


def test_owned_client_is_closed():
    f = PoliteFetcher(delay=0)
    with f:
        assert not f.client.is_closed
    assert f.client.is_closed
