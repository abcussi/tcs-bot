"""Polite HTTP fetcher: robots.txt, per-domain delay, timeouts, retries; never raises (AC-005.2)."""

from __future__ import annotations

import io
import logging
import time
from collections.abc import Callable
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import httpx
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential

from gfinance_crawler.news.models import FetchResult

log = logging.getLogger(__name__)

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (compatible; VendorRiskBot/0.1; TCS vendor-risk research crawler)"
)
ROBOTS_AGENT = "VendorRiskBot"
BLOCKED_STATUSES = {401, 403, 451}
MAX_TEXT_CHARS = 2_000_000  # bronze column is MEDIUMTEXT (16 MiB of utf8mb4)
_TEXTUAL = ("text/", "json", "xml", "rss", "atom", "javascript")


class _RetryableStatus(Exception):
    def __init__(self, response: httpx.Response) -> None:
        super().__init__(f"HTTP {response.status_code}")
        self.response = response


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.ProxyError):  # proxy CONNECT refusal: retrying will not help
        return False
    return isinstance(exc, _RetryableStatus | httpx.TransportError)


def pdf_to_text(data: bytes) -> str | None:
    """Text of a PDF via pypdf when it is installed; None otherwise / on failure."""
    try:
        from pypdf import PdfReader
    except ImportError:  # pragma: no cover - pypdf is installed in the workspace
        return None
    try:
        reader = PdfReader(io.BytesIO(data))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages).strip()
    except Exception as exc:  # noqa: BLE001 - tolerant
        log.warning("pdf extraction failed: %s", exc)
        return None


class PoliteFetcher:
    """GET with robots.txt (cached per origin, fetched through the same client; any robots fetch
    failure => allow), per-domain delay, tenacity retries on transport errors / 5xx / 429.

    ``fetch`` never raises. 401/403/451, robots disallow and proxy CONNECT refusals give
    ``blocked=True``.
    """

    def __init__(
        self,
        client: httpx.Client | None = None,
        user_agent: str = DEFAULT_USER_AGENT,
        timeout: float = 15.0,
        delay: float = 1.0,
        robots: bool = True,
        *,
        max_attempts: int = 3,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        robots_agent: str = ROBOTS_AGENT,
    ) -> None:
        self._owns_client = client is None
        self.client = client or httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers={
                "User-Agent": user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml,application/json;"
                "q=0.9,*/*;q=0.8",
                "Accept-Language": "en-IN,en;q=0.9",
            },
        )
        self.user_agent = user_agent
        self.delay = max(0.0, delay)
        self.robots = robots
        self.robots_agent = robots_agent
        self._sleep = sleep
        self._clock = clock
        self._last: dict[str, float] = {}
        self._robots: dict[str, RobotFileParser | None] = {}
        self._retrying = Retrying(
            stop=stop_after_attempt(max(1, max_attempts)),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception(_is_retryable),
            sleep=sleep,
            reraise=True,
        )
        self.requests = 0

    # ------------------------------------------------------------------ lifecycle
    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def __enter__(self) -> PoliteFetcher:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ politeness
    def _wait_turn(self, host: str) -> None:
        last = self._last.get(host)
        if last is not None and self.delay > 0:
            remaining = self.delay - (self._clock() - last)
            if remaining > 0:
                self._sleep(remaining)
        self._last[host] = self._clock()

    def _get(self, url: str) -> httpx.Response:
        self._wait_turn(urlsplit(url).netloc.lower())
        self.requests += 1
        response = self.client.get(url)
        if response.status_code == 429 or response.status_code >= 500:
            raise _RetryableStatus(response)
        return response

    def _robots_for(self, url: str) -> RobotFileParser | None:
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}".lower()
        if origin in self._robots:
            return self._robots[origin]
        parser: RobotFileParser | None = None
        try:
            self._wait_turn(parts.netloc.lower())
            self.requests += 1
            response = self.client.get(f"{origin}/robots.txt")
            if 200 <= response.status_code < 300:
                parser = RobotFileParser()
                parser.parse(response.text.splitlines())
        except Exception as exc:  # noqa: BLE001 - robots failure => allow
            log.info("robots.txt unavailable for %s: %s", origin, exc)
        self._robots[origin] = parser
        return parser

    def allowed(self, url: str) -> bool:
        if not self.robots:
            return True
        parser = self._robots_for(url)
        return True if parser is None else parser.can_fetch(self.robots_agent, url)

    # ------------------------------------------------------------------ fetch
    def fetch(self, url: str) -> FetchResult:
        try:
            scheme = urlsplit(url).scheme
        except ValueError as exc:
            return FetchResult(url=url, final_url=url, error=f"invalid url: {exc}")
        if scheme not in ("http", "https"):
            return FetchResult(url=url, final_url=url, error=f"unsupported scheme {scheme!r}")
        if not self.allowed(url):
            return FetchResult(url=url, final_url=url, error="robots.txt disallow", blocked=True)
        try:
            response = self._retrying(self._get, url)
        except _RetryableStatus as exc:
            r = exc.response
            return FetchResult(
                url=url,
                final_url=str(r.url),
                status=r.status_code,
                content_type=_ctype(r),
                error=f"HTTP {r.status_code} after retries",
            )
        except httpx.ProxyError as exc:
            return FetchResult(url=url, final_url=url, error=f"proxy: {exc}", blocked=True)
        except Exception as exc:  # noqa: BLE001 - fetch never raises
            return FetchResult(url=url, final_url=url, error=f"{type(exc).__name__}: {exc}")
        return self._result(url, response)

    def _result(self, url: str, r: httpx.Response) -> FetchResult:
        ctype = _ctype(r)
        base = {"url": url, "final_url": str(r.url), "status": r.status_code, "content_type": ctype}
        if r.status_code in BLOCKED_STATUSES:
            return FetchResult(**base, error=f"HTTP {r.status_code}", blocked=True)
        if not 200 <= r.status_code < 300:
            return FetchResult(**base, error=f"HTTP {r.status_code}")
        if "pdf" in ctype:
            text = pdf_to_text(r.content)
            if text is None:
                return FetchResult(**base, error="pdf without extractable text")
            return FetchResult(**{**base, "content_type": "text/plain"}, text=text[:MAX_TEXT_CHARS])
        if ctype and not any(t in ctype for t in _TEXTUAL):
            return FetchResult(**base, error=f"unsupported content-type {ctype}")
        try:
            text = r.text
        except Exception as exc:  # noqa: BLE001 - undecodable body
            return FetchResult(**base, error=f"decode: {exc}")
        return FetchResult(**base, text=text[:MAX_TEXT_CHARS])


def _ctype(r: httpx.Response) -> str:
    return r.headers.get("content-type", "").split(";")[0].strip().lower()
