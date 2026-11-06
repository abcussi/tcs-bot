"""Level-0 sources: search/listing endpoints that seed the BFS (AC-005.1).

Every source turns one query into depth-0 ``RawDocument``s whose ``raw_content`` is the raw item
(an RSS ``<item>`` XML snippet or the item's JSON) and whose ``follow_url`` is where depth 1 lives.
Sources fetch through a shared :class:`PoliteFetcher` (robots, delay, retries) and raise
:class:`SearchError` when the endpoint is blocked or fails, so the crawler records it and moves on.

Depth-0 ``url`` is the item's identity inside the source (Google News article link, Bing
``apiclick`` link, HN/Reddit permalink). GDELT has no item permalink, so its depth-0 url is the
DOC-API endpoint scoped to the article (``...doc?format=json&mode=artlist&item=<article url>``);
this keeps depth-0 and depth-1 rows distinct for dedupe.
"""

from __future__ import annotations

import json
import re
from calendar import timegm
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable
from urllib.parse import parse_qs, quote, urlencode, urlsplit

import feedparser

from gfinance_crawler.news.fetch import PoliteFetcher, pdf_to_text
from gfinance_crawler.news.links import canonical_url, host_of, html_title
from gfinance_crawler.news.models import FetchResult, RawDocument, SourceType
from gfinance_crawler.news.queries import COMMUNITY_SUBREDDITS

GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"
BING_NEWS_RSS = "https://www.bing.com/news/search"
GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"
HN_SEARCH_API = "https://hn.algolia.com/api/v1/search"
HN_ITEM_API = "https://hn.algolia.com/api/v1/items/{id}"
HN_ITEM_WEB = "https://news.ycombinator.com/item?id={id}"
REDDIT = "https://www.reddit.com"

_COMMUNITY_HOSTS = ("reddit.com", "ycombinator.com", "hn.algolia.com", "redd.it")


class SearchError(RuntimeError):
    """A source endpoint could not be searched. ``blocked`` = robots/401/403/451/proxy."""

    def __init__(self, source: str, url: str, error: str | None, blocked: bool) -> None:
        super().__init__(f"{source}: {error or 'failed'} ({url})")
        self.source = source
        self.url = url
        self.blocked = blocked


@runtime_checkable
class Source(Protocol):
    name: str
    source_type: str

    def search(self, query: str, limit: int) -> list[RawDocument]: ...


def source_type_for(url: str, default: SourceType = "news") -> SourceType:
    host = host_of(url)
    if host == "tcs.com" or host.endswith(".tcs.com"):
        return "corporate"
    if any(host == h or host.endswith("." + h) for h in _COMMUNITY_HOSTS):
        return "community"
    return default


def _struct_to_dt(value: Any) -> datetime | None:
    try:
        return datetime.fromtimestamp(timegm(value), tz=UTC) if value else None
    except (TypeError, ValueError, OverflowError):
        return None


def _epoch_to_dt(value: Any) -> datetime | None:
    try:
        return datetime.fromtimestamp(float(value), tz=UTC)
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def _iso_to_dt(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    for fmt in ("%Y%m%dT%H%M%SZ", "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=UTC)
        except ValueError:
            continue
    try:
        dt = datetime.fromisoformat(value)
        return dt if dt.tzinfo else dt.replace(tzinfo=UTC)
    except ValueError:
        return None


class _HttpSource:
    name = "base"
    source_type: SourceType = "news"

    def __init__(self, fetcher: PoliteFetcher | None = None) -> None:
        self.fetcher = fetcher or PoliteFetcher()

    def _get(self, url: str) -> FetchResult:
        result = self.fetcher.fetch(url)
        if not result.ok:
            raise SearchError(self.name, url, result.error, result.blocked)
        return result

    def _get_json(self, url: str) -> Any:
        result = self._get(url)
        try:
            return json.loads(result.text)
        except ValueError as exc:
            raise SearchError(self.name, url, f"invalid JSON: {exc}", False) from exc


# ------------------------------------------------------------------ RSS (Google / Bing)
_ITEM_RE = re.compile(r"<item\b.*?</item>", re.IGNORECASE | re.DOTALL)


def _rss_items(xml: str) -> list[tuple[Any, str]]:
    """``[(feedparser entry, raw <item> XML), ...]`` in feed order."""
    feed = feedparser.parse(xml)
    raws = _ITEM_RE.findall(xml)
    out: list[tuple[Any, str]] = []
    for i, entry in enumerate(feed.entries):
        raw = raws[i] if len(raws) == len(feed.entries) else json.dumps(dict(entry), default=str)
        out.append((entry, raw))
    return out


class GoogleNewsRSS(_HttpSource):
    """Google News search RSS (India edition). Item links are ``news.google.com`` redirects and
    are kept as-is (url + follow_url); the publisher (``<source url=...>``) stays in raw_content."""

    name = "google_news"
    source_type: SourceType = "news"

    def search_url(self, query: str) -> str:
        params = {"q": query, "hl": "en-IN", "gl": "IN", "ceid": "IN:en"}
        return f"{GOOGLE_NEWS_RSS}?{urlencode(params, quote_via=quote)}"

    def search(self, query: str, limit: int) -> list[RawDocument]:
        result = self._get(self.search_url(query))
        docs: list[RawDocument] = []
        for entry, raw in _rss_items(result.text)[: max(0, limit)]:
            link = entry.get("link")
            if not link:
                continue
            docs.append(
                RawDocument(
                    url=link,
                    canonical_url=canonical_url(link),
                    depth=0,
                    source=self.name,
                    source_type=self.source_type,
                    query=query,
                    title=entry.get("title"),
                    raw_content=raw,
                    content_type="application/rss+xml",
                    http_status=result.status,
                    published_at=_struct_to_dt(entry.get("published_parsed")),
                    follow_url=link,
                )
            )
        return docs


def unwrap_bing(link: str) -> str:
    """``https://www.bing.com/news/apiclick.aspx?...&url=<publisher>`` -> publisher URL."""
    try:
        parts = urlsplit(link)
    except ValueError:
        return link
    if "bing.com" in (parts.hostname or ""):
        target = parse_qs(parts.query).get("url")
        if target and target[0].startswith("http"):
            return target[0]
    return link


class BingNewsRSS(_HttpSource):
    """Bing News RSS. ``apiclick`` item links are unwrapped to the publisher URL for depth 1."""

    name = "bing_news"
    source_type: SourceType = "news"

    def search_url(self, query: str) -> str:
        params = {"q": query, "format": "rss", "setlang": "en-IN"}
        return f"{BING_NEWS_RSS}?{urlencode(params, quote_via=quote)}"

    def search(self, query: str, limit: int) -> list[RawDocument]:
        result = self._get(self.search_url(query))
        docs: list[RawDocument] = []
        for entry, raw in _rss_items(result.text)[: max(0, limit)]:
            link = entry.get("link")
            if not link:
                continue
            target = unwrap_bing(link)
            docs.append(
                RawDocument(
                    url=link,
                    canonical_url=canonical_url(link),
                    depth=0,
                    source=self.name,
                    source_type=source_type_for(target, self.source_type),
                    query=query,
                    title=entry.get("title"),
                    raw_content=raw,
                    content_type="application/rss+xml",
                    http_status=result.status,
                    published_at=_struct_to_dt(entry.get("published_parsed")),
                    follow_url=target,
                )
            )
        return docs


# ------------------------------------------------------------------ GDELT
class GdeltDoc(_HttpSource):
    """GDELT 2.0 DOC API, article list mode (direct publisher URLs)."""

    name = "gdelt"
    source_type: SourceType = "news"

    def search_url(self, query: str, limit: int) -> str:
        params = {
            "query": query,
            "mode": "artlist",
            "format": "json",
            "maxrecords": max(1, min(int(limit), 250)),
            "sort": "datedesc",
        }
        return f"{GDELT_DOC_API}?{urlencode(params, quote_via=quote)}"

    @staticmethod
    def item_url(article_url: str) -> str:
        return f"{GDELT_DOC_API}?format=json&mode=artlist&item={quote(article_url, safe='')}"

    def search(self, query: str, limit: int) -> list[RawDocument]:
        if limit <= 0:
            return []
        data = self._get_json(self.search_url(query, limit))
        articles = data.get("articles") or [] if isinstance(data, dict) else []
        docs: list[RawDocument] = []
        for art in articles[:limit]:
            url = art.get("url") if isinstance(art, dict) else None
            if not url:
                continue
            item = self.item_url(url)
            docs.append(
                RawDocument(
                    url=item,
                    canonical_url=canonical_url(item),
                    depth=0,
                    source=self.name,
                    source_type=source_type_for(url, self.source_type),
                    query=query,
                    title=art.get("title"),
                    raw_content=json.dumps(art, ensure_ascii=False),
                    content_type="application/json",
                    published_at=_iso_to_dt(art.get("seendate")),
                    follow_url=url,
                )
            )
        return docs


# ------------------------------------------------------------------ Hacker News
class HackerNews(_HttpSource):
    """Hacker News via the Algolia search API (stories + comments); depth 1 = thread JSON."""

    name = "hackernews"
    source_type: SourceType = "community"

    def search_url(self, query: str, limit: int) -> str:
        params = {"query": query, "tags": "(story,comment)", "hitsPerPage": max(1, int(limit))}
        return f"{HN_SEARCH_API}?{urlencode(params, quote_via=quote)}"

    def search(self, query: str, limit: int) -> list[RawDocument]:
        if limit <= 0:
            return []
        data = self._get_json(self.search_url(query, limit))
        hits = data.get("hits") or [] if isinstance(data, dict) else []
        docs: list[RawDocument] = []
        for hit in hits[:limit]:
            oid = hit.get("objectID") if isinstance(hit, dict) else None
            if not oid:
                continue
            thread_id = hit.get("story_id") or oid
            web = HN_ITEM_WEB.format(id=oid)
            docs.append(
                RawDocument(
                    url=web,
                    canonical_url=canonical_url(web),
                    depth=0,
                    source=self.name,
                    source_type="community",
                    query=query,
                    title=hit.get("title") or hit.get("story_title"),
                    raw_content=json.dumps(hit, ensure_ascii=False),
                    content_type="application/json",
                    published_at=_iso_to_dt(hit.get("created_at"))
                    or _epoch_to_dt(hit.get("created_at_i")),
                    follow_url=HN_ITEM_API.format(id=thread_id),
                )
            )
        return docs


# ------------------------------------------------------------------ Reddit
class RedditSearch(_HttpSource):
    """Reddit search JSON: site-wide plus per-subreddit (``restrict_sr=1``); depth 1 = the
    thread's ``<permalink>.json``. Note reddit.com/robots.txt may disallow crawling, in which
    case every endpoint is reported as blocked."""

    name = "reddit"
    source_type: SourceType = "community"

    def __init__(
        self,
        fetcher: PoliteFetcher | None = None,
        subreddits: list[str] | None = None,
        include_global: bool = True,
    ) -> None:
        super().__init__(fetcher)
        self.subreddits = list(COMMUNITY_SUBREDDITS if subreddits is None else subreddits)
        self.include_global = include_global

    def search_urls(self, query: str, limit: int) -> list[str]:
        base = {"q": query, "sort": "relevance", "t": "year", "limit": max(1, int(limit))}
        urls = []
        if self.include_global:
            urls.append(f"{REDDIT}/search.json?{urlencode(base, quote_via=quote)}")
        for sub in self.subreddits:
            params = {**base, "restrict_sr": 1}
            urls.append(f"{REDDIT}/r/{sub}/search.json?{urlencode(params, quote_via=quote)}")
        return urls

    def search(self, query: str, limit: int) -> list[RawDocument]:
        if limit <= 0:
            return []
        docs: list[RawDocument] = []
        seen: set[str] = set()
        errors: list[SearchError] = []
        for url in self.search_urls(query, limit):
            try:
                data = self._get_json(url)
            except SearchError as exc:
                errors.append(exc)
                if exc.blocked:  # robots/proxy apply to the whole host: stop early
                    break
                continue
            children = ((data or {}).get("data") or {}).get("children") or []
            for child in children[:limit]:
                post = child.get("data") if isinstance(child, dict) else None
                permalink = (post or {}).get("permalink")
                if not permalink:
                    continue
                web = f"{REDDIT}{permalink}"
                key = canonical_url(web)
                if key in seen:
                    continue
                seen.add(key)
                docs.append(
                    RawDocument(
                        url=web,
                        canonical_url=key,
                        depth=0,
                        source=self.name,
                        source_type="community",
                        query=query,
                        title=post.get("title"),
                        raw_content=json.dumps(child, ensure_ascii=False),
                        content_type="application/json",
                        published_at=_epoch_to_dt(post.get("created_utc")),
                        follow_url=f"{REDDIT}{permalink.rstrip('/')}.json",
                    )
                )
        if not docs and errors:
            raise errors[0]
        return docs


# ------------------------------------------------------------------ local inbox
_INBOX_TYPES = {
    ".html": "text/html",
    ".htm": "text/html",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".json": "application/json",
    ".pdf": "text/plain",  # stored as extracted text
}


class LocalInbox:
    """Offline/manual documents dropped in a folder (``data/bronze/inbox``) -> depth-0 docs.

    Query-independent (the crawler calls it once) and trusted (no relevance filter). PDFs are
    stored as their extracted text (pypdf) and skipped when no text can be extracted.
    """

    name = "local_inbox"
    source_type: SourceType = "other"
    query_independent = True
    relevance_filter = False

    def __init__(self, path: str | Path, max_files: int = 500) -> None:
        self.path = Path(path)
        self.max_files = max_files
        self.skipped: list[str] = []

    def files(self) -> list[Path]:
        if not self.path.is_dir():
            return []
        return sorted(
            p
            for p in self.path.iterdir()
            if p.is_file() and p.suffix.lower() in _INBOX_TYPES and not p.name.startswith(".")
        )

    def _read(self, p: Path) -> str | None:
        if p.suffix.lower() == ".pdf":
            return pdf_to_text(p.read_bytes()) or None
        return p.read_text(encoding="utf-8", errors="replace")

    def search(self, query: str = "", limit: int | None = None) -> list[RawDocument]:
        cap = self.max_files if limit is None else min(limit, self.max_files)
        docs: list[RawDocument] = []
        for p in self.files():
            if len(docs) >= cap:
                break
            if p.name.lower() == "readme.md":
                continue
            try:
                text = self._read(p)
            except OSError as exc:
                self.skipped.append(f"{p.name}: {exc}")
                continue
            if not text or not text.strip():
                self.skipped.append(f"{p.name}: empty")
                continue
            ctype = _INBOX_TYPES[p.suffix.lower()]
            title, url, published = self._metadata(p, text, ctype)
            docs.append(
                RawDocument(
                    url=url,
                    canonical_url=canonical_url(url),
                    depth=0,
                    source=self.name,
                    source_type=source_type_for(url, self.source_type)
                    if url.startswith("http")
                    else self.source_type,
                    query=query,
                    title=title,
                    raw_content=text,
                    content_type=ctype,
                    published_at=published,
                    fetched_at=datetime.fromtimestamp(p.stat().st_mtime, tz=UTC),
                )
            )
        return docs

    @staticmethod
    def _metadata(p: Path, text: str, ctype: str) -> tuple[str | None, str, datetime | None]:
        url = p.resolve().as_uri()
        title: str | None = p.stem.replace("_", " ")
        published = None
        if ctype == "text/html":
            title = html_title(text) or title
        elif ctype == "application/json":
            try:
                data = json.loads(text)
            except ValueError:
                data = None
            if isinstance(data, dict):
                title = data.get("title") or title
                if isinstance(data.get("url"), str) and data["url"].startswith("http"):
                    url = data["url"]
                published = _iso_to_dt(data.get("published_at"))
        return title, url, published
