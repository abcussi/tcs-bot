"""Breadth-first TCS risk crawler with hard caps (AC-005.2, AC-005.3).

Depth 0: every source x query search item (TCS-relevant ones are kept).
Depth 1: the item's ``follow_url`` (article / HN thread JSON / Reddit thread JSON), stored only if
         TCS-relevant.
Depth 2: outbound links of depth-1 pages whose anchor/URL text is TCS-relevant and whose domain is
         allow-listed; the fetched page is stored only if TCS-relevant. Depth 2 is never expanded.

Hard caps: ``max_depth <= 2`` and ``max_docs <= 500`` (yielded documents). Dedupe is on the
canonical URL (both of what is fetched and of what is yielded). A blocked or failing source /
fetch is counted in ``stats`` and never stops the run.
"""

from __future__ import annotations

import logging
import re
from collections import deque
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from typing import Any

from bs4 import BeautifulSoup

from gfinance_crawler.news.fetch import PoliteFetcher
from gfinance_crawler.news.links import (
    canonical_url,
    extract_links,
    extract_links_json,
    host_of,
    html_title,
    is_tcs_relevant,
    visible_text,
)
from gfinance_crawler.news.models import FetchResult, RawDocument
from gfinance_crawler.news.queries import DEFAULT_ALLOWED_DOMAINS, domain_allowed
from gfinance_crawler.news.sources import SearchError, Source, source_type_for

log = logging.getLogger(__name__)

HARD_MAX_DOCS = 500
HARD_MAX_DEPTH = 2
_MAX_SOURCE_FAILURES = 3


def _new_stats() -> dict[str, Any]:
    return {
        "discovered": 0,
        "fetched": 0,
        "stored": 0,
        "duplicates": 0,
        "skipped_irrelevant": 0,
        "blocked": 0,
        "failed": 0,
        "by_depth": {},
        "by_source": {},
        "links_skipped": 0,
        "errors": [],
    }


def _published_from_html(html: str) -> datetime | None:
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:  # noqa: BLE001
        return None
    for attrs in (
        {"property": "article:published_time"},
        {"name": "pubdate"},
        {"name": "publish-date"},
        {"itemprop": "datePublished"},
    ):
        tag = soup.find("meta", attrs=attrs)
        if tag is not None and tag.get("content"):
            try:
                dt = datetime.fromisoformat(str(tag["content"]).replace("Z", "+00:00"))
            except ValueError:
                continue
            return dt if dt.tzinfo else dt.replace(tzinfo=UTC)
    return None


def _json_title(text: str) -> str | None:
    m = re.search(r'"title"\s*:\s*"((?:[^"\\]|\\.){1,300})"', text)
    return m.group(1) if m else None


class RiskCrawler:
    """``crawl(queries)`` yields ``RawDocument`` breadth-first; ``stats`` reports the run."""

    def __init__(
        self,
        sources: Sequence[Source],
        fetcher: PoliteFetcher,
        allowed_domains: Sequence[str] | None = None,
        max_docs: int = HARD_MAX_DOCS,
        max_depth: int = HARD_MAX_DEPTH,
        per_query_limit: int | None = None,
    ) -> None:
        self.sources = list(sources)
        self.fetcher = fetcher
        self.allowed_domains = list(
            DEFAULT_ALLOWED_DOMAINS if allowed_domains is None else allowed_domains
        )
        self.max_docs = max(0, min(int(max_docs), HARD_MAX_DOCS))
        self.max_depth = max(0, min(int(max_depth), HARD_MAX_DEPTH))
        self.per_query_limit = per_query_limit
        self._stats = _new_stats()

    @property
    def stats(self) -> dict[str, Any]:
        s = dict(self._stats)
        s["by_depth"] = dict(s["by_depth"])
        s["by_source"] = dict(s["by_source"])
        s["errors"] = list(s["errors"])
        return s

    # ------------------------------------------------------------------ helpers
    def _limit_for(self, n_queries: int) -> int:
        if self.per_query_limit is not None:
            return max(0, self.per_query_limit)
        n_sources = sum(1 for s in self.sources if not getattr(s, "query_independent", False))
        # Leave ~2/3 of the budget for depth 1-2.
        return max(2, self.max_docs // max(1, 3 * n_queries * max(1, n_sources)))

    def _error(self, msg: str) -> None:
        if len(self._stats["errors"]) < 50:
            self._stats["errors"].append(msg[:300])

    def _record(self, doc: RawDocument) -> None:
        s = self._stats
        s["stored"] += 1
        d = str(doc.depth)
        s["by_depth"][d] = s["by_depth"].get(d, 0) + 1
        s["by_source"][doc.source] = s["by_source"].get(doc.source, 0) + 1

    @property
    def _full(self) -> bool:
        return self._stats["stored"] >= self.max_docs

    def _fetch(self, url: str) -> FetchResult | None:
        res = self.fetcher.fetch(url)
        if res.blocked:
            self._stats["blocked"] += 1
            self._error(f"blocked {url}: {res.error}")
            return None
        if not res.ok:
            self._stats["failed"] += 1
            self._error(f"failed {url}: {res.error}")
            return None
        self._stats["fetched"] += 1
        return res

    # ------------------------------------------------------------------ crawl
    def crawl(self, queries: Sequence[str]) -> Iterator[RawDocument]:
        self._stats = _new_stats()
        if self.max_docs == 0:
            return
        stored: set[str] = set()  # canonical urls yielded (or rejected as off-topic)
        fetched: set[str] = set()  # canonical urls fetched / queued for fetch
        frontier1: deque[RawDocument] = deque()
        frontier2: deque[tuple[str, RawDocument]] = deque()
        queries = list(queries)
        limit = self._limit_for(len(queries))

        # ---------------- depth 0
        for doc in self._depth0(queries, limit):
            key = doc.canonical_url or canonical_url(doc.url)
            if key in stored:
                self._stats["duplicates"] += 1
                continue
            stored.add(key)
            src = next((s for s in self.sources if s.name == doc.source), None)
            if getattr(src, "relevance_filter", True) and not is_tcs_relevant(
                f"{doc.title or ''} {visible_text(doc.raw_content, doc.content_type)}"
            ):
                self._stats["skipped_irrelevant"] += 1
                continue
            yield doc
            self._record(doc)
            if self._full:
                return
            if doc.follow_url and self.max_depth >= 1:
                frontier1.append(doc)

        # ---------------- depth 1
        while frontier1 and not self._full:
            parent = frontier1.popleft()
            doc = self._fetch_child(parent.follow_url or "", parent, 1, stored, fetched)
            if doc is None:
                continue
            yield doc
            self._record(doc)
            if self.max_depth >= 2:
                self._enqueue_links(doc, frontier2, fetched, stored)

        # ---------------- depth 2
        while frontier2 and not self._full:
            url, parent = frontier2.popleft()
            doc = self._fetch_child(url, parent, 2, stored, fetched, queued=True)
            if doc is not None:
                yield doc
                self._record(doc)
        log.info("news crawl done: %s", {k: v for k, v in self._stats.items() if k != "errors"})

    def _depth0(self, queries: list[str], limit: int) -> Iterator[RawDocument]:
        failures: dict[str, int] = {}
        disabled: set[str] = set()

        def run(source: Source, query: str, n: int) -> list[RawDocument]:
            try:
                return source.search(query, n)
            except SearchError as exc:
                if exc.blocked:
                    self._stats["blocked"] += 1
                    disabled.add(source.name)  # one blocked source never stops the others
                else:
                    self._stats["failed"] += 1
                    failures[source.name] = failures.get(source.name, 0) + 1
                self._error(str(exc))
            except Exception as exc:  # noqa: BLE001 - a broken source must not stop the run
                self._stats["failed"] += 1
                failures[source.name] = failures.get(source.name, 0) + 1
                self._error(f"{source.name}: {type(exc).__name__}: {exc}")
            if failures.get(source.name, 0) >= _MAX_SOURCE_FAILURES:
                disabled.add(source.name)
            return []

        for source in self.sources:
            if getattr(source, "query_independent", False):
                for doc in run(source, "", self.max_docs):
                    self._stats["discovered"] += 1
                    yield doc
        for query in queries:
            for source in self.sources:
                if getattr(source, "query_independent", False) or source.name in disabled:
                    continue
                for doc in run(source, query, limit):
                    self._stats["discovered"] += 1
                    yield doc

    def _fetch_child(
        self,
        url: str,
        parent: RawDocument,
        depth: int,
        stored: set[str],
        fetched: set[str],
        queued: bool = False,
    ) -> RawDocument | None:
        key = canonical_url(url)
        if not queued:
            if key in fetched:
                self._stats["duplicates"] += 1
                return None
            fetched.add(key)
        res = self._fetch(url)
        if res is None:
            return None
        final_key = canonical_url(res.final_url)
        fetched.add(final_key)
        if final_key in stored:
            self._stats["duplicates"] += 1
            return None
        stored.add(final_key)
        is_html = "html" in res.content_type or res.text.lstrip()[:1] == "<"
        is_json = "json" in res.content_type
        title = html_title(res.text) if is_html else (_json_title(res.text) if is_json else None)
        if not is_tcs_relevant(f"{title or ''} {visible_text(res.text, res.content_type)}"):
            self._stats["skipped_irrelevant"] += 1
            return None
        default_type = parent.source_type if parent.source_type != "other" else "news"
        return RawDocument(
            url=res.final_url,
            canonical_url=final_key,
            parent_url=parent.url,
            depth=depth,
            source=parent.source if depth == 1 else "web",
            source_type=source_type_for(res.final_url, default_type),
            query=parent.query,
            title=title or (parent.title if depth == 1 else None),
            raw_content=res.text,
            content_type=res.content_type or "text/html",
            http_status=res.status,
            published_at=(_published_from_html(res.text) if is_html else None)
            or (parent.published_at if depth == 1 else None),
        )

    def _enqueue_links(
        self,
        doc: RawDocument,
        frontier2: deque[tuple[str, RawDocument]],
        fetched: set[str],
        stored: set[str],
    ) -> None:
        if "json" in doc.content_type:
            links = extract_links_json(doc.raw_content)
        elif "html" in doc.content_type or doc.raw_content.lstrip()[:1] == "<":
            links = extract_links(doc.raw_content, doc.url)
        else:
            return
        for url, anchor in links:
            key = canonical_url(url)
            if key in fetched or key in stored or key == doc.canonical_url:
                continue
            if not domain_allowed(host_of(url), self.allowed_domains) or not is_tcs_relevant(
                f"{anchor} {url}"
            ):
                self._stats["links_skipped"] += 1
                continue
            fetched.add(key)
            self._stats["discovered"] += 1
            frontier2.append((url, doc))
