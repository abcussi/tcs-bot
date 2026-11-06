"""Main-text extraction for bronze documents (AC-005.5).

``extract_main_text(raw, content_type, url) -> (text, title | None)`` understands:

* HTML: trafilatura (``favor_recall``, no comments) with a BeautifulSoup fallback that drops
  script/style/nav/footer/header/aside/form; title from trafilatura metadata, ``og:title`` or
  ``<title>``.
* JSON: Reddit thread (``[post_listing, comments_listing]``), Reddit listing / single ``t3``
  item, Hacker News Algolia item or search hit (recursive ``children``), GDELT artlist item or
  ``{"articles": [...]}`` response. Author names are never copied into the text.
* XML: RSS 2.0 / Atom items (Google News, Bing News): title + description with HTML stripped.
* ``text/plain`` / ``text/markdown``: passthrough.

Everything is normalized with :func:`tcs_bot.ingestion.normalize.normalize_text`. The function
never raises on malformed input: it degrades to a best-effort plain-text rendering.
"""

from __future__ import annotations

import html
import json
import re
from typing import Any

from bs4 import BeautifulSoup

from tcs_bot.ingestion.normalize import normalize_text
from tcs_bot.logging_setup import get_logger

log = get_logger(__name__)

MAX_REDDIT_COMMENTS = 40
MAX_HN_COMMENTS = 60
_MIN_TRAFILATURA_WORDS = 20
_DROP_TAGS = ("script", "style", "noscript", "nav", "footer", "header", "aside", "form", "svg")
_REMOVED = {"[deleted]", "[removed]", ""}
_WORD = re.compile(r"\w+")


# --------------------------------------------------------------------------- helpers
def strip_html(fragment: str | None) -> str:
    """HTML fragment (e.g. HN ``comment_text``, RSS ``description``) -> plain text."""
    if not fragment:
        return ""
    if "<" not in fragment and "&" not in fragment:
        return fragment
    soup = BeautifulSoup(fragment, "lxml")
    for br in soup.find_all(["br", "p", "li"]):
        br.insert_before("\n")
    return html.unescape(soup.get_text(" "))


def _join(parts: list[str]) -> str:
    return normalize_text("\n\n".join(p.strip() for p in parts if p and p.strip()))


def _kind(content_type: str, raw: str) -> str:
    ct = (content_type or "").split(";")[0].strip().lower()
    head = raw.lstrip()[:400].lower()
    if "json" in ct:
        return "json"
    if ct in ("text/plain", "text/markdown", "text/x-markdown"):
        return "text"
    if "xhtml" in ct or "html" in ct:
        # Some feeds are served as text/html; sniff RSS/Atom roots.
        return "xml" if head.startswith(("<?xml", "<rss", "<feed", "<item")) else "html"
    if "xml" in ct or "rss" in ct or "atom" in ct:
        return "xml" if "<html" not in head else "html"
    # unknown content type -> sniff
    if head.startswith(("{", "[")):
        return "json"
    if head.startswith(("<rss", "<feed", "<item")) or (head.startswith("<?xml") and "<rss" in head):
        return "xml"
    if head.startswith("<"):
        return "html"
    return "text"


# --------------------------------------------------------------------------- HTML
def _html_title(raw: str, soup: BeautifulSoup | None = None) -> str | None:
    try:
        import trafilatura

        meta = trafilatura.extract_metadata(raw)
        if meta is not None and meta.title:
            return normalize_text(meta.title)
    except Exception:  # noqa: BLE001 - metadata is best effort
        pass
    soup = soup or BeautifulSoup(raw, "lxml")
    og = soup.find("meta", attrs={"property": "og:title"}) or soup.find(
        "meta", attrs={"name": "twitter:title"}
    )
    if og is not None and og.get("content"):
        return normalize_text(str(og["content"]))
    if soup.title and soup.title.string:
        return normalize_text(soup.title.string)
    h1 = soup.find("h1")
    return normalize_text(h1.get_text(" ")) if h1 else None


def _bs4_text(soup: BeautifulSoup) -> str:
    for tag in soup(_DROP_TAGS):
        tag.decompose()
    body = soup.body or soup
    return normalize_text(body.get_text("\n"))


def _extract_html(raw: str, url: str) -> tuple[str, str | None]:
    text = ""
    try:
        import trafilatura

        text = (
            trafilatura.extract(
                raw,
                url=url or None,
                include_comments=False,
                favor_recall=True,
                include_tables=True,
            )
            or ""
        )
    except Exception as exc:  # noqa: BLE001 - fall back to bs4
        log.debug("silver.extract.trafilatura_failed", url=url, error=str(exc)[:200])
    soup = BeautifulSoup(raw, "lxml")
    title = _html_title(raw, soup)
    if len(_WORD.findall(text)) < _MIN_TRAFILATURA_WORDS:
        fallback = _bs4_text(soup)
        if len(fallback) > len(text):
            text = fallback
    return normalize_text(text), title


# --------------------------------------------------------------------------- JSON
def _reddit_comment_bodies(listing: Any, out: list[str], limit: int) -> None:
    """Depth-first over a Reddit comments listing, collecting ``body`` (never ``author``)."""
    if not isinstance(listing, dict):
        return
    for child in (listing.get("data") or {}).get("children") or []:
        if len(out) >= limit:
            return
        if not isinstance(child, dict) or child.get("kind") != "t1":
            continue
        data = child.get("data") or {}
        body = strip_html(str(data.get("body") or "")).strip()
        if body not in _REMOVED:
            out.append(body)
        _reddit_comment_bodies(data.get("replies"), out, limit)


def _reddit_post(data: dict[str, Any]) -> tuple[list[str], str | None]:
    title = str(data.get("title") or "").strip() or None
    parts = [title or "", strip_html(str(data.get("selftext") or ""))]
    return [p for p in parts if p.strip() not in _REMOVED], title


def _extract_reddit(obj: Any) -> tuple[list[str], str | None] | None:
    # Thread page: [post_listing, comments_listing]
    if isinstance(obj, list) and obj and all(isinstance(x, dict) for x in obj):
        if obj[0].get("kind") == "Listing":
            posts = (obj[0].get("data") or {}).get("children") or []
            parts: list[str] = []
            title = None
            for p in posts:
                if isinstance(p, dict) and p.get("kind") == "t3":
                    post_parts, title = _reddit_post(p.get("data") or {})
                    parts.extend(post_parts)
            comments: list[str] = []
            for listing in obj[1:]:
                _reddit_comment_bodies(listing, comments, MAX_REDDIT_COMMENTS)
            return parts + comments, title
        return None
    if not isinstance(obj, dict):
        return None
    kind = obj.get("kind")
    if kind == "t3":
        return _reddit_post(obj.get("data") or {})
    if kind == "Listing":  # search listing: concatenate posts
        parts, title = [], None
        for child in (obj.get("data") or {}).get("children") or []:
            if isinstance(child, dict) and child.get("kind") == "t3":
                post_parts, t = _reddit_post(child.get("data") or {})
                parts.extend(post_parts)
                title = title or t
        return parts, title
    if "selftext" in obj and "subreddit" in obj:  # bare post data
        return _reddit_post(obj)
    return None


def _hn_walk(node: Any, out: list[str], limit: int) -> None:
    if not isinstance(node, dict) or len(out) >= limit:
        return
    for key in ("story_text", "comment_text", "text"):
        val = node.get(key)
        if val:
            out.append(strip_html(str(val)))
            break
    for child in node.get("children") or []:
        if len(out) >= limit:
            return
        _hn_walk(child, out, limit)


def _extract_hn(obj: Any) -> tuple[list[str], str | None] | None:
    if not isinstance(obj, dict):
        return None
    is_item = "children" in obj and obj.get("type") in ("story", "comment", "poll", "job")
    is_hit = "objectID" in obj or "_tags" in obj or "story_text" in obj or "comment_text" in obj
    if not (is_item or is_hit):
        return None
    title = str(obj.get("title") or obj.get("story_title") or "").strip() or None
    parts: list[str] = [title or ""]
    _hn_walk(obj, parts, MAX_HN_COMMENTS + 1)
    return parts, title


def _extract_gdelt(obj: Any) -> tuple[list[str], str | None] | None:
    if isinstance(obj, dict) and isinstance(obj.get("articles"), list):
        titles = [str(a.get("title") or "") for a in obj["articles"] if isinstance(a, dict)]
        return titles, (titles[0] if titles else None)
    if isinstance(obj, dict) and "seendate" in obj and "title" in obj:
        title = str(obj.get("title") or "").strip() or None
        return [title or ""], title
    return None


def _extract_generic_json(obj: Any) -> tuple[list[str], str | None]:
    if isinstance(obj, dict):
        title = next((str(obj[k]) for k in ("title", "headline", "name") if obj.get(k)), None)
        parts = [title or ""]
        for k in ("description", "summary", "selftext", "text", "body", "content", "abstract"):
            if isinstance(obj.get(k), str):
                parts.append(strip_html(obj[k]))
        return parts, title
    if isinstance(obj, list):
        parts, title = [], None
        for item in obj[:50]:
            p, t = _extract_generic_json(item)
            parts.extend(p)
            title = title or t
        return parts, title
    return [str(obj)], None


def _extract_json(raw: str) -> tuple[str, str | None]:
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return normalize_text(raw), None
    for extractor in (_extract_reddit, _extract_hn, _extract_gdelt):
        res = extractor(obj)
        if res is not None:
            parts, title = res
            return _join(parts), (normalize_text(title) if title else None)
    parts, title = _extract_generic_json(obj)
    return _join(parts), (normalize_text(title) if title else None)


# --------------------------------------------------------------------------- RSS / Atom
def _extract_xml(raw: str) -> tuple[str, str | None]:
    try:
        soup = BeautifulSoup(raw, "xml")
    except Exception:  # noqa: BLE001 - lxml-xml unavailable / broken markup
        soup = BeautifulSoup(raw, "lxml")
    items = soup.find_all(["item", "entry"])
    if not items:
        return _extract_html(raw, "")
    parts: list[str] = []
    first_title: str | None = None
    for it in items[:50]:
        t = it.find("title")
        title = strip_html(t.get_text(" ")) if t else ""
        d = it.find(["description", "summary", "content"])
        desc = strip_html(d.get_text(" ")) if d else ""
        first_title = first_title or (normalize_text(title) or None)
        parts.extend([title, desc])
    return _join(parts), first_title


# --------------------------------------------------------------------------- public API
def extract_main_text(raw: str, content_type: str, url: str) -> tuple[str, str | None]:
    """Return ``(clean_text, title | None)`` for a bronze ``raw_content``."""
    if not raw or not raw.strip():
        return "", None
    kind = _kind(content_type, raw)
    try:
        if kind == "html":
            return _extract_html(raw, url)
        if kind == "json":
            return _extract_json(raw)
        if kind == "xml":
            return _extract_xml(raw)
    except Exception as exc:  # noqa: BLE001 - never crash the silver batch on bad markup
        log.warning("silver.extract.failed", url=url, kind=kind, error=str(exc)[:200])
        return normalize_text(strip_html(raw)), None
    return normalize_text(raw), None
