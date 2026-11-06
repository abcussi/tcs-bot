"""URL canonicalisation, link extraction and TCS relevance (AC-005.2, AC-005.3)."""

from __future__ import annotations

import html as html_lib
import json
import re
from typing import Any
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

from bs4 import BeautifulSoup

_TRACKING_PARAMS = {"fbclid", "gclid", "dclid", "msclkid", "igshid", "mc_cid", "mc_eid"}
_DEFAULT_PORTS = {"http": 80, "https": 443}


def canonical_url(url: str) -> str:
    """Normalise a URL for dedupe: lowercase scheme/host, drop default port, fragment,
    ``utm_*``/click-id params and the trailing slash; sort the remaining query params."""
    url = (url or "").strip()
    try:
        parts = urlsplit(url)
    except ValueError:
        return url
    scheme = parts.scheme.lower()
    host = (parts.hostname or "").lower().rstrip(".")
    netloc = host
    try:
        port = parts.port
    except ValueError:
        port = None
    if port and _DEFAULT_PORTS.get(scheme) != port:
        netloc = f"{host}:{port}"
    if parts.username:
        netloc = f"{parts.username}@{netloc}"
    query = [
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if not k.lower().startswith("utm_") and k.lower() not in _TRACKING_PARAMS
    ]
    query.sort()
    path = parts.path or ""
    if len(path) > 1:
        path = path.rstrip("/")
    elif path == "/":
        path = ""
    if scheme == "file" and not path:
        path = parts.path
    return urlunsplit((scheme, netloc, path, urlencode(query, doseq=True), ""))


def host_of(url: str) -> str:
    try:
        return (urlsplit(url).hostname or "").lower()
    except ValueError:
        return ""


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def extract_links(html: str, base_url: str) -> list[tuple[str, str]]:
    """Absolute http(s) links of an HTML page as ``(url, anchor text)``, in document order,
    de-duplicated on canonical URL. Never raises."""
    try:
        soup = BeautifulSoup(html or "", "lxml")
    except Exception:  # noqa: BLE001 - tolerant parsing
        return []
    base = base_url
    base_tag = soup.find("base", href=True)
    if base_tag is not None:
        base = urljoin(base_url, str(base_tag["href"]))
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for a in soup.find_all("a", href=True):
        href = str(a["href"]).strip()
        if not href or href.startswith(("#", "mailto:", "javascript:", "tel:", "data:")):
            continue
        absolute = urljoin(base, href)
        if urlsplit(absolute).scheme not in ("http", "https"):
            continue
        key = canonical_url(absolute)
        if key in seen:
            continue
        seen.add(key)
        anchor = _clean(a.get_text(" ")) or _clean(str(a.get("title") or ""))
        out.append((absolute, anchor))
    return out


def extract_links_json(text: str) -> list[tuple[str, str]]:
    """Links inside a JSON payload (HN item / Reddit thread): ``url`` fields (anchor = sibling
    ``title``) plus ``<a href>`` inside HTML string fields (HN ``text``, Reddit ``*_html``)."""
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        return []
    out: list[tuple[str, str]] = []
    seen: set[str] = set()

    def add(url: str, anchor: str) -> None:
        if urlsplit(url).scheme not in ("http", "https"):
            return
        key = canonical_url(url)
        if key not in seen:
            seen.add(key)
            out.append((url, anchor))

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            url = node.get("url")
            if isinstance(url, str) and url.startswith("http"):
                add(url, _clean(str(node.get("title") or "")))
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
        elif isinstance(node, str) and "<a " in node:
            for url, anchor in extract_links(html_lib.unescape(node), "https://invalid.local/"):
                add(url, anchor)

    walk(data)
    return out


def html_title(html: str) -> str | None:
    try:
        soup = BeautifulSoup(html or "", "lxml")
    except Exception:  # noqa: BLE001
        return None
    og = soup.find("meta", attrs={"property": "og:title"})
    if og is not None and og.get("content"):
        return _clean(str(og["content"])) or None
    if soup.title and soup.title.string:
        return _clean(soup.title.string) or None
    h1 = soup.find("h1")
    if h1 is None:
        return None
    return _clean(h1.get_text(" ")) or None


def _json_strings(node: Any, out: list[str]) -> None:
    if isinstance(node, str):
        out.append(node)
    elif isinstance(node, dict):
        for key, value in node.items():
            if key == "_highlightResult":  # Algolia search metadata (echoes the query terms)
                continue
            _json_strings(value, out)
    elif isinstance(node, list):
        for value in node:
            _json_strings(value, out)


def visible_text(content: str, content_type: str) -> str:
    """Rough plain text used for relevance checks (not the silver extractor).

    HTML -> visible text; JSON -> its string values only (keys and Algolia ``_highlightResult``
    metadata are dropped so that the *query terms* echoed back by a search API never make an
    unrelated hit look TCS-relevant).
    """
    if "html" in (content_type or "").lower() or content.lstrip()[:1] == "<":
        try:
            soup = BeautifulSoup(content, "lxml")
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()
            return _clean(soup.get_text(" "))
        except Exception:  # noqa: BLE001
            return content
    if "json" in (content_type or "").lower() or content.lstrip()[:1] in "{[":
        try:
            data = json.loads(content)
        except ValueError:
            return content
        parts: list[str] = []
        _json_strings(data, parts)
        return " ".join(html_lib.unescape(p) for p in parts)
    return content


# ------------------------------------------------------------------ relevance
# Unambiguous references to the company.
_STRONG = re.compile(
    r"tata\s+consultancy|tcs\.com|krithivasan|\bTCS\s+(?:Ltd|Limited|iON|BaNCS)\b"
    r"|\bNSE\s*:\s*TCS\b|\bTCS\.NS\b",
    re.IGNORECASE,
)
_BARE = re.compile(r"(?<![A-Za-z0-9])tcs(?![A-Za-z0-9])", re.IGNORECASE)
# "TCS" as Indian *Tax Collected at Source* (e.g. on LRS / foreign remittances).
_TAX_DOC = re.compile(
    r"tax\s+collected\s+at\s+source|liberali[sz]ed\s+remittance|\bsection\s*206\s*C\b",
    re.IGNORECASE,
)
_TAX_NEAR = re.compile(
    r"\bLRS\b|remittance|\bTDS\b|\bGST\b|income[\s-]tax|tax\s+credit|\bITR\b|206\s*C"
    r"|\bTCS\s+(?:rate|deduct|refund|credit|on)\b",
    re.IGNORECASE,
)
# "TCS" used for something else (e.g. the "Taking Children Seriously" parenting philosophy).
_OTHER_TCS = re.compile(
    r"taking\s+children\s+seriously|unconditional\s+parenting",
    re.IGNORECASE,
)
_WINDOW = 80


def is_tcs_relevant(text: str) -> bool:
    """True when ``text`` refers to Tata Consultancy Services.

    Matches "Tata Consultancy", ``tcs.com``, CEO K Krithivasan, "TCS Ltd", ticker forms, and bare
    "TCS"/"tcs" (incl. URL slugs) unless it is used as *Tax Collected at Source*: a document that
    talks about tax-collected-at-source/LRS and has no unambiguous company reference is rejected,
    and bare mentions within ~80 chars of tax/remittance vocabulary do not count.
    """
    if not text:
        return False
    if _STRONG.search(text):
        return True
    if _TAX_DOC.search(text):
        return False
    for m in _BARE.finditer(text):
        window = text[max(0, m.start() - _WINDOW) : m.end() + _WINDOW]
        if not _TAX_NEAR.search(window) and not _OTHER_TCS.search(window):
            return True
    return False
