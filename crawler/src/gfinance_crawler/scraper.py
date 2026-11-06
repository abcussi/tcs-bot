"""Google Finance quote page fetcher + tolerant HTML parser (AC-002.2, AC-002.3).

Parsing strategy, in order of preference for each field:
1. ``data-*`` attributes on the main ``div[data-last-price]`` element,
2. known CSS classes (``zzDege`` name, ``YMlKec fxKbKc`` price, ``JwB6zf`` change %, ...),
3. label text of the "key stats" rows
   (``div.gyFHrc`` -> ``div.mfs7Fc`` label / ``div.P6K39c`` value).

The parser never raises: anything it cannot read becomes ``None``.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from typing import Any

import httpx
from bs4 import BeautifulSoup, Tag
from tenacity import (
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from gfinance_crawler.config import CrawlerSettings
from gfinance_crawler.models import Quote

log = logging.getLogger(__name__)

# ---------------------------------------------------------------- number parsing helpers

_MISSING = {"", "-", "—", "–", "n/a", "na", "none", "null", "--"}
_SUFFIX = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}
_NUMBER_RE = re.compile(r"([+\-]?)\s*[^\d+\-.]*?(\d[\d,]*(?:\.\d+)?|\.\d+)\s*([KMBT])?(?![A-Za-z])")
_CURRENCY_SYMBOLS = {
    "$": "USD",
    "€": "EUR",
    "£": "GBP",
    "¥": "JPY",
    "₹": "INR",
    "₩": "KRW",
    "CHF": "CHF",
}


def _normalize(text: str | None) -> str:
    if text is None:
        return ""
    return (
        text.replace("−", "-")  # unicode minus
        .replace("\xa0", " ")
        .replace(" ", " ")
        .strip()
    )


def parse_number(text: str | None) -> float | None:
    """Parse ``"$1,234.56"``, ``"-2.5%"``, ``"2.45T USD"``, ``"1.2B"`` -> float; junk -> None.

    Magnitude suffixes K/M/B/T are expanded to absolute units. Never raises.
    """
    try:
        s = _normalize(text)
        if s.lower() in _MISSING:
            return None
        m = _NUMBER_RE.search(s)
        if not m:
            return None
        sign, digits, suffix = m.groups()
        value = float(digits.replace(",", ""))
        if suffix:
            value *= _SUFFIX[suffix.upper()]
        if sign == "-" or (not sign and s.startswith("(") and s.endswith(")")):
            value = -value
        return value
    except (ValueError, TypeError, KeyError):
        return None


def parse_percent(text: str | None) -> float | None:
    """``"1.23%"`` -> 1.23 (percent points, not a fraction). Never raises."""
    return parse_number(text)


def parse_range(text: str | None) -> str | None:
    """Normalise ``"$150.10 - $155.20"`` -> ``"150.10 - 155.20"``; missing -> None."""
    s = _normalize(text)
    if s.lower() in _MISSING:
        return None
    parts = [p.strip() for p in re.split(r"\s+[-–—]\s+", s) if p.strip()]
    nums = [parse_number(p) for p in parts]
    if len(parts) == 2 and all(n is not None for n in nums):
        return " - ".join(re.sub(r"[^\d.,\-]", "", p) for p in parts)
    return s or None


def detect_currency(text: str | None) -> str | None:
    """Best-effort ISO currency from a price string (``"$12"`` -> USD, ``"2.4T EUR"`` -> EUR)."""
    s = _normalize(text)
    if not s:
        return None
    code = re.search(r"\b([A-Z]{3})\b", s)
    if code:
        return code.group(1)
    for symbol, iso in _CURRENCY_SYMBOLS.items():
        if symbol in s:
            return iso
    return None


# ---------------------------------------------------------------- HTML parsing

_LABELS: dict[str, tuple[str, ...]] = {
    "previous_close": ("previous close", "prev close", "prev. close"),
    "day_range": ("day range", "day's range", "days range"),
    "year_range": ("year range", "52-week range", "52 week range", "52wk range"),
    "market_cap": ("market cap", "market capitalization"),
    "pe_ratio": ("p/e ratio", "pe ratio", "p/e"),
    "dividend_yield": ("dividend yield", "div yield", "dividend"),
    "primary_exchange": ("primary exchange",),
}


def _text(node: Tag | None) -> str | None:
    if node is None:
        return None
    t = _normalize(node.get_text(" ", strip=True))
    return t or None


def _key_stats(soup: BeautifulSoup) -> dict[str, str]:
    """Map normalised label -> raw value text from the key stats panel."""
    stats: dict[str, str] = {}
    rows = soup.select("div.gyFHrc")
    for row in rows:
        label = _text(row.select_one("div.mfs7Fc"))
        value = _text(row.select_one("div.P6K39c"))
        if label and value is not None:
            stats.setdefault(label.lower(), value)
    if not stats:  # rows without the wrapper class: pair each label with its next sibling
        for lab in soup.select("div.mfs7Fc"):
            sib = lab.find_next_sibling()
            label, value = _text(lab), _text(sib if isinstance(sib, Tag) else None)
            if label and value is not None:
                stats.setdefault(label.lower(), value)
    return stats


def _stat_by_text(soup: BeautifulSoup, aliases: Iterable[str]) -> str | None:
    """Last-resort lookup: find an element whose whole text is the label, use the next element."""
    wanted = set(aliases)
    for node in soup.find_all(string=True):
        if _normalize(str(node)).lower() in wanted:
            nxt = node.find_next(string=lambda s: bool(_normalize(s)))
            if nxt is not None:
                return _normalize(str(nxt)) or None
    return None


def _lookup(stats: dict[str, str], soup: BeautifulSoup, field: str) -> str | None:
    aliases = _LABELS[field]
    for alias in aliases:
        if alias in stats:
            return stats[alias]
    for label, value in stats.items():  # labels sometimes carry tooltips / extra words
        if any(label.startswith(alias) for alias in aliases):
            return value
    return _stat_by_text(soup, aliases)


def _main_price_node(soup: BeautifulSoup, ticker: str) -> Tag | None:
    nodes = soup.select("[data-last-price]")
    for node in nodes:
        if str(node.get("data-symbol", "")).upper() == ticker.upper():
            return node
    return nodes[0] if nodes else None


def _attr(node: Tag | None, name: str) -> str | None:
    if node is None:
        return None
    value = node.get(name)
    if isinstance(value, list):
        value = " ".join(value)
    return _normalize(value) or None


def _is_down(node: Tag | None) -> bool:
    """Google renders the direction with an arrow icon / aria-label rather than a sign."""
    if node is None:
        return False
    for el in [node, *node.find_all(True)]:
        label = (_attr(el, "aria-label") or "").lower()
        if label.startswith("down"):
            return True
    return False


def _name(soup: BeautifulSoup) -> str | None:
    name = _text(soup.select_one("div.zzDege"))
    if name:
        return name
    title = _text(soup.find("title"))
    if title:
        name = re.split(r"\s+\(|\s+[-|]\s+", title)[0].strip()
        if name and "google finance" not in name.lower():
            return name
    return None


def _safe(fn: Callable[[], Any]) -> Any:
    try:
        return fn()
    except Exception:  # noqa: BLE001 - the parser must never crash the run
        log.debug("field parse failed", exc_info=True)
        return None


def parse_quote_html(html: str, ticker: str, exchange: str, url: str) -> Quote:
    """Parse a Google Finance quote page into a :class:`Quote` (AC-002.3). Never raises."""
    try:
        soup = BeautifulSoup(html or "", "lxml")
    except Exception:  # noqa: BLE001
        log.warning("unparseable HTML for %s:%s", ticker, exchange)
        return Quote(ticker=ticker, exchange=exchange, source_url=url)

    data_node = _safe(lambda: _main_price_node(soup, ticker))
    stats: dict[str, str] = _safe(lambda: _key_stats(soup)) or {}

    def stat(field: str) -> str | None:
        return _safe(lambda: _lookup(stats, soup, field))

    price_text = _safe(lambda: _text(soup.select_one("div.YMlKec.fxKbKc")))
    price = parse_number(_attr(data_node, "data-last-price")) or parse_number(price_text)

    market_cap_text = stat("market_cap")
    currency = (
        _attr(data_node, "data-currency-code")
        or detect_currency(price_text)
        or detect_currency(market_cap_text)
    )

    previous_close = parse_number(stat("previous_close"))

    # change % : JwB6zf (header badge), else inside enJeMd / P2Luy
    pct_node = _safe(lambda: soup.select_one("div.JwB6zf")) or _safe(
        lambda: soup.select_one("div.enJeMd span.P2Luy, span.P2Luy")
    )
    change_pct: float | None = None
    pct_text = _text(pct_node)
    if pct_text and "%" in pct_text:
        pct_match = re.search(r"[+\-]?\s*[\d.,]+\s*%", _normalize(pct_text))
        change_pct = parse_percent(pct_match.group(0)) if pct_match else None

    # absolute change: first signed number in span.P2Luy (e.g. "+2.50 Today" / "-$1.10")
    change: float | None = None
    change_node = _safe(lambda: soup.select_one("span.P2Luy"))
    change_text = _text(change_node)
    if change_text:
        abs_part = re.sub(r"\(.*?\)|[+\-]?[\d.,]+\s*%", "", _normalize(change_text))
        change = parse_number(abs_part)

    down = _safe(lambda: _is_down(pct_node)) or False
    if down:
        change_pct = -abs(change_pct) if change_pct is not None else None
        change = -abs(change) if change is not None else None

    # fallbacks / sign repair from price vs previous close
    if price is not None and previous_close:
        computed = price - previous_close
        if change is None:
            change = round(computed, 4)
        if change_pct is None:
            change_pct = round(computed / previous_close * 100, 4)
        if computed < 0:
            change = -abs(change)
            change_pct = -abs(change_pct)

    pe = stat("pe_ratio")
    dividend = stat("dividend_yield")

    return Quote(
        ticker=ticker,
        exchange=exchange,
        name=_safe(lambda: _name(soup)),
        price=price,
        currency=currency,
        change=change,
        change_pct=change_pct,
        previous_close=previous_close,
        day_range=parse_range(stat("day_range")),
        year_range=parse_range(stat("year_range")),
        market_cap=parse_number(market_cap_text),
        pe_ratio=parse_number(pe),
        dividend_yield=parse_percent(dividend),
        scraped_at=datetime.now(UTC),
        source_url=url,
    )


# ---------------------------------------------------------------- fetching


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status == 429 or status >= 500
    return False


class GoogleFinanceScraper:
    """Fetches quote pages with timeout, retries and a polite delay (AC-002.2)."""

    def __init__(
        self,
        settings: CrawlerSettings | None = None,
        client: httpx.Client | None = None,
        *,
        sleep: Callable[[float], None] = time.sleep,
        max_attempts: int = 3,
    ) -> None:
        self.settings = settings or CrawlerSettings()
        self._owns_client = client is None
        self.client = client or httpx.Client(
            timeout=self.settings.crawler_timeout_seconds,
            follow_redirects=True,
            headers={
                "User-Agent": self.settings.crawler_user_agent,
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "en-US,en;q=0.9",
            },
        )
        self._sleep = sleep
        self._retrying = Retrying(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception(_is_retryable),
            sleep=sleep,
            reraise=True,
        )
        self.fetched = 0
        self.errors: dict[str, str] = {}

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def __enter__(self) -> GoogleFinanceScraper:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _get(self, url: str) -> str:
        response = self.client.get(url, params={"hl": "en"})
        response.raise_for_status()
        return response.text

    def fetch(self, ticker: str, exchange: str) -> Quote:
        """Fetch and parse one quote. Raises httpx errors once retries are exhausted."""
        url = self.settings.quote_url(ticker, exchange)
        html = self._retrying(self._get, url)
        return parse_quote_html(html, ticker, exchange, url)

    def fetch_all(self, pairs: Iterable[tuple[str, str]] | None = None) -> list[Quote]:
        """Fetch every pair; failures are logged and recorded in ``self.errors``, never raised."""
        pairs = list(pairs if pairs is not None else self.settings.ticker_pairs)
        self.fetched = 0
        self.errors = {}
        quotes: list[Quote] = []
        for i, (ticker, exchange) in enumerate(pairs):
            if i and self.settings.crawler_delay_seconds > 0:
                self._sleep(self.settings.crawler_delay_seconds)
            key = f"{ticker}:{exchange}"
            try:
                quote = self.fetch(ticker, exchange)
            except Exception as exc:  # noqa: BLE001 - one bad ticker must not stop the run
                self.errors[key] = f"{type(exc).__name__}: {exc}"
                log.warning("fetch failed for %s: %s", key, self.errors[key])
                continue
            self.fetched += 1
            quotes.append(quote)
            log.info("fetched %s price=%s %s", key, quote.price, quote.currency or "")
        return quotes

    @property
    def failed(self) -> int:
        return len(self.errors)
