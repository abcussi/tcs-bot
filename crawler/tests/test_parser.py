from __future__ import annotations

import pytest
from gfinance_crawler.scraper import (
    detect_currency,
    parse_number,
    parse_percent,
    parse_quote_html,
    parse_range,
)

URL = "https://www.google.com/finance/quote/IBM:NYSE"


def test_parse_full_fixture(ibm_html: str) -> None:
    """AC-002.3 AC-002.7 every Quote field is parsed from the saved fixture."""
    q = parse_quote_html(ibm_html, "IBM", "NYSE", URL)
    assert q.ticker == "IBM" and q.exchange == "NYSE" and q.source_url == URL
    assert q.name == "International Business Machines Corp"
    assert q.price == pytest.approx(187.45)  # main quote, not the MSFT carousel entry
    assert q.currency == "USD"
    assert q.change == pytest.approx(-2.33)
    assert q.change_pct == pytest.approx(-1.23)  # sign from aria-label "Down by"
    assert q.previous_close == pytest.approx(189.78)
    assert q.day_range == "186.10 - 190.02"
    assert q.year_range == "135.87 - 199.18"
    assert q.market_cap == pytest.approx(172.30e9)
    assert q.pe_ratio == pytest.approx(21.34)
    assert q.dividend_yield == pytest.approx(3.56)
    assert q.scraped_at.tzinfo is not None


def test_parse_degraded_fixture_gives_nones(degraded_html: str) -> None:
    """AC-002.3 missing fields => None, no exception."""
    q = parse_quote_html(degraded_html, "ACME", "LON", URL)
    assert q.name == "Acme Holdings Plc"
    for field in (
        "price",
        "currency",
        "change",
        "change_pct",
        "previous_close",
        "day_range",
        "year_range",
        "market_cap",
        "pe_ratio",
        "dividend_yield",
    ):
        assert getattr(q, field) is None, field


@pytest.mark.parametrize(
    "html", ["", "<html", "not html at all \x00", "<div class='YMlKec fxKbKc'>"]
)
def test_parser_never_raises(html: str) -> None:
    """AC-002.3 parser is tolerant of garbage input."""
    q = parse_quote_html(html, "X", "Y", URL)
    assert q.price is None and q.ticker == "X"


def test_css_fallback_without_data_attributes() -> None:
    """AC-002.3 falls back to CSS classes / label text when data-* attributes are absent."""
    html = """
    <title>Foo Corp (FOO) Stock Price &amp; News - Google Finance</title>
    <div class="YMlKec fxKbKc">€1.234,5</div>
    <div class="YMlKec fxKbKc">€12.50</div>
    <table><tr><td>Previous close</td><td>€12.00</td></tr>
           <tr><td>Market cap</td><td>3.1M EUR</td></tr></table>
    """
    q = parse_quote_html(html, "FOO", "ETR", URL)
    assert q.name == "Foo Corp"
    assert q.currency == "EUR"
    assert q.previous_close == pytest.approx(12.0)
    assert q.market_cap == pytest.approx(3.1e6)
    assert q.price is not None


def test_change_computed_from_previous_close_when_missing() -> None:
    """AC-002.3 change / change_pct derived from price and previous close if not rendered."""
    html = """
    <div data-last-price="110" data-currency-code="USD"></div>
    <div class="gyFHrc">
      <div class="mfs7Fc">Previous close</div><div class="P6K39c">$100.00</div>
    </div>
    """
    q = parse_quote_html(html, "X", "Y", URL)
    assert q.change == pytest.approx(10.0)
    assert q.change_pct == pytest.approx(10.0)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("$1,234.56", 1234.56),
        ("2.45T USD", 2.45e12),
        ("1.2B", 1.2e9),
        ("350.5M", 350.5e6),
        ("12K", 12_000.0),
        ("-0.75%", -0.75),
        ("−1.10", -1.10),  # unicode minus
        ("+$2.50", 2.50),
        ("(3.00)", -3.0),
        ("21.34", 21.34),
        (".5", 0.5),
        ("—", None),
        ("-", None),
        ("", None),
        (None, None),
        ("n/a", None),
        ("abc", None),
    ],
)
def test_parse_number(text: str | None, expected: float | None) -> None:
    """AC-002.3 robust number parsing (currency, separators, K/M/B/T, dashes)."""
    result = parse_number(text)
    if expected is None:
        assert result is None
    else:
        assert result == pytest.approx(expected)


def test_parse_percent_and_range_and_currency() -> None:
    """AC-002.3 helper parsers."""
    assert parse_percent("3.56%") == pytest.approx(3.56)
    assert parse_percent("—") is None
    assert parse_range("$135.87 - $199.18") == "135.87 - 199.18"
    assert parse_range("—") is None
    assert detect_currency("$12.00") == "USD"
    assert detect_currency("172.30B USD") == "USD"
    assert detect_currency("£5") == "GBP"
    assert detect_currency("12.00") is None
