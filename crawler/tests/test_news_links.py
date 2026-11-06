"""Spec 005 bronze crawler: canonical URLs, link extraction, TCS relevance (AC-005.2, AC-005.3)."""

from __future__ import annotations

import pytest
from conftest import news_fixture
from gfinance_crawler.news.links import (
    canonical_url,
    extract_links,
    extract_links_json,
    html_title,
    is_tcs_relevant,
    visible_text,
)
from gfinance_crawler.news.queries import (
    COMMUNITY_SUBREDDITS,
    DEFAULT_ALLOWED_DOMAINS,
    TCS_RISK_QUERIES,
    domain_allowed,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("HTTPS://WWW.Example.COM/a/b/", "https://www.example.com/a/b"),
        ("https://example.com/a#frag", "https://example.com/a"),
        (
            "https://example.com/a?utm_source=x&b=2&fbclid=1&a=1&gclid=z&UTM_medium=y",
            "https://example.com/a?a=1&b=2",
        ),
        ("https://example.com:443/x", "https://example.com/x"),
        ("http://example.com:8080/x/", "http://example.com:8080/x"),
        ("https://example.com/", "https://example.com"),
        ("https://example.com", "https://example.com"),
    ],
)
def test_canonical_url(raw, expected):
    """AC-005.2: dedupe key normalisation."""
    assert canonical_url(raw) == expected


def test_canonical_url_is_idempotent_and_tolerant():
    u = "https://Example.com/a/?b=2&a=1#x"
    assert canonical_url(canonical_url(u)) == canonical_url(u)
    assert canonical_url("not a url") == "not a url"
    assert canonical_url("") == ""


def test_extract_links_resolves_dedupes_and_skips_non_http():
    """AC-005.3: (url, anchor) pairs, absolute, deduped on canonical URL."""
    html = news_fixture("article_lawsuit.html")
    links = extract_links(html, "https://economictimes.indiatimes.com/tech/x/articleshow/1001.cms")
    urls = [u for u, _ in links]
    assert all(u.startswith("http") for u in urls)
    assert not any(u.startswith("mailto:") for u in urls)
    assert (
        "https://economictimes.indiatimes.com/tech/tcs-synthetic-lawsuit/articleshow/1001.cms#comments"
        in urls
    )
    anchors = dict(links)
    assert (
        anchors["https://www.financialexpress.com/money/lrs-rules/"]
        == "TCS on LRS remittance rules"
    )
    dup = extract_links('<a href="/a?utm_source=1">x</a><a href="/a">y</a>', "https://h.com/")
    assert dup == [("https://h.com/a?utm_source=1", "x")]
    assert extract_links("", "https://h.com") == []


def test_extract_links_json_finds_url_fields_and_html_anchors():
    links = extract_links_json(news_fixture("hn_item.json"))
    urls = [u for u, _ in links]
    assert urls[0] == "https://www.reuters.com/technology/tcs-synthetic-data-breach-2026-11-01/"
    assert links[0][1] == "SYNTHETIC: TCS data breach discussion"
    assert "https://www.business-standard.com/companies/tcs-synthetic-security-review.html" in urls
    assert extract_links_json("not json") == []


def test_html_title_prefers_og_title():
    assert html_title(news_fixture("article_whistleblower.html")).startswith(
        "SYNTHETIC: Tata Consultancy Services"
    )
    assert html_title("<html><body><h1> Head </h1></body></html>") == "Head"
    assert html_title("<p>no title</p>") is None


@pytest.mark.parametrize(
    "text",
    [
        "TCS reports Q2 results",
        "Tata Consultancy Services faces lawsuit",
        "tata  consultancy services",
        "Statement published on www.tcs.com today",
        "TCS's CEO K Krithivasan said the company will comply",
        "Krithivasan addressed employees",
        "https://economictimes.indiatimes.com/tech/tcs-layoffs/articleshow/1.cms",
        "Shares of NSE: TCS fell",
        "TCS on LRS is explained here, but Tata Consultancy shares also fell",  # strong wins
    ],
)
def test_is_tcs_relevant_positive(text):
    """AC-005.3: references to the company are relevant."""
    assert is_tcs_relevant(text)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "Tax Collected at Source (TCS) on foreign remittance under LRS",
        "TCS on LRS remittance rules",
        "How is TCS refunded in ITR after a foreign remittance?",
        "New tax collected at source rules. The TCS applies to tour packages.",
        "Section 206C: TCS on sale of goods",
        "etcs and tcsh are not words we look for",
        "Infosys and Wipro results",
    ],
)
def test_is_tcs_relevant_negative(text):
    """AC-005.3: Tax-Collected-at-Source contexts and off-topic text are not relevant."""
    assert not is_tcs_relevant(text)


def test_query_set_and_domains_cover_spec():
    """AC-005.1: fixed TCS risk query set; community subreddits; depth-2 allow-list."""
    joined = " ".join(TCS_RISK_QUERIES).lower()
    for word in (
        "ethics",
        "compliance",
        "lawsuit",
        "fine",
        "layoffs",
        "data breach",
        "visa",
        "investigation",
        "controversy",
        "outage",
        "whistleblower",
        "bribery",
        "sebi",
        "esg",
        "trade secrets",
    ):
        assert word in joined, word
    assert all(("TCS" in q or "Tata Consultancy" in q) for q in TCS_RISK_QUERIES)
    assert 18 <= len(TCS_RISK_QUERIES) <= 25
    assert "developersIndia" in COMMUNITY_SUBREDDITS
    assert domain_allowed("economictimes.indiatimes.com", DEFAULT_ALLOWED_DOMAINS)
    assert domain_allowed("www.tcs.com", DEFAULT_ALLOWED_DOMAINS)
    assert domain_allowed("www.livemint.com", DEFAULT_ALLOWED_DOMAINS)
    assert not domain_allowed("example.org", DEFAULT_ALLOWED_DOMAINS)
    assert not domain_allowed("eviltcs.com", DEFAULT_ALLOWED_DOMAINS)


def test_visible_text_json_ignores_keys_and_algolia_highlight_metadata():
    """Real HN Algolia hits echo the query terms in ``_highlightResult.matchedWords``; those
    must not make an unrelated hit look TCS-relevant."""
    raw = news_fixture("hn_search_highlight.json")
    import json

    first = json.dumps(json.loads(raw)["hits"][0])
    assert is_tcs_relevant(first)  # the raw JSON string alone is fooled by matchedWords ["tcs"]
    assert not is_tcs_relevant(visible_text(first, "application/json"))
    second = json.dumps(json.loads(raw)["hits"][1])
    assert is_tcs_relevant(visible_text(second, "application/json"))
    assert visible_text("not json {", "application/json") == "not json {"


def test_is_tcs_relevant_rejects_other_tcs_meanings():
    assert not is_tcs_relevant("Ideas such as Taking Children Seriously (TCS) and unschooling")
    assert not is_tcs_relevant("Unconditional Parenting and TCS are discussed in the thread")
    assert is_tcs_relevant("TCS reported a data breach; the T&Cs were updated later.")
