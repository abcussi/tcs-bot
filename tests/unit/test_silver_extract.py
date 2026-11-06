"""Silver main-text extraction per content type (AC-005.5). Fixtures are synthetic."""

from __future__ import annotations

import json
from pathlib import Path

from tcs_bot.medallion.silver.extract import extract_main_text, strip_html

FIXTURES = Path(__file__).parents[1] / "fixtures" / "silver"


def _read(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def test_html_uses_main_text_and_og_title():
    """AC-005.5: trafilatura main text; nav/script/footer boilerplate dropped; og:title."""
    text, title = extract_main_text(_read("article.html"), "text/html; charset=utf-8", "https://x")
    assert title == "TCS client data exposed in fictional breach scenario"
    assert "Northwind Mutual Bank" in text
    assert "data breach" in text
    for junk in ("window.tracking", "font-family", "Subscribe to our newsletter", "gadgets"):
        assert junk not in text


def test_html_bs4_fallback_for_tiny_pages():
    """AC-005.5: bs4 fallback (script/style/nav/footer removed) when trafilatura finds nothing."""
    raw = (
        "<html><head><title>Short page</title><script>var a=1;</script></head><body>"
        "<nav>Menu</nav><div>TCS update: fictional short note.</div><footer>Footer</footer>"
        "</body></html>"
    )
    text, title = extract_main_text(raw, "text/html", "https://x")
    assert title == "Short page"
    assert "TCS update" in text
    assert "var a" not in text and "Menu" not in text and "Footer" not in text


def test_reddit_thread_json_post_and_comments_without_authors():
    """AC-005.5: Reddit thread JSON -> title + selftext + comment bodies (no usernames)."""
    text, title = extract_main_text(_read("reddit_thread.json"), "application/json", "https://r")
    assert title and title.startswith("TCS bench policy")
    assert "asked them to resign" in text
    assert "work-life balance" in text  # top-level comment
    assert "Forced resignation is illegal" in text  # nested reply
    assert "fictional_user" not in text
    assert "[deleted]" not in text


def test_reddit_listing_item():
    item = {
        "kind": "t3",
        "data": {"title": "TCS question", "selftext": "Fictional body.", "author": "someone"},
    }
    text, title = extract_main_text(json.dumps(item), "application/json", "https://r")
    assert title == "TCS question"
    assert "Fictional body." in text and "someone" not in text


def test_hn_thread_recursive_comments_strip_html():
    """AC-005.5: HN Algolia item -> title + recursive comment text, HTML stripped."""
    text, title = extract_main_text(_read("hn_thread.json"), "application/json", "https://hn")
    assert title and title.startswith("TCS outage")
    assert "nobody had a runbook" in text
    assert "&amp;" not in text and "<p>" not in text
    assert "Contract termination is likely" in text  # grandchild
    assert "fictional_commenter" not in text


def test_hn_search_hit_story_text():
    hit = {"objectID": "1", "title": "Ask HN: TCS", "story_text": "<p>Fictional <i>text</i></p>"}
    text, title = extract_main_text(json.dumps(hit), "application/json", "https://hn")
    assert title == "Ask HN: TCS"
    assert "Fictional text" in text


def test_gdelt_item_title_only():
    text, title = extract_main_text(_read("gdelt_item.json"), "application/json", "https://g")
    assert title and "H-1B visa fraud" in title
    assert text == title


def test_rss_item_title_and_description_stripped():
    """AC-005.5: RSS item -> title + description with the HTML stripped."""
    text, title = extract_main_text(_read("rss_item.xml"), "application/rss+xml", "https://n")
    assert title == "TCS faces fictional lawsuit over trade secrets - Example Daily"
    assert "trade secrets lawsuit" in text
    assert "<a" not in text and "&lt;" not in text and "font" not in text


def test_plain_text_and_markdown_passthrough_normalized():
    raw = "Line one​  with   spaces\r\n\r\n\r\n\r\n# Heading"
    for ct in ("text/plain", "text/markdown"):
        text, title = extract_main_text(raw, ct, "file://x")
        assert title is None
        assert text == "Line one with spaces\n\n# Heading"


def test_sniffs_unknown_content_type_and_handles_garbage():
    assert extract_main_text('{"title": "T", "description": "D"}', "", "u")[0] == "T\n\nD"
    assert extract_main_text("", "text/html", "u") == ("", None)
    text, _ = extract_main_text("{not json", "application/json", "u")
    assert text == "{not json"


def test_strip_html():
    assert strip_html("a &amp; <b>b</b>").strip() == "a & b"
    assert strip_html(None) == ""
