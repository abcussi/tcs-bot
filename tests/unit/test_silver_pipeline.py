"""Silver pipeline with a fake bronze repo + fake store (AC-005.5 / AC-005.6). Synthetic data."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from tcs_bot.medallion.models import BronzeDocument, SilverRecord
from tcs_bot.medallion.silver import pipeline
from tcs_bot.medallion.silver.pipeline import (
    attribute_risk,
    build_silver_record,
    detect_language,
    domain_of,
    is_relevant,
    run_silver,
    summarize,
)

FIXTURES = Path(__file__).parents[1] / "fixtures" / "silver"
FILES = {
    "article.html": "text/html; charset=utf-8",
    "tax_collected_at_source.html": "text/html",
    "tata_motors.html": "text/html",
    "reddit_thread.json": "application/json",
    "hn_thread.json": "application/json",
    "rss_item.xml": "application/rss+xml",
    "gdelt_item.json": "application/json",
    "hn_comment_hit_incidental.json": "application/json",
    "hn_story_hit_tcs_title.json": "application/json",
    "hn_story_hit_tcs_title_neutral.json": "application/json",
    "hn_long_generic_thread.json": "application/json",
    "hn_group_thread_tata_title.json": "application/json",
}


def _doc(doc_id: int, name: str, **kw) -> BronzeDocument:
    data = {
        "id": doc_id,
        "url": f"https://www.example.com/{name}",
        "canonical_url": f"https://example.com/{name}",
        "depth": 1,
        "source": "web",
        "raw_content": (FIXTURES / name).read_text(encoding="utf-8"),
        "content_type": FILES[name],
    }
    data.update(kw)
    return BronzeDocument(**data)


class FakeRepo:
    def __init__(self, docs: list[BronzeDocument]) -> None:
        self.docs = docs
        self.marked: list[int] = []
        self.calls: list[tuple] = []

    def iter_unprocessed(self, layer: str = "silver", limit: int | None = None) -> Iterator:
        self.calls.append((layer, limit))
        pending = [d for d in self.docs if d.id not in self.marked]
        yield from pending[:limit] if limit else pending

    def mark_processed(self, ids: list[int], layer: str = "silver") -> None:
        assert layer == "silver"
        self.marked.extend(ids)


class FakeStore:
    def __init__(self, fail_ids: set[int] | None = None, down: bool = False) -> None:
        self.records: dict[int, SilverRecord] = {}
        self.fail_ids = fail_ids or set()
        self.down = down
        self.ensured = 0
        self.batches: list[int] = []
        self.last_failed_ids: set[int] = set()

    def ensure_index(self) -> bool:
        self.ensured += 1
        return True

    def upsert(self, records, refresh: bool = False) -> int:
        records = list(records)
        self.batches.append(len(records))
        if self.down:
            self.last_failed_ids = {r.doc_id for r in records}
            return 0
        self.last_failed_ids = {r.doc_id for r in records if r.doc_id in self.fail_ids}
        for r in records:
            if r.doc_id not in self.last_failed_ids:
                self.records[r.doc_id] = r
        return len(records) - len(self.last_failed_ids)


def test_build_record_article(settings):
    """AC-005.5: extraction, entities, relevance, signals, sentiment, score, summary."""
    rec = build_silver_record(_doc(1, "article.html", title=None, query="TCS data breach"))
    assert rec.doc_id == 1 and rec.domain == "example.com" and rec.query == "TCS data breach"
    assert rec.title == "TCS client data exposed in fictional breach scenario"
    assert rec.relevant and rec.tcs_relevance >= settings.medallion_relevance_threshold
    assert "data_privacy_cyber" in rec.risk_categories
    assert rec.risk_score > 0 and rec.sentiment < 0
    assert rec.extractor == "rules-v1" and rec.language == "en"
    assert rec.word_count >= 30
    assert "Tata Consultancy Services" in rec.entities.organizations
    assert "Pune" in rec.entities.places
    assert rec.summary and len(rec.summary) <= 600
    assert "Subscribe" not in rec.summary


def test_bronze_title_preferred_over_extracted():
    rec = build_silver_record(_doc(2, "article.html", title="Listing title"))
    assert rec.title == "Listing title"


def test_false_positives_marked_irrelevant():
    """AC-005.6: tax 'TCS' and Tata-Motors-only stories are kept but relevant=false."""
    for name in ("tax_collected_at_source.html", "tata_motors.html"):
        rec = build_silver_record(_doc(3, name))
        assert rec.relevant is False, name
        assert rec.clean_text  # kept for audit


def test_title_naming_tcs_is_relevant_without_minimum_words():
    """AC-005.6: a headline-only TCS item (< 30 words) counts when the TITLE names TCS."""
    rec = build_silver_record(_doc(4, "gdelt_item.json"))
    assert rec.tcs_relevance >= 0.35
    assert rec.word_count < 30 and rec.relevant is True
    assert "geopolitical_immigration" in rec.risk_categories


@pytest.mark.parametrize(
    "name, category",
    [
        ("hn_story_hit_tcs_title.json", "labor_hr"),  # "TCS announces ... layoffs ..."
        ("hn_story_hit_tcs_title_neutral.json", None),  # "TCS tops ... estimates ..."
    ],
)
def test_title_only_hn_story_hits_are_relevant(name, category):
    """AC-005.6: Algolia story hits are title-only (< 30 words) yet strongly about TCS."""
    rec = build_silver_record(_doc(8, name, source="hackernews", depth=0))
    assert rec.word_count < 30
    assert rec.tcs_relevance >= 0.35
    assert rec.relevant is True
    if category:
        assert category in rec.risk_categories


def test_is_relevant_title_rule():
    """AC-005.6: short docs need TCS in the title; long docs and low relevance unchanged."""
    assert is_relevant(0.8, 11, "TCS layoffs herald AI shakeup", 0.35) is True
    assert is_relevant(0.8, 11, "Outsourcing layoffs herald AI shakeup", 0.35) is False
    assert is_relevant(0.2, 11, "TCS layoffs", 0.35) is False  # below threshold
    assert is_relevant(0.5, 400, "A thread about jobs", 0.35) is True
    assert is_relevant(0.3, 400, "TCS layoffs", 0.35) is False


def test_short_body_only_tcs_mention_is_not_relevant():
    """AC-005.6: < 30 words and TCS only in the body (title silent) stays irrelevant."""
    raw = json.dumps({"title": "Weekend plans", "text": "SYNTHETIC FIXTURE: TCS TCS TCS TCS TCS."})
    rec = build_silver_record(_doc(9, "hn_thread.json", title="Weekend plans", raw_content=raw))
    assert rec.word_count < 30 and rec.tcs_relevance >= 0.35
    assert rec.relevant is False


def test_incidental_hn_comment_hit_is_irrelevant_and_demoted():
    """AC-005.6: HN comment hit that names TCS once in passing is not a TCS risk document."""
    rec = build_silver_record(_doc(11, "hn_comment_hit_incidental.json", source="hackernews"))
    assert rec.word_count >= 30
    assert rec.tcs_relevance < 0.35 and rec.relevant is False
    assert rec.risk_score < 25  # layoff words in the story are not TCS risk


def test_tata_group_headline_with_tcs_in_body_is_irrelevant():
    """AC-005.6: 'chairman of Tata group retires' thread with a few TCS asides is a group story."""
    rec = build_silver_record(_doc(12, "hn_group_thread_tata_title.json", source="hackernews"))
    assert rec.tcs_relevance < 0.35 and rec.relevant is False


def _inflated_thread(extra_comments: int) -> str:
    thread = json.loads((FIXTURES / "hn_long_generic_thread.json").read_text(encoding="utf-8"))
    filler = "Invented filler remark about commuting, coffee, keyboards and weekend plans. " * 4
    thread["children"] += [
        {"id": 2000 + i, "type": "comment", "author": "f", "text": f"{filler}{i}.", "children": []}
        for i in range(extra_comments)
    ]
    return json.dumps(thread)


def test_long_generic_thread_does_not_inflate_risk_with_length():
    """AC-005.5/.6: many generic risk words far from the only TCS aside are not a TCS risk."""
    scores = []
    for extra in (0, 15, 35):
        rec = build_silver_record(
            _doc(13, "hn_long_generic_thread.json", raw_content=_inflated_thread(extra))
        )
        assert rec.relevant is False
        assert len(rec.risk_categories) >= 6  # the generic words are still detected
        scores.append(rec.risk_score)
    assert scores[0] < 40 and scores == sorted(scores, reverse=True)  # saturates, never grows
    focused = build_silver_record(
        _bing_doc(json.loads((FIXTURES / "bing_news_snippets.json").read_text("utf-8"))["cases"][1])
    )
    assert focused.relevant and focused.risk_score > scores[0] + 20


def _bing_doc(case: dict) -> BronzeDocument:
    raw = (
        f"<rss><channel><item><title>{case['title']}</title>"
        f"<description>{case['description']}</description></item></channel></rss>"
    )
    return _doc(20, "rss_item.xml", source="bing_news", depth=0, raw_content=raw)


BING_CASES = json.loads((FIXTURES / "bing_news_snippets.json").read_text("utf-8"))["cases"]


@pytest.mark.parametrize("case", BING_CASES, ids=[c["id"] for c in BING_CASES])
def test_bing_style_tcs_snippets_stay_relevant_with_sensible_category(case):
    """AC-005.6: focused Bing-News snippets about TCS remain relevant, score and categorised."""
    rec = build_silver_record(_bing_doc(case))
    assert rec.relevant is True and rec.tcs_relevance >= 0.8
    assert case["category"] in rec.risk_categories
    assert rec.risk_score >= 40


def test_attribute_risk_scales_only_below_threshold():
    """AC-005.5: off-topic docs (relevance < threshold) get a proportionally lower risk score."""
    assert attribute_risk(80.0, 0.5, 0.35) == 80.0
    assert attribute_risk(80.0, 0.35, 0.35) == 80.0
    assert attribute_risk(80.0, 0.175, 0.35) == 40.0
    assert attribute_risk(80.0, 0.0, 0.35) == 0.0


def test_threshold_from_settings(monkeypatch):
    from tcs_bot.config import get_settings

    monkeypatch.setenv("MEDALLION_RELEVANCE_THRESHOLD", "0.99")
    get_settings.cache_clear()
    assert build_silver_record(_doc(5, "article.html")).relevant is False


def test_community_sources():
    reddit = build_silver_record(_doc(6, "reddit_thread.json", source="reddit"))
    assert "labor_hr" in reddit.risk_categories
    hn = build_silver_record(_doc(7, "hn_thread.json", source="hackernews"))
    assert "operational_delivery" in hn.risk_categories


def test_run_silver_batches_and_marks_only_indexed(monkeypatch):
    """AC-005.5: batches of 50; bad doc skipped; mark processed only after ES upsert."""
    monkeypatch.setattr(pipeline, "BATCH_SIZE", 3)
    names = list(FILES)
    docs = [_doc(i + 1, names[i % len(names)]) for i in range(8)]
    bad = _doc(99, "article.html")
    docs.insert(4, bad)
    real_build = pipeline.build_silver_record

    def build(doc, settings=None):
        if doc.id == 99:
            raise ValueError("boom")
        return real_build(doc, settings)

    monkeypatch.setattr(pipeline, "build_silver_record", build)
    repo, store = FakeRepo(docs), FakeStore(fail_ids={2})
    stats = run_silver(repo=repo, store=store)
    assert store.ensured == 1
    assert store.batches == [3, 3, 2]
    assert stats["processed"] == 8
    assert stats["relevant"] + stats["irrelevant"] == 8
    assert stats["indexed"] == 7
    assert stats["failed"] == 2  # doc 99 (build) + doc 2 (index)
    assert sorted(repo.marked) == [1, 3, 4, 5, 6, 7, 8]
    assert repo.calls == [("silver", None)]


def test_run_silver_es_down_marks_nothing():
    repo = FakeRepo([_doc(1, "article.html"), _doc(2, "rss_item.xml")])
    stats = run_silver(limit=5, repo=repo, store=FakeStore(down=True))
    assert stats["indexed"] == 0 and stats["failed"] == 2
    assert repo.marked == []
    assert repo.calls == [("silver", 5)]


def test_run_silver_empty():
    stats = run_silver(repo=FakeRepo([]), store=FakeStore())
    assert stats == {"processed": 0, "relevant": 0, "irrelevant": 0, "indexed": 0, "failed": 0}


def test_helpers():
    assert domain_of("https://WWW.Example.COM:443/a?b") == "example.com"
    assert detect_language("") == "und"
    assert detect_language("The company said that it was in the process of hiring.") == "en"
    assert detect_language("यह एक परीक्षण वाक्य है जो हिंदी में लिखा गया है") == "hi"
    text = "Short. " + "This sentence has more than eight words in it for sure. " * 5
    s = summarize(text, "")
    assert s.count("This sentence") == 3 and "Short." not in s


@pytest.mark.parametrize("name", list(FILES))
def test_every_fixture_builds(name):
    rec = build_silver_record(_doc(10, name))
    assert 0 <= rec.tcs_relevance <= 1 and 0 <= rec.risk_score <= 100
    assert -1 <= rec.sentiment <= 1
