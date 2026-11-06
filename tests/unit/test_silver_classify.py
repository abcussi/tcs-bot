"""Rules classifier + TCS relevance (AC-005.5 / AC-005.6). Fixtures are synthetic."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tcs_bot.medallion.models import RISK_CATEGORIES
from tcs_bot.medallion.silver.classify import (
    MAX_EVIDENCE_CHARS,
    SEVERITIES,
    TAXONOMY,
    classify,
    combine_points,
    split_sentences,
    tcs_relevance,
)
from tcs_bot.medallion.silver.extract import extract_main_text

FIXTURES = Path(__file__).parents[1] / "fixtures" / "silver"
CASES = json.loads((FIXTURES / "category_cases.json").read_text(encoding="utf-8"))["cases"]
THRESHOLD = 0.35  # settings.medallion_relevance_threshold default


def _sev(signals, category):
    return max(
        (SEVERITIES.index(s.severity) for s in signals if s.category == category), default=-1
    )


def test_taxonomy_covers_every_category():
    assert {r.category for r in TAXONOMY} == set(RISK_CATEGORIES)
    assert {c["category"] for c in CASES} == set(RISK_CATEGORIES)


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_category_detected_with_expected_severity(case):
    """AC-005.5: each category fires with at least the expected severity + evidence sentence."""
    signals, score = classify(case["text"], case["title"])
    assert _sev(signals, case["category"]) >= SEVERITIES.index(case["min_severity"])
    assert 0 < score <= 100
    for s in signals:
        assert s.evidence and len(s.evidence) <= MAX_EVIDENCE_CHARS
        assert s.evidence in case["title"] or s.evidence in case["text"]


def test_severity_examples():
    signals, _ = classify("TCS was hit by ransomware.", "")
    assert signals[0].category == "data_privacy_cyber" and signals[0].severity == "critical"
    signals, _ = classify("The SEC fined TCS for record keeping lapses.", "")
    assert _sev(signals, "compliance_regulatory") == SEVERITIES.index("high")
    signals, _ = classify("Boycott calls grew against TCS.", "")
    assert _sev(signals, "reputational") == SEVERITIES.index("medium")


def test_negation_lowers_severity_one_level():
    """AC-005.5: 'denied' / 'dismissed the lawsuit' / 'no evidence of' lower severity."""
    plain, _ = classify("TCS was accused of fraud.", "")
    denied, _ = classify("TCS denied the fraud allegations.", "")
    assert _sev(plain, "ethics") == SEVERITIES.index("critical")
    assert _sev(denied, "ethics") == SEVERITIES.index("high")
    dismissed, _ = classify("The court dismissed the lawsuit against TCS.", "")
    assert _sev(dismissed, "legal_litigation") == SEVERITIES.index("medium")
    none, _ = classify("Auditors found no evidence of bribery at TCS.", "")
    assert _sev(none, "ethics") == SEVERITIES.index("high")
    # "dismissed 200 employees" is a labour event, not a negation
    fired, _ = classify(
        "TCS dismissed 200 employees in a lawsuit-free restructuring of layoffs.", ""
    )
    assert _sev(fired, "labor_hr") == SEVERITIES.index("high")


def test_signals_deduped_by_category_and_keyword_and_prefer_tcs_sentence():
    text = "Another firm faced a data breach. TCS also faced a data breach. A data breach again."
    signals, _ = classify(text, "")
    breaches = [s for s in signals if s.keyword == "data breach"]
    assert len(breaches) == 1
    assert "TCS" in breaches[0].evidence


def test_title_hits_boost_score_and_tcs_sentences_weigh_more():
    body = "The company is in a lawsuit."
    _, plain = classify(body, "Quarterly update")
    _, titled = classify(body, "TCS lawsuit filed")
    assert titled > plain
    _, other = classify("A rival firm faces a lawsuit.", "")
    _, tcs = classify("TCS faces a lawsuit.", "")
    assert tcs > other


def test_score_diminishing_returns_and_bounds():
    assert combine_points([]) == 0
    assert combine_points([40]) == 40
    assert 40 < combine_points([40, 40]) < 80
    assert combine_points([95] * 20) <= 100
    text = " ".join(c["text"] for c in CASES)
    signals, score = classify(text, "TCS")
    assert len({s.category for s in signals}) == len(RISK_CATEGORIES)
    assert 90 < score <= 100


def test_no_signals_for_neutral_text():
    signals, score = classify("TCS opened a new office and hired graduates.", "TCS update")
    assert signals == [] and score == 0


def test_evidence_is_capped():
    long = "TCS faced a lawsuit " + "with many words " * 60 + "."
    signals, _ = classify(long, "")
    assert all(len(s.evidence) <= MAX_EVIDENCE_CHARS for s in signals)


def test_split_sentences():
    assert split_sentences("One. Two! Three?\nFour") == ["One.", "Two!", "Three?", "Four"]
    assert split_sentences("Paid $4.5 million. Then more.") == ["Paid $4.5 million.", "Then more."]


# ------------------------------------------------------------------ relevance
def test_relevance_high_for_tcs_articles():
    """AC-005.6: TCS-centred documents clear the relevance threshold."""
    text, title = extract_main_text(
        (FIXTURES / "article.html").read_text(encoding="utf-8"), "text/html", "u"
    )
    assert tcs_relevance(text, title or "") > 0.7
    for case in CASES:
        assert tcs_relevance(case["text"], case["title"]) >= THRESHOLD


def test_relevance_tax_collected_at_source_false_positive():
    """AC-005.6: 'TCS' as Tax Collected at Source is not about the company."""
    raw = (FIXTURES / "tax_collected_at_source.html").read_text(encoding="utf-8")
    text, title = extract_main_text(raw, "text/html", "u")
    assert "TCS" in text
    assert tcs_relevance(text, title or "") < 0.15


def test_relevance_tata_motors_only_article():
    """AC-005.6: other Tata companies (with the Tata Sons chairman) are not TCS."""
    raw = (FIXTURES / "tata_motors.html").read_text(encoding="utf-8")
    text, title = extract_main_text(raw, "text/html", "u")
    assert tcs_relevance(text, title or "") < 0.1


def test_relevance_zero_without_mentions_and_low_for_passing_mention():
    assert tcs_relevance("Infosys and Wipro reported results.", "IT results") == 0
    passing = "An article about outsourcing. TCS was mentioned once. " + "filler " * 300
    assert tcs_relevance(passing, "") < THRESHOLD
    assert tcs_relevance("Fictional note.", "TCS wins a fictional award") >= THRESHOLD


def test_relevance_tcs_with_tax_words_but_explicit_company_kept():
    text = (
        "Tata Consultancy Services said its CEO K Krithivasan expects deals to grow. "
        "TCS also noted that remittances from overseas staff were unaffected."
    )
    assert tcs_relevance(text, "TCS outlook") >= THRESHOLD
