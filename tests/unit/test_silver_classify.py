"""Rules classifier + TCS relevance (AC-005.5 / AC-005.6). Fixtures are synthetic."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tcs_bot.medallion.models import RISK_CATEGORIES
from tcs_bot.medallion.silver.classify import (
    FAR_CAP,
    MAX_EVIDENCE_CHARS,
    SEVERITIES,
    TAXONOMY,
    classify,
    combine_points,
    length_damping,
    split_sentences,
    tcs_relevance,
    title_names_tcs,
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


def _hn_text(name: str) -> tuple[str, str]:
    text, title = extract_main_text(
        (FIXTURES / name).read_text(encoding="utf-8"), "application/json", "u"
    )
    return text, title or ""


def test_relevance_incidental_hn_comment_hit_below_threshold():
    """AC-005.6: one passing TCS aside in a short comment is not 'about TCS'."""
    text, title = _hn_text("hn_comment_hit_incidental.json")
    assert "TCS" in text and not title_names_tcs(title)
    assert tcs_relevance(text, title) < THRESHOLD
    # the very same text under a headline naming TCS is relevant
    assert tcs_relevance(text, "TCS vendor dispute") >= THRESHOLD


def test_relevance_tata_group_headline_with_tcs_only_in_body():
    """AC-005.6: a Tata group / chairman headline with TCS asides is not a TCS document."""
    text, title = _hn_text("hn_group_thread_tata_title.json")
    assert text.count("TCS") >= 3 and not title_names_tcs(title)
    assert tcs_relevance(text, title) < THRESHOLD
    assert tcs_relevance(text, "TCS and the Tata group") >= THRESHOLD


def test_relevance_counts_company_name_and_acronym_pair_once():
    """AC-005.6: 'Tata Consultancy Services (TCS)' is one mention, not two."""
    filler = "Invented sentence about something unrelated to any company at all. " * 8
    once = tcs_relevance(f"A note on TCS. {filler}", "Industry note")
    pair = tcs_relevance(f"A note on Tata Consultancy Services (TCS). {filler}", "Industry note")
    twice = tcs_relevance(f"A note on TCS and Tata Consultancy Services. {filler}", "Industry note")
    assert abs(pair - once) < 0.01 and twice > pair + 0.1


def test_title_names_tcs():
    assert title_names_tcs("TCS tops estimates")
    assert title_names_tcs("Tata Consultancy Services wins deal")
    assert not title_names_tcs("Tata chairman retires")
    assert not title_names_tcs("Tax Collected at Source rules")
    assert not title_names_tcs("")


# ------------------------------------------------------------------ risk score (focus + length)
def _generic_risk_sentences(n: int) -> str:
    """Generic risk talk (many categories, no TCS) - pads a thread far from the TCS mention."""
    base = [
        "A rival firm was sued over visa fraud and discrimination.",
        "Regulators opened an investigation and imposed a hefty fine.",
        "There was a data breach, a ransomware outage and a boycott.",
        "Auditors resigned after a bribery scandal and an antitrust probe.",
        "Workers faced layoffs, unpaid overtime and a forced resignation.",
        "Shares fell, margins declined and tariffs and sanctions followed.",
    ]
    return " ".join(base[i % len(base)] for i in range(n))


def test_far_evidence_weighs_less_than_evidence_in_the_tcs_sentence():
    """AC-005.5: the same generic sentence scores less when it is far from any TCS mention."""
    near = "TCS faces a lawsuit."
    far = (
        "TCS opened an office. "
        + "Nothing else happened that week. " * 8
        + "A firm faces a lawsuit."
    )
    _, near_score = classify(near, "")
    _, far_score = classify(far, "")
    assert far_score < near_score / 2
    signals, _ = classify(far, "")
    assert [s.keyword for s in signals] == ["lawsuit"]  # still reported as a signal


def test_adjacent_sentence_still_counts_for_short_tcs_items():
    """AC-005.5: evidence right next to the TCS mention (short snippets) keeps its weight."""
    title = "TCS vendor update"
    body = "The retailer paused orders. A cyberattack disrupted them for weeks."
    _, adjacent = classify(body, title)
    far_body = "The retailer paused orders. " + "Quiet week followed. " * 6 + "A cyberattack hit."
    _, distant = classify(far_body, title)
    assert adjacent >= 20 > distant + 5


def test_far_evidence_total_is_capped():
    """AC-005.5: any amount of distant generic evidence adds at most FAR_CAP points."""
    text = (
        "TCS opened an office. " + "Quiet filler sentence here. " * 6 + _generic_risk_sentences(40)
    )
    _, score = classify(text, "")
    assert 0 < score <= FAR_CAP


def test_long_thread_scores_below_focused_tcs_article():
    """AC-005.5: a long generic thread (TCS in the title) < a short focused TCS lawsuit article."""
    thread = (
        "TCS was mentioned. " + "Quiet filler sentence here. " * 4 + _generic_risk_sentences(300)
    )
    _, long_score = classify(thread, "TCS megathread")
    _, longer_score = classify(thread + " " + _generic_risk_sentences(300), "TCS megathread")
    focused = (
        "A jury found that TCS misappropriated trade secrets in the lawsuit. "
        "TCS must pay damages and plans to appeal the verdict."
    )
    _, focused_score = classify(focused, "TCS loses trade secrets lawsuit")
    assert long_score < 25 and longer_score <= long_score
    assert focused_score > long_score + 25


def test_length_damping():
    assert length_damping(50) == length_damping(300) == 1.0
    assert 1.0 > length_damping(600) > length_damping(6000) > 0.4
    signals, score = classify("TCS faces a lawsuit. " + "Filler words here today. " * 400, "")
    assert [x.keyword for x in signals] == ["lawsuit"]  # detected; only its weight is damped
    _, short = classify("TCS faces a lawsuit.", "")
    assert score < short
