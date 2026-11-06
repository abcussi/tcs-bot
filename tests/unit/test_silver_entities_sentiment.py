"""Silver entity extraction and lexicon sentiment (AC-005.5). Text is synthetic."""

from __future__ import annotations

from tcs_bot.medallion.models import Entities
from tcs_bot.medallion.silver.entities import MAX_PER_TYPE, extract_entities, merge_entities
from tcs_bot.medallion.silver.sentiment import sentiment

TEXT = (
    "SYNTHETIC FIXTURE. Tata Consultancy Services Ltd said CEO K Krithivasan met SEBI officials "
    "in Mumbai. The US Department of Labor and the Enforcement Directorate (ED) were cited. "
    "A penalty of ₹1,200 crore and $45 million was discussed, plus Rs 5 lakh. Epic Systems "
    "sued TCS in Wisconsin, U.S. Mr. Rahul Sharma of Acme Widgets Inc. said Infosys and Wipro "
    "were not involved. Chairman N Chandrasekaran spoke in London, UK."
)


def test_regulators_canonical():
    """AC-005.5: regulators gazetteer with canonical names."""
    ents = extract_entities(TEXT + " The Securities and Exchange Commission and GDPR apply.")
    for r in ("SEBI", "US Department of Labor", "Enforcement Directorate", "SEC", "GDPR"):
        assert r in ents.regulators
    assert extract_entities("The sec ed of the report").regulators == []


def test_money_amounts():
    ents = extract_entities(TEXT + " It paid USD 2.5 billion and 300 crore rupees.")
    for m in ("₹1,200 crore", "$45 million", "Rs 5 lakh", "USD 2.5 billion", "300 crore rupees"):
        assert m in ents.money
    assert extract_entities("200 million people use phones").money == []


def test_places_with_aliases():
    ents = extract_entities(TEXT + " Offices in Bangalore.")
    assert {"Mumbai", "Wisconsin", "United States", "London", "United Kingdom", "Bengaluru"} <= set(
        ents.places
    )
    assert ents.places.count("United States") == 1


def test_organizations_known_and_suffix_pattern():
    ents = extract_entities(TEXT)
    assert ents.organizations[0] == "Tata Consultancy Services"
    assert ents.organizations.count("Tata Consultancy Services") == 1  # Ltd + TCS deduped
    for o in ("Epic Systems", "Acme Widgets Inc", "Infosys", "Wipro"):
        assert o in ents.organizations


def test_people_roles_honorifics_known_execs():
    ents = extract_entities(TEXT)
    assert {"K Krithivasan", "N Chandrasekaran", "Rahul Sharma"} <= set(ents.people)


def test_caps_and_merge():
    many = " ".join(f"Company{i} Holdings bought it." for i in range(40))
    assert len(extract_entities(many).organizations) == MAX_PER_TYPE
    merged = merge_entities(Entities(people=["A B"]), Entities(people=["a b", "C D"]))
    assert merged.people == ["A B", "C D"]


def test_sentiment_bounds_polarity_and_negation():
    """AC-005.5: sentiment in [-1, 1], negation flips."""
    pos = sentiment("TCS reported strong growth and record profit.")
    neg = sentiment("TCS faces fraud lawsuit and layoffs; shares plunged.")
    assert 0 < pos <= 1 and -1 <= neg < 0
    assert sentiment("Regulators found fraud.") < 0 < sentiment("Regulators did not find fraud.")
    assert sentiment("") == 0 and sentiment("The office is on the third floor.") == 0
    long_neg = "TCS lawsuit. " + "Neutral words here. " * 400
    assert -1 < sentiment(long_neg) < 0
