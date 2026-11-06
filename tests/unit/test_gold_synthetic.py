"""SYNTHETIC gold test corpus (spec 005) + the in-memory silver store used by the gold tests.

Every record is fabricated: titles start with ``[SYNTHETIC]``, URLs live under the reserved
``.example`` TLD and ``extractor`` is ``synthetic-test-fixture``. Nothing here describes a real
event. Three clearly separated topics, 10 documents each (30 total).
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from datetime import UTC, datetime

from tcs_bot.medallion.models import RiskSignal, SilverRecord

SYNTHETIC_EXTRACTOR = "synthetic-test-fixture"

TOPICS: dict[str, dict] = {
    "cyber": {
        "categories": ["data_privacy_cyber", "operational_delivery"],
        "risk": (70, 95),
        "titles": [
            "ransomware attack encrypts client servers",
            "customer data breach exposes personal records",
            "leaked credentials found on hacker forum",
            "phishing campaign compromises admin accounts",
        ],
        "sentences": [
            "A ransomware attack encrypted servers holding client data.",
            "Attackers exfiltrated personal records and passwords from a database.",
            "The breach exposed customer credentials and payment card details.",
            "Security researchers found leaked credentials on a hacker forum.",
            "Malware spread through an unpatched VPN appliance.",
            "Incident responders restored backups after the encryption.",
            "Regulators asked for a data protection impact assessment after the breach.",
            "Phishing emails compromised administrator accounts and firewall settings.",
            "Forensic investigators traced the intrusion to stolen passwords.",
            "Clients were notified that personal data had been exfiltrated.",
        ],
    },
    "labour": {
        "categories": ["labor_hr", "reputational"],
        "risk": (35, 60),
        "titles": [
            "employees protest delayed onboarding of graduates",
            "staff report forced resignations and salary cuts",
            "union complains about bench policy and layoffs",
            "workers criticise return to office mandate",
        ],
        "sentences": [
            "Employees said the appraisal cycle was delayed and bonuses were cut.",
            "Graduates complained that onboarding letters were postponed for months.",
            "A union alleged forced resignations of senior staff on the bench.",
            "Workers criticised the return to office mandate and long commutes.",
            "Staff described salary hikes far below inflation and high attrition.",
            "Former employees reported layoffs announced with little notice.",
            "The labour ministry received complaints about unpaid overtime.",
            "Managers pressured engineers to resign instead of offering severance.",
            "Attrition rose as employees quit over variable pay and appraisals.",
            "Freshers waited a year for joining dates after campus hiring.",
        ],
    },
    "legal": {
        "categories": ["legal_litigation", "ethics"],
        "risk": (55, 85),
        "titles": [
            "jury verdict in trade secrets lawsuit",
            "court upholds damages award in software theft case",
            "appeal filed against misappropriation ruling",
            "plaintiff seeks injunction over stolen source code",
        ],
        "sentences": [
            "A federal jury found misappropriation of trade secrets and awarded damages.",
            "The court upheld the verdict and the punitive damages award on appeal.",
            "The plaintiff alleged that confidential source code was copied without licence.",
            "Lawyers filed an appeal against the judgment in the circuit court.",
            "The judge denied a motion for a new trial in the lawsuit.",
            "The complaint sought an injunction and compensatory damages.",
            "Testimony at trial described downloads of proprietary software documentation.",
            "The appellate panel reduced punitive damages but affirmed liability.",
            "Settlement talks failed before the jury returned its verdict.",
            "The litigation concerned a licence agreement for insurance software.",
        ],
    },
}
TOPIC_NAMES = list(TOPICS)


def synthetic_records(per_topic: int = 10, seed: int = 7) -> list[SilverRecord]:
    """``3 * per_topic`` SYNTHETIC silver records; ``doc_id`` order interleaves topics."""
    rng = random.Random(seed)
    out: list[SilverRecord] = []
    doc_id = 1000
    for i in range(per_topic):
        for t_idx, name in enumerate(TOPIC_NAMES):
            topic = TOPICS[name]
            doc_id += 1
            sentences = rng.sample(topic["sentences"], 6)
            title = f"[SYNTHETIC] {topic['titles'][i % len(topic['titles'])]} ({name} #{i})"
            text = " ".join(sentences)
            lo, hi = topic["risk"]
            out.append(
                SilverRecord(
                    doc_id=doc_id,
                    url=f"https://synthetic-{['news', 'forum', 'wire'][i % 3]}.example/{name}/{i}",
                    domain=f"synthetic-{['news', 'forum', 'wire'][i % 3]}.example",
                    source=["synthetic_news", "synthetic_forum", "synthetic_wire"][i % 3],
                    source_type="community" if i % 3 == 1 else "news",
                    depth=i % 3,
                    title=title,
                    clean_text=text,
                    summary=sentences[0],
                    word_count=len(text.split()),
                    published_at=datetime(2026, 1 + (i + t_idx) % 9, 1 + i, tzinfo=UTC),
                    tcs_relevance=0.9,
                    relevant=True,
                    risk_signals=[
                        RiskSignal(
                            category=topic["categories"][0],
                            severity="high" if hi > 80 else "medium",
                            keyword=f"synthetic-{name}",
                            evidence=sentences[0],
                        )
                    ],
                    risk_categories=list(topic["categories"]),
                    sentiment=-0.4,
                    risk_score=float(lo + (hi - lo) * rng.random()),
                    extractor=SYNTHETIC_EXTRACTOR,
                    processed_at=datetime(2026, 10, 1, tzinfo=UTC),
                )
            )
    return out


def topic_of(rec: SilverRecord) -> str:
    return rec.url.split("/")[-2]


class FakeSilverStore:
    """In-memory stand-in for ``SilverStore.iter_relevant`` (no Elasticsearch)."""

    def __init__(self, records: list[SilverRecord]) -> None:
        self.records = records

    def iter_relevant(self, min_relevance: float | None = None) -> Iterator[SilverRecord]:
        for r in self.records:
            if r.relevant and (min_relevance is None or r.tcs_relevance >= min_relevance):
                yield r


def test_synthetic_corpus_is_labelled_and_balanced() -> None:
    """AC-005.11: the gold fixture corpus is clearly synthetic and has 3 balanced topics."""
    recs = synthetic_records()
    assert len(recs) == 30
    assert all(r.title.startswith("[SYNTHETIC]") for r in recs)
    assert all(r.domain.endswith(".example") and r.extractor == SYNTHETIC_EXTRACTOR for r in recs)
    assert {topic_of(r) for r in recs} == set(TOPIC_NAMES)
    assert synthetic_records() == synthetic_records()  # deterministic
