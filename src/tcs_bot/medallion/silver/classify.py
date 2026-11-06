"""Deterministic, explainable TCS risk classifier (AC-005.5 rules path, "rules-v1").

* :data:`TAXONOMY` - regex rule families for every :data:`RiskCategory`, each with a severity.
* :func:`classify` - sentence-level matching -> deduped :class:`RiskSignal` list + 0-100 score.
* :func:`tcs_relevance` - 0..1 score of how much a document is *about* Tata Consultancy Services,
  with strong penalties for the Indian tax "Tax Collected at Source" (TCS) and for other Tata
  group companies (Tata Motors, Tata Steel, ...) mentioned without TCS.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from tcs_bot.medallion.models import RISK_CATEGORIES, RiskSignal

SEVERITIES: tuple[str, ...] = ("low", "medium", "high", "critical")
SEVERITY_POINTS: dict[str, float] = {"low": 5.0, "medium": 12.0, "high": 25.0, "critical": 40.0}
MAX_EVIDENCE_CHARS = 300
TITLE_BOOST = 1.5
NON_TCS_SENTENCE_WEIGHT = 0.7  # evidence without document context (LLM-merged signals)
# Evidence weight by sentence distance to the nearest sentence that names TCS (the title counts
# as the sentence before the body). Far evidence is generic discussion, not a TCS risk.
PROXIMITY_WEIGHTS: tuple[float, ...] = (1.0, 0.8, 0.6)
FAR_WEIGHT = 0.25
FAR_CAP = 18.0  # far evidence can never contribute more than this many points in total
LENGTH_REF_WORDS = 300
LENGTH_DAMPING = 0.12  # score /= 1 + 0.12 * log2(words / 300) for long texts


@dataclass(frozen=True)
class Rule:
    category: str
    severity: str
    keyword: str
    pattern: re.Pattern[str]


def _r(category: str, severity: str, keyword: str, pattern: str) -> Rule:
    assert category in RISK_CATEGORIES and severity in SEVERITIES, (category, severity)
    return Rule(category, severity, keyword, re.compile(pattern, re.IGNORECASE))


# Regulators/agencies that make a "probe"/"notice" a compliance event. Acronyms are matched
# case-sensitively via the scoped ``(?-i:...)`` flag so "ed"/"sec" in prose do not fire.
_REG = (
    r"(?:(?-i:SEC|SEBI|DOJ|FTC|EEOC|CBI|ED|SFIO|CCI|RBI|ICO|CNIL|NLRB|USCIS|OFAC|FCA|DOL)"
    r"|Enforcement Directorate|regulators?|regulatory|authorit(?:y|ies)|Department of Labou?r"
    r"|Income Tax Department|tax authorities|Competition Commission|European Commission)"
)

TAXONOMY: tuple[Rule, ...] = (
    # ------------------------------------------------------------------ ethics
    _r("ethics", "critical", "bribery", r"\bbrib(?:e|es|ed|ery|ing)\b|\bkickbacks?\b"),
    _r("ethics", "critical", "corruption", r"\bcorrupt(?:ion|ed)?\b|(?-i:\bFCPA\b)"),
    _r("ethics", "critical", "fraud", r"(?<!visa )\bfraud(?:ulent|ulently|s)?\b"),
    _r("ethics", "critical", "money laundering", r"\bmoney[- ]laundering\b|\bembezzle\w*"),
    _r("ethics", "high", "insider trading", r"\binsider trading\b"),
    _r(
        "ethics", "high", "misconduct", r"\bmisconduct\b|\bethic(?:s|al) (?:breach|violation|lapse)"
    ),
    _r("ethics", "high", "whistleblower", r"\bwhistle-?blow\w*"),
    _r(
        "ethics", "high", "falsification", r"\bfalsif\w+|\bforg(?:ed|ery)\b|\bfake (?:invoice|bill)"
    ),
    _r("ethics", "medium", "conflict of interest", r"\bconflicts? of interest\b"),
    # ------------------------------------------------------------------ compliance / regulatory
    _r(
        "compliance_regulatory",
        "high",
        "regulatory probe",
        _REG + r"[^.;]{0,80}?\b(?:probe|investigat\w+|inquiry|enquiry|scrutiny|notice|subpoena)"
        r"|\b(?:probe|investigation|inquiry|subpoena)\w*\b[^.;]{0,40}?\bby (?:the )?" + _REG,
    ),
    _r(
        "compliance_regulatory",
        "high",
        "penalty/fine",
        r"\bpenalt(?:y|ies)\b|\bfined\b|\bfine of\b|\bfines\b"
        r"|\b(?:a|the|hefty|record|regulatory|civil)\s+fine\b",
    ),
    _r("compliance_regulatory", "high", "show-cause notice", r"\bshow[- ]cause notice\b"),
    _r(
        "compliance_regulatory",
        "high",
        "regulatory violation",
        r"\bnon-?compliance\b|\bviolat(?:ed|es|ion|ions|ing)\b[^.;]{0,40}?"
        r"\b(?:laws?|rules?|regulations?|act|norms|guidelines|code)\b",
    ),
    _r("compliance_regulatory", "high", "antitrust", r"\banti-?trust\b|\bcartel\w*\b"),
    _r(
        "compliance_regulatory",
        "high",
        "audit qualification",
        r"\bqualified (?:audit )?opinion\b|\bauditors? resign\w*",
    ),
    _r(
        "compliance_regulatory",
        "medium",
        "investigation",
        r"\binvestigat(?:ion|ions|ing|ed)\b|\bprobe[ds]?\b",
    ),
    _r(
        "compliance_regulatory",
        "medium",
        "tax dispute",
        r"\btax (?:demand|dispute|notice|evasion)\b|(?-i:\bGST\b) (?:demand|notice)",
    ),
    # ------------------------------------------------------------------ legal / litigation
    _r(
        "legal_litigation",
        "high",
        "lawsuit",
        r"\blaw ?suits?\b|\bsue[ds]?\b|\bsuing\b|\bclass[- ]action\b|\blitigation\b",
    ),
    _r(
        "legal_litigation",
        "high",
        "settlement",
        r"\bsettlement\b|\bagreed to (?:pay|settle)\b"
        r"|\bsettled (?:the|a|with|for|claims?|lawsuit|case|charges)\b",
    ),
    _r("legal_litigation", "high", "verdict/damages", r"\bverdict\b|\bjury\b|\bdamages\b"),
    _r("legal_litigation", "high", "trade secrets", r"\btrade secrets?\b|\bmisappropriat\w+"),
    _r(
        "legal_litigation",
        "high",
        "infringement",
        r"\b(?:patent|copyright|trademark) infringement\b|\binfring\w+",
    ),
    _r(
        "legal_litigation",
        "high",
        "criminal charges",
        r"\bindicted\b|\bcharged with\b|\barrested\b|(?-i:\bFIR\b)|\bcriminal (?:case|complaint"
        r"|charges?)\b",
    ),
    _r(
        "legal_litigation",
        "medium",
        "court proceedings",
        r"\b(?:court|tribunal|injunction|arbitration|plaintiffs?|defendants?)\b",
    ),
    # ------------------------------------------------------------------ data privacy / cyber
    _r(
        "data_privacy_cyber",
        "critical",
        "data breach",
        r"\bdata (?:breach|leak|theft|exposure)\w*|\bbreach(?:ed)?\b[^.;]{0,30}?\b(?:data|systems?|"
        r"network|security|records)\b|\b(?:security|cyber) breach\w*",
    ),
    _r("data_privacy_cyber", "critical", "ransomware", r"\bransomware\b"),
    _r(
        "data_privacy_cyber",
        "critical",
        "cyberattack",
        r"\bcyber[- ]?attacks?\b|\bsupply[- ]chain attack\b",
    ),
    _r("data_privacy_cyber", "critical", "leak", r"\bleak(?:ed|s|ing)?\b"),
    _r(
        "data_privacy_cyber",
        "high",
        "hacking",
        r"\bhack(?:ed|ers?|ing)\b|\bmalware\b|\bphishing\b|\bcompromised\b",
    ),
    _r(
        "data_privacy_cyber",
        "high",
        "privacy violation",
        r"\bprivacy (?:violation|breach|lawsuit)\w*|(?-i:\bGDPR\b)[^.;]{0,30}?\b(?:breach"
        r"|violation|fine)",
    ),
    _r(
        "data_privacy_cyber",
        "medium",
        "vulnerability",
        r"\bvulnerabilit(?:y|ies)\b|\bsocial engineering\b",
    ),
    # ------------------------------------------------------------------ labour / HR
    _r(
        "labor_hr",
        "high",
        "layoffs",
        r"\blay-?offs?\b|\blaid off\b|\bjob cuts?\b|\bretrench\w*|\bdownsiz\w+"
        r"|\bheadcount (?:cut|reduction)s?\b|\bsack(?:ed|ing)\b",
    ),
    _r(
        "labor_hr",
        "high",
        "discrimination",
        r"\bdiscriminat\w+|\bharass\w*|\bbias(?:ed)? against\b|\bageis[mt]\b",
    ),
    _r(
        "labor_hr",
        "high",
        "forced resignation",
        r"\bforced (?:to )?resign\w*|\bforced resignations?\b|\basked to resign\b|\bcoerced\b",
    ),
    _r(
        "labor_hr",
        "high",
        "unpaid wages",
        r"\bunpaid\b|\bwage theft\b|\bsalary (?:delay|cut|freeze)s?\b|\bdelayed (?:salar\w+|pay"
        r"|wages|appraisals?|onboarding|joining)",
    ),
    _r(
        "labor_hr",
        "medium",
        "toxic culture",
        r"\btoxic\b|\bburn-?out\b|\bexploit\w*|\bwork[- ]life balance\b|\bovertime\b",
    ),
    _r(
        "labor_hr",
        "medium",
        "bench policy",
        r"\bbench (?:policy|period|time|employees|staff)\b|\bon (?:the )?bench\b",
    ),
    _r(
        "labor_hr",
        "medium",
        "attrition",
        r"\battrition\b|\bemployee (?:protest|strike)s?\b|\bunion\b",
    ),
    # ------------------------------------------------------------------ financial
    _r(
        "financial",
        "medium",
        "profit warning",
        r"\bprofit warning\b|\bguidance cut\b|\b(?:cut|cuts|lower(?:ed|s)?|slash(?:ed"
        r"|es)?) (?:its |the |full-year |revenue )?(?:guidance|outlook|forecast)\b",
    ),
    _r(
        "financial",
        "medium",
        "margin decline",
        r"\bmargins? (?:declin\w+|contract\w+|fell|fall\w*|drop\w*|shr[ai]nk\w*|squeez\w+"
        r"|pressure|erosion)|\bmargin (?:decline|contraction)\b",
    ),
    _r(
        "financial",
        "medium",
        "revenue decline",
        r"\b(?:revenue|profit|earnings|income|sales) (?:declin\w+|fell|drop\w*|miss\w*|slump\w*)"
        r"|\bmiss(?:ed|es)? (?:estimates|expectations)\b",
    ),
    _r(
        "financial",
        "medium",
        "share price fall",
        r"\b(?:shares?|stock) (?:fell|fall\w*|slump\w*|plung\w+|tumbl\w+|crash\w*|sank|slid)\b",
    ),
    _r("financial", "medium", "downgrade", r"\bdowngrad\w+"),
    _r("financial", "medium", "impairment", r"\bimpairment\b|\bwrite-?(?:down|off)s?\b"),
    _r("financial", "critical", "going concern", r"\bgoing concern\b|\binsolven\w+|\bbankrupt\w*"),
    # ------------------------------------------------------------------ operational / delivery
    _r(
        "operational_delivery",
        "high",
        "outage",
        r"\boutages?\b|\bdowntime\b|\bservice disruptions?\b|\bsystem failures?\b",
    ),
    _r(
        "operational_delivery",
        "high",
        "project failure",
        r"\bproject (?:failure|failed|delay\w*|overrun\w*)\b|\bfailed (?:implementation"
        r"|migration|rollout|project|go-live)\b|\bbotched\b|\bcost overruns?\b",
    ),
    _r(
        "operational_delivery",
        "high",
        "contract termination",
        r"\b(?:terminat\w+|cancel\w+|scrap\w+|pull(?:ed|s)? out of)\b[^.;]{0,40}?\b(?:contract"
        r"|deal|engagement)s?\b"
        r"|\b(?:contract|deal|engagement)s?\b[^.;]{0,40}?\b(?:terminat\w+|cancel\w+|scrapp\w+)",
    ),
    _r(
        "operational_delivery",
        "high",
        "client loss",
        r"\b(?:lost|loses|losing|loss of)\b[^.;]{0,30}?\b(?:clients?|customers?|deals?|accounts?"
        r"|mandates?)\b|\bclient (?:exit|loss|attrition)\b",
    ),
    _r(
        "operational_delivery",
        "medium",
        "delivery issues",
        r"\bmissed deadlines?\b|\b(?:SLA|service level) breach\w*|\bdelivery (?:issues|failures?"
        r"|delays?|problems)\b",
    ),
    # ------------------------------------------------------------------ reputational
    _r("reputational", "high", "scandal", r"\bscandals?\b|\breputational (?:damage|harm|risk)\b"),
    _r("reputational", "medium", "boycott", r"\bboycott\w*"),
    _r("reputational", "medium", "backlash", r"\bbacklash\b|\boutrage\b|\buproar\b"),
    _r("reputational", "medium", "controversy", r"\bcontrovers(?:y|ies|ial)\b"),
    _r(
        "reputational",
        "medium",
        "criticism",
        r"\bcriticis(?:m|ed|es)\b|\bcriticiz(?:ed|es)\b|\bnegative publicity\b|\bslammed\b",
    ),
    # ------------------------------------------------------------------ ESG
    _r(
        "esg",
        "medium",
        "emissions",
        r"\b(?:carbon |greenhouse gas |GHG )?emissions\b|\bcarbon footprint\b"
        r"|\bnet[- ]zero (?:miss|delay)\w*",
    ),
    _r("esg", "medium", "greenwashing", r"\bgreenwash\w*"),
    _r("esg", "medium", "human rights", r"\bhuman rights\b"),
    _r("esg", "high", "forced labour", r"\b(?:forced|child) labou?r\b|\bmodern slavery\b"),
    _r(
        "esg",
        "medium",
        "ESG downgrade",
        r"(?-i:\bESG\b)[^.;]{0,30}?\b(?:downgrade\w*|controvers\w+|concerns?|rating cut)",
    ),
    # ------------------------------------------------------------------ geopolitical / immigration
    _r(
        "geopolitical_immigration",
        "high",
        "visa fraud",
        r"\bvisa (?:fraud|abuse|misuse|violations?|scam)\b",
    ),
    _r("geopolitical_immigration", "high", "H-1B", r"\bH-?1B\b|\bL-?1 visas?\b|\bwork visas?\b"),
    _r(
        "geopolitical_immigration",
        "high",
        "sanctions",
        r"\bsanctions\b|\bsanctioned (?:entit|countr|compan|individual)\w*",
    ),
    _r(
        "geopolitical_immigration",
        "high",
        "tariffs",
        r"\btariffs?\b|\btrade war\b|\bexport controls?\b|\boutsourcing (?:tax|ban|curbs?)\b",
    ),
    _r("geopolitical_immigration", "medium", "immigration", r"\bimmigration\b|\bdeportat\w+"),
)

# "denied"/"dismissed the case"/"cleared of" etc. lower severity one level.
_NEGATION = re.compile(
    r"\bno evidence (?:of|that)\b|\bden(?:ied|ies|y|ying)\b|\bcleared of\b|\bacquitted\b"
    r"|\bwithout merit\b|\bno wrongdoing\b|\bbaseless\b|\bunfounded\b|\bfalse (?:claims?"
    r"|reports?)\b"
    r"|\bdismiss(?:ed|es)\b(?! (?:\d+|over|nearly|around|about|hundreds|thousands|several|some"
    r"|many)"
    r"\b|\s+(?:employees|staff|workers))"
    r"|\bnot (?:involved|affected|impacted|found)\b|\brumou?rs?\b",
    re.IGNORECASE,
)

# --------------------------------------------------------------------------- relevance
BODY_ONLY_FACTOR = 0.6  # relevance multiplier when the title does not name TCS
BODY_FOCUS_FLOOR_PCT = 1.0  # TCS mentions per 100 words below which a body-only hit is incidental
BODY_FOCUS_SPAN_PCT = 2.0  # ... and above floor+span the body-only penalty is gone
TATA_TITLE_FACTOR = 0.6  # headline about Tata (Sons/group/chairman) with TCS only in the body
_TCS_STRONG = re.compile(
    r"(?-i:\bTCS\b)|\bTata Consultancy\b|\btcs\.com\b|\bKrithivasan\b", re.IGNORECASE
)
_TCS_WEAK = re.compile(r"\bChandrasekaran\b", re.IGNORECASE)
# "TCS (Tata Consultancy Services)" / "Tata Consultancy Services (TCS)" name the company once.
_TCS_DOUBLE = re.compile(
    r"\bTata Consultancy(?: Services)?(?:\s*\(\s*(?-i:TCS)\s*\)|,?\s+(?-i:TCS)\b)"
    r"|(?-i:\bTCS\b)\s*\(\s*Tata Consultancy(?: Services)?\s*\)",
    re.IGNORECASE,
)
_TCS_ANY = re.compile(_TCS_STRONG.pattern + r"|\bChandrasekaran\b", re.IGNORECASE)
_TCS_EXPLICIT = re.compile(r"\bTata Consultancy\b|\btcs\.com\b|\bKrithivasan\b", re.IGNORECASE)
_TAX_STRONG = re.compile(
    r"\btax collected at (?:the )?source\b|\b206 ?C\b|\bForm 27EQ\b"
    r"|\bLiberali[sz]ed Remittance Scheme\b",
    re.IGNORECASE,
)
_TCS_TAX_ACRONYM = (
    r"(?-i:\bTCS\b) (?:rate|rates|rule|rules|on|provisions?|credit|certificate|refund|liability"
    r"|deduct\w*|threshold|under section|u/s)\b"
)
_TCS_TAX = re.compile(_TCS_TAX_ACRONYM, re.IGNORECASE)
_TAX_WEAK = re.compile(
    _TCS_TAX_ACRONYM + r"|(?-i:\bLRS\b|\bTDS\b|\bITR\b|\bCBDT\b)|\bremittances?\b"
    r"|\bforeign tour packages?\b|\bincome tax return\b",
    re.IGNORECASE,
)
_TATA_OTHER = re.compile(
    r"\bTata (?:Motors|Steel|Power|Chemicals|Consumer|Elxsi|Communications|Projects|Capital|Play"
    r"|Sky|Technologies|Electronics|Digital|Neu|AutoComp|Realty|Housing|Coffee|Tea|Trusts"
    r"|Advanced Systems)\b|\bAir India\b|\bJaguar Land Rover\b|(?-i:\bJLR\b)|\bTitan\b"
    r"|\bTanishq\b|\bTaj Hotels\b|(?-i:\bIHCL\b)|\bVoltas\b|\bTrent\b|\bBigBasket\b|\bVistara\b",
    re.IGNORECASE,
)
_TATA_TITLE = re.compile(r"\bTata\b", re.IGNORECASE)  # group/chairman headline without TCS
_WORD = re.compile(r"\w+")
_SENT_SPLIT = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[\"'(\[]?[A-Z0-9₹$£€])|\n+")


def split_sentences(text: str) -> list[str]:
    """Cheap sentence splitter (punctuation + capital, or newline)."""
    return [s.strip() for s in _SENT_SPLIT.split(text or "") if s and s.strip()]


def mentions_tcs(text: str) -> bool:
    return bool(_TCS_ANY.search(text or ""))


def title_names_tcs(title: str) -> bool:
    """True if the headline itself names TCS (company name, ``TCS``, CEO...)."""
    return bool(_TCS_ANY.search(title or ""))


def _clip(sentence: str) -> str:
    s = " ".join(sentence.split())
    return s if len(s) <= MAX_EVIDENCE_CHARS else s[: MAX_EVIDENCE_CHARS - 1].rstrip() + "…"


def _lower(severity: str) -> str:
    return SEVERITIES[max(0, SEVERITIES.index(severity) - 1)]


def _signal_points(severity: str, weight: float, title_hit: bool) -> float:
    return SEVERITY_POINTS[severity] * weight * (TITLE_BOOST if title_hit else 1.0)


def _tcs_distances(flags: list[bool]) -> list[int | None]:
    """Distance (in units) from every unit to the nearest TCS-naming unit; ``None`` if none."""
    n = len(flags)
    dist: list[int | None] = [None] * n
    last: int | None = None
    for i in range(n):
        if flags[i]:
            last = i
        if last is not None:
            dist[i] = i - last
    last = None
    for i in range(n - 1, -1, -1):
        if flags[i]:
            last = i
        if last is not None:
            d = last - i
            dist[i] = d if dist[i] is None else min(dist[i], d)  # type: ignore[type-var]
    return dist


def _proximity_weight(distance: int | None) -> float:
    if distance is None or distance >= len(PROXIMITY_WEIGHTS):
        return FAR_WEIGHT
    return PROXIMITY_WEIGHTS[distance]


def length_damping(words: int) -> float:
    """1.0 up to :data:`LENGTH_REF_WORDS`, then ``1 / (1 + 0.12 * log2(words / ref))``."""
    if words <= LENGTH_REF_WORDS:
        return 1.0
    return 1.0 / (1.0 + LENGTH_DAMPING * math.log2(words / LENGTH_REF_WORDS))


def combine_points(points: list[float]) -> float:
    """Diminishing returns: ``100 * (1 - prod(1 - p/100))`` over the strongest signals first."""
    remaining = 1.0
    for p in sorted(points, reverse=True):
        remaining *= 1.0 - min(max(p, 0.0), 95.0) / 100.0
    return round(100.0 * (1.0 - remaining), 1)


def score_signals(signals: list[RiskSignal], title: str = "") -> float:
    """Risk score 0-100 for a signal list (used for LLM-merged signals too)."""
    title_l = (title or "").lower()
    points = []
    for s in signals:
        title_hit = bool(title_l) and s.evidence.lower().strip(" …") in title_l
        weight = 1.0 if mentions_tcs(s.evidence) else NON_TCS_SENTENCE_WEIGHT
        points.append(_signal_points(s.severity, weight, title_hit))
    return combine_points(points)


def classify(text: str, title: str = "") -> tuple[list[RiskSignal], float]:
    """Return ``(signals deduped by (category, keyword), risk_score 0..100)``.

    Every sentence (the title counts as one) is matched against :data:`TAXONOMY`. For each
    ``(category, keyword)`` the strongest occurrence is kept: highest (negation-adjusted) severity,
    preferring sentences at/near a TCS mention and title hits. Evidence is the sentence (<= 300
    chars).

    The score is TCS-focused: a signal is weighted by its sentence distance to the nearest sentence
    naming TCS (:data:`PROXIMITY_WEIGHTS`, :data:`FAR_WEIGHT`), the contribution of distant
    evidence is capped at :data:`FAR_CAP` points, and long texts are damped (:func:`length_damping`)
    so a long thread full of generic words is not rated above a focused TCS article.
    """
    units: list[tuple[str, bool]] = []
    if title and title.strip():
        units.append((title.strip(), True))
    units.extend((s, False) for s in split_sentences(text))
    tcs_flags = [mentions_tcs(sentence) for sentence, _ in units]
    distances = _tcs_distances(tcs_flags)

    # (points, signal, weight, title_hit)
    best: dict[tuple[str, str], tuple[float, RiskSignal, float, bool]] = {}
    for (sentence, is_title), distance in zip(units, distances, strict=True):
        weight = _proximity_weight(distance)
        negated = bool(_NEGATION.search(sentence))
        for rule in TAXONOMY:
            if not rule.pattern.search(sentence):
                continue
            severity = _lower(rule.severity) if negated else rule.severity
            pts = _signal_points(severity, weight, is_title)
            key = (rule.category, rule.keyword)
            prev = best.get(key)
            if prev is None or pts > prev[0]:
                sig = RiskSignal(
                    category=rule.category,  # type: ignore[arg-type]
                    severity=severity,  # type: ignore[arg-type]
                    keyword=rule.keyword,
                    evidence=_clip(sentence),
                )
                title_hit = is_title or (prev[3] if prev else False)
                best[key] = (pts, sig, weight, title_hit)
            elif is_title:
                best[key] = (prev[0], prev[1], prev[2], True)

    near: list[float] = []
    far: list[float] = []
    signals: list[RiskSignal] = []
    for _pts, sig, weight, title_hit in sorted(best.values(), key=lambda v: -v[0]):
        signals.append(sig)
        # the title boost applies even when the evidence kept is a body sentence
        pts = _signal_points(sig.severity, weight, title_hit)
        (near if weight > FAR_WEIGHT else far).append(pts)
    if far:
        near.append(min(combine_points(far), FAR_CAP))
    words = len(_WORD.findall(f"{title} {text}"))
    return signals, round(combine_points(near) * length_damping(words), 1)


def categories_of(signals: list[RiskSignal]) -> list[str]:
    """Distinct categories in taxonomy order."""
    present = {s.category for s in signals}
    return [c for c in RISK_CATEGORIES if c in present]


def tcs_relevance(text: str, title: str = "") -> float:
    """How much the document is about Tata Consultancy Services, 0..1.

    ``0.45 * mention saturation + 0.25 * density (mentions / 100 words) + 0.30 * title hit``,
    then multiplied by penalties for tax-at-source and non-TCS Tata group contexts.
    """
    text = text or ""
    title = title or ""
    full = f"{title}\n{text}"
    words = max(len(_WORD.findall(full)), 1)

    tax_strong = len(_TAX_STRONG.findall(full))
    tax_weak = len(_TAX_WEAK.findall(full))
    # "TCS" acronyms inside tax phrases are not company mentions.
    strong = len(_TCS_STRONG.findall(full)) - len(_TCS_TAX.findall(full))
    strong = max(strong - len(_TCS_DOUBLE.findall(full)), 0)
    weak = len(_TCS_WEAK.findall(full))
    mentions = strong + 0.5 * weak
    if mentions <= 0:
        return 0.0

    mention_score = 1.0 - math.exp(-mentions / 3.0)
    density_pct = (mentions / words) * 100.0
    density_score = min(1.0, density_pct)
    title_hit = 1.0 if _TCS_ANY.search(title) else 0.0
    score = 0.45 * mention_score + 0.25 * density_score + 0.30 * title_hit
    if not title_hit:
        # TCS is only named in the body: a passing mention in a thread about something else
        # unless TCS is a large share of the text. Group headlines (Tata & Sons...) count less.
        focus = min(max((density_pct - BODY_FOCUS_FLOOR_PCT) / BODY_FOCUS_SPAN_PCT, 0.0), 1.0)
        score *= BODY_ONLY_FACTOR + (1.0 - BODY_ONLY_FACTOR) * focus
        if _TATA_TITLE.search(title):
            score *= TATA_TITLE_FACTOR

    explicit = bool(_TCS_EXPLICIT.search(full))
    if tax_strong or tax_weak >= 2:
        score *= 0.6 if explicit else 0.1
    elif tax_weak == 1 and not explicit:
        score *= 0.5

    tata_other = len(_TATA_OTHER.findall(full))
    if tata_other:
        if strong == 0:
            score *= 0.15  # only the Tata Sons chairman is named, in a non-TCS story
        elif tata_other > 2 * strong:
            score *= 0.5
    return round(min(max(score, 0.0), 1.0), 4)
