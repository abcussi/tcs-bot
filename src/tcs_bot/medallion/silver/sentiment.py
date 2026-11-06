"""Small finance/risk-tuned lexicon sentiment (AC-005.5): ``sentiment(text) -> -1..1``.

Token scores are summed with negation flipping (``not``, ``no``, ``never``, ``n't``, ``without``
within the 3 preceding tokens), length-damped (``/ sqrt(tokens / 25)``) and normalized
VADER-style: ``x / sqrt(x^2 + ALPHA)``.
"""

from __future__ import annotations

import math
import re

ALPHA = 15.0
LENGTH_UNIT = 25.0  # tokens
_NEGATORS = {"not", "no", "never", "without", "nor", "neither", "cannot", "hardly", "barely"}
_TOKEN = re.compile(r"[a-z][a-z'-]*")

LEXICON: dict[str, float] = {
    # negative - risk / legal / ethics
    "fraud": -3.0,
    "fraudulent": -3.0,
    "bribery": -3.0,
    "bribe": -3.0,
    "corruption": -3.0,
    "scandal": -2.5,
    "lawsuit": -2.0,
    "sued": -2.0,
    "litigation": -1.5,
    "penalty": -2.0,
    "fined": -2.0,
    "fine": -0.5,
    "probe": -1.5,
    "investigation": -1.5,
    "violation": -2.0,
    "violated": -2.0,
    "breach": -2.5,
    "breached": -2.5,
    "ransomware": -3.0,
    "hacked": -2.5,
    "leak": -2.0,
    "leaked": -2.0,
    "cyberattack": -3.0,
    "outage": -2.0,
    "disruption": -1.5,
    "failure": -2.0,
    "failed": -2.0,
    "layoffs": -2.0,
    "layoff": -2.0,
    "laid": -1.0,
    "discrimination": -2.5,
    "harassment": -2.5,
    "toxic": -2.0,
    "unpaid": -2.0,
    "exploitation": -2.5,
    "boycott": -2.0,
    "backlash": -2.0,
    "controversy": -1.5,
    "criticism": -1.5,
    "criticised": -1.5,
    "criticized": -1.5,
    "decline": -1.5,
    "declined": -1.5,
    "fell": -1.5,
    "fall": -1.0,
    "drop": -1.0,
    "dropped": -1.5,
    "slump": -2.0,
    "plunge": -2.5,
    "plunged": -2.5,
    "loss": -1.5,
    "losses": -1.5,
    "lost": -1.5,
    "weak": -1.5,
    "weaker": -1.5,
    "downgrade": -2.0,
    "downgraded": -2.0,
    "warning": -1.5,
    "risk": -0.5,
    "risks": -0.5,
    "concern": -1.0,
    "concerns": -1.0,
    "allegations": -2.0,
    "alleged": -1.5,
    "accused": -2.0,
    "misconduct": -2.5,
    "terminated": -2.0,
    "termination": -2.0,
    "cancelled": -1.5,
    "delay": -1.0,
    "delayed": -1.0,
    "sanctions": -1.5,
    "tariffs": -1.0,
    "uncertainty": -1.0,
    "crisis": -2.5,
    "damages": -1.5,
    "guilty": -2.5,
    "greenwashing": -2.0,
    "attrition": -1.0,
    # positive
    "growth": 1.5,
    "grew": 1.5,
    "record": 1.0,
    "strong": 1.5,
    "stronger": 1.5,
    "profit": 1.0,
    "profitable": 1.5,
    "beat": 1.5,
    "gain": 1.0,
    "gains": 1.0,
    "rose": 1.0,
    "rise": 1.0,
    "surge": 2.0,
    "win": 1.5,
    "won": 1.5,
    "wins": 1.5,
    "award": 1.5,
    "awarded": 1.0,
    "recognised": 1.5,
    "recognized": 1.5,
    "upgrade": 1.5,
    "upgraded": 1.5,
    "expansion": 1.0,
    "hiring": 1.0,
    "hires": 1.0,
    "partnership": 1.0,
    "innovation": 1.0,
    "resilient": 1.5,
    "stable": 1.0,
    "improved": 1.5,
    "improvement": 1.5,
    "compliant": 1.0,
    "cleared": 1.5,
    "acquitted": 2.0,
    "dismissed": 0.5,
    "settled": 0.5,
    "resolved": 1.5,
    "success": 2.0,
    "successful": 2.0,
    "trust": 1.0,
    "leader": 1.0,
    "best": 1.5,
}


def sentiment(text: str) -> float:
    """Return a score in ``[-1, 1]`` (0 for empty / neutral text)."""
    tokens = _TOKEN.findall((text or "").lower())
    total = 0.0
    for i, tok in enumerate(tokens):
        score = LEXICON.get(tok)
        if score is None:
            continue
        window = tokens[max(0, i - 3) : i]
        if any(t in _NEGATORS or t.endswith("n't") for t in window):
            score = -0.5 * score
        total += score
    if total == 0:
        return 0.0
    # Damp length so a long article is not saturated by sheer token count.
    x = total / math.sqrt(max(1.0, len(tokens) / LENGTH_UNIT))
    return round(max(-1.0, min(1.0, x / math.sqrt(x * x + ALPHA))), 4)
