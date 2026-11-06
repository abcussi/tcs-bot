"""Fixed TCS risk query set, community subreddits and the depth-2 domain allow-list (AC-005.1/3)."""

from __future__ import annotations

TCS_RISK_QUERIES: list[str] = [
    '"Tata Consultancy Services" ethics',
    '"Tata Consultancy Services" compliance violation',
    "TCS lawsuit",
    '"Tata Consultancy Services" lawsuit verdict',
    "TCS fine penalty",
    "TCS layoffs",
    "TCS data breach",
    "TCS cyber attack",
    "TCS H-1B visa fraud",
    "TCS discrimination lawsuit",
    '"Tata Consultancy Services" investigation',
    "TCS SEBI",
    "TCS controversy",
    "TCS outage client",
    "TCS client loss contract termination",
    "TCS whistleblower",
    "TCS bribery corruption",
    "TCS tax dispute",
    "TCS ESG",
    "TCS employee union",
    "TCS trade secrets Epic Systems",
    "TCS DXC CSC trade secrets",
]

COMMUNITY_SUBREDDITS: list[str] = [
    "developersIndia",
    "india",
    "cscareerquestions",
    "IndianWorkplace",
    "TCS",
]

# Suffix-matched (``host == d or host.endswith("." + d)``).
DEFAULT_ALLOWED_DOMAINS: list[str] = [
    "news.google.com",
    "bing.com",
    "gdeltproject.org",
    "hn.algolia.com",
    "news.ycombinator.com",
    "www.reddit.com",
    "economictimes.indiatimes.com",
    "livemint.com",
    "moneycontrol.com",
    "business-standard.com",
    "thehindubusinessline.com",
    "reuters.com",
    "timesofindia.indiatimes.com",
    "financialexpress.com",
    "cnbctv18.com",
    "ndtvprofit.com",
    "hindustantimes.com",
    "indianexpress.com",
    "thehindu.com",
    "tcs.com",
]


def domain_allowed(host: str, allowed: list[str] | tuple[str, ...]) -> bool:
    host = (host or "").lower().rstrip(".")
    return any(host == d or host.endswith("." + d) for d in (a.lower() for a in allowed))
