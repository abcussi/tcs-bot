"""Rule/gazetteer entity extraction for silver records (AC-005.5).

``extract_entities(text) -> Entities`` with five deduped lists (max :data:`MAX_PER_TYPE` each):
regulators (canonical names), money amounts, places, organizations and people.
"""

from __future__ import annotations

import re

from tcs_bot.medallion.models import Entities

MAX_PER_TYPE = 20

# (pattern, canonical). Acronyms are case-sensitive; long names case-insensitive.
_REGULATORS: tuple[tuple[str, str], ...] = (
    (r"(?-i:\bSEC\b)|Securities and Exchange Commission", "SEC"),
    (r"(?-i:\bSEBI\b)|Securities and Exchange Board of India", "SEBI"),
    (r"(?-i:\bDOJ\b)|(?:US |U\.S\. )?Department of Justice|Justice Department", "DOJ"),
    (r"(?-i:\bFTC\b)|Federal Trade Commission", "FTC"),
    (r"(?-i:\bEEOC\b)|Equal Employment Opportunity Commission", "EEOC"),
    (r"(?-i:\bRBI\b)|Reserve Bank of India", "RBI"),
    (r"(?-i:\bNSE\b)|National Stock Exchange", "NSE"),
    (r"(?-i:\bBSE\b)|Bombay Stock Exchange", "BSE"),
    (r"(?-i:\bED\b)|Enforcement Directorate", "Enforcement Directorate"),
    (r"(?-i:\bCBI\b)|Central Bureau of Investigation", "CBI"),
    (r"Income[- ]Tax Department|(?-i:\bCBDT\b)", "Income Tax Department"),
    (r"(?:US |U\.S\. )?Department of Labou?r|(?-i:\bDOL\b)", "US Department of Labor"),
    (r"(?-i:\bICO\b)|Information Commissioner'?s Office", "ICO"),
    (r"(?-i:\bCNIL\b)", "CNIL"),
    (r"European Commission", "European Commission"),
    (r"(?-i:\bGDPR\b)|General Data Protection Regulation", "GDPR"),
    (r"(?-i:\bDPDP\b)(?: Act)?|Digital Personal Data Protection", "DPDP Act"),
    (r"(?-i:\bCCI\b)|Competition Commission of India", "CCI"),
    (r"(?-i:\bMCA\b)|Ministry of Corporate Affairs", "Ministry of Corporate Affairs"),
    (r"(?-i:\bSFIO\b)|Serious Fraud Investigation Office", "SFIO"),
    (r"(?-i:\bNCLT\b)|National Company Law Tribunal", "NCLT"),
    (r"(?-i:\bUSCIS\b)|Citizenship and Immigration Services", "USCIS"),
    (r"(?-i:\bDHS\b)|Department of Homeland Security", "DHS"),
    (r"(?-i:\bFCA\b)|Financial Conduct Authority", "FCA"),
    (r"(?-i:\bOFAC\b)|Office of Foreign Assets Control", "OFAC"),
    (r"(?-i:\bNLRB\b)|National Labor Relations Board", "NLRB"),
    (r"(?-i:\bFBI\b)|Federal Bureau of Investigation", "FBI"),
    (r"(?-i:\bCERT-In\b)", "CERT-In"),
    (r"(?-i:\bMeitY\b)|Ministry of Electronics and Information Technology", "MeitY"),
    (r"Ministry of Labou?r(?: and Employment)?", "Ministry of Labour and Employment"),
)
_REGULATOR_RX = [(re.compile(p, re.IGNORECASE), c) for p, c in _REGULATORS]

_NUM = r"\d[\d,]*(?:\.\d+)?"
_SCALE = r"(?:crores?|lakhs?|million|billion|trillion|mn|bn|cr|thousand|[kKmMbB])\b"
_MONEY_RX = (
    re.compile(rf"(?:₹|Rs\.?|INR|US\$|USD|\$|€|EUR|£|GBP)\s?{_NUM}(?:\s?{_SCALE})?", re.IGNORECASE),
    re.compile(
        rf"\b{_NUM}\s?(?:crores?|lakhs?)(?:\s(?:rupees|INR))?\b"
        rf"|\b{_NUM}\s?(?:million|billion)\s(?:dollars|rupees|euros|pounds|USD|INR)\b",
        re.IGNORECASE,
    ),
)

_COUNTRIES = (
    "India",
    "United States",
    "United Kingdom",
    "Canada",
    "Mexico",
    "Brazil",
    "Argentina",
    "Chile",
    "Germany",
    "France",
    "Netherlands",
    "Belgium",
    "Ireland",
    "Spain",
    "Portugal",
    "Italy",
    "Switzerland",
    "Sweden",
    "Norway",
    "Denmark",
    "Finland",
    "Poland",
    "Hungary",
    "Romania",
    "Czech Republic",
    "Austria",
    "Russia",
    "Ukraine",
    "Turkey",
    "Israel",
    "Saudi Arabia",
    "United Arab Emirates",
    "UAE",
    "Qatar",
    "Egypt",
    "South Africa",
    "Nigeria",
    "Kenya",
    "China",
    "Japan",
    "South Korea",
    "Singapore",
    "Malaysia",
    "Indonesia",
    "Philippines",
    "Vietnam",
    "Thailand",
    "Australia",
    "New Zealand",
    "Sri Lanka",
    "Bangladesh",
    "Pakistan",
    "Nepal",
)
_CITIES = (
    "Mumbai",
    "Bengaluru",
    "Bangalore",
    "Chennai",
    "Hyderabad",
    "Pune",
    "New Delhi",
    "Delhi",
    "Noida",
    "Gurugram",
    "Gurgaon",
    "Kolkata",
    "Ahmedabad",
    "Gandhinagar",
    "Kochi",
    "Thiruvananthapuram",
    "Nagpur",
    "Indore",
    "Bhubaneswar",
    "Lucknow",
    "Coimbatore",
    "New York",
    "San Francisco",
    "Los Angeles",
    "Chicago",
    "Dallas",
    "Houston",
    "Seattle",
    "Boston",
    "Atlanta",
    "Phoenix",
    "Washington",
    "Texas",
    "California",
    "New Jersey",
    "Wisconsin",
    "Florida",
    "Illinois",
    "Massachusetts",
    "Ohio",
    "London",
    "Manchester",
    "Birmingham",
    "Leeds",
    "Edinburgh",
    "Glasgow",
    "Toronto",
    "Montreal",
    "Sydney",
    "Melbourne",
    "Dubai",
    "Tokyo",
    "Paris",
    "Berlin",
    "Frankfurt",
    "Amsterdam",
    "Brussels",
    "Dublin",
    "Zurich",
    "Madrid",
    "Budapest",
    "Johannesburg",
    "São Paulo",
    "Mexico City",
)
_PLACE_ALIASES: dict[str, str] = {
    "US": "United States",
    "U.S.": "United States",
    "USA": "United States",
    "U.S.A.": "United States",
    "UK": "United Kingdom",
    "U.K.": "United Kingdom",
    "Britain": "United Kingdom",
    "England": "United Kingdom",
    "Bangalore": "Bengaluru",
    "Gurgaon": "Gurugram",
    "UAE": "United Arab Emirates",
}


def _alt(names: tuple[str, ...] | list[str]) -> str:
    return "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))


_PLACE_RX = re.compile(
    rf"(?<![\w.])(?:{_alt(list(_COUNTRIES) + list(_CITIES) + list(_PLACE_ALIASES))})(?![\w])"
)

# Known organizations (clients / competitors / group companies); canonical spelling.
_KNOWN_ORGS: dict[str, str] = {
    "Tata Consultancy Services": "Tata Consultancy Services",
    "Tata Consultancy": "Tata Consultancy Services",
    "TCS": "Tata Consultancy Services",
    "Tata Sons": "Tata Sons",
    "Tata Group": "Tata Group",
    "Tata Motors": "Tata Motors",
    "Tata Steel": "Tata Steel",
    "Jaguar Land Rover": "Jaguar Land Rover",
    "Air India": "Air India",
    "Infosys": "Infosys",
    "Wipro": "Wipro",
    "HCLTech": "HCLTech",
    "HCL Technologies": "HCLTech",
    "Tech Mahindra": "Tech Mahindra",
    "LTIMindtree": "LTIMindtree",
    "Mphasis": "Mphasis",
    "Accenture": "Accenture",
    "Cognizant": "Cognizant",
    "Capgemini": "Capgemini",
    "Genpact": "Genpact",
    "IBM": "IBM",
    "Deloitte": "Deloitte",
    "PwC": "PwC",
    "KPMG": "KPMG",
    "Epic Systems": "Epic Systems",
    "DXC Technology": "DXC Technology",
    "DXC": "DXC Technology",
    "Computer Sciences Corporation": "CSC",
    "CSC": "CSC",
    "Marks & Spencer": "Marks & Spencer",
    "M&S": "Marks & Spencer",
    "Microsoft": "Microsoft",
    "Google": "Google",
    "Amazon": "Amazon",
    "AWS": "Amazon Web Services",
    "Apple": "Apple",
    "Meta": "Meta",
    "Nvidia": "Nvidia",
    "Transamerica": "Transamerica",
    "Nielsen": "Nielsen",
    "Bank of America": "Bank of America",
    "JPMorgan": "JPMorgan",
    "Citigroup": "Citigroup",
    "Morgan Stanley": "Morgan Stanley",
}
_KNOWN_ORG_RX = re.compile(rf"(?<![\w&])(?:{_alt(list(_KNOWN_ORGS))})(?![\w&])")
_ORG_SUFFIX = (
    r"(?:Ltd|Limited|Inc|Incorporated|Corp|Corporation|LLC|LLP|PLC|plc|AG|GmbH|SA|NV|Bank"
    r"|Group|Systems|Technologies|Technology|Holdings|Services|Solutions|Consulting|Partners)"
)
_ORG_RX = re.compile(rf"\b((?:[A-Z][\w&'.-]*\s+){{0,4}}[A-Z][\w&'.-]*\s+{_ORG_SUFFIX})\b\.?")
_LEGAL_SUFFIX_RX = re.compile(
    r"\s+(?:Ltd|Limited|Inc|Incorporated|Corp|Corporation|LLC|PLC|plc)\.?$"
)
_LEADING_NOISE = {
    "The",
    "A",
    "An",
    "In",
    "On",
    "At",
    "As",
    "By",
    "For",
    "From",
    "With",
    "And",
    "But",
    "After",
    "Before",
    "While",
    "When",
    "According",
    "Yesterday",
    "Today",
    "Last",
    "This",
    "That",
    "Its",
    "Their",
    "Shares",
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
}

_KNOWN_PEOPLE: dict[str, str] = {
    "K Krithivasan": "K Krithivasan",
    "K. Krithivasan": "K Krithivasan",
    "Krithivasan": "K Krithivasan",
    "N Chandrasekaran": "N Chandrasekaran",
    "N. Chandrasekaran": "N Chandrasekaran",
    "Chandrasekaran": "N Chandrasekaran",
    "Rajesh Gopinathan": "Rajesh Gopinathan",
    "Samir Seksaria": "Samir Seksaria",
    "Aarthi Subramanian": "Aarthi Subramanian",
    "Milind Lakkad": "Milind Lakkad",
    "Ratan Tata": "Ratan Tata",
    "Noel Tata": "Noel Tata",
}
_KNOWN_PEOPLE_RX = re.compile(rf"(?<![\w.])(?:{_alt(list(_KNOWN_PEOPLE))})(?!\w)")
_NAME = r"((?:[A-Z]\.?\s)?[A-Z][a-z]+(?:\s[A-Z][a-z]+){0,2})"
_ROLE_RX = re.compile(
    r"(?i:\b(?:CEO|CFO|COO|CTO|CHRO|chairman|chairperson|chief executive(?: officer)?"
    r"|managing director|founder|president|spokesperson|spokesman|judge|justice|minister"
    r"|senator|commissioner)\b),?\s+" + _NAME
)
_HONORIFIC_RX = re.compile(r"\b(?:Mr|Mrs|Ms|Dr|Shri|Smt)\.?\s+" + _NAME)
_PEOPLE_STOP = {"Said", "The", "Of", "And", "Has", "Was", "Is", "Will", "Who", "For"}


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        it = " ".join(it.split()).strip(" ,.;:")
        if it and it.lower() not in seen:
            seen.add(it.lower())
            out.append(it)
        if len(out) >= MAX_PER_TYPE:
            break
    return out


def _regulators(text: str) -> list[str]:
    hits: list[tuple[int, str]] = []
    for rx, canonical in _REGULATOR_RX:
        m = rx.search(text)
        if m:
            hits.append((m.start(), canonical))
    return _dedupe([c for _, c in sorted(hits)])


def _money(text: str) -> list[str]:
    hits: list[tuple[int, str]] = []
    covered: list[tuple[int, int]] = []
    for rx in _MONEY_RX:
        for m in rx.finditer(text):
            if any(a <= m.start() < b for a, b in covered):
                continue
            covered.append((m.start(), m.end()))
            hits.append((m.start(), m.group(0)))
    return _dedupe([h for _, h in sorted(hits)])


def _places(text: str) -> list[str]:
    return _dedupe([_PLACE_ALIASES.get(m.group(0), m.group(0)) for m in _PLACE_RX.finditer(text)])


def _organizations(text: str) -> list[str]:
    hits: list[tuple[int, str]] = []
    for m in _KNOWN_ORG_RX.finditer(text):
        hits.append((m.start(), _KNOWN_ORGS[m.group(0)]))
    for m in _ORG_RX.finditer(text):
        tokens = m.group(1).split()
        while len(tokens) > 1 and tokens[0] in _LEADING_NOISE:
            tokens.pop(0)
        if len(tokens) < 2:
            continue
        name = " ".join(tokens)
        canonical = _KNOWN_ORGS.get(_LEGAL_SUFFIX_RX.sub("", name), name)
        hits.append((m.start(), canonical))
    return _dedupe([h for _, h in sorted(hits)])


def _people(text: str) -> list[str]:
    hits: list[tuple[int, str]] = []
    for m in _KNOWN_PEOPLE_RX.finditer(text):
        hits.append((m.start(), _KNOWN_PEOPLE[m.group(0)]))
    for rx in (_ROLE_RX, _HONORIFIC_RX):
        for m in rx.finditer(text):
            tokens = [t for t in m.group(1).split() if t not in _PEOPLE_STOP]
            name = " ".join(tokens)
            if not name or name in _KNOWN_ORGS:
                continue
            hits.append((m.start(1), _KNOWN_PEOPLE.get(name, name)))
    return _dedupe([h for _, h in sorted(hits)])


def extract_entities(text: str) -> Entities:
    """Regulators, money, places, organizations and people found in ``text``."""
    text = text or ""
    return Entities(
        organizations=_organizations(text),
        people=_people(text),
        regulators=_regulators(text),
        places=_places(text),
        money=_money(text),
    )


def merge_entities(a: Entities, b: Entities) -> Entities:
    """Union of two entity sets (order: ``a`` first), deduped and capped."""
    return Entities(
        **{
            field: _dedupe(list(getattr(a, field)) + list(getattr(b, field)))
            for field in Entities.model_fields
        }
    )
