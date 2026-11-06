"""LCEL report chain (AC-003.5).

Input dict::

    {"vendor": Vendor, "market": dict | None, "chunks": [(ChunkRecord, score)],
     "focus_areas": [str], "data_gaps": [str]  # optional, merged into the report}

Output: a validated ``RiskReport``. Every path ends in ``finalize_report`` which forces
vendor_id / vendor_name / model / sources from the real inputs and makes ``risk_level``
consistent with ``score_to_level``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from langchain_core.runnables import Runnable, RunnableLambda, RunnableParallel, RunnablePassthrough

from tcs_bot.llm.prompts import STRUCTURED_OUTPUT_INSTRUCTIONS, build_prompt
from tcs_bot.schemas.domain import ChunkRecord, Vendor
from tcs_bot.schemas.report import (
    ComplianceRecommendation,
    FinancialHealth,
    RedFlag,
    RiskReport,
    SourceRef,
    score_to_level,
)

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI

    from tcs_bot.config import Settings

FAKE_MODEL = "fake-heuristic-v1"
REPORT_MODEL_METADATA_KEY = "report_model"
_MAX_CONTEXT_CHARS_PER_CHUNK = 1500

ScoredChunks = list[tuple[ChunkRecord, float]]


# --------------------------------------------------------------------------- input formatting
def _vendor_profile(v: Vendor) -> str:
    return "\n".join(
        [
            f"- vendor_id: {v.id}",
            f"- name: {v.name}",
            f"- external_id: {v.external_id}",
            f"- sector: {v.sector}",
            f"- country: {v.country}",
            f"- ticker: {v.ticker or 'not listed'}",
            f"- criticality to the firm: {v.criticality}",
            f"- services provided: {v.services or 'unknown'}",
        ]
    )


def _market_data(market: Mapping[str, Any] | None) -> str:
    if not market:
        return "No market data available."
    return json.dumps(dict(market), default=str, indent=2, sort_keys=True)


def _context(chunks: ScoredChunks) -> str:
    if not chunks:
        return "No document excerpts available for this vendor."
    blocks = []
    for i, (c, score) in enumerate(chunks, start=1):
        header = (
            f"[S{i}] document_id={c.document_id} chunk={c.chunk_index} "
            f"type={c.doc_type or 'unknown'} title={c.title or 'untitled'} relevance={score:.3f}"
        )
        blocks.append(f"{header}\n{c.text[:_MAX_CONTEXT_CHARS_PER_CHUNK]}")
    return "\n\n".join(blocks)


def format_inputs(inputs: Mapping[str, Any], format_instructions: str = "") -> dict[str, str]:
    focus = inputs.get("focus_areas") or []
    return {
        "vendor_profile": _vendor_profile(inputs["vendor"]),
        "market_data": _market_data(inputs.get("market")),
        "context": _context(inputs.get("chunks") or []),
        "focus_areas": ", ".join(focus) if focus else "general (all risk domains)",
        "format_instructions": format_instructions,
    }


# --------------------------------------------------------------------------- post-processing
def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for item in items:
        key = item.strip()
        if key and key not in seen:
            seen.add(key)
            out.append(key)
    return out


def finalize_report(
    raw: RiskReport | Mapping[str, Any] | None, inputs: Mapping[str, Any], model: str
) -> RiskReport:
    """Force identity/provenance fields from the real inputs; make risk_level consistent."""
    if raw is None:
        raise ValueError("LLM returned no structured output")
    vendor: Vendor = inputs["vendor"]
    chunks: ScoredChunks = inputs.get("chunks") or []
    data = raw.model_dump() if isinstance(raw, RiskReport) else dict(raw)

    score = round(min(100.0, max(0.0, float(data.get("overall_risk_score") or 0.0))), 2)
    data.update(
        vendor_id=vendor.id,
        vendor_name=vendor.name,
        model=model,
        overall_risk_score=score,
        risk_level=score_to_level(score),
        sources=[
            SourceRef(
                document_id=c.document_id,
                chunk_index=c.chunk_index,
                title=c.title,
                score=round(min(1.0, max(-1.0, float(s))), 6),
            )
            for c, s in chunks
        ],
        data_gaps=_dedupe(list(data.get("data_gaps") or []) + list(inputs.get("data_gaps") or [])),
        generated_at=datetime.now(UTC),
    )
    return RiskReport.model_validate(data)


# --------------------------------------------------------------------------- real (OpenAI) path
def build_llm(settings: Settings) -> ChatOpenAI:
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.openai_chat_model,
        temperature=settings.openai_temperature,
        api_key=settings.openai_api_key,
        max_retries=2,
        timeout=60,
    )


def _with_finalize(generator: Runnable, model: str) -> Runnable[dict, RiskReport]:
    """Run ``generator`` on the inputs, then post-process with the original inputs."""
    return RunnableParallel(inputs=RunnablePassthrough(), report=generator) | RunnableLambda(
        lambda d: finalize_report(d["report"], d["inputs"], model)
    ).with_config(run_name="finalize_report")


def _structured_generator(llm: ChatOpenAI) -> Runnable:
    """``format | prompt | llm.with_structured_output(RiskReport)`` (function calling)."""
    return (
        RunnableLambda(lambda x: format_inputs(x, STRUCTURED_OUTPUT_INSTRUCTIONS)).with_config(
            run_name="format_inputs"
        )
        | build_prompt()
        | llm.with_structured_output(RiskReport, method="function_calling")
    )


def _parser_generator(llm: ChatOpenAI) -> Runnable:
    """``format | prompt | llm | PydanticOutputParser(RiskReport)`` (plain completion)."""
    from langchain_core.output_parsers import PydanticOutputParser

    parser = PydanticOutputParser(pydantic_object=RiskReport)
    instructions = parser.get_format_instructions()
    return (
        RunnableLambda(lambda x: format_inputs(x, instructions)).with_config(
            run_name="format_inputs"
        )
        | build_prompt()
        | llm
        | parser
    )


def _tag_model(chain: Runnable[dict, RiskReport], run_name: str, model: str) -> Runnable:
    return chain.with_config(run_name=run_name, metadata={REPORT_MODEL_METADATA_KEY: model})


def report_model_id(chain: Runnable, settings: Settings) -> str:
    """Model id the chain stamps on its reports (``RiskReport.model``), used in cache keys.

    Read from the chain's metadata (set by the builders here); falls back to the settings for
    injected chains without it.
    """
    config = getattr(chain, "config", None) or {}
    model = (config.get("metadata") or {}).get(REPORT_MODEL_METADATA_KEY)
    if isinstance(model, str) and model:
        return model
    return FAKE_MODEL if settings.use_fake_llm else settings.openai_chat_model


def build_report_chain(settings: Settings) -> Runnable[dict, RiskReport]:
    """``format | ChatPromptTemplate | ChatOpenAI(...).with_structured_output(RiskReport)``.

    Uses function calling (non-strict) because RiskReport has defaulted fields that OpenAI's
    strict json_schema mode rejects. If the structured call fails (no tool call, invalid tool
    arguments, ...), ``.with_fallbacks`` retries with the ``PydanticOutputParser`` path on the
    same ``ChatOpenAI`` instance (AC-003.5). Returns the offline heuristic chain when
    ``settings.use_fake_llm``.
    """
    if settings.use_fake_llm:
        return build_fake_chain()
    llm = build_llm(settings)
    generator = _structured_generator(llm).with_fallbacks([_parser_generator(llm)])
    return _tag_model(
        _with_finalize(generator, settings.openai_chat_model),
        "risk_report_chain",
        settings.openai_chat_model,
    )


def build_report_chain_with_parser(settings: Settings) -> Runnable[dict, RiskReport]:
    """Parser-only chain: plain completion + ``PydanticOutputParser`` (models without tools)."""
    if settings.use_fake_llm:
        return build_fake_chain()
    return _tag_model(
        _with_finalize(_parser_generator(build_llm(settings)), settings.openai_chat_model),
        "risk_report_chain_parser",
        settings.openai_chat_model,
    )


# --------------------------------------------------------------------------- fake (offline) path
@dataclass(frozen=True)
class FlagRule:
    key: str
    category: str
    severity: str
    weight: float
    patterns: tuple[str, ...]
    description: str
    recommendation: str
    regulation: str
    priority: str = "high"


RED_FLAG_RULES: tuple[FlagRule, ...] = (
    FlagRule(
        "going_concern",
        "financial",
        "critical",
        30,
        (r"going[- ]concern", r"substantial doubt"),
        "Substantial doubt disclosed about the vendor's ability to continue as a going concern.",
        "Activate the exit strategy: confirm substitutability, test transition plans and data "
        "return, and escalate to the risk committee.",
        "DORA Art. 28(8) exit strategies; OCC 2023-17 (termination)",
    ),
    FlagRule(
        "enforcement",
        "compliance",
        "high",
        15,
        (
            r"\bfine[sd]?\b",
            r"\bpenalt(?:y|ies)\b",
            r"enforcement action",
            r"consent order",
            r"cease[- ]and[- ]desist",
            r"\bsanction(?:s|ed)?\b",
        ),
        "Regulatory enforcement, fines or penalties reported against the vendor.",
        "Obtain the regulatory orders and the vendor's remediation plan; re-perform enhanced "
        "due diligence and track remediation milestones.",
        "OCC 2023-17 (ongoing monitoring); FINRA Rule 3110",
    ),
    FlagRule(
        "control_exceptions",
        "compliance",
        "high",
        12,
        (
            r"soc ?[12][^.]{0,60}exception",
            r"exceptions? (?:were )?noted",
            r"qualified opinion",
            r"adverse opinion",
            r"material weakness",
            r"significant deficienc",
        ),
        "Control deficiencies reported (SOC exceptions, qualified opinion or material weakness).",
        "Request the full SOC 2 Type II report, bridge letter and evidence that exceptions are "
        "remediated; assess impact on our ICFR reliance.",
        "SOX 404; DORA Art. 28(4)",
    ),
    FlagRule(
        "cyber_incident",
        "cyber",
        "critical",
        18,
        (
            r"(?:data|security|cyber) breach",
            r"breach of (?:its |our )?(?:systems?|networks?|data)",
            r"ransomware",
            r"cyber[- ]?attack",
            r"unauthori[sz]ed access",
            r"exfiltrat",
        ),
        "Cybersecurity incident (breach / ransomware / unauthorized access) involving the vendor.",
        "Obtain the incident report and root-cause analysis, confirm containment and data "
        "exposure, and assess our own incident-notification obligations.",
        "DORA Art. 19 (major ICT incident reporting); SEC Reg S-P",
    ),
    FlagRule(
        "outage",
        "operational",
        "high",
        12,
        (
            r"\boutages?\b",
            r"service disruption",
            r"downtime",
            r"failover fail",
            r"business continuity (?:plan )?fail",
        ),
        "Material service outage or disruption reported.",
        "Review BCP/DR test results, SLA performance and credits; assess impact on SCI systems "
        "and require a resilience improvement plan.",
        "SEC Reg SCI; DORA Art. 11; FINRA Rule 4370",
    ),
    FlagRule(
        "leverage",
        "financial",
        "high",
        12,
        (
            r"covenant (?:breach|waiver|violation|default)",
            r"\bdowngrade[sd]?\b",
            r"high(?:ly)? leverage",
            r"leverage ratio",
            r"debt[- ]to[- ]equity",
            r"liquidity (?:shortfall|constraint|pressure)",
            r"\bdefault(?:ed)? on\b",
        ),
        "Leverage, covenant or credit-rating pressure.",
        "Move the vendor to quarterly financial monitoring; track covenant compliance and "
        "rating actions.",
        "OCC 2023-17 (financial condition)",
        "medium",
    ),
    FlagRule(
        "concentration",
        "concentration",
        "medium",
        8,
        (
            r"customer concentration",
            r"single (?:largest )?(?:customer|client)",
            r"\d{2,3}(?:\.\d+)?\s?% of (?:total |its )?revenues?",
            r"largest (?:customer|client)",
        ),
        "Revenue concentration in a small number of customers.",
        "Assess the vendor's dependency on few clients and our own concentration exposure; "
        "document it in the ICT third-party register.",
        "DORA Art. 29 (concentration risk); OCC 2023-17",
        "medium",
    ),
    FlagRule(
        "litigation",
        "reputational",
        "medium",
        8,
        (r"litigation", r"lawsuits?", r"class action", r"\bsued\b", r"legal proceedings"),
        "Material litigation or legal proceedings disclosed.",
        "Obtain counsel's assessment of the litigation and its potential impact on service "
        "delivery.",
        "OCC 2023-17 (legal and regulatory compliance)",
        "medium",
    ),
    FlagRule(
        "late_filing",
        "compliance",
        "high",
        10,
        (
            r"late filing",
            r"\bNT 10-[KQ]\b",
            r"delayed (?:annual |quarterly )?(?:report|filing)",
            r"failed to (?:timely )?file",
            r"notification of late filing",
        ),
        "Late or delayed regulatory/financial filing.",
        "Obtain the reason for the delayed filing and the expected filing date; flag for "
        "accounting-quality review.",
        "SEC Rule 12b-25; SOX 302",
    ),
    FlagRule(
        "auditor_resignation",
        "financial",
        "high",
        12,
        (
            r"auditor (?:has )?resign",
            r"resignation of (?:the |its )?(?:independent )?"
            r"(?:registered public accounting firm|auditor)",
            r"dismiss(?:ed|al) (?:of )?(?:the |its )?auditor",
            r"change in (?:the )?(?:certifying )?accountant",
        ),
        "Auditor resignation or dismissal disclosed.",
        "Inquire about the reasons for the auditor change and any disagreements disclosed.",
        "SEC Form 8-K Item 4.01; SOX 404",
    ),
)

_COMPILED = {r.key: [re.compile(p, re.IGNORECASE) for p in r.patterns] for r in RED_FLAG_RULES}
_NEGATION = re.compile(r"\b(?:no|not|without|none|nor|never|zero)\b", re.IGNORECASE)
_SENTENCE_SPLIT = re.compile(r"[.!?;\n]")
_SEVERITY_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}
CRITICALITY_BASE = {"low": 10.0, "medium": 20.0, "high": 28.0, "critical": 35.0}


def _snippet(text: str, start: int, end: int, before: int = 80, after: int = 140) -> str:
    lo, hi = max(0, start - before), min(len(text), end + after)
    s = " ".join(text[lo:hi].split())
    return f"{'...' if lo > 0 else ''}{s}{'...' if hi < len(text) else ''}"


def _find(rule: FlagRule, text: str) -> re.Match[str] | None:
    for pat in _COMPILED[rule.key]:
        for m in pat.finditer(text):
            # negation in the same sentence just before the match, or inside the match itself
            before = _SENTENCE_SPLIT.split(text[max(0, m.start() - 40) : m.start()])[-1]
            if not _NEGATION.search(before + m.group(0)):
                return m
    return None


def _market_adjustment(market: Mapping[str, Any] | None) -> tuple[float, str, RedFlag | None]:
    if not market:
        return 0.0, "unknown - no market data available", None
    pct = market.get("change_pct")
    price, cur = market.get("price"), market.get("currency") or ""
    pe = market.get("pe_ratio")
    parts = []
    if price is not None:
        parts.append(f"last price {price} {cur}".strip())
    if pct is not None:
        parts.append(f"daily change {float(pct):+.2f}%")
    if pe is not None:
        parts.append(f"P/E {pe}")
    desc = ", ".join(parts) or "quote available but without price fields"
    add, flag = 0.0, None
    if pct is not None:
        pct = float(pct)
        if pct <= -10:
            add = 12.0
            flag = RedFlag(
                category="financial",
                severity="high",
                description=f"Sharp negative market move ({pct:+.2f}%) in the vendor's stock.",
                evidence=f"Market quote: {desc}",
            )
        elif pct <= -5:
            add = 6.0
            flag = RedFlag(
                category="financial",
                severity="medium",
                description=f"Notable negative market move ({pct:+.2f}%) in the vendor's stock.",
                evidence=f"Market quote: {desc}",
            )
        elif pct <= -2:
            add = 2.0
    if pe is not None and float(pe) < 0:
        add += 3.0
    signal = "negative" if add >= 6 else "neutral" if add > 0 else "no adverse signal"
    return add, f"{signal}: {desc}", flag


def _any(chunks: ScoredChunks, pattern: str) -> bool:
    rx = re.compile(pattern, re.IGNORECASE)
    return any(rx.search(c.text) for c, _ in chunks)


def heuristic_report(inputs: Mapping[str, Any]) -> RiskReport:
    """Deterministic offline analyzer: score = criticality base + weighted flags + market."""
    vendor: Vendor = inputs["vendor"]
    chunks: ScoredChunks = inputs.get("chunks") or []
    focus: list[str] = list(inputs.get("focus_areas") or [])

    hits: dict[str, tuple[FlagRule, ChunkRecord, re.Match[str], int]] = {}
    for rule in RED_FLAG_RULES:
        for idx, (chunk, _score) in enumerate(chunks, start=1):
            m = _find(rule, chunk.text)
            if m is not None:
                hits[rule.key] = (rule, chunk, m, idx)
                break

    red_flags: list[RedFlag] = []
    score = CRITICALITY_BASE.get(vendor.criticality, 20.0)
    for rule, chunk, m, idx in hits.values():
        score += rule.weight
        red_flags.append(
            RedFlag(
                category=rule.category,  # type: ignore[arg-type]
                severity=rule.severity,  # type: ignore[arg-type]
                description=rule.description,
                evidence=f"[S{idx}] {_snippet(chunk.text, m.start(), m.end())}",
            )
        )

    market_add, market_signal, market_flag = _market_adjustment(inputs.get("market"))
    score += market_add
    if market_flag is not None:
        red_flags.append(market_flag)
    if not chunks:
        score += 10.0  # unknown is not safe: lack of evidence raises residual risk
    score = round(min(100.0, max(0.0, score)), 2)
    red_flags.sort(key=lambda f: -_SEVERITY_ORDER[f.severity])

    # financial health (deterministic, evidence-driven)
    if "going_concern" in hits:
        liquidity = "Stressed: going-concern doubt disclosed."
    elif _any(chunks, r"liquidity (?:shortfall|constraint|pressure)|cash burn"):
        liquidity = "Under pressure: liquidity constraints mentioned."
    elif _any(chunks, r"cash and cash equivalents|strong liquidity|undrawn (?:credit|revolv)"):
        liquidity = "Adequate: cash / liquidity resources are described."
    else:
        liquidity = "unknown"
    if _any(chunks, r"net loss|operating loss|loss before tax"):
        profitability = "Loss-making per the provided documents."
    elif _any(chunks, r"net income|operating profit|profitab|ebitda margin"):
        profitability = "Profitable per the provided documents."
    else:
        profitability = "unknown"
    if "leverage" in hits:
        leverage = "Elevated: covenant, rating or leverage pressure reported."
    elif _any(chunks, r"\bdebt\b|leverage|borrowing"):
        leverage = "Debt is mentioned without covenant or rating concerns."
    else:
        leverage = "unknown"
    fin_flags = [f for f in red_flags if f.category == "financial"]
    summary = (
        f"{len(fin_flags)} financial red flag(s) identified."
        if fin_flags
        else "No financial red flags identified in the retrieved documents."
        if chunks
        else "No financial documents available; financial health could not be assessed."
    )

    # compliance recommendations
    recs: list[ComplianceRecommendation] = [
        ComplianceRecommendation(
            priority=rule.priority,  # type: ignore[arg-type]
            recommendation=rule.recommendation,
            regulation_reference=rule.regulation,
        )
        for rule, *_ in hits.values()
    ]
    if not chunks:
        recs.append(
            ComplianceRecommendation(
                priority="high",
                recommendation="Collect current audited financial statements, the latest SOC 2 "
                "Type II report and recent regulatory filings before relying on this vendor.",
                regulation_reference="OCC 2023-17 (due diligence)",
            )
        )
    if vendor.criticality in ("high", "critical"):
        recs.append(
            ComplianceRecommendation(
                priority="medium",
                recommendation="Keep the vendor in the register of information as supporting a "
                "critical or important function, with annual due diligence and a tested exit plan.",
                regulation_reference="DORA Art. 28(3)",
            )
        )
    else:
        recs.append(
            ComplianceRecommendation(
                priority="low",
                recommendation="Continue annual monitoring proportionate to criticality.",
                regulation_reference="OCC 2023-17 (ongoing monitoring)",
            )
        )
    recs.sort(key=lambda r: {"high": 0, "medium": 1, "low": 2}[r.priority])

    level = score_to_level(score)
    flag_names = ", ".join(r.key.replace("_", " ") for r, *_ in hits.values()) or "none"
    narrative = (
        f"{vendor.name} ({vendor.sector}, {vendor.criticality} criticality) has an overall risk "
        f"score of {score:.0f}/100 ({level}). Document red flags: {flag_names}. "
        f"Market signal: {market_signal}. "
        f"The assessment is based on {len(chunks)} retrieved document excerpt(s)"
        + (f" with focus on {', '.join(focus)}." if focus else ".")
        + " Generated by the offline heuristic analyzer; validate with an analyst review."
    )

    data_gaps = []
    caller_gaps = " ".join(inputs.get("data_gaps") or []).lower()
    if not inputs.get("market") and "market" not in caller_gaps:
        data_gaps.append("No market data available for the vendor.")
    if chunks and not _any(chunks, r"soc ?[12]|iso ?27001"):
        data_gaps.append("No SOC 2 / ISO 27001 assurance report found in the retrieved context.")

    return RiskReport(
        vendor_id=vendor.id,
        vendor_name=vendor.name,
        overall_risk_score=score,
        risk_level=level,
        red_flags=red_flags,
        financial_health=FinancialHealth(
            summary=summary,
            liquidity=liquidity,
            profitability=profitability,
            leverage=leverage,
            market_signal=market_signal,
        ),
        compliance_recommendations=recs,
        risk_narrative=narrative,
        data_gaps=data_gaps,
        model=FAKE_MODEL,
    )


def build_fake_chain() -> Runnable[dict, RiskReport]:
    return _tag_model(
        _with_finalize(
            RunnableLambda(heuristic_report).with_config(run_name="heuristic_analyzer"), FAKE_MODEL
        ),
        "risk_report_chain_fake",
        FAKE_MODEL,
    )
