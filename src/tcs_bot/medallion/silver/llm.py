"""Optional LLM extraction for silver (AC-005.5, OpenAI path).

``ChatPromptTemplate | ChatOpenAI(gpt-4.1-mini, T=0).with_structured_output(LlmSilverOutput,
method="function_calling")``. The LLM result is merged with the deterministic rules record:
signals are unioned (LLM signals must be grounded in the document text), relevance and sentiment
are averaged, the LLM summary wins, entities are unioned. Any error -> ``None`` (caller keeps the
rules record), so silver never depends on OpenAI being reachable (constitution §6).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from tcs_bot.logging_setup import get_logger
from tcs_bot.medallion.models import BronzeDocument, Entities, RiskSignal, SilverRecord

if TYPE_CHECKING:
    from langchain_core.runnables import Runnable

    from tcs_bot.config import Settings

log = get_logger(__name__)

MAX_LLM_CHARS = 12_000
MIN_GROUNDING = 0.6  # share of evidence tokens that must appear in the document

SYSTEM_PROMPT = (
    "You are a third-party (vendor) risk analyst monitoring Tata Consultancy Services (TCS), the "
    "Indian IT services company (CEO K Krithivasan; parent Tata Sons). Extract structured risk "
    "intelligence from ONE document.\n"
    "Rules:\n"
    "- 'TCS' can also mean the Indian tax 'Tax Collected at Source'; such documents and stories "
    "only about other Tata companies (Tata Motors, Tata Steel, Air India, ...) have "
    "tcs_relevance close to 0.\n"
    "- tcs_relevance: 0..1, how much the document is about Tata Consultancy Services.\n"
    "- risk_signals: one per distinct risk; category from the allowed list; severity "
    "low|medium|high|critical; keyword = short concept (e.g. 'data breach'); evidence = a "
    "sentence copied verbatim from the document. Report allegations, denials and dismissed "
    "cases with lower severity. No signals if there is no risk.\n"
    "- entities: organizations, people, regulators, places, money amounts mentioned.\n"
    "- summary: 2-3 factual sentences. sentiment: -1 (very negative for TCS) .. 1.\n"
    "Use only the document; never invent facts."
)
HUMAN_PROMPT = "Source: {source}\nURL: {url}\nTitle: {title}\n\nDocument:\n{text}"


class LlmSilverOutput(BaseModel):
    """Structured output requested from the LLM."""

    tcs_relevance: float = Field(ge=0, le=1, description="0..1 aboutness for Tata Consultancy")
    risk_signals: list[RiskSignal] = Field(default_factory=list)
    entities: Entities = Field(default_factory=Entities)
    summary: str = Field(default="", description="2-3 factual sentences")
    sentiment: float = Field(ge=-1, le=1, default=0.0)


def build_silver_chain(settings: Settings) -> Runnable:
    """``prompt | ChatOpenAI(...).with_structured_output(LlmSilverOutput)`` (no network here)."""
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)])
    llm = ChatOpenAI(
        model=settings.openai_chat_model,
        temperature=0.0,
        api_key=settings.openai_api_key,
        max_retries=1,
        timeout=60,
    )
    structured = llm.with_structured_output(LlmSilverOutput, method="function_calling")
    return (prompt | structured).with_config(run_name="silver_llm_extract")


_TOKEN = re.compile(r"\w+")


def is_grounded(evidence: str, text: str) -> bool:
    """True if ``evidence`` (mostly) occurs in ``text`` - drops hallucinated quotes."""
    ev = " ".join(evidence.split()).strip(" .…\"'").lower()
    if not ev:
        return False
    hay = " ".join(text.split()).lower()
    if ev in hay:
        return True
    toks = _TOKEN.findall(ev)
    if not toks:
        return False
    vocab = set(_TOKEN.findall(hay))
    return sum(t in vocab for t in toks) / len(toks) >= MIN_GROUNDING


def merge_llm_output(
    base: SilverRecord, out: LlmSilverOutput, text: str, model: str, threshold: float
) -> SilverRecord:
    """Merge an LLM result into the rules-based record."""
    from tcs_bot.medallion.silver.classify import MAX_EVIDENCE_CHARS, categories_of, score_signals
    from tcs_bot.medallion.silver.entities import merge_entities

    seen = {(s.category, s.keyword.lower()) for s in base.risk_signals}
    signals = list(base.risk_signals)
    for s in out.risk_signals:
        key = (s.category, s.keyword.strip().lower())
        if key in seen or not is_grounded(s.evidence, text):
            continue
        seen.add(key)
        ev = " ".join(s.evidence.split())[:MAX_EVIDENCE_CHARS]
        signals.append(s.model_copy(update={"keyword": s.keyword.strip(), "evidence": ev}))
    relevance = round((base.tcs_relevance + out.tcs_relevance) / 2.0, 4)
    return base.model_copy(
        update={
            "risk_signals": signals,
            "risk_categories": categories_of(signals),
            "risk_score": max(base.risk_score, score_signals(signals, base.title)),
            "tcs_relevance": relevance,
            "relevant": relevance >= threshold and base.word_count >= 30,
            "sentiment": round((base.sentiment + out.sentiment) / 2.0, 4),
            "summary": out.summary.strip() or base.summary,
            "entities": merge_entities(base.entities, out.entities),
            "extractor": model,
        }
    )


def llm_extract(
    doc: BronzeDocument,
    text: str,
    settings: Settings,
    base: SilverRecord | None = None,
    chain: Any | None = None,
) -> SilverRecord | None:
    """LLM-enriched :class:`SilverRecord`, or ``None`` (fake provider or any error)."""
    if settings.use_fake_llm and chain is None:
        return None
    try:
        if base is None:
            from tcs_bot.medallion.silver.pipeline import build_rules_record

            base = build_rules_record(doc, settings)
        chain = chain if chain is not None else build_silver_chain(settings)
        raw = chain.invoke(
            {
                "source": doc.source,
                "url": doc.url,
                "title": base.title,
                "text": (text or base.clean_text)[:MAX_LLM_CHARS],
            }
        )
        out = raw if isinstance(raw, LlmSilverOutput) else LlmSilverOutput.model_validate(raw)
        return merge_llm_output(
            base,
            out,
            text or base.clean_text,
            settings.openai_chat_model,
            settings.medallion_relevance_threshold,
        )
    except Exception as exc:  # noqa: BLE001 - any LLM failure falls back to rules
        log.warning("silver.llm.failed", doc_id=doc.id, error=str(exc)[:200])
        return None
