"""LCEL chain: prompt, real-chain construction (no network), fake heuristic chain (AC-003.5/.6)."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from vendor_risk_ai.config import Settings
from vendor_risk_ai.llm import chain as chain_mod
from vendor_risk_ai.llm.chain import (
    FAKE_MODEL,
    build_report_chain,
    build_report_chain_with_parser,
    finalize_report,
    format_inputs,
)
from vendor_risk_ai.llm.prompts import SYSTEM_PROMPT, build_prompt
from vendor_risk_ai.schemas.domain import ChunkRecord, Vendor
from vendor_risk_ai.schemas.report import RiskReport, score_to_level

from .conftest import CLEAN_DOC, GOING_CONCERN_DOC

VENDOR = Vendor(
    id=7,
    external_id="V-007",
    name="Acme Clearing Services",
    sector="post-trade",
    country="US",
    ticker="ACME:NYSE",
    criticality="critical",
    services="clearing",
)
LOW_VENDOR = VENDOR.model_copy(update={"criticality": "low", "ticker": None})


def _chunks(*texts: str) -> list[tuple[ChunkRecord, float]]:
    return [
        (ChunkRecord(document_id=i + 1, chunk_index=0, text=t, title=f"doc{i}"), 0.9 - i * 0.1)
        for i, t in enumerate(texts)
    ]


def _real_settings() -> Settings:
    return Settings(llm_provider="openai", openai_api_key="sk-test-not-real", _env_file=None)


def _walk(obj: Any, seen: set[int] | None = None) -> list[Any]:
    """Flatten a LangChain runnable graph (sequences, parallels, bindings)."""
    seen = seen if seen is not None else set()
    if id(obj) in seen:
        return []
    seen.add(id(obj))
    out = [obj]
    for attr in ("first", "last", "bound", "runnable"):
        if hasattr(obj, attr):
            out += _walk(getattr(obj, attr), seen)
    for attr in ("middle", "steps", "fallbacks"):
        if isinstance(getattr(obj, attr, None), list):
            for s in getattr(obj, attr):
                out += _walk(s, seen)
    if isinstance(getattr(obj, "steps__", None), dict):
        for s in obj.steps__.values():
            out += _walk(s, seen)
    return out


# ----------------------------------------------------------------------------- prompt
def test_prompt_variables_and_content() -> None:
    """AC-003.5: ChatPromptTemplate with the documented variables; rubric matches score_to_level."""
    prompt = build_prompt()
    assert isinstance(prompt, ChatPromptTemplate)
    assert set(prompt.input_variables) == {
        "vendor_profile",
        "market_data",
        "context",
        "focus_areas",
        "format_instructions",
    }
    for term in ("DORA", "Reg SCI", "FINRA", "SOX", "2023-17", "data_gaps"):
        assert term in SYSTEM_PROMPT
    for lo, level in ((0, "low"), (25, "medium"), (50, "high"), (75, "critical")):
        assert score_to_level(lo) == level
        assert f'"{level}"' in SYSTEM_PROMPT
    msgs = prompt.format_messages(**format_inputs({"vendor": VENDOR, "chunks": _chunks("abc")}))
    assert "Acme Clearing Services" in msgs[1].content and "[S1]" in msgs[1].content
    assert "No market data available." in msgs[1].content


# ----------------------------------------------------------------------------- real chain
def test_real_chain_uses_gpt41mini_temperature0_structured_output() -> None:
    """AC-003.5: prompt | ChatOpenAI(gpt-4.1-mini, T=0).with_structured_output(RiskReport)."""
    settings = _real_settings()
    assert not settings.use_fake_llm
    nodes = _walk(build_report_chain(settings))
    llms = [n for n in nodes if isinstance(n, ChatOpenAI)]
    assert len(llms) == 1
    assert llms[0].model_name == "gpt-4.1-mini"
    assert llms[0].temperature == 0.0
    assert any(isinstance(n, ChatPromptTemplate) for n in nodes)
    # with_structured_output binds the RiskReport tool and forces it
    bindings = [n for n in nodes if getattr(n, "bound", None) is llms[0]]
    kwargs = bindings[0].kwargs
    assert kwargs["tools"][0]["function"]["name"] == "RiskReport"
    assert kwargs["tool_choice"]["function"]["name"] == "RiskReport"


def test_parser_chain_uses_pydantic_output_parser() -> None:
    """AC-003.5: PydanticOutputParser fallback path is constructible offline."""
    nodes = _walk(build_report_chain_with_parser(_real_settings()))
    assert any(isinstance(n, PydanticOutputParser) for n in nodes)
    assert any(isinstance(n, ChatOpenAI) and n.model_name == "gpt-4.1-mini" for n in nodes)


def test_parser_path_end_to_end_with_fake_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """AC-003.5/.6: parser path output is post-processed (ids, model, sources, level)."""
    llm_json = RiskReport(
        vendor_id=999,
        vendor_name="WRONG",
        overall_risk_score=80,
        risk_level="low",
        financial_health={"summary": "weak"},
        risk_narrative="n",
        data_gaps=["no SOC 2"],
    ).model_dump_json()
    monkeypatch.setattr(chain_mod, "build_llm", lambda s: FakeListChatModel(responses=[llm_json]))
    report = build_report_chain_with_parser(_real_settings()).invoke(
        {
            "vendor": VENDOR,
            "market": None,
            "chunks": _chunks("x"),
            "focus_areas": [],
            "data_gaps": ["No market quote"],
        }
    )
    assert (report.vendor_id, report.vendor_name) == (7, "Acme Clearing Services")
    assert report.model == "gpt-4.1-mini"
    assert report.risk_level == "critical"
    assert [s.document_id for s in report.sources] == [1]
    assert report.data_gaps == ["no SOC 2", "No market quote"]


def test_finalize_report_rejects_missing_output_and_clamps() -> None:
    """AC-003.6: None from the LLM is an error; scores clamped to 0..100."""
    with pytest.raises(ValueError):
        finalize_report(None, {"vendor": VENDOR}, "m")
    raw = {
        "overall_risk_score": 140,
        "risk_level": "low",
        "vendor_id": 0,
        "vendor_name": "x",
        "financial_health": {"summary": "s"},
        "risk_narrative": "n",
    }
    r = finalize_report(raw, {"vendor": VENDOR, "chunks": []}, "m")
    assert r.overall_risk_score == 100 and r.risk_level == "critical"


# ----------------------------------------------------------------------------- fake chain
def test_fake_chain_flags_going_concern_document(settings: Settings) -> None:
    """AC-003.5/.6: offline heuristic chain yields a valid RiskReport with expected red flags."""
    assert settings.use_fake_llm
    market = {"price": 12.3, "currency": "USD", "change_pct": -14.2}
    report = build_report_chain(settings).invoke(
        {
            "vendor": VENDOR,
            "market": market,
            "chunks": _chunks(GOING_CONCERN_DOC),
            "focus_areas": ["liquidity"],
        }
    )
    assert isinstance(report, RiskReport)
    RiskReport.model_validate_json(report.model_dump_json())  # round-trips as JSON
    assert report.model == FAKE_MODEL
    assert report.vendor_id == 7 and report.vendor_name == "Acme Clearing Services"
    assert report.risk_level == "critical" and report.overall_risk_score >= 75
    assert report.risk_level == score_to_level(report.overall_risk_score)
    descriptions = " ".join(f.description.lower() for f in report.red_flags)
    for needle in (
        "going concern",
        "enforcement",
        "outage",
        "leverage",
        "concentration",
        "market move",
    ):
        assert needle in descriptions, needle
    gc = next(f for f in report.red_flags if "going concern" in f.description.lower())
    assert gc.severity == "critical" and gc.category == "financial"
    assert "substantial doubt" in gc.evidence and gc.evidence.startswith("[S1]")
    assert not any(f.category == "cyber" for f in report.red_flags)  # "covenant breach" != cyber
    assert report.red_flags[0].severity == "critical"  # sorted by severity
    fh = report.financial_health
    assert fh.liquidity.startswith("Stressed") and fh.profitability.startswith("Loss")
    assert fh.leverage.startswith("Elevated") and "-14.20%" in fh.market_signal
    refs = " ".join(r.regulation_reference for r in report.compliance_recommendations)
    assert "DORA" in refs and "OCC 2023-17" in refs
    assert report.compliance_recommendations[0].priority == "high"
    assert len(report.sources) == 1 and report.sources[0].score == pytest.approx(0.9)
    assert "liquidity" in report.risk_narrative


def test_fake_chain_clean_vendor_is_low_risk(settings: Settings) -> None:
    """AC-003.5: no red flags at a low-criticality vendor -> low risk; negations ignored."""
    doc = CLEAN_DOC + " There is no going concern doubt and no material weakness."
    report = build_report_chain(settings).invoke(
        {"vendor": LOW_VENDOR, "market": None, "chunks": _chunks(doc), "focus_areas": []}
    )
    assert report.red_flags == []
    assert report.risk_level == "low"
    assert report.financial_health.profitability.startswith("Profitable")
    assert report.financial_health.liquidity.startswith("Adequate")
    assert "No market data available for the vendor." in report.data_gaps


def test_fake_chain_without_chunks_reports_data_gaps(settings: Settings) -> None:
    """AC-003.6: no documents -> still a valid report, data gaps and collection recommendation."""
    report = build_report_chain(settings).invoke(
        {
            "vendor": VENDOR,
            "market": None,
            "chunks": [],
            "focus_areas": [],
            "data_gaps": ["No documents ingested."],
        }
    )
    assert report.sources == [] and "No documents ingested." in report.data_gaps
    assert any("audited financial" in r.recommendation for r in report.compliance_recommendations)
    assert report.overall_risk_score == 45.0  # critical base 35 + 10 for missing evidence
