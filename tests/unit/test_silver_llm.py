"""Silver LLM path (AC-005.5): constructed only, never invoked against the network."""

from __future__ import annotations

from langchain_core.runnables import RunnableLambda

from tcs_bot.config import Settings
from tcs_bot.medallion.models import BronzeDocument, Entities, RiskSignal
from tcs_bot.medallion.silver.llm import (
    LlmSilverOutput,
    build_silver_chain,
    is_grounded,
    llm_extract,
)
from tcs_bot.medallion.silver.pipeline import build_rules_record

TEXT = (
    "SYNTHETIC FIXTURE: fictional test text. Tata Consultancy Services (TCS) said a fictional "
    "client paused a project. The client cited delivery delays in its statement. Analysts in the "
    "invented scenario expect TCS to address the issue next quarter, and the company said its "
    "team is working with the client to restore the schedule quickly and transparently."
)


def _doc() -> BronzeDocument:
    return BronzeDocument(
        id=7,
        url="https://www.example.com/fictional",
        canonical_url="https://example.com/fictional",
        depth=1,
        source="web",
        title="Fictional TCS project pause",
        raw_content=TEXT,
        content_type="text/plain",
    )


def _openai_settings() -> Settings:
    return Settings(llm_provider="openai", openai_api_key="sk-test-not-real", _env_file=None)


def test_chain_constructs_with_structured_output_no_network():
    """AC-005.5: prompt | ChatOpenAI(gpt-4.1-mini, T=0).with_structured_output(...)."""
    s = _openai_settings()
    assert not s.use_fake_llm
    chain = build_silver_chain(s)
    prompt, binding, *_ = chain.bound.steps
    assert set(prompt.input_variables) == {"source", "url", "title", "text"}
    llm = binding.bound  # RunnableBinding(ChatOpenAI, tools=[LlmSilverOutput])
    assert llm.model_name == "gpt-4.1-mini"
    assert llm.temperature == 0.0
    tool = binding.kwargs["tools"][0]["function"]
    assert tool["name"] == "LlmSilverOutput"
    assert {"risk_signals", "entities", "summary", "sentiment", "tcs_relevance"} <= set(
        tool["parameters"]["properties"]
    )


def test_fake_provider_returns_none(settings):
    assert settings.use_fake_llm
    assert llm_extract(_doc(), TEXT, settings) is None


def test_merge_with_rules_grounded_signals_only():
    s = _openai_settings()
    base = build_rules_record(_doc(), s)
    out = LlmSilverOutput(
        tcs_relevance=0.9,
        risk_signals=[
            RiskSignal(
                category="operational_delivery",
                severity="medium",
                keyword="project pause",
                evidence="Tata Consultancy Services (TCS) said a fictional client paused a "
                "project.",
            ),
            RiskSignal(  # hallucinated evidence -> dropped
                category="ethics",
                severity="critical",
                keyword="bribery",
                evidence="Executives were arrested for bribery in Atlantis.",
            ),
        ],
        entities=Entities(organizations=["Fictional Client Corp"]),
        summary="A fictional client paused a TCS project.",
        sentiment=-0.4,
    )
    rec = llm_extract(_doc(), base.clean_text, s, base=base, chain=RunnableLambda(lambda _: out))
    assert rec is not None
    assert rec.extractor == "gpt-4.1-mini"
    keywords = {sig.keyword for sig in rec.risk_signals}
    assert "project pause" in keywords and "bribery" not in keywords
    assert {sig.keyword for sig in base.risk_signals} <= keywords
    assert "operational_delivery" in rec.risk_categories
    assert rec.summary == "A fictional client paused a TCS project."
    assert rec.tcs_relevance == round((base.tcs_relevance + 0.9) / 2, 4)
    assert "Fictional Client Corp" in rec.entities.organizations
    assert rec.risk_score >= base.risk_score


def test_any_llm_error_falls_back_to_none():
    def boom(_):
        raise RuntimeError("network blocked")

    s = _openai_settings()
    assert llm_extract(_doc(), TEXT, s, chain=RunnableLambda(boom)) is None


def test_is_grounded():
    assert is_grounded("The client cited delivery delays", TEXT)
    assert not is_grounded("Completely unrelated invented quote about rockets", TEXT)
    assert not is_grounded("", TEXT)


def test_merge_applies_title_rule_and_keeps_rules_risk_score():
    """AC-005.6: a short TCS-titled item stays relevant after merge; LLM adds no inflation."""
    s = _openai_settings()
    doc = _doc().model_copy(
        update={
            "title": "TCS announces fictional layoffs",
            "raw_content": "SYNTHETIC FIXTURE: note.",
        }
    )
    base = build_rules_record(doc, s)
    assert base.word_count < 30 and base.relevant
    out = LlmSilverOutput(
        tcs_relevance=0.9, risk_signals=[], entities=Entities(), summary="s", sentiment=0.0
    )
    rec = llm_extract(doc, base.clean_text, s, base=base, chain=RunnableLambda(lambda _: out))
    assert rec is not None and rec.relevant is True
    assert rec.risk_score == base.risk_score
