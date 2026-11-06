"""Real OpenAI smoke test (AC-003.3, AC-003.5). Skipped unless a real key is configured.

Run locally:  OPENAI_API_KEY=sk-... LLM_PROVIDER=openai uv run pytest -m openai
In CI it runs only when the repository secret OPENAI_API_KEY is set.
"""

from __future__ import annotations

import os

import pytest

from tcs_bot.config import get_settings
from tcs_bot.llm.chain import build_report_chain
from tcs_bot.rag.embeddings import get_embedder
from tcs_bot.rag.retriever import Retriever
from tcs_bot.schemas import ChunkRecord, RiskReport, Vendor, score_to_level

pytestmark = pytest.mark.openai

DOCS = [
    "FY2025 annual report: revenue grew 4% but the auditor raised substantial doubt about the "
    "company's ability to continue as a going concern due to a covenant breach on its term loan.",
    "News: the vendor suffered a 9-hour outage of its market-data feed affecting three exchanges.",
    "Regulatory filing: the SOC 2 Type II report contains two exceptions in change management.",
    "The company opened a new office in Lisbon and hired 40 engineers.",
]


@pytest.fixture
def openai_settings(monkeypatch):
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key.startswith("sk-") or key == "sk-replace-me":
        pytest.skip("OPENAI_API_KEY not configured")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    get_settings.cache_clear()
    yield get_settings()
    get_settings.cache_clear()


def test_real_embeddings_and_structured_report(openai_settings):
    """AC-003.3 / AC-003.4 / AC-003.5: real embeddings + gpt-4.1-mini structured output."""
    s = openai_settings
    assert not s.use_fake_llm
    embedder = get_embedder(s)
    assert embedder.model == "text-embedding-3-small" and embedder.dim == 1536

    chunks = [ChunkRecord(document_id=1, chunk_index=i, text=t) for i, t in enumerate(DOCS)]
    ranked = Retriever(embedder).retrieve("going concern and covenant breach risk", chunks, 3)
    assert ranked[0][0].chunk_index == 0, "going-concern chunk should rank first"

    vendor = Vendor(id=1, external_id="SMOKE-1", name="Smoke Test Vendor", criticality="high")
    report = build_report_chain(s).invoke(
        {"vendor": vendor, "market": None, "chunks": ranked, "focus_areas": ["financial"]}
    )
    assert isinstance(report, RiskReport)
    assert report.vendor_id == 1 and report.vendor_name == "Smoke Test Vendor"
    assert report.risk_level == score_to_level(report.overall_risk_score)
    assert report.red_flags, "a going-concern + outage context must yield red flags"
