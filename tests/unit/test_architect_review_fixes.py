"""Architect review 2026-11-06 fixes (specs/003 tasks.md "Open (architect review)")."""

from __future__ import annotations

from typing import Any

import pytest
from elastic_transport import ConnectionError as TransportConnectionError
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import Runnable, RunnableLambda, RunnableWithFallbacks
from langchain_openai import ChatOpenAI
from pydantic import ValidationError
from structlog.testing import capture_logs

from tcs_bot.config import Settings
from tcs_bot.llm import chain as chain_mod
from tcs_bot.llm.chain import FAKE_MODEL, build_report_chain, report_model_id
from tcs_bot.rag.embeddings import HashEmbedder
from tcs_bot.schemas.api import ReportRequest
from tcs_bot.schemas.domain import ChunkRecord, DocumentCreate, Vendor
from tcs_bot.schemas.report import RiskReport
from tcs_bot.search import elastic
from tcs_bot.services.report_service import ReportService, report_cache_key

from .conftest import CLEAN_DOC, GOING_CONCERN_DOC, FakeDocumentRepo, FakeIndexer, InMemoryCache
from .test_ai_chain import _walk

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


def _doc(content: str = GOING_CONCERN_DOC, title: str = "FY2025") -> DocumentCreate:
    return DocumentCreate(doc_type="financial_report", title=title, content=content)


def _real_settings(**kw: Any) -> Settings:
    return Settings(llm_provider="openai", openai_api_key="sk-test-not-real", _env_file=None, **kw)


def _report_keys(service: ReportService) -> list[str]:
    cache: InMemoryCache = service.cache  # type: ignore[assignment]
    return [k for k in cache.store if k.startswith("report:")]


# ----------------------------------------------------------------------------- 1. cache key
def test_cache_key_includes_top_k_and_model() -> None:
    """AC-003.8: different effective top_k or generating model -> different cache key."""
    refs = [(1, 0), (1, 1)]
    base = report_cache_key(1, refs, ["Cyber", "liquidity"], 6, "gpt-4.1-mini")
    assert base != report_cache_key(1, refs, ["cyber", "liquidity"], 10, "gpt-4.1-mini")
    assert base != report_cache_key(1, refs, ["cyber", "liquidity"], 6, FAKE_MODEL)
    # focus normalization + order independence, and id pairs == full records
    records = [ChunkRecord(document_id=d, chunk_index=i, text="t") for d, i in reversed(refs)]
    assert base == report_cache_key(1, records, [" liquidity", "cyber"], 6, "gpt-4.1-mini")


def test_generate_report_does_not_serve_other_top_k_or_model(service: ReportService) -> None:
    """AC-003.8: a top_k=2 report is not served for top_k=10; provider switch re-generates."""
    service.ingest_document(1, _doc(GOING_CONCERN_DOC * 3))
    small = service.generate_report(ReportRequest(vendor_id=1, top_k=2))
    big = service.generate_report(ReportRequest(vendor_id=1, top_k=10))
    assert len(small.sources) == 2 and len(big.sources) > 2
    assert len(service.reports.rows) == 2 and len(_report_keys(service)) == 2
    # explicit top_k equal to the default shares the default request's entry
    service.generate_report(ReportRequest(vendor_id=1))
    service.generate_report(ReportRequest(vendor_id=1, top_k=service.settings.rag_top_k))
    assert len(service.reports.rows) == 3

    # same chunks/focus/top_k but a chain stamping another model id -> cache miss
    other = build_report_chain(service.settings).with_config(
        metadata={chain_mod.REPORT_MODEL_METADATA_KEY: "gpt-4.1-mini"}
    )
    service.chain = other
    assert service.model_id == "gpt-4.1-mini"
    service.generate_report(ReportRequest(vendor_id=1, top_k=2))
    assert len(service.reports.rows) == 4


def test_report_model_id_from_chain_or_settings(settings: Settings) -> None:
    """AC-003.8: model id comes from the chain metadata; settings fallback for foreign chains."""
    assert report_model_id(build_report_chain(settings), settings) == FAKE_MODEL
    real = _real_settings()
    assert report_model_id(build_report_chain(real), real) == "gpt-4.1-mini"
    assert report_model_id(RunnableLambda(lambda x: x), real) == "gpt-4.1-mini"
    assert report_model_id(RunnableLambda(lambda x: x), settings) == FAKE_MODEL


# ----------------------------------------------------------------------------- 6. cache hit
def test_cache_hit_does_not_load_chunk_embeddings(service: ReportService) -> None:
    """AC-003.8: the cache key uses chunk ids only; a hit never loads texts/embeddings."""
    service.ingest_document(1, _doc())
    docs: FakeDocumentRepo = service.documents  # type: ignore[assignment]
    first = service.generate_report(ReportRequest(vendor_id=1))
    loads = docs.full_loads
    second = service.generate_report(ReportRequest(vendor_id=1))
    assert second == first and docs.full_loads == loads
    service.ingest_document(1, _doc(CLEAN_DOC, title="dup check"))
    assert docs.full_loads == loads  # ingest dedupe check is id-only too


# ----------------------------------------------------------------------------- 3. indexed
def test_document_indexed_means_es_accepted_chunks(service: ReportService) -> None:
    """AC-003.2: DocumentOut.indexed is True only if the indexer reports >0 chunks indexed."""
    service.chunk_indexer = FakeIndexer(accepted=False)  # index_chunks swallowed an ES error
    assert service.ingest_document(1, _doc()).indexed is False
    service.chunk_indexer = FakeIndexer()
    assert service.ingest_document(1, _doc(CLEAN_DOC)).indexed is True
    service.chunk_indexer = None
    assert service.ingest_document(2, _doc(CLEAN_DOC)).indexed is False


# ----------------------------------------------------------------------------- 2. re-embed
class OtherEmbedder(HashEmbedder):
    model = "other-embedder-v1"


def test_embed_all_pending_reembeds_other_model_and_persists(
    service: ReportService, indexer: FakeIndexer
) -> None:
    """AC-003.3: chunks embedded by another model are re-embedded, persisted and re-indexed."""
    service.ingest_document(1, _doc(GOING_CONCERN_DOC * 3))
    service.ingest_document(2, _doc(CLEAN_DOC))
    service.generate_report(ReportRequest(vendor_id=1))
    assert _report_keys(service)
    docs: FakeDocumentRepo = service.documents  # type: ignore[assignment]
    n_chunks = len(docs.chunks_for_vendor(1)) + len(docs.chunks_for_vendor(2))

    # provider switch: stored vectors are hash-embedder-v1, current embedder is another model
    service.embedder = OtherEmbedder()
    service.retriever.embedder = service.embedder
    indexer.indexed.clear()
    assert service.embed_all_pending() == 2  # documents processed
    stored = docs.chunks_for_vendor(1) + docs.chunks_for_vendor(2)
    assert len(stored) == n_chunks
    assert {c.embedding_model for c in stored} == {"other-embedder-v1"}
    assert all(len(c.embedding or []) == 1536 for c in stored)
    assert len(indexer.indexed) == n_chunks
    assert {c.embedding_model for c in indexer.indexed} == {"other-embedder-v1"}
    assert not _report_keys(service)  # vendor report caches invalidated
    assert service.embed_all_pending() == 0  # nothing stale any more


def test_embed_all_pending_reembed_flag_and_missing_embeddings(service: ReportService) -> None:
    """AC-003.3: reembed=True forces every document; chunks without embedding are filled."""
    service.ingest_document(1, _doc())
    service.ingest_document(2, _doc(CLEAN_DOC))
    assert service.embed_all_pending() == 0
    assert service.embed_all_pending(reembed=True) == 2

    docs: FakeDocumentRepo = service.documents  # type: ignore[assignment]
    did = next(iter(docs.chunks))
    docs.chunks[did] = [
        c.model_copy(update={"embedding": None, "embedding_model": None}) for c in docs.chunks[did]
    ]
    assert service.embed_all_pending() == 1
    assert all(c.embedding_model == "hash-embedder-v1" for c in docs.chunks[did])


def test_reembed_failure_is_skipped(service: ReportService) -> None:
    """AC-003.3: an embedding outage during re-embed skips the document, nothing persisted."""
    service.ingest_document(1, _doc())

    class Broken(OtherEmbedder):
        def embed(self, texts: list[str]):  # type: ignore[override]
            raise TimeoutError("embeddings down")

    service.embedder = Broken()
    with capture_logs():
        assert service.embed_all_pending() == 0
    stored = service.documents.chunks_for_vendor(1)
    assert {c.embedding_model for c in stored} == {"hash-embedder-v1"}


# ----------------------------------------------------------------------------- 4. ES down
class DownES:
    def search(self, **_: Any) -> Any:
        raise TransportConnectionError("connection refused")


def test_latest_quote_strict_raises_but_default_does_not(monkeypatch: pytest.MonkeyPatch) -> None:
    """Constitution §6: strict lookup distinguishes ES down (raise) from no quote (None)."""
    monkeypatch.setattr(elastic, "get_es", lambda: DownES())
    with capture_logs():
        assert elastic.latest_quote("IBM:NYSE") is None
        with pytest.raises(elastic.SearchUnavailable):
            elastic.latest_quote("IBM:NYSE", raise_on_error=True)
        with pytest.raises(elastic.SearchUnavailable):
            elastic.latest_quote_strict("IBM:NYSE")
    assert elastic.latest_quote_strict("") is None


def test_service_with_strict_lookup_reports_es_down_gap(
    service: ReportService, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Constitution §6: ES down -> report from MySQL chunks + explicit 'unavailable' data gap."""
    monkeypatch.setattr(elastic, "get_es", lambda: DownES())
    service.quote_lookup = elastic.latest_quote_strict
    service.ingest_document(1, _doc(CLEAN_DOC))
    with capture_logs():
        report = service.generate_report(ReportRequest(vendor_id=1))
    gaps = " ".join(report.data_gaps)
    assert "Market data unavailable (search backend error)" in gaps
    assert "No market quote found" not in gaps
    assert report.sources  # still generated from stored chunks


def test_build_default_service_wires_strict_quote_lookup() -> None:
    """Constitution §6: the real wiring uses the raising quote lookup."""
    import inspect

    from tcs_bot.services import report_service

    src = inspect.getsource(report_service.build_default_service)
    assert "quote_lookup=latest_quote_strict" in src


# ----------------------------------------------------------------------------- 5. fallback
def _fallbacks(chain: Runnable) -> list[RunnableWithFallbacks]:
    return [n for n in _walk(chain) if isinstance(n, RunnableWithFallbacks)]


def test_real_chain_has_pydantic_parser_fallback() -> None:
    """AC-003.5: structured-output chain falls back to the PydanticOutputParser path."""
    nodes = _walk(build_report_chain(_real_settings()))
    fbs = [n for n in nodes if isinstance(n, RunnableWithFallbacks)]
    assert len(fbs) == 1
    primary, fallback = _walk(fbs[0].runnable), _walk(fbs[0].fallbacks[0])
    assert not any(isinstance(n, PydanticOutputParser) for n in primary)
    assert any(isinstance(n, PydanticOutputParser) for n in fallback)
    llms = {id(n) for n in nodes if isinstance(n, ChatOpenAI)}
    assert len(llms) == 1  # one ChatOpenAI(gpt-4.1-mini, T=0) shared by both paths


class _NoToolsLLM(FakeListChatModel):
    """Plain-text fake: structured output (tool calling) always fails."""

    def with_structured_output(self, *_: Any, **__: Any) -> Runnable:  # type: ignore[override]
        def fail(_: Any) -> None:
            raise ValueError("model returned no tool call")

        return RunnableLambda(fail)


def test_fallback_path_produces_report_when_structured_output_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC-003.5/.6: structured call fails -> parser fallback -> validated, finalized RiskReport."""
    llm_json = RiskReport(
        vendor_id=999,
        vendor_name="WRONG",
        overall_risk_score=81,
        risk_level="low",
        financial_health={"summary": "weak"},
        risk_narrative="n",
    ).model_dump_json()
    monkeypatch.setattr(chain_mod, "build_llm", lambda s: _NoToolsLLM(responses=[llm_json]))
    report = build_report_chain(_real_settings()).invoke(
        {"vendor": VENDOR, "market": None, "chunks": [], "focus_areas": [], "data_gaps": ["g"]}
    )
    assert isinstance(report, RiskReport)
    assert (report.vendor_id, report.model, report.risk_level) == (7, "gpt-4.1-mini", "critical")
    assert "g" in report.data_gaps


# ----------------------------------------------------------------------------- 7. config
def test_config_pins_1536_dim_embedding_model() -> None:
    """AC-003.3 / constitution §3: only 1536-dim embedding models are accepted."""
    assert _real_settings().openai_embedding_model == "text-embedding-3-small"
    with pytest.raises(ValidationError, match="1536"):
        _real_settings(openai_embedding_model="text-embedding-3-large")


def test_config_warns_on_temperature_or_model_drift() -> None:
    """AC-003.5 / constitution §3: temperature != 0.0 or another chat model is logged loudly."""
    with capture_logs() as logs:
        _real_settings()
    assert not logs
    with capture_logs() as logs:
        s = _real_settings(openai_temperature=0.7, openai_chat_model="gpt-4o")
    assert s.openai_temperature == 0.7
    events = {e["event"] for e in logs}
    assert {"config.temperature_override", "config.chat_model_override"} <= events


# ----------------------------------------------------------------------------- 11. errors
def test_errors_live_in_neutral_module_and_are_reexported() -> None:
    """AC-003.1 / constitution §8: AppError hierarchy importable without FastAPI handlers."""
    from tcs_bot import errors
    from tcs_bot.api import errors as api_errors

    for name in (
        "AppError",
        "NotFoundError",
        "ValidationAppError",
        "UpstreamError",
        "DependencyUnavailable",
    ):
        assert getattr(api_errors, name) is getattr(errors, name)
    exc = errors.DependencyUnavailable("x", {"a": 1})
    assert isinstance(exc, errors.AppError) and exc.status_code == 503
    assert exc.to_dict() == {
        "error": {"code": "dependency_unavailable", "message": "x", "details": {"a": 1}}
    }
    assert callable(api_errors.register_error_handlers)


# ----------------------------------------------------------------------------- QA: empty doc
def test_empty_document_rejected_before_persisting(service: ReportService) -> None:
    """AC-003.2 / constitution §5: whitespace-only content -> 422 and no document row."""
    from tcs_bot.errors import ValidationAppError

    with pytest.raises(ValidationAppError) as ei:
        service.ingest_document(1, DocumentCreate(title="t", content=" \n\t ​ "))
    assert ei.value.code == "empty_document" and ei.value.status_code == 422
    assert service.documents.list_for_vendor(1) == []
