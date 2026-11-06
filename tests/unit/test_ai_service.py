"""ReportService orchestration with in-memory fakes (AC-003.2/.4/.5/.8)."""

from __future__ import annotations

import pytest
from langchain_core.runnables import RunnableLambda

from tcs_bot.api.errors import (
    DependencyUnavailable,
    NotFoundError,
    UpstreamError,
    ValidationAppError,
)
from tcs_bot.schemas.api import ReportRequest
from tcs_bot.schemas.domain import DocumentCreate
from tcs_bot.services.report_service import ReportService, report_cache_key

from .conftest import CLEAN_DOC, GOING_CONCERN_DOC, FakeIndexer, FakeQuotes, InMemoryCache


def _doc(content: str = GOING_CONCERN_DOC, title: str = "FY2025 annual report") -> DocumentCreate:
    return DocumentCreate(doc_type="financial_report", title=title, content=content)


def test_ingest_chunks_embeds_indexes_and_dedupes(
    service: ReportService, indexer: FakeIndexer
) -> None:
    """AC-003.2: ingest normalizes, chunks, embeds, saves chunks and indexes in ES."""
    out = service.ingest_document(1, _doc(GOING_CONCERN_DOC * 3))
    assert out.vendor_id == 1 and out.chunks >= 2 and out.indexed is True
    stored = service.documents.chunks_for_vendor(1)
    assert len(stored) == out.chunks
    assert all(
        len(c.embedding or []) == 1536 and c.embedding_model == "hash-embedder-v1" for c in stored
    )
    assert all(len(c.text) <= service.settings.rag_chunk_size for c in stored)
    assert indexer.calls == [(1, out.chunks)]
    again = service.ingest_document(1, _doc(GOING_CONCERN_DOC * 3))  # same sha256
    assert again.id == out.id and again.chunks == out.chunks
    assert len(service.documents.chunks_for_vendor(1)) == out.chunks


def test_ingest_unknown_vendor_and_es_failure(service: ReportService) -> None:
    """AC-003.2: unknown vendor -> 404; ES failure is logged, ingestion still succeeds."""
    with pytest.raises(NotFoundError):
        service.ingest_document(999, _doc())
    service.chunk_indexer = FakeIndexer(fail=True)
    out = service.ingest_document(2, _doc(CLEAN_DOC))
    assert out.indexed is False and out.chunks == 1


def test_generate_cache_hit_and_latest(service: ReportService, quotes: FakeQuotes) -> None:
    """AC-003.8: ingest -> generate (persisted + cached) -> cache hit -> latest."""
    service.ingest_document(1, _doc())
    req = ReportRequest(vendor_id=1, focus_areas=["liquidity"])
    first = service.generate_report(req)
    assert first.risk_level == "critical" and first.model == "fake-heuristic-v1"
    assert first.sources and quotes.calls == ["ACME:NYSE"]
    assert len(service.reports.rows) == 1

    cache: InMemoryCache = service.cache  # type: ignore[assignment]
    k = service.settings.rag_top_k
    key = report_cache_key(
        1, service.documents.chunks_for_vendor(1), ["liquidity"], k, "fake-heuristic-v1"
    )
    assert key.startswith("report:1:") and key.endswith(f":k{k}:fake-heuristic-v1:liquidity")
    assert key in cache.store

    second = service.generate_report(req)
    assert second == first  # served from cache, not regenerated
    assert len(service.reports.rows) == 1 and quotes.calls == ["ACME:NYSE"]

    third = service.generate_report(
        ReportRequest(vendor_id=1, focus_areas=["liquidity"], use_cache=False)
    )
    assert len(service.reports.rows) == 2 and third.generated_at > first.generated_at
    assert service.latest_report(1) == third


def test_new_document_invalidates_cached_reports(service: ReportService) -> None:
    """AC-003.8: cache key includes doc fingerprint; ingest invalidates vendor report keys."""
    service.ingest_document(1, _doc(CLEAN_DOC))
    service.generate_report(ReportRequest(vendor_id=1))
    cache: InMemoryCache = service.cache  # type: ignore[assignment]
    assert any(k.startswith("report:1:") for k in cache.store)
    service.ingest_document(1, _doc(GOING_CONCERN_DOC, title="filing"))
    assert not any(k.startswith("report:1:") for k in cache.store)
    report = service.generate_report(ReportRequest(vendor_id=1))
    assert any("going concern" in f.description.lower() for f in report.red_flags)


def test_generate_without_documents_or_market(service: ReportService) -> None:
    """AC-003.6: no chunks / no ticker -> report still produced with explicit data_gaps."""
    report = service.generate_report(ReportRequest(vendor_name="northwind data feeds"))
    assert report.vendor_id == 2 and report.sources == []
    gaps = " ".join(report.data_gaps)
    assert "No documents ingested" in gaps and "no listed ticker" in gaps


def test_generate_degrades_when_quote_lookup_fails(service: ReportService) -> None:
    """Constitution §6: ES down -> report from MySQL chunks with a data_gaps note."""
    service.quote_lookup = FakeQuotes(fail=True)
    service.ingest_document(1, _doc(CLEAN_DOC))
    report = service.generate_report(ReportRequest(vendor_id=1))
    assert any("Market data unavailable" in g for g in report.data_gaps)
    assert report.financial_health.market_signal.startswith("unknown")


def test_vendor_resolution_errors(service: ReportService) -> None:
    """AC-003.7: unknown id/name -> 404; ambiguous name -> 422; latest without report -> 404."""
    with pytest.raises(NotFoundError):
        service.generate_report(ReportRequest(vendor_id=42))
    with pytest.raises(NotFoundError):
        service.generate_report(ReportRequest(vendor_name="Globex"))
    with pytest.raises(ValidationAppError):
        service.generate_report(ReportRequest(vendor_name="a"))  # matches both vendors
    assert service.generate_report(ReportRequest(vendor_name="V-001")).vendor_id == 1
    with pytest.raises(NotFoundError):
        service.latest_report(2)
    with pytest.raises(NotFoundError):
        service.get_vendor(42)
    assert [v.id for v in service.list_vendors(limit=1, offset=1)] == [2]


def test_llm_failure_maps_to_upstream_error(service: ReportService) -> None:
    """AC-003.5: LLM exceptions -> UpstreamError (502), nothing persisted or cached."""

    def boom(_: dict) -> None:
        raise TimeoutError("openai timeout")

    service.chain = RunnableLambda(boom)
    with pytest.raises(UpstreamError) as ei:
        service.generate_report(ReportRequest(vendor_id=1))
    assert ei.value.status_code == 502 and ei.value.code == "llm_failed"
    assert service.reports.rows == []


def test_repository_failure_maps_to_dependency_unavailable(service: ReportService) -> None:
    """Constitution §6/§8: infra exceptions surface as 503 AppErrors, not bare 500s."""

    def down(*_: object, **__: object) -> None:
        raise ConnectionError("mysql down")

    service.vendors.get = down  # type: ignore[method-assign]
    with pytest.raises(DependencyUnavailable):
        service.get_vendor(1)


def test_embed_all_pending(service: ReportService, indexer: FakeIndexer) -> None:
    """AC-004.1 (index step): chunk+embed every stored document lacking chunks."""
    d1 = service.documents.add(1, _doc())
    service.documents.add(2, _doc(CLEAN_DOC))
    assert service.embed_all_pending() == 2
    assert {c.document_id for c in service.documents.chunks_for_vendor(1)} == {d1}
    assert len(indexer.calls) == 2
    assert service.embed_all_pending() == 0  # idempotent
