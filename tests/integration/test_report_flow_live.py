"""ReportService wired by ``build_default_service()`` against live MySQL / Redis / ES.

Covers ingest -> MySQL chunks + ES ``vendor-chunks`` -> report -> ``risk_reports`` -> cache hit ->
invalidation on ingest, and market-quote enrichment from ES ``market-quotes``.
Uses the shared seeded database; everything a test creates is removed afterwards.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

import pytest

from tcs_bot.db.mysql import transaction
from tcs_bot.schemas.api import ReportRequest
from tcs_bot.schemas.domain import DocumentCreate
from tcs_bot.schemas.report import RiskReport, score_to_level
from tcs_bot.search.elastic import chunk_doc_id
from tcs_bot.services.report_service import ReportService, build_default_service

pytestmark = pytest.mark.integration

FLOW_VENDOR_EXT = "VND-005"  # Tidewater Execution Systems LLC (ticker ORCL:NYSE)
QUOTE_VENDOR_EXT = "VND-003"  # Clearpath Settlement Technologies AG (ticker IBM:NYSE)


class CountingChain:
    """Delegates to the real (offline) LCEL chain and counts invocations."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.calls = 0

    def invoke(self, inputs: dict, *args: Any, **kwargs: Any) -> RiskReport:
        self.calls += 1
        return self.inner.invoke(inputs, *args, **kwargs)


def _doc(tag: str, extra: str = "") -> DocumentCreate:
    body = (
        f"Integration test filing {tag}. The company reported a net loss and negative operating "
        "cash flow for the year. The auditor issued a qualified opinion and the SOC 2 Type II "
        "report noted two exceptions in change management. A regional outage of the order "
        "routing platform lasted six hours. The largest customer accounted for 38% of revenue. "
    ) * 3
    return DocumentCreate(
        doc_type="news",
        title=f"qa-integration {tag}",
        source="qa-integration",
        content=body + extra,
    )


@pytest.fixture
def service(live_stack) -> Iterator[tuple[ReportService, CountingChain]]:
    svc = build_default_service()
    counting = CountingChain(svc.chain)
    svc.chain = counting  # type: ignore[assignment]
    yield svc, counting


@pytest.fixture
def cleanup(live_stack) -> Iterator[dict[str, Any]]:
    """Track created docs / quote ids; delete rows, ES docs and audit rows afterwards."""
    with transaction() as cur:
        cur.execute("SELECT COALESCE(MAX(id), 0) AS m FROM risk_reports")
        max_report = int(cur.fetchone()["m"])
        cur.execute("SELECT COALESCE(MAX(id), 0) AS m FROM audit_log")
        max_audit = int(cur.fetchone()["m"])
    track: dict[str, Any] = {"docs": [], "quotes": [], "vendors": set()}
    yield track
    es = live_stack["es"]
    s = live_stack["settings"]
    for vendor_id, doc_id, n_chunks in track["docs"]:
        for i in range(n_chunks):
            es.options(ignore_status=404).delete(
                index=s.elastic_chunks_index, id=f"{vendor_id}:{doc_id}:{i}"
            )
    es.indices.refresh(index=s.elastic_chunks_index)
    for qid in track["quotes"]:
        es.options(ignore_status=404).delete(index=s.elastic_quotes_index, id=qid, refresh=True)
    with transaction() as cur:
        for _, doc_id, _ in track["docs"]:
            cur.execute("DELETE FROM vendor_documents WHERE id = %s", (doc_id,))  # cascades chunks
        cur.execute("DELETE FROM risk_reports WHERE id > %s", (max_report,))
        cur.execute(
            "DELETE FROM audit_log WHERE id > %s AND action IN ('document.add', 'report.save')",
            (max_audit,),
        )
    svc = build_default_service()
    for vendor_id in track["vendors"]:
        svc.invalidate_vendor_reports(vendor_id)


def _vendor_id(ext: str) -> int:
    with transaction() as cur:
        cur.execute("SELECT id FROM vendors WHERE external_id = %s", (ext,))
        row = cur.fetchone()
    if not row:
        pytest.skip(f"seed vendor {ext} missing (run `uv run tcs-bot seed`)")
    return int(row["id"])


def _reports_count(vendor_id: int) -> int:
    with transaction() as cur:
        cur.execute("SELECT COUNT(*) AS n FROM risk_reports WHERE vendor_id = %s", (vendor_id,))
        return int(cur.fetchone()["n"])


def _check_report(report: RiskReport, vendor_id: int) -> None:
    RiskReport.model_validate(report.model_dump(mode="json"))
    assert report.vendor_id == vendor_id
    assert 0 <= report.overall_risk_score <= 100
    assert report.risk_level == score_to_level(report.overall_risk_score)


def test_ingest_report_cache_and_invalidation(live_stack, service, cleanup):
    """AC-003.2, AC-003.8, AC-001.7, AC-001.6: ingest persists chunks in MySQL and ES; the report
    is persisted to risk_reports, cached in Redis (second call skips the chain) and the cache is
    invalidated when another document is ingested."""
    svc, chain = service
    es, r = live_stack["es"], live_stack["redis"]
    s = live_stack["settings"]
    vendor_id = _vendor_id(FLOW_VENDOR_EXT)
    cleanup["vendors"].add(vendor_id)
    tag = uuid.uuid4().hex[:12]

    # -- ingest: chunks in MySQL + ES
    out = svc.ingest_document(vendor_id, _doc(tag))
    cleanup["docs"].append((vendor_id, out.id, out.chunks))
    assert out.vendor_id == vendor_id and out.chunks >= 2 and out.indexed is True
    with transaction() as cur:
        cur.execute(
            "SELECT chunk_index, embedding_model, JSON_LENGTH(embedding) AS dims "
            "FROM document_chunks WHERE document_id = %s ORDER BY chunk_index",
            (out.id,),
        )
        rows = cur.fetchall()
    assert [row["chunk_index"] for row in rows] == list(range(out.chunks))
    assert all(row["dims"] == 1536 and row["embedding_model"] for row in rows)
    chunks = [c for c in svc.documents.chunks_for_vendor(vendor_id) if c.document_id == out.id]
    for c in chunks:
        hit = es.get(index=s.elastic_chunks_index, id=chunk_doc_id(vendor_id, c))
        assert hit["_source"]["vendor_id"] == vendor_id
        assert hit["_source"]["document_id"] == out.id
        assert len(hit["_source"]["embedding"]) == 1536

    # -- duplicate ingest is deduped (same doc id, no new chunks)
    dup = svc.ingest_document(vendor_id, _doc(tag))
    assert dup.id == out.id and dup.chunks == out.chunks and dup.indexed is False

    # -- first report: chain invoked, persisted, cached in Redis
    def report_keys() -> set[bytes]:
        return set(r.scan_iter(f"vra:report:{vendor_id}:*"))

    keys_before = report_keys()
    before = _reports_count(vendor_id)
    req = ReportRequest(vendor_id=vendor_id, focus_areas=["cyber"])
    first = svc.generate_report(req)
    _check_report(first, vendor_id)
    assert chain.calls == 1
    assert _reports_count(vendor_id) == before + 1
    assert any(src.document_id == out.id for src in first.sources)
    new_keys = report_keys() - keys_before
    assert len(new_keys) == 1, new_keys  # the report is in Redis (L2)
    key = next(iter(new_keys)).decode().removeprefix("vra:")
    assert svc.cache.get(key) is not None
    latest = svc.latest_report(vendor_id)
    assert latest.generated_at == first.generated_at

    # -- second call: cache hit, chain not re-invoked, nothing new persisted
    second = svc.generate_report(req)
    assert chain.calls == 1
    assert second.model_dump(mode="json") == first.model_dump(mode="json")
    assert _reports_count(vendor_id) == before + 1

    # -- use_cache=False bypasses the cache
    svc.generate_report(ReportRequest(vendor_id=vendor_id, focus_areas=["cyber"], use_cache=False))
    assert chain.calls == 2

    # -- ingesting another document invalidates cached reports for the vendor
    out2 = svc.ingest_document(vendor_id, _doc(tag + "-b", extra="Ransomware attack disclosed."))
    cleanup["docs"].append((vendor_id, out2.id, out2.chunks))
    assert r.exists(f"vra:{key}") == 0
    assert svc.cache.get(key) is None
    assert not (report_keys() & new_keys)
    third = svc.generate_report(req)
    assert chain.calls == 3
    _check_report(third, vendor_id)
    assert third.generated_at != first.generated_at


def test_report_by_vendor_name_live(live_stack, service, cleanup):
    """AC-003.7, AC-003.9: vendor_name resolution (case-insensitive partial) on live data."""
    svc, _ = service
    vendor_id = _vendor_id(FLOW_VENDOR_EXT)
    cleanup["vendors"].add(vendor_id)
    report = svc.generate_report(ReportRequest(vendor_name="tidewater", use_cache=False))
    _check_report(report, vendor_id)


def test_market_quote_enrichment(live_stack, service, cleanup):
    """AC-002.4, AC-003.6: a quote in ES market-quotes for the vendor's ticker is used by the
    report (market_signal populated, no 'No market quote' data gap)."""
    svc, _ = service
    es = live_stack["es"]
    s = live_stack["settings"]
    vendor_id = _vendor_id(QUOTE_VENDOR_EXT)
    cleanup["vendors"].add(vendor_id)
    vendor = svc.get_vendor(vendor_id)
    assert vendor.ticker and ":" in vendor.ticker
    symbol, exchange = vendor.ticker.split(":", 1)

    # Baseline: without a quote the report states the gap (index is empty offline).
    if svc.quote_lookup(vendor.ticker) is None:
        base = svc.generate_report(ReportRequest(vendor_id=vendor_id, use_cache=False))
        assert any("No market quote" in g for g in base.data_gaps)
        assert base.financial_health.market_signal.startswith("unknown")

    now = datetime.now(UTC)
    qid = f"{symbol}:{exchange}:{now:%Y%m%d%H%M}:qa-{uuid.uuid4().hex[:6]}"
    cleanup["quotes"].append(qid)
    es.index(
        index=s.elastic_quotes_index,
        id=qid,
        refresh=True,
        document={
            "ticker": symbol,
            "exchange": exchange,
            "name": "QA Synthetic Quote",
            "price": 123.45,
            "currency": "USD",
            "change": -15.2,
            "change_pct": -11.0,
            "previous_close": 138.65,
            "pe_ratio": 21.5,
            "scraped_at": now.isoformat(),
            "source_url": "https://example.invalid/qa",
        },
    )
    assert svc.quote_lookup(vendor.ticker)["price"] == pytest.approx(123.45)

    report = svc.generate_report(ReportRequest(vendor_id=vendor_id, use_cache=False))
    _check_report(report, vendor_id)
    assert not any("No market quote" in g for g in report.data_gaps)
    assert not any("market data" in g.lower() for g in report.data_gaps)
    signal = report.financial_health.market_signal
    assert "123.45" in signal and "-11.00%" in signal and signal.startswith("negative")
    assert any("market move" in f.description for f in report.red_flags)


def test_rejected_empty_document_leaves_no_row(live_stack, service, cleanup):
    """AC-003.2, constitution §5: a document rejected as empty must not be persisted."""
    from tcs_bot.api.errors import ValidationAppError

    svc, _ = service
    vendor_id = _vendor_id(FLOW_VENDOR_EXT)
    title = f"qa-empty-{uuid.uuid4().hex[:8]}"
    try:
        with pytest.raises(ValidationAppError):
            svc.ingest_document(vendor_id, DocumentCreate(title=title, content=" \n\t ​ "))
        with transaction() as cur:
            cur.execute("SELECT id FROM vendor_documents WHERE title = %s", (title,))
            leftovers = cur.fetchall()
        assert leftovers == []
    finally:
        with transaction() as cur:
            cur.execute("DELETE FROM vendor_documents WHERE title = %s", (title,))
