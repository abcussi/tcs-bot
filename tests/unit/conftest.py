"""Shared unit-test fakes: in-memory repositories (contract: specs/001-data-layer/plan.md),
in-memory cache, offline embedder/chain. No network, no infra."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from typing import Any

import pytest

from tcs_bot.config import Settings, get_settings
from tcs_bot.schemas.domain import (
    ChunkRecord,
    DocumentCreate,
    Vendor,
    VendorCreate,
    VendorDocument,
)
from tcs_bot.schemas.report import RiskReport, StoredReport

GOING_CONCERN_DOC = (
    "Annual report FY2025. The company reported a net loss of $48 million and negative operating "
    "cash flow. Our independent auditor included an explanatory paragraph expressing substantial "
    "doubt about the company's ability to continue as a going concern. The company is in "
    "negotiations with lenders after a covenant breach under its revolving credit facility, and "
    "Moody's downgraded the senior notes to Caa1.\n\n"
    "In March the SEC imposed a civil penalty of $2 million for recordkeeping failures. "
    "A regional outage of the trade-matching platform lasted 9 hours in June. "
    "Our largest customer accounted for 41% of total revenue."
)
CLEAN_DOC = (
    "Quarterly update. Revenue grew 8% with net income of $12 million and an EBITDA margin of 31%. "
    "Cash and cash equivalents were $220 million. The SOC 2 Type II report was issued with no "
    "exceptions. Customer onboarding continued at a steady pace."
)


# ----------------------------------------------------------------------------- settings
@pytest.fixture(autouse=True)
def _fake_llm_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Force the offline provider for every unit test (constitution §4)."""
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def settings() -> Settings:
    return get_settings()


# ----------------------------------------------------------------------------- fakes
class FakeVendorRepo:
    def __init__(self) -> None:
        self._rows: dict[int, Vendor] = {}
        self._next = 1

    def list(self, limit: int = 100, offset: int = 0) -> list[Vendor]:
        return [self._rows[k] for k in sorted(self._rows)][offset : offset + limit]

    def get(self, vendor_id: int) -> Vendor | None:
        return self._rows.get(vendor_id)

    def get_by_external_id(self, external_id: str) -> Vendor | None:
        return next((v for v in self._rows.values() if v.external_id == external_id), None)

    def upsert(self, v: VendorCreate) -> int:
        existing = self.get_by_external_id(v.external_id)
        vid = existing.id if existing else self._next
        if not existing:
            self._next += 1
        self._rows[vid] = Vendor(id=vid, **v.model_dump())
        return vid


class FakeDocumentRepo:
    def __init__(self) -> None:
        self.docs: dict[int, VendorDocument] = {}
        self.chunks: dict[int, list[ChunkRecord]] = {}
        self._next = 1
        self.full_loads = 0  # chunks_for_vendor calls (texts + embeddings)

    def add(self, vendor_id: int, doc: DocumentCreate) -> int:
        sha = hashlib.sha256(doc.content.encode()).hexdigest()
        for d in self.docs.values():
            if d.vendor_id == vendor_id and d.content_sha256 == sha:
                return d.id
        did = self._next
        self._next += 1
        self.docs[did] = VendorDocument(
            id=did, vendor_id=vendor_id, content_sha256=sha, **doc.model_dump()
        )
        return did

    def list_for_vendor(self, vendor_id: int) -> list[VendorDocument]:
        return [d for d in self.docs.values() if d.vendor_id == vendor_id]

    def save_chunks(self, document_id: int, chunks: list[ChunkRecord]) -> int:
        self.chunks[document_id] = [
            c.model_copy(update={"document_id": document_id}) for c in chunks
        ]
        return len(chunks)

    def chunk_ids_for_vendor(self, vendor_id: int) -> list[tuple[int, int]]:
        return [
            (did, c.chunk_index)
            for did in sorted(self.chunks)
            if self.docs[did].vendor_id == vendor_id
            for c in self.chunks[did]
        ]

    def chunks_for_vendor(self, vendor_id: int) -> list[ChunkRecord]:
        self.full_loads += 1
        out: list[ChunkRecord] = []
        for did in sorted(self.chunks):
            if self.docs[did].vendor_id == vendor_id:
                out.extend(self.chunks[did])
        return out


class FakeReportRepo:
    def __init__(self) -> None:
        self.rows: list[StoredReport] = []

    def save(self, vendor_id: int, report: RiskReport, model: str) -> int:
        rid = len(self.rows) + 1
        self.rows.append(
            StoredReport(
                id=rid,
                vendor_id=vendor_id,
                report=report,
                model=model,
                created_at=datetime.now(UTC),
            )
        )
        return rid

    def latest(self, vendor_id: int) -> StoredReport | None:
        rows = [r for r in self.rows if r.vendor_id == vendor_id]
        return rows[-1] if rows else None


class InMemoryCache:
    """JSON round-trips values, like the real L2 (Redis) tier."""

    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.hits = 0

    def get(self, key: str) -> Any | None:
        raw = self.store.get(key)
        if raw is None:
            return None
        self.hits += 1
        return json.loads(raw)

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        self.store[key] = json.dumps(value, default=str)

    def delete(self, key: str) -> None:
        self.store.pop(key, None)

    def get_or_set(self, key: str, factory: Callable[[], Any], ttl: int | None = None) -> Any:
        value = self.get(key)
        if value is None:
            value = factory()
            self.set(key, value, ttl)
        return value


class FakeQuotes:
    def __init__(self, quotes: dict[str, dict] | None = None, fail: bool = False) -> None:
        self.quotes = quotes or {}
        self.fail = fail
        self.calls: list[str] = []

    def __call__(self, ticker: str) -> dict | None:
        self.calls.append(ticker)
        if self.fail:
            raise ConnectionError("elasticsearch down")
        return self.quotes.get(ticker)


class FakeIndexer:
    """``accepted=False`` mimics ``index_chunks`` swallowing ES errors (returns 0)."""

    def __init__(self, fail: bool = False, accepted: bool = True) -> None:
        self.fail = fail
        self.accepted = accepted
        self.calls: list[tuple[int, int]] = []
        self.indexed: list[ChunkRecord] = []

    def __call__(self, vendor_id: int, chunks: list[ChunkRecord]) -> int:
        if self.fail:
            raise ConnectionError("elasticsearch down")
        self.calls.append((vendor_id, len(chunks)))
        self.indexed.extend(chunks)
        return len(chunks) if self.accepted else 0


# ----------------------------------------------------------------------------- fixtures
@pytest.fixture
def vendor_repo() -> FakeVendorRepo:
    repo = FakeVendorRepo()
    repo.upsert(
        VendorCreate(
            external_id="V-001",
            name="Acme Clearing Services",
            sector="post-trade",
            country="US",
            ticker="ACME:NYSE",
            criticality="critical",
            services="trade matching and clearing connectivity",
        )
    )
    repo.upsert(
        VendorCreate(
            external_id="V-002",
            name="Northwind Data Feeds",
            sector="market data",
            country="UK",
            ticker=None,
            criticality="low",
            services="reference data",
        )
    )
    return repo


@pytest.fixture
def quotes() -> FakeQuotes:
    return FakeQuotes(
        {
            "ACME:NYSE": {
                "ticker": "ACME",
                "exchange": "NYSE",
                "price": 12.3,
                "currency": "USD",
                "change_pct": -14.2,
                "pe_ratio": -3.1,
            }
        }
    )


@pytest.fixture
def indexer() -> FakeIndexer:
    return FakeIndexer()


@pytest.fixture
def service(
    vendor_repo: FakeVendorRepo, quotes: FakeQuotes, indexer: FakeIndexer, settings: Settings
):
    from tcs_bot.llm.chain import build_report_chain
    from tcs_bot.rag.embeddings import HashEmbedder
    from tcs_bot.services.report_service import ReportService

    return ReportService(
        vendors=vendor_repo,
        documents=FakeDocumentRepo(),
        reports=FakeReportRepo(),
        cache=InMemoryCache(),
        embedder=HashEmbedder(),
        chain=build_report_chain(settings),
        quote_lookup=quotes,
        chunk_indexer=indexer,
        settings=settings,
    )
