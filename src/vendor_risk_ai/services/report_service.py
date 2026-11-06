"""ReportService: ingestion + retrieval + LLM + persistence + cache (AC-003.2/.4/.5/.8).

Repositories, cache and quote lookup are injected (contracts in specs/001-data-layer/plan.md),
so unit tests run with in-memory fakes and ``build_default_service()`` wires the real ones.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

from langchain_core.runnables import Runnable

from vendor_risk_ai.errors import (
    AppError,
    DependencyUnavailable,
    NotFoundError,
    UpstreamError,
    ValidationAppError,
)
from vendor_risk_ai.ingestion.chunker import chunk_text
from vendor_risk_ai.ingestion.normalize import normalize_text
from vendor_risk_ai.logging_setup import get_logger
from vendor_risk_ai.rag.embeddings import Embedder
from vendor_risk_ai.rag.retriever import Retriever
from vendor_risk_ai.schemas.api import DocumentOut, ReportRequest
from vendor_risk_ai.schemas.domain import ChunkRecord, DocumentCreate, Vendor, VendorCreate
from vendor_risk_ai.schemas.report import RiskReport, StoredReport

if TYPE_CHECKING:
    from vendor_risk_ai.config import Settings
    from vendor_risk_ai.schemas.domain import VendorDocument

log = get_logger(__name__)
T = TypeVar("T")

RETRIEVAL_HINT = (
    "financial health red flags compliance going concern liquidity leverage covenant downgrade "
    "fines penalties enforcement SOC 2 exceptions audit opinion outage breach ransomware "
    "litigation customer concentration late filing auditor resignation"
)
_VENDOR_PAGE = 500


# --------------------------------------------------------------------------- injected contracts
class VendorRepo(Protocol):
    def list(self, limit: int = 100, offset: int = 0) -> list[Vendor]: ...
    def get(self, vendor_id: int) -> Vendor | None: ...
    def get_by_external_id(self, external_id: str) -> Vendor | None: ...
    def upsert(self, v: VendorCreate) -> int: ...


class DocumentRepo(Protocol):
    def add(self, vendor_id: int, doc: DocumentCreate) -> int: ...
    def list_for_vendor(self, vendor_id: int) -> list[VendorDocument]: ...
    def save_chunks(self, document_id: int, chunks: list[ChunkRecord]) -> int: ...
    def chunks_for_vendor(self, vendor_id: int) -> list[ChunkRecord]: ...
    # optional (light, no embeddings); the service falls back to chunks_for_vendor without it
    def chunk_ids_for_vendor(self, vendor_id: int) -> list[tuple[int, int]]: ...


class ReportRepo(Protocol):
    def save(self, vendor_id: int, report: RiskReport, model: str) -> int: ...
    def latest(self, vendor_id: int) -> StoredReport | None: ...


class Cache(Protocol):
    def get(self, key: str) -> Any | None: ...
    def set(self, key: str, value: Any, ttl: int | None = None) -> None: ...
    def delete(self, key: str) -> None: ...


QuoteLookup = Callable[[str], dict | None]
ChunkIndexer = Callable[[int, list[ChunkRecord]], int]


ChunkRef = ChunkRecord | tuple[int, int]  # full record or (document_id, chunk_index)


def _chunk_ref(c: ChunkRef) -> tuple[int | None, int]:
    return (c.document_id, c.chunk_index) if isinstance(c, ChunkRecord) else (c[0], c[1])


def chunk_fingerprint(chunks: Sequence[ChunkRef]) -> str:
    """Order-independent hash of the vendor's chunk set (``ChunkRecord``s or id pairs)."""
    ids = sorted(f"{d}:{i}" for d, i in map(_chunk_ref, chunks))
    return hashlib.sha1(f"{len(ids)}|{','.join(ids)}".encode()).hexdigest()[:16]


def report_cache_key(
    vendor_id: int,
    chunks: Sequence[ChunkRef],
    focus_areas: list[str],
    top_k: int,
    model: str,
) -> str:
    """AC-003.8: vendor + doc fingerprint + normalized focus + effective top_k + model id."""
    focus = ",".join(sorted({f.strip().lower() for f in focus_areas if f.strip()}))
    return f"report:{vendor_id}:{chunk_fingerprint(chunks)}:k{int(top_k)}:{model}:{focus}"


def _keys_index(vendor_id: int) -> str:
    return f"report-keys:{vendor_id}"


class ReportService:
    def __init__(
        self,
        vendors: VendorRepo,
        documents: DocumentRepo,
        reports: ReportRepo,
        cache: Cache,
        embedder: Embedder,
        chain: Runnable[dict, RiskReport],
        quote_lookup: QuoteLookup,
        chunk_indexer: ChunkIndexer | None = None,
        settings: Settings | None = None,
    ) -> None:
        if settings is None:
            from vendor_risk_ai.config import get_settings

            settings = get_settings()
        self.vendors = vendors
        self.documents = documents
        self.reports = reports
        self.cache = cache
        self.embedder = embedder
        self.chain = chain
        self.quote_lookup = quote_lookup
        self.chunk_indexer = chunk_indexer
        self.settings = settings
        self.retriever = Retriever(embedder)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _db(what: str, fn: Callable[[], T]) -> T:
        """Run a repository call; infra failures become 503 instead of a bare 500."""
        try:
            return fn()
        except AppError:
            raise
        except Exception as exc:
            log.error("repository_failed", op=what, error_type=type(exc).__name__)
            raise DependencyUnavailable(
                f"Database operation failed: {what}", {"operation": what}
            ) from exc

    def _cache_get(self, key: str) -> Any | None:
        try:
            return self.cache.get(key)
        except Exception as exc:
            log.warning("cache_get_failed", key=key, error_type=type(exc).__name__)
            return None

    def _cache_set(self, key: str, value: Any) -> None:
        try:
            self.cache.set(key, value)
        except Exception as exc:
            log.warning("cache_set_failed", key=key, error_type=type(exc).__name__)

    def _remember_key(self, vendor_id: int, key: str) -> None:
        keys = self._cache_get(_keys_index(vendor_id)) or []
        if key not in keys:
            self._cache_set(_keys_index(vendor_id), [*keys, key])

    def invalidate_vendor_reports(self, vendor_id: int) -> int:
        """Delete every cached report of ``vendor_id``. Returns keys removed."""
        keys = self._cache_get(_keys_index(vendor_id)) or []
        for key in [*keys, _keys_index(vendor_id)]:
            try:
                self.cache.delete(key)
            except Exception as exc:
                log.warning("cache_delete_failed", key=key, error_type=type(exc).__name__)
        return len(keys)

    def _require_vendor(self, vendor_id: int) -> Vendor:
        vendor = self._db("get_vendor", lambda: self.vendors.get(vendor_id))
        if vendor is None:
            raise NotFoundError(f"Vendor {vendor_id} not found", {"vendor_id": vendor_id})
        return vendor

    def _resolve_vendor(self, req: ReportRequest) -> Vendor:
        if req.vendor_id is not None:
            return self._require_vendor(req.vendor_id)
        name = (req.vendor_name or "").strip().lower()
        candidates = self._all_vendors()
        exact = [v for v in candidates if v.name.lower() == name or v.external_id.lower() == name]
        if len(exact) == 1:
            return exact[0]
        partial = exact or [v for v in candidates if name in v.name.lower()]
        if len(partial) == 1:
            return partial[0]
        if not partial:
            raise NotFoundError(
                f"Vendor '{req.vendor_name}' not found", {"vendor_name": req.vendor_name}
            )
        raise ValidationAppError(
            f"Vendor name '{req.vendor_name}' is ambiguous",
            {"matches": [{"id": v.id, "name": v.name} for v in partial[:10]]},
            code="ambiguous_vendor",
        )

    def _all_vendors(self) -> list[Vendor]:
        out: list[Vendor] = []
        offset = 0
        while True:
            page = self._db(
                "list_vendors", lambda o=offset: self.vendors.list(limit=_VENDOR_PAGE, offset=o)
            )
            out.extend(page)
            if len(page) < _VENDOR_PAGE:
                return out
            offset += _VENDOR_PAGE

    @property
    def model_id(self) -> str:
        """Id of the model that generates reports with the current chain (cache-key input)."""
        from vendor_risk_ai.llm.chain import report_model_id

        return report_model_id(self.chain, self.settings)

    def _chunk_ids(self, vendor_id: int) -> list[tuple[int, int]]:
        """``(document_id, chunk_index)`` pairs without loading texts/embeddings."""
        light = getattr(self.documents, "chunk_ids_for_vendor", None)
        if light is not None:
            return self._db("chunk_ids_for_vendor", lambda: list(light(vendor_id)))
        chunks = self._db("chunks_for_vendor", lambda: self.documents.chunks_for_vendor(vendor_id))
        return [(int(c.document_id or 0), c.chunk_index) for c in chunks]

    def _index(self, vendor_id: int, chunks: list[ChunkRecord]) -> bool:
        """Mirror chunks into ES; True only if ES accepted at least one (constitution §6)."""
        if self.chunk_indexer is None or not chunks:
            return False
        try:
            return self.chunk_indexer(vendor_id, chunks) > 0
        except Exception as exc:  # ES is optional (constitution §6)
            log.warning("chunk_index_failed", vendor_id=vendor_id, error_type=type(exc).__name__)
            return False

    def _split(self, content: str) -> list[str]:
        """Normalize + chunk; ``empty_document`` (422) when nothing is left (AC-003.2)."""
        pieces = chunk_text(
            normalize_text(content), self.settings.rag_chunk_size, self.settings.rag_chunk_overlap
        )
        if not pieces:
            raise ValidationAppError(
                "Document has no text after normalization", code="empty_document"
            )
        return pieces

    def _chunk_and_store(
        self,
        vendor_id: int,
        document_id: int,
        doc_type: str | None,
        title: str | None,
        content: str | list[str],
    ) -> tuple[int, bool]:
        """Embed + save (+ index) a document's chunks; ``content`` may be pre-split pieces."""
        pieces = content if isinstance(content, list) else self._split(content)
        embeddings: list[list[float]] | None
        try:
            embeddings = self.embedder.embed(pieces).tolist()
        except Exception as exc:  # store chunks anyway; retriever embeds on the fly later
            log.warning("embedding_failed", document_id=document_id, error_type=type(exc).__name__)
            embeddings = None
        chunks = [
            ChunkRecord(
                document_id=document_id,
                chunk_index=i,
                text=text,
                embedding=embeddings[i] if embeddings is not None else None,
                embedding_model=self.embedder.model if embeddings is not None else None,
                doc_type=doc_type,
                title=title,
            )
            for i, text in enumerate(pieces)
        ]
        self._db("save_chunks", lambda: self.documents.save_chunks(document_id, chunks))
        return len(chunks), self._index(vendor_id, chunks)

    def _is_stale(self, chunk: ChunkRecord) -> bool:
        """Missing, wrong-sized or other-model embedding (not comparable with the embedder)."""
        return (
            not chunk.embedding
            or len(chunk.embedding) != self.embedder.dim
            or chunk.embedding_model != self.embedder.model
        )

    def _reembed_document(self, vendor_id: int, chunks: list[ChunkRecord], force: bool) -> bool:
        """Re-embed one document's stale (or all, if ``force``) chunks, persist and re-index.

        ``chunks`` is the document's full chunk set: ``save_chunks`` replaces the set in one
        transaction, so unchanged chunks are passed through as-is.
        """
        document_id = int(chunks[0].document_id or 0)
        todo = [i for i, c in enumerate(chunks) if force or self._is_stale(c)]
        if not todo:
            return False
        try:
            vectors = self.embedder.embed([chunks[i].text for i in todo]).tolist()
        except Exception as exc:
            log.warning("reembed_failed", document_id=document_id, error_type=type(exc).__name__)
            return False
        updated = list(chunks)
        for i, vec in zip(todo, vectors, strict=True):
            updated[i] = chunks[i].model_copy(
                update={"embedding": vec, "embedding_model": self.embedder.model}
            )
        self._db("save_chunks", lambda: self.documents.save_chunks(document_id, updated))
        self._index(vendor_id, updated)
        log.info("document_reembedded", document_id=document_id, chunks=len(todo))
        return True

    # ------------------------------------------------------------------ public API
    def list_vendors(self, limit: int = 100, offset: int = 0) -> list[Vendor]:
        return self._db("list_vendors", lambda: self.vendors.list(limit=limit, offset=offset))

    def get_vendor(self, vendor_id: int) -> Vendor:
        return self._require_vendor(vendor_id)

    def ingest_document(self, vendor_id: int, doc: DocumentCreate) -> DocumentOut:
        self._require_vendor(vendor_id)
        # Validate BEFORE inserting: rejected content must not leave a document/audit row.
        pieces = self._split(doc.content)
        document_id = self._db("add_document", lambda: self.documents.add(vendor_id, doc))
        existing = [ref for ref in self._chunk_ids(vendor_id) if ref[0] == document_id]
        if existing:  # duplicate content (dedupe by sha256): already chunked
            log.info("document_duplicate", vendor_id=vendor_id, document_id=document_id)
            return DocumentOut(
                id=document_id, vendor_id=vendor_id, chunks=len(existing), indexed=False
            )
        n, indexed = self._chunk_and_store(vendor_id, document_id, doc.doc_type, doc.title, pieces)
        self.invalidate_vendor_reports(vendor_id)
        log.info("document_ingested", vendor_id=vendor_id, document_id=document_id, chunks=n)
        return DocumentOut(id=document_id, vendor_id=vendor_id, chunks=n, indexed=indexed)

    def embed_all_pending(self, reembed: bool = False) -> int:
        """Bring every stored document's chunks up to date with the current embedder.

        - documents without chunks are chunked + embedded (+ indexed);
        - documents whose chunks have a missing / other-model embedding (e.g. after a
          fake <-> openai switch) are re-embedded, persisted and re-indexed;
        - ``reembed=True`` forces re-embedding of every chunk.

        Returns the number of documents processed. Used by the orchestrator ``index`` step.
        """
        processed = 0
        for vendor in self._all_vendors():
            docs = self._db("list_documents", lambda v=vendor: self.documents.list_for_vendor(v.id))
            by_doc: dict[int, list[ChunkRecord]] = {}
            for c in self._db(
                "chunks_for_vendor", lambda v=vendor: self.documents.chunks_for_vendor(v.id)
            ):
                by_doc.setdefault(int(c.document_id or 0), []).append(c)
            changed = 0
            for d in docs:
                if d.id in by_doc:
                    changed += self._reembed_document(vendor.id, by_doc[d.id], force=reembed)
                    continue
                try:
                    self._chunk_and_store(vendor.id, d.id, d.doc_type, d.title, d.content)
                except ValidationAppError as exc:
                    log.warning("document_skipped", document_id=d.id, reason=exc.message)
                    continue
                changed += 1
            if changed:
                self.invalidate_vendor_reports(vendor.id)
            processed += changed
        log.info("embed_all_pending_done", documents=processed, reembed=reembed)
        return processed

    def generate_report(self, req: ReportRequest) -> RiskReport:
        vendor = self._resolve_vendor(req)
        top_k = req.top_k or self.settings.rag_top_k
        model = self.model_id
        # Cheap key first (ids only): a cache hit never loads texts/embeddings.
        key = report_cache_key(vendor.id, self._chunk_ids(vendor.id), req.focus_areas, top_k, model)

        if req.use_cache:
            cached = self._cache_get(key)
            if cached is not None:
                try:
                    report = RiskReport.model_validate(cached)
                    log.info("report_cache_hit", vendor_id=vendor.id)
                    return report
                except ValueError:
                    log.warning("report_cache_corrupt", key=key)

        chunks = self._db("chunks_for_vendor", lambda: self.documents.chunks_for_vendor(vendor.id))
        # Re-key from what is actually used (a document may have been added in between).
        key = report_cache_key(vendor.id, chunks, req.focus_areas, top_k, model)

        data_gaps: list[str] = []
        retrieved: list[tuple[ChunkRecord, float]] = []
        if not chunks:
            data_gaps.append(
                "No documents ingested for this vendor; assessment relies on the vendor profile "
                "only."
            )
        else:
            query = " ".join([vendor.name, vendor.services, *req.focus_areas, RETRIEVAL_HINT])
            try:
                retrieved = self.retriever.retrieve(query, chunks, top_k)
            except Exception as exc:
                raise UpstreamError(
                    "Embedding provider failed during retrieval",
                    {"error_type": type(exc).__name__},
                    code="embedding_failed",
                ) from exc

        market: dict | None = None
        if vendor.ticker:
            try:
                market = self.quote_lookup(vendor.ticker)
                if market is None:
                    data_gaps.append(f"No market quote found for ticker {vendor.ticker}.")
            except Exception as exc:
                log.warning(
                    "quote_lookup_failed", ticker=vendor.ticker, error_type=type(exc).__name__
                )
                data_gaps.append(
                    "Market data unavailable (search backend error); no market signal used."
                )
        else:
            data_gaps.append("Vendor has no listed ticker; no market signal available.")

        try:
            result = self.chain.invoke(
                {
                    "vendor": vendor,
                    "market": market,
                    "chunks": retrieved,
                    "focus_areas": req.focus_areas,
                    "data_gaps": data_gaps,
                }
            )
        except AppError:
            raise
        except Exception as exc:
            log.error("llm_failed", vendor_id=vendor.id, error_type=type(exc).__name__)
            raise UpstreamError(
                "LLM failed to produce a risk report",
                {"error_type": type(exc).__name__},
                code="llm_failed",
            ) from exc
        report = result if isinstance(result, RiskReport) else RiskReport.model_validate(result)

        self._db("save_report", lambda: self.reports.save(vendor.id, report, report.model))
        self._cache_set(key, report.model_dump(mode="json"))
        self._remember_key(vendor.id, key)
        log.info(
            "report_generated",
            vendor_id=vendor.id,
            score=report.overall_risk_score,
            level=report.risk_level,
            model=report.model,
        )
        return report

    def latest_report(self, vendor_id: int) -> RiskReport:
        stored = self._db("latest_report", lambda: self.reports.latest(vendor_id))
        if stored is None:
            raise NotFoundError(f"No report found for vendor {vendor_id}", {"vendor_id": vendor_id})
        return stored.report


def build_default_service(settings: Settings | None = None) -> ReportService:
    """Wire the real implementations. Imports are local so unit tests never need infra modules."""
    from vendor_risk_ai.cache.cache import get_cache
    from vendor_risk_ai.config import get_settings
    from vendor_risk_ai.db.repositories import (
        DocumentRepository,
        ReportRepository,
        VendorRepository,
    )
    from vendor_risk_ai.llm.chain import build_report_chain
    from vendor_risk_ai.rag.embeddings import get_embedder
    from vendor_risk_ai.search.elastic import index_chunks, latest_quote_strict

    settings = settings or get_settings()
    return ReportService(
        vendors=VendorRepository(),
        documents=DocumentRepository(),
        reports=ReportRepository(),
        cache=get_cache(),
        embedder=get_embedder(settings),
        chain=build_report_chain(settings),
        # strict: ES down raises -> explicit data_gaps note, distinct from "no quote" (§6)
        quote_lookup=latest_quote_strict,
        chunk_indexer=index_chunks,
        settings=settings,
    )
