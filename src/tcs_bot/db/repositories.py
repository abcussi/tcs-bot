"""Repositories over MySQL (contract: specs/001-data-layer/plan.md). All use ``transaction()``.

Cursor-level helpers (``upsert_vendor_cur``, ``add_document_cur``, ``write_audit``) are exposed so
callers that need several writes in ONE transaction (e.g. ``db.seed``) can compose them.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from mysql.connector import errors as mysql_errors

from tcs_bot.db.mysql import transaction
from tcs_bot.schemas.domain import (
    ChunkRecord,
    DocumentCreate,
    Vendor,
    VendorCreate,
    VendorDocument,
)
from tcs_bot.schemas.report import RiskReport, StoredReport

_VENDOR_COLS = (
    "id, external_id, name, sector, country, ticker, criticality, services, created_at, updated_at"
)
_DOC_COLS = (
    "id, vendor_id, doc_type, title, source, published_at, content, content_sha256, created_at"
)


# --------------------------------------------------------------------------- helpers
def content_sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _json_default(o: Any) -> Any:
    if isinstance(o, datetime | date):
        return o.isoformat()
    if isinstance(o, Decimal):
        return float(o)
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


def _dumps(value: Any) -> str:
    return json.dumps(value, default=_json_default, separators=(",", ":"))


def _loads(value: Any) -> Any:
    """MySQL JSON columns come back as str, bytes or (C-ext) already decoded."""
    if value is None:
        return None
    if isinstance(value, bytes | bytearray):
        value = value.decode("utf-8")
    if isinstance(value, str):
        return json.loads(value)
    return value


def write_audit(
    cur: Any, action: str, entity: str, entity_id: Any, payload: dict[str, Any] | None = None
) -> None:
    """Insert an ``audit_log`` row using the caller's cursor (same transaction)."""
    cur.execute(
        "INSERT INTO audit_log (action, entity, entity_id, payload) VALUES (%s, %s, %s, %s)",
        (action, entity, None if entity_id is None else str(entity_id), _dumps(payload or {})),
    )


def upsert_vendor_cur(cur: Any, v: VendorCreate) -> int:
    """Insert or update a vendor by ``external_id``; returns its id."""
    cur.execute(
        "INSERT INTO vendors (external_id, name, sector, country, ticker, criticality, services) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s) AS new "
        "ON DUPLICATE KEY UPDATE name = new.name, sector = new.sector, country = new.country, "
        "ticker = new.ticker, criticality = new.criticality, services = new.services",
        (v.external_id, v.name, v.sector, v.country, v.ticker, v.criticality, v.services),
    )
    cur.execute("SELECT id FROM vendors WHERE external_id = %s", (v.external_id,))
    row = cur.fetchone()
    return int(row["id"])


def add_document_cur(cur: Any, vendor_id: int, doc: DocumentCreate) -> tuple[int, bool]:
    """Insert a document deduped by (vendor_id, sha256). Returns ``(id, inserted)``.

    Writes an ``audit_log`` row (same transaction) when a new document is inserted.
    """
    sha = content_sha256(doc.content)
    select = "SELECT id FROM vendor_documents WHERE vendor_id = %s AND content_sha256 = %s"
    cur.execute(select, (vendor_id, sha))
    row = cur.fetchone()
    if row:
        return int(row["id"]), False
    try:
        cur.execute(
            "INSERT INTO vendor_documents "
            "(vendor_id, doc_type, title, source, published_at, content, content_sha256) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (vendor_id, doc.doc_type, doc.title, doc.source, doc.published_at, doc.content, sha),
        )
    except mysql_errors.IntegrityError as exc:
        # Concurrent insert of the same content (duplicate key) -> return the winner's id.
        if getattr(exc, "errno", None) != 1062:
            raise
        cur.execute(select, (vendor_id, sha))
        return int(cur.fetchone()["id"]), False
    doc_id = int(cur.lastrowid)
    write_audit(
        cur,
        "document.add",
        "vendor_document",
        doc_id,
        {"vendor_id": vendor_id, "doc_type": doc.doc_type, "title": doc.title, "sha256": sha},
    )
    return doc_id, True


def _row_to_vendor(row: dict[str, Any]) -> Vendor:
    data = dict(row)
    data["services"] = data.get("services") or ""
    return Vendor.model_validate(data)


# --------------------------------------------------------------------------- repositories
class VendorRepository:
    def list(self, limit: int = 100, offset: int = 0) -> list[Vendor]:
        with transaction() as cur:
            cur.execute(
                f"SELECT {_VENDOR_COLS} FROM vendors ORDER BY id LIMIT %s OFFSET %s",
                (int(limit), int(offset)),
            )
            rows = cur.fetchall()
        return [_row_to_vendor(r) for r in rows]

    def get(self, vendor_id: int) -> Vendor | None:
        with transaction() as cur:
            cur.execute(f"SELECT {_VENDOR_COLS} FROM vendors WHERE id = %s", (vendor_id,))
            row = cur.fetchone()
        return _row_to_vendor(row) if row else None

    def get_by_external_id(self, external_id: str) -> Vendor | None:
        with transaction() as cur:
            cur.execute(
                f"SELECT {_VENDOR_COLS} FROM vendors WHERE external_id = %s", (external_id,)
            )
            row = cur.fetchone()
        return _row_to_vendor(row) if row else None

    def upsert(self, v: VendorCreate) -> int:
        with transaction() as cur:
            return upsert_vendor_cur(cur, v)


class DocumentRepository:
    def add(self, vendor_id: int, doc: DocumentCreate) -> int:
        """Insert (deduped by sha256 per vendor) and audit; returns the document id."""
        with transaction() as cur:
            doc_id, _ = add_document_cur(cur, vendor_id, doc)
        return doc_id

    def list_for_vendor(self, vendor_id: int) -> list[VendorDocument]:
        with transaction() as cur:
            cur.execute(
                f"SELECT {_DOC_COLS} FROM vendor_documents WHERE vendor_id = %s ORDER BY id",
                (vendor_id,),
            )
            rows = cur.fetchall()
        return [VendorDocument.model_validate(r) for r in rows]

    def save_chunks(self, document_id: int, chunks: list[ChunkRecord]) -> int:
        """Replace the document's chunk set (upsert by index, drop stale tail). Returns count."""
        with transaction() as cur:
            for c in chunks:
                cur.execute(
                    "INSERT INTO document_chunks "
                    "(document_id, chunk_index, text, embedding, embedding_model) "
                    "VALUES (%s, %s, %s, %s, %s) AS new "
                    "ON DUPLICATE KEY UPDATE text = new.text, embedding = new.embedding, "
                    "embedding_model = new.embedding_model",
                    (
                        document_id,
                        c.chunk_index,
                        c.text,
                        None if c.embedding is None else _dumps(c.embedding),
                        c.embedding_model,
                    ),
                )
            keep = [c.chunk_index for c in chunks]
            if keep:
                placeholders = ", ".join(["%s"] * len(keep))
                cur.execute(
                    "DELETE FROM document_chunks WHERE document_id = %s "
                    f"AND chunk_index NOT IN ({placeholders})",
                    (document_id, *keep),
                )
            else:
                cur.execute("DELETE FROM document_chunks WHERE document_id = %s", (document_id,))
        return len(chunks)

    def chunk_ids_for_vendor(self, vendor_id: int) -> list[tuple[int, int]]:
        """``(document_id, chunk_index)`` of every chunk of the vendor — no text, no embeddings.

        Cheap input for the report cache key (AC-003.8) and "already chunked?" checks.
        """
        with transaction() as cur:
            cur.execute(
                "SELECT c.document_id, c.chunk_index FROM document_chunks c "
                "JOIN vendor_documents d ON d.id = c.document_id "
                "WHERE d.vendor_id = %s ORDER BY c.document_id, c.chunk_index",
                (vendor_id,),
            )
            rows = cur.fetchall()
        return [(int(r["document_id"]), int(r["chunk_index"])) for r in rows]

    def chunks_for_vendor(self, vendor_id: int) -> list[ChunkRecord]:
        with transaction() as cur:
            cur.execute(
                "SELECT c.document_id, c.chunk_index, c.text, c.embedding, c.embedding_model, "
                "d.doc_type, d.title FROM document_chunks c "
                "JOIN vendor_documents d ON d.id = c.document_id "
                "WHERE d.vendor_id = %s ORDER BY c.document_id, c.chunk_index",
                (vendor_id,),
            )
            rows = cur.fetchall()
        return [
            ChunkRecord(
                document_id=int(r["document_id"]),
                chunk_index=int(r["chunk_index"]),
                text=r["text"],
                embedding=_loads(r["embedding"]),
                embedding_model=r["embedding_model"],
                doc_type=r["doc_type"],
                title=r["title"],
            )
            for r in rows
        ]


class ReportRepository:
    def save(self, vendor_id: int, report: RiskReport, model: str) -> int:
        """Persist ``report.model_dump_json()`` and an audit row in one transaction."""
        with transaction() as cur:
            cur.execute(
                "INSERT INTO risk_reports "
                "(vendor_id, overall_score, risk_level, report_json, model) "
                "VALUES (%s, %s, %s, %s, %s)",
                (
                    vendor_id,
                    round(float(report.overall_risk_score), 2),
                    report.risk_level,
                    report.model_dump_json(),
                    model,
                ),
            )
            report_id = int(cur.lastrowid)
            write_audit(
                cur,
                "report.save",
                "risk_report",
                report_id,
                {
                    "vendor_id": vendor_id,
                    "overall_risk_score": report.overall_risk_score,
                    "risk_level": report.risk_level,
                    "model": model,
                },
            )
        return report_id

    def latest(self, vendor_id: int) -> StoredReport | None:
        with transaction() as cur:
            cur.execute(
                "SELECT id, vendor_id, report_json, model, created_at FROM risk_reports "
                "WHERE vendor_id = %s ORDER BY created_at DESC, id DESC LIMIT 1",
                (vendor_id,),
            )
            row = cur.fetchone()
        if not row:
            return None
        return StoredReport(
            id=int(row["id"]),
            vendor_id=int(row["vendor_id"]),
            report=RiskReport.model_validate(_loads(row["report_json"])),
            model=row["model"],
            created_at=row["created_at"],
        )
