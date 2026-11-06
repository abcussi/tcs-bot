"""Read-only bronze access (MySQL ``bronze_documents``): lineage without the raw body by default.

``MysqlBronzeReader`` goes through :func:`tcs_bot.db.mysql.transaction` (constitution rule 5) and
never logs row content. ``raw_content`` is selected only when ``include_raw=True``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Protocol

from mysql.connector import errors as mysql_errors

from tcs_bot.errors import DependencyUnavailable
from tcs_bot.logging_setup import get_logger
from tcs_bot.medallion.read.models import BronzeLineage

log = get_logger(__name__)

_LINEAGE_COLS = (
    "id, run_id, url, canonical_url, parent_url, depth, source, source_type, `query`, title, "
    "content_type, http_status, published_at, fetched_at, silver_processed_at, "
    "CHAR_LENGTH(raw_content) AS raw_chars"
)


class BronzeReader(Protocol):
    """What the read layer needs from bronze (tests pass an in-memory fake)."""

    def get(self, doc_id: int, include_raw: bool = False) -> BronzeLineage | None: ...


def _utc(value: Any) -> datetime | None:
    """MySQL DATETIME is naive UTC (see ``medallion.bronze._from_db_dt``)."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    return datetime.fromisoformat(str(value)).replace(tzinfo=UTC)


def _lineage(row: dict[str, Any]) -> BronzeLineage:
    return BronzeLineage(
        id=int(row["id"]),
        run_id=row.get("run_id"),
        url=row["url"],
        canonical_url=row["canonical_url"],
        parent_url=row.get("parent_url"),
        depth=int(row["depth"]),
        source=row["source"],
        source_type=row["source_type"],
        query=row.get("query") or "",
        title=row.get("title"),
        content_type=row.get("content_type") or "",
        http_status=int(row.get("http_status") or 0),
        published_at=_utc(row.get("published_at")),
        fetched_at=_utc(row.get("fetched_at")),
        silver_processed_at=_utc(row.get("silver_processed_at")),
        raw_chars=int(row["raw_chars"]) if row.get("raw_chars") is not None else None,
        raw_content=row.get("raw_content"),
    )


class MysqlBronzeReader:
    def get(self, doc_id: int, include_raw: bool = False) -> BronzeLineage | None:
        from tcs_bot.db.mysql import transaction

        cols = _LINEAGE_COLS + (", raw_content" if include_raw else "")
        try:
            with transaction() as cur:
                cur.execute(f"SELECT {cols} FROM bronze_documents WHERE id = %s", (int(doc_id),))
                row = cur.fetchone()
        except (mysql_errors.Error, OSError) as exc:
            log.error("bronze.read_failed", error_type=type(exc).__name__)
            raise DependencyUnavailable(
                f"MySQL bronze read failed ({type(exc).__name__}); is MySQL up and migrated?",
                code="bronze_unavailable",
            ) from exc
        if row is None:
            return None
        return _lineage(row)
