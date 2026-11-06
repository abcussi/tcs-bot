"""Bronze layer: raw TCS-risk documents in MySQL (AC-005.1, AC-005.2, AC-005.4).

``run_bronze`` crawls (``gfinance_crawler.news``: sources -> breadth-first crawler) and writes every
yielded document to ``bronze_documents`` in batches, each batch in one ``transaction()``,
idempotent on ``url_sha256 = sha256(canonical_url)``. Every run is recorded in ``crawl_runs``
(params + stats). Bronze rows are immutable except for the ``silver_processed_at`` watermark.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from gfinance_crawler.news.bfs import HARD_MAX_DEPTH, HARD_MAX_DOCS, RiskCrawler
from gfinance_crawler.news.fetch import PoliteFetcher
from gfinance_crawler.news.models import RawDocument
from gfinance_crawler.news.queries import DEFAULT_ALLOWED_DOMAINS, TCS_RISK_QUERIES
from gfinance_crawler.news.sources import (
    BingNewsRSS,
    GdeltDoc,
    GoogleNewsRSS,
    HackerNews,
    LocalInbox,
    RedditSearch,
    Source,
)

from tcs_bot.config import get_settings
from tcs_bot.db.mysql import transaction
from tcs_bot.logging_setup import get_logger
from tcs_bot.medallion.models import BronzeDocument, CrawlStats

log = get_logger(__name__)

BATCH_SIZE = 25
DEFAULT_INBOX = Path("data/bronze/inbox")
#: layer name -> watermark column. Only silver reads bronze.
_LAYER_COLUMNS = {"silver": "silver_processed_at"}
_PAGE = 200
_MAX_RAW_CHARS = 3_000_000  # MEDIUMTEXT = 16 MiB; utf8mb4 is up to 4 bytes/char

_COLS = (
    "id, run_id, url, canonical_url, parent_url, depth, source, source_type, query, title, "
    "raw_content, content_type, http_status, published_at, fetched_at"
)


# --------------------------------------------------------------------------- helpers
def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def url_sha256(doc: BronzeDocument) -> str:
    """Bronze identity: sha256 of the canonical URL (falls back to the raw URL)."""
    return sha256_hex(doc.canonical_url or doc.url)


def _to_db_dt(value: datetime | None) -> datetime | None:
    """MySQL DATETIME is naive: store UTC."""
    if value is None:
        return None
    if value.tzinfo is not None:
        value = value.astimezone(UTC).replace(tzinfo=None)
    return value.replace(microsecond=0)


def _from_db_dt(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    return datetime.fromisoformat(str(value)).replace(tzinfo=UTC)


def _column(layer: str) -> str:
    try:
        return _LAYER_COLUMNS[layer]
    except KeyError:
        raise ValueError(
            f"unknown layer {layer!r}; expected one of {sorted(_LAYER_COLUMNS)}"
        ) from None


def _row_to_doc(row: dict[str, Any]) -> BronzeDocument:
    return BronzeDocument(
        id=int(row["id"]),
        run_id=row["run_id"],
        url=row["url"],
        canonical_url=row["canonical_url"],
        parent_url=row["parent_url"],
        depth=int(row["depth"]),
        source=row["source"],
        source_type=row["source_type"],
        query=row["query"] or "",
        title=row["title"],
        raw_content=row["raw_content"],
        content_type=row["content_type"],
        http_status=int(row["http_status"]),
        published_at=_from_db_dt(row["published_at"]),
        fetched_at=_from_db_dt(row["fetched_at"]) or datetime.now(UTC),
    )


def to_bronze(raw: RawDocument, run_id: int | None) -> BronzeDocument:
    """Map a crawler ``RawDocument`` 1:1 to a ``BronzeDocument`` (``follow_url`` is dropped)."""
    data = raw.model_dump()  # follow_url is excluded at the model level
    data.pop("follow_url", None)
    return BronzeDocument(**data, run_id=run_id)


# --------------------------------------------------------------------------- repository
class BronzeRepository:
    """``crawl_runs`` + ``bronze_documents``. Every statement goes through ``transaction()``."""

    def start_run(self, params: dict[str, Any]) -> int:
        with transaction() as cur:
            cur.execute(
                "INSERT INTO crawl_runs (started_at, params) VALUES (UTC_TIMESTAMP(), %s)",
                (json.dumps(params, default=str),),
            )
            return int(cur.lastrowid)

    def finish_run(self, run_id: int, stats: CrawlStats | dict[str, Any]) -> None:
        payload = stats.model_dump() if isinstance(stats, CrawlStats) else dict(stats)
        with transaction() as cur:
            cur.execute(
                "UPDATE crawl_runs SET finished_at = UTC_TIMESTAMP(), stats = %s WHERE id = %s",
                (json.dumps(payload, default=str), run_id),
            )

    def get_run(self, run_id: int) -> dict[str, Any] | None:
        with transaction() as cur:
            cur.execute(
                "SELECT id, started_at, finished_at, params, stats FROM crawl_runs WHERE id = %s",
                (run_id,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        for key in ("params", "stats"):
            value = row[key]
            if isinstance(value, bytes | bytearray):
                value = value.decode("utf-8")
            row[key] = json.loads(value) if isinstance(value, str) else value
        return dict(row)

    def save_batch(self, run_id: int | None, docs: Sequence[BronzeDocument]) -> int:
        """Insert ``docs`` in ONE transaction, idempotent on ``url_sha256``
        (``INSERT ... ON DUPLICATE KEY UPDATE id = id``). Returns the number of NEW rows."""
        unique: dict[str, BronzeDocument] = {}
        for doc in docs:
            unique.setdefault(url_sha256(doc), doc)
        if not unique:
            return 0
        shas = list(unique)
        placeholders = ", ".join(["%s"] * len(shas))
        with transaction() as cur:
            # mysql-connector reports FOUND_ROWS, so rowcount cannot tell insert from duplicate:
            # count pre-existing keys inside the same transaction instead.
            cur.execute(
                f"SELECT url_sha256 FROM bronze_documents WHERE url_sha256 IN ({placeholders})",
                tuple(shas),
            )
            existing = {r["url_sha256"] for r in cur.fetchall()}
            for sha, doc in unique.items():
                if sha in existing:
                    continue
                raw = doc.raw_content[:_MAX_RAW_CHARS]
                cur.execute(
                    "INSERT INTO bronze_documents (run_id, url, url_sha256, canonical_url, "
                    "parent_url, depth, source, source_type, query, title, raw_content, "
                    "content_type, http_status, published_at, fetched_at, content_sha256) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
                    "ON DUPLICATE KEY UPDATE id = id",
                    (
                        doc.run_id if doc.run_id is not None else run_id,
                        doc.url,
                        sha,
                        doc.canonical_url or doc.url,
                        doc.parent_url,
                        doc.depth,
                        doc.source[:32],
                        doc.source_type,
                        (doc.query or "")[:255],
                        doc.title,
                        raw,
                        (doc.content_type or "text/html")[:64],
                        doc.http_status,
                        _to_db_dt(doc.published_at),
                        _to_db_dt(doc.fetched_at),
                        sha256_hex(raw),
                    ),
                )
        return len(unique) - len(existing)

    def iter_unprocessed(
        self, layer: str = "silver", limit: int | None = None
    ) -> Iterator[BronzeDocument]:
        """Stream documents whose ``<layer>_processed_at IS NULL`` in id order.

        Keyset-paginated (one short transaction per page), so callers may ``mark_processed``
        while iterating; documents that stay unprocessed are not re-yielded in the same pass.
        """
        col = _column(layer)
        last_id = 0
        remaining = limit
        while remaining is None or remaining > 0:
            page = _PAGE if remaining is None else min(_PAGE, remaining)
            with transaction() as cur:
                cur.execute(
                    f"SELECT {_COLS} FROM bronze_documents "
                    f"WHERE {col} IS NULL AND id > %s ORDER BY id LIMIT %s",
                    (last_id, page),
                )
                rows = cur.fetchall()
            if not rows:
                return
            for row in rows:
                yield _row_to_doc(row)
            last_id = int(rows[-1]["id"])
            if remaining is not None:
                remaining -= len(rows)
            if len(rows) < page:
                return

    def mark_processed(self, ids: Iterable[int], layer: str = "silver") -> int:
        col = _column(layer)
        id_list = sorted({int(i) for i in ids})
        if not id_list:
            return 0
        placeholders = ", ".join(["%s"] * len(id_list))
        with transaction() as cur:
            cur.execute(
                f"UPDATE bronze_documents SET {col} = UTC_TIMESTAMP() "
                f"WHERE id IN ({placeholders}) AND {col} IS NULL",
                tuple(id_list),
            )
            return int(cur.rowcount or 0)

    def count(self, unprocessed_layer: str | None = None) -> int:
        """All bronze rows, or only those not yet processed by ``unprocessed_layer``."""
        where = f" WHERE {_column(unprocessed_layer)} IS NULL" if unprocessed_layer else ""
        with transaction() as cur:
            cur.execute(f"SELECT COUNT(*) AS n FROM bronze_documents{where}")
            row = cur.fetchone()
        return int(row["n"]) if row else 0


# --------------------------------------------------------------------------- run
class _Repo(Protocol):
    def start_run(self, params: dict[str, Any]) -> int: ...
    def finish_run(self, run_id: int, stats: CrawlStats | dict[str, Any]) -> None: ...
    def save_batch(self, run_id: int | None, docs: Sequence[BronzeDocument]) -> int: ...


def default_sources(
    fetcher: PoliteFetcher, inbox_dir: Path | str | None = DEFAULT_INBOX
) -> list[Source]:
    """Real level-0 sources (AC-005.1) sharing one polite fetcher, plus the local inbox."""
    sources: list[Source] = [
        GoogleNewsRSS(fetcher),
        BingNewsRSS(fetcher),
        GdeltDoc(fetcher),
        HackerNews(fetcher),
        RedditSearch(fetcher),
    ]
    if inbox_dir is not None:
        sources.insert(0, LocalInbox(inbox_dir))
    return sources


def run_bronze(
    max_docs: int | None = None,
    max_depth: int | None = None,
    queries: Sequence[str] | None = None,
    sources: Sequence[Source] | None = None,
    fetcher: PoliteFetcher | None = None,
    *,
    repo: _Repo | None = None,
    allowed_domains: Sequence[str] | None = None,
    per_query_limit: int | None = None,
    inbox_dir: Path | str | None = DEFAULT_INBOX,
) -> CrawlStats:
    """Crawl -> bronze. Caps default to settings and are clamped to the hard caps (500 docs,
    depth 2). Writes batches of :data:`BATCH_SIZE`; records the run in ``crawl_runs``.

    In the returned stats ``stored`` is the number of NEW bronze rows; documents the crawler
    yielded that were already in bronze (earlier runs) are added to ``duplicates``.
    """
    s = get_settings()
    max_docs = min(max_docs if max_docs is not None else s.medallion_max_docs, HARD_MAX_DOCS)
    max_depth = min(max_depth if max_depth is not None else s.medallion_max_depth, HARD_MAX_DEPTH)
    queries = list(queries) if queries is not None else list(TCS_RISK_QUERIES)
    repo = repo if repo is not None else BronzeRepository()
    owns_fetcher = fetcher is None
    fetcher = fetcher if fetcher is not None else PoliteFetcher()
    sources = list(sources) if sources is not None else default_sources(fetcher, inbox_dir)
    crawler = RiskCrawler(
        sources,
        fetcher,
        allowed_domains=allowed_domains if allowed_domains is not None else DEFAULT_ALLOWED_DOMAINS,
        max_docs=max_docs,
        max_depth=max_depth,
        per_query_limit=per_query_limit,
    )
    params = {
        "max_docs": max_docs,
        "max_depth": max_depth,
        "queries": queries,
        "sources": [src.name for src in sources],
        "per_query_limit": per_query_limit,
    }
    run_id = repo.start_run(params)
    log.info("bronze.run_started", run_id=run_id, max_docs=max_docs, max_depth=max_depth)
    inserted = 0
    yielded = 0
    batch: list[BronzeDocument] = []
    error: str | None = None
    try:
        for raw in crawler.crawl(queries):
            batch.append(to_bronze(raw, run_id))
            yielded += 1
            if len(batch) >= BATCH_SIZE:
                inserted += repo.save_batch(run_id, batch)
                batch = []
        if batch:
            inserted += repo.save_batch(run_id, batch)
            batch = []
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        raw_stats = crawler.stats
        stats = CrawlStats(run_id=run_id, **raw_stats)
        stats.stored = inserted
        stats.duplicates += yielded - inserted - len(batch)
        payload: dict[str, Any] = stats.model_dump()
        payload["yielded"] = yielded
        payload["errors"] = raw_stats.get("errors", [])
        payload["links_skipped"] = raw_stats.get("links_skipped", 0)
        if error:
            payload["error"] = error[:500]
        try:
            repo.finish_run(run_id, payload)
        except Exception as exc:  # noqa: BLE001 - never mask the original error
            log.error("bronze.finish_run_failed", run_id=run_id, error=str(exc))
        if owns_fetcher:
            fetcher.close()
    log.info(
        "bronze.run_done",
        run_id=run_id,
        stored=stats.stored,
        duplicates=stats.duplicates,
        blocked=stats.blocked,
        failed=stats.failed,
        by_depth=stats.by_depth,
    )
    return stats
