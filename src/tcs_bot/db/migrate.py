"""Versioned schema migrations (AC-001.3, AC-001.4, AC-005.4).

Migrations are applied in order (:data:`MIGRATIONS`): ``001_initial`` = ``schema.sql`` (spec 001),
``002_medallion`` = ``schema_002_medallion.sql`` (spec 005 bronze: ``crawl_runs``,
``bronze_documents``). Each SQL file may only contain ``CREATE TABLE IF NOT EXISTS`` statements.

Caveat — MySQL DDL is *not* transactional
-----------------------------------------
Every ``CREATE TABLE`` causes an implicit COMMIT in MySQL, so a classic BEGIN/ROLLBACK cannot undo
a half-applied schema. Each migration version is therefore one *logical* transaction:

1. record the tables that already exist,
2. execute each ``CREATE TABLE IF NOT EXISTS`` of that version in order,
3. insert the version row into ``schema_migrations``,
4. on ANY failure, run a compensating ``DROP TABLE`` (reverse order) for every table created
   *by this version in this run* — pre-existing tables and earlier, already-recorded versions are
   never touched — and re-raise (later versions are not attempted).

It is idempotent: a version that is recorded and whose tables all exist is skipped; when every
version is skipped ``migrate()`` is a no-op returning ``[]``.
"""

from __future__ import annotations

import re
from importlib import resources
from typing import Any

from tcs_bot.db.mysql import transaction
from tcs_bot.logging_setup import get_logger

log = get_logger(__name__)

SCHEMA_VERSION = "001_initial"  # the base schema (schema.sql)
#: ``(version, sql file in tcs_bot.db)`` in apply order.
MIGRATIONS: list[tuple[str, str]] = [
    (SCHEMA_VERSION, "schema.sql"),
    ("002_medallion", "schema_002_medallion.sql"),
]
LATEST_VERSION = MIGRATIONS[-1][0]
_CREATE_RE = re.compile(r"CREATE\s+TABLE\s+IF\s+NOT\s+EXISTS\s+`?(\w+)`?", re.IGNORECASE)


def load_statements(sql: str | None = None) -> list[tuple[str, str]]:
    """Parse a schema file (default ``schema.sql``) into ``[(table_name, statement), ...]``
    (comments stripped)."""
    if sql is None:
        sql = resources.files("tcs_bot.db").joinpath("schema.sql").read_text("utf-8")
    body = "\n".join(line for line in sql.splitlines() if not line.strip().startswith("--"))
    out: list[tuple[str, str]] = []
    for raw in body.split(";"):
        stmt = raw.strip()
        if not stmt:
            continue
        m = _CREATE_RE.search(stmt)
        if not m:
            raise ValueError(
                f"schema.sql must only contain CREATE TABLE IF NOT EXISTS: {stmt[:60]!r}"
            )
        out.append((m.group(1), stmt))
    return out


def _read_sql(filename: str) -> str:
    return resources.files("tcs_bot.db").joinpath(filename).read_text("utf-8")


def statements_for(version: str) -> list[tuple[str, str]]:
    """Statements of one migration version (``001`` goes through :func:`load_statements`)."""
    filename = dict(MIGRATIONS)[version]
    if version == SCHEMA_VERSION:
        return load_statements()
    return load_statements(_read_sql(filename))


def all_statements() -> list[tuple[str, str]]:
    """Every migration's statements, in apply order."""
    return [st for version, _ in MIGRATIONS for st in statements_for(version)]


def _existing_tables(cur: Any) -> set[str]:
    cur.execute(
        "SELECT TABLE_NAME AS table_name FROM information_schema.TABLES "
        "WHERE TABLE_SCHEMA = DATABASE()"
    )
    return {str(row["table_name"]) for row in cur.fetchall()}


def _version_applied(cur: Any, version: str = SCHEMA_VERSION) -> bool:
    cur.execute("SELECT version FROM schema_migrations WHERE version = %s", (version,))
    return cur.fetchone() is not None


def migrate() -> list[str]:
    """Apply every pending migration; return the tables created in this run (``[]`` if no-op)."""
    created: list[str] = []
    for version, _ in MIGRATIONS:
        created.extend(_apply(version, statements_for(version)))
    return created


def _apply(version: str, statements: list[tuple[str, str]]) -> list[str]:
    wanted = [name for name, _ in statements]
    created: list[str] = []
    with transaction() as cur:
        existing = _existing_tables(cur)
        if set(wanted) <= existing and _version_applied(cur, version):
            log.info("db.migrate.noop", version=version)
            return []
        try:
            for name, stmt in statements:
                cur.execute(stmt)
                if name not in existing:
                    created.append(name)
            cur.execute(
                "INSERT IGNORE INTO schema_migrations (version, applied_at) "
                "VALUES (%s, UTC_TIMESTAMP())",
                (version,),
            )
        except Exception as exc:
            log.error("db.migrate.failed", version=version, error=str(exc), created=created)
            _compensate(cur, created)
            raise
    log.info("db.migrate.applied", version=version, created=created)
    return created


def _compensate(cur: Any, created: list[str]) -> None:
    """Compensating rollback: drop tables created in this run, children first."""
    for name in reversed(created):
        try:
            cur.execute(f"DROP TABLE IF EXISTS `{name}`")
            log.warning("db.migrate.compensating_drop", table=name)
        except Exception as exc:  # keep dropping the rest
            log.error("db.migrate.compensating_drop_failed", table=name, error=str(exc))
