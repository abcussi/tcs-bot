"""Schema migration (AC-001.3, AC-001.4).

Caveat — MySQL DDL is *not* transactional
-----------------------------------------
Every ``CREATE TABLE`` causes an implicit COMMIT in MySQL, so a classic BEGIN/ROLLBACK cannot undo
a half-applied schema. ``migrate()`` therefore implements one *logical* transaction:

1. record the tables that already exist,
2. execute each ``CREATE TABLE IF NOT EXISTS`` from ``schema.sql`` in order,
3. insert the version row into ``schema_migrations``,
4. on ANY failure, run a compensating ``DROP TABLE`` (reverse order) for every table created
   *in this run* — pre-existing tables are never touched — and re-raise.

It is idempotent: if the version is recorded and every table exists, it is a no-op returning ``[]``.
"""

from __future__ import annotations

import re
from importlib import resources
from typing import Any

from tcs_bot.db.mysql import transaction
from tcs_bot.logging_setup import get_logger

log = get_logger(__name__)

SCHEMA_VERSION = "001_initial"
_CREATE_RE = re.compile(r"CREATE\s+TABLE\s+IF\s+NOT\s+EXISTS\s+`?(\w+)`?", re.IGNORECASE)


def load_statements(sql: str | None = None) -> list[tuple[str, str]]:
    """Parse ``schema.sql`` into ``[(table_name, statement), ...]`` (comments stripped)."""
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


def _existing_tables(cur: Any) -> set[str]:
    cur.execute(
        "SELECT TABLE_NAME AS table_name FROM information_schema.TABLES "
        "WHERE TABLE_SCHEMA = DATABASE()"
    )
    return {str(row["table_name"]) for row in cur.fetchall()}


def _version_applied(cur: Any) -> bool:
    cur.execute("SELECT version FROM schema_migrations WHERE version = %s", (SCHEMA_VERSION,))
    return cur.fetchone() is not None


def migrate() -> list[str]:
    """Create all tables; return the names of tables created in this run (``[]`` if no-op)."""
    statements = load_statements()
    wanted = [name for name, _ in statements]
    created: list[str] = []
    with transaction() as cur:
        existing = _existing_tables(cur)
        if set(wanted) <= existing and _version_applied(cur):
            log.info("db.migrate.noop", version=SCHEMA_VERSION)
            return []
        try:
            for name, stmt in statements:
                cur.execute(stmt)
                if name not in existing:
                    created.append(name)
            cur.execute(
                "INSERT IGNORE INTO schema_migrations (version, applied_at) "
                "VALUES (%s, UTC_TIMESTAMP())",
                (SCHEMA_VERSION,),
            )
        except Exception as exc:
            log.error("db.migrate.failed", version=SCHEMA_VERSION, error=str(exc), created=created)
            _compensate(cur, created)
            raise
    log.info("db.migrate.applied", version=SCHEMA_VERSION, created=created)
    return created


def _compensate(cur: Any, created: list[str]) -> None:
    """Compensating rollback: drop tables created in this run, children first."""
    for name in reversed(created):
        try:
            cur.execute(f"DROP TABLE IF EXISTS `{name}`")
            log.warning("db.migrate.compensating_drop", table=name)
        except Exception as exc:  # keep dropping the rest
            log.error("db.migrate.compensating_drop_failed", table=name, error=str(exc))
