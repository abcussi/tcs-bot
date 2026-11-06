"""Unit tests for migrate() compensating rollback and transaction() semantics (AC-001.2/3/4)."""

from __future__ import annotations

from contextlib import contextmanager

import pytest
from mysql.connector import errors as mysql_errors
from structlog.testing import capture_logs

from tcs_bot.db import migrate as migrate_mod
from tcs_bot.db import mysql as mysql_mod

TABLES = [name for name, _ in migrate_mod.load_statements()]
MEDALLION_TABLES = [name for name, _ in migrate_mod.statements_for("002_medallion")]
ALL_TABLES = TABLES + MEDALLION_TABLES


class FakeCursor:
    def __init__(
        self,
        existing: set[str] | None = None,
        fail_on_create: int | None = None,
        version_applied: bool = False,
    ) -> None:
        self.existing = existing or set()
        self.fail_on_create = fail_on_create
        self.version_applied = version_applied
        self.executed: list[str] = []
        self.creates = 0
        self._result: list[dict] = []

    def execute(self, sql: str, params=None) -> None:
        self.executed.append(sql)
        up = sql.lstrip().upper()
        if up.startswith("SELECT TABLE_NAME"):
            self._result = [{"table_name": t} for t in sorted(self.existing)]
        elif up.startswith("SELECT VERSION"):
            self._result = [{"version": "x"}] if self.version_applied else []
        elif up.startswith("CREATE TABLE"):
            self.creates += 1
            if self.creates == self.fail_on_create:
                raise mysql_errors.ProgrammingError(msg="boom on create", errno=1064)
        else:
            self._result = []

    def fetchall(self) -> list[dict]:
        return self._result

    def fetchone(self) -> dict | None:
        return self._result[0] if self._result else None

    @property
    def drops(self) -> list[str]:
        return [s for s in self.executed if s.upper().startswith("DROP TABLE")]


def _patch_tx(monkeypatch, cursor: FakeCursor) -> None:
    @contextmanager
    def fake_tx():
        yield cursor

    monkeypatch.setattr(migrate_mod, "transaction", fake_tx)


def test_schema_has_all_plan_tables():
    assert TABLES == [
        "schema_migrations",
        "vendors",
        "vendor_documents",
        "document_chunks",
        "risk_reports",
        "audit_log",
    ]


def test_failure_on_third_create_drops_two_created_tables(monkeypatch):
    """AC-001.3: failure on the 3rd CREATE drops the 2 tables created in this run (reverse)."""
    cur = FakeCursor(fail_on_create=3)
    _patch_tx(monkeypatch, cur)
    with capture_logs(), pytest.raises(mysql_errors.ProgrammingError):
        migrate_mod.migrate()
    assert cur.drops == [
        "DROP TABLE IF EXISTS `vendors`",
        "DROP TABLE IF EXISTS `schema_migrations`",
    ]
    assert not any("INSERT" in s.upper() for s in cur.executed)


def test_compensation_never_drops_preexisting_tables(monkeypatch):
    """AC-001.3: only tables created in this run are dropped."""
    cur = FakeCursor(existing={"schema_migrations"}, fail_on_create=3)
    _patch_tx(monkeypatch, cur)
    with capture_logs(), pytest.raises(mysql_errors.ProgrammingError):
        migrate_mod.migrate()
    assert cur.drops == ["DROP TABLE IF EXISTS `vendors`"]


def test_fresh_migrate_returns_created_and_records_version(monkeypatch):
    cur = FakeCursor()
    _patch_tx(monkeypatch, cur)
    with capture_logs():
        assert migrate_mod.migrate() == ALL_TABLES  # 001 then 002 (spec 005)
    assert sum("INSERT IGNORE INTO schema_migrations" in s for s in cur.executed) == 2


def test_second_run_is_noop(monkeypatch):
    """AC-001.4: all tables + version present -> no DDL, returns []."""
    cur = FakeCursor(existing=set(ALL_TABLES), version_applied=True)
    _patch_tx(monkeypatch, cur)
    with capture_logs():
        assert migrate_mod.migrate() == []
    assert cur.creates == 0


# --------------------------------------------------------------------------- versioned (spec 005)
def test_medallion_migration_tables_and_order():
    """AC-005.4: migration 002 adds crawl_runs then bronze_documents (FK order), after 001."""
    assert [v for v, _ in migrate_mod.MIGRATIONS] == ["001_initial", "002_medallion"]
    assert migrate_mod.LATEST_VERSION == "002_medallion"
    assert MEDALLION_TABLES == ["crawl_runs", "bronze_documents"]
    ddl = dict(migrate_mod.statements_for("002_medallion"))["bronze_documents"]
    for needle in (
        "UNIQUE KEY uq_bronze_documents_url_sha (url_sha256)",
        "silver_processed_at",
        "ix_bronze_documents_run",
        "ix_bronze_documents_source",
        "ENGINE=InnoDB",
        "utf8mb4",
        "ENUM('news','community','corporate','other')",
        "MEDIUMTEXT",
    ):
        assert needle in ddl, needle


def test_001_applied_only_002_runs(monkeypatch):
    """AC-001.4/AC-005.4: an existing 001 database only gets the 002 tables."""

    class Cur(FakeCursor):
        def execute(self, sql, params=None):
            super().execute(sql, params)
            if sql.lstrip().upper().startswith("SELECT VERSION"):
                self._result = [{"version": "x"}] if params == ("001_initial",) else []

    cur = Cur(existing=set(TABLES))
    _patch_tx(monkeypatch, cur)
    with capture_logs():
        assert migrate_mod.migrate() == MEDALLION_TABLES
    assert cur.creates == len(MEDALLION_TABLES)


def test_failure_in_002_compensates_only_002(monkeypatch):
    """AC-001.3 semantics per version: a failing 002 DDL drops only what 002 created."""
    cur = FakeCursor(fail_on_create=len(TABLES) + 2)  # 2nd CREATE of migration 002
    _patch_tx(monkeypatch, cur)
    with capture_logs(), pytest.raises(mysql_errors.ProgrammingError):
        migrate_mod.migrate()
    assert cur.drops == ["DROP TABLE IF EXISTS `crawl_runs`"]
    assert sum("INSERT IGNORE INTO schema_migrations" in s for s in cur.executed) == 1


# --------------------------------------------------------------------------- transaction()
class FakeConn:
    def __init__(self) -> None:
        self.committed = self.rolled_back = self.closed = False
        self.cursor_obj = FakeCursor()
        self.cursor_closed = False

    def cursor(self, dictionary: bool = False, buffered: bool = False):
        assert dictionary is True
        conn = self

        class C(FakeCursor):
            def close(self_inner):
                conn.cursor_closed = True

        return C()

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True

    def close(self):
        self.closed = True


class FakePool:
    def __init__(self) -> None:
        self.conn = FakeConn()

    def get_connection(self):
        return self.conn


def test_transaction_commits_and_returns_connection(monkeypatch):
    """AC-001.2: commit on success; cursor closed; connection returned (close())."""
    pool = FakePool()
    monkeypatch.setattr(mysql_mod, "get_pool", lambda: pool)
    with mysql_mod.transaction() as cur:
        cur.execute("SELECT 1")
    c = pool.conn
    assert c.committed and not c.rolled_back and c.closed and c.cursor_closed


def test_transaction_rolls_back_and_reraises(monkeypatch):
    """AC-001.2: rollback on any exception, re-raised; connection still returned."""
    pool = FakePool()
    monkeypatch.setattr(mysql_mod, "get_pool", lambda: pool)
    with pytest.raises(RuntimeError, match="nope"), mysql_mod.transaction():
        raise RuntimeError("nope")
    c = pool.conn
    assert c.rolled_back and not c.committed and c.closed and c.cursor_closed


def test_get_pool_is_singleton(monkeypatch):
    """AC-001.1: pool created once per process and reused."""
    created = []

    class P:
        def __init__(self, **kw):
            created.append(kw)

        def _remove_connections(self):
            pass

    monkeypatch.setattr(mysql_mod, "MySQLConnectionPool", P)
    mysql_mod.reset_pool()
    try:
        with capture_logs():
            assert mysql_mod.get_pool() is mysql_mod.get_pool()
        assert len(created) == 1
        kw = created[0]
        assert kw["autocommit"] is False and kw["charset"] == "utf8mb4"
        assert isinstance(kw["password"], str)
    finally:
        mysql_mod.reset_pool()
