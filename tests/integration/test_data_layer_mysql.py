"""Integration tests for the MySQL data layer (AC-001.1..AC-001.5) against the live server.

Runs in an isolated database ``vendor_risk_dltest`` (created via root, granted to the app user)
so it never touches the shared ``vendor_risk`` schema. Skips cleanly if MySQL is unreachable.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import pytest
from structlog.testing import capture_logs

from tcs_bot.config import get_settings
from tcs_bot.db import migrate as migrate_mod
from tcs_bot.db import mysql as mysql_mod
from tcs_bot.db.migrate import migrate
from tcs_bot.db.mysql import ping, transaction
from tcs_bot.db.repositories import (
    DocumentRepository,
    ReportRepository,
    VendorRepository,
    content_sha256,
)
from tcs_bot.db.seed import seed
from tcs_bot.schemas.domain import ChunkRecord, DocumentCreate, VendorCreate
from tcs_bot.schemas.report import FinancialHealth, RedFlag, RiskReport

pytestmark = pytest.mark.integration

TEST_DB = "vendor_risk_dltest"


def _root_conn():
    import mysql.connector

    s = get_settings()
    return mysql.connector.connect(
        host=s.mysql_host,
        port=s.mysql_port,
        user="root",
        password=os.environ.get("MYSQL_ROOT_PASSWORD", "root_pw"),
        connection_timeout=3,
    )


@pytest.fixture(scope="module")
def test_db():
    try:
        conn = _root_conn()
    except Exception as exc:
        pytest.skip(f"MySQL not reachable as root: {exc}")
    app_user = get_settings().mysql_user
    cur = conn.cursor()
    cur.execute(f"CREATE DATABASE IF NOT EXISTS `{TEST_DB}` CHARACTER SET utf8mb4")
    cur.execute(f"GRANT ALL PRIVILEGES ON `{TEST_DB}`.* TO '{app_user}'@'%'")
    conn.commit()
    old = os.environ.get("MYSQL_DATABASE")
    os.environ["MYSQL_DATABASE"] = TEST_DB
    get_settings.cache_clear()
    mysql_mod.reset_pool()
    try:
        yield TEST_DB
    finally:
        mysql_mod.reset_pool()
        if old is None:
            os.environ.pop("MYSQL_DATABASE", None)
        else:
            os.environ["MYSQL_DATABASE"] = old
        get_settings.cache_clear()
        cur.execute(f"DROP DATABASE IF EXISTS `{TEST_DB}`")
        conn.commit()
        conn.close()


def _tables() -> set[str]:
    with transaction() as cur:
        cur.execute(
            "SELECT TABLE_NAME AS t FROM information_schema.TABLES WHERE TABLE_SCHEMA = DATABASE()"
        )
        return {r["t"] for r in cur.fetchall()}


def _drop_all() -> None:
    with transaction() as cur:
        cur.execute("SET FOREIGN_KEY_CHECKS = 0")
        for t in _tables_cur(cur):
            cur.execute(f"DROP TABLE IF EXISTS `{t}`")
        cur.execute("SET FOREIGN_KEY_CHECKS = 1")


def _tables_cur(cur) -> list[str]:
    cur.execute(
        "SELECT TABLE_NAME AS t FROM information_schema.TABLES WHERE TABLE_SCHEMA = DATABASE()"
    )
    return [r["t"] for r in cur.fetchall()]


def _count(sql: str, params: tuple = ()) -> int:
    with transaction() as cur:
        cur.execute(sql, params)
        return int(next(iter(cur.fetchone().values())))


@pytest.fixture
def empty_db(test_db):
    _drop_all()
    return test_db


@pytest.fixture
def migrated_db(empty_db):
    with capture_logs():
        migrate()
    return empty_db


# --------------------------------------------------------------------------- pool / tx
def test_pool_singleton_and_ping(test_db):
    """AC-001.1: one pool per process, reused."""
    assert mysql_mod.get_pool() is mysql_mod.get_pool()
    assert ping() is True


def test_pool_handles_more_threads_than_pool_size(test_db):
    """AC-001.1/2: connections are returned to the pool, so 3x pool_size callers all succeed."""
    n = get_settings().mysql_pool_size * 3

    def work(_: int) -> int:
        with transaction() as cur:
            cur.execute("SELECT SLEEP(0.02) AS s, 1 AS ok")
            return cur.fetchone()["ok"]

    with ThreadPoolExecutor(max_workers=n) as ex:
        assert sum(ex.map(work, range(n))) == n


def test_transaction_rollback_leaves_no_rows(migrated_db):
    """AC-001.2: exception inside transaction() rolls back every statement."""
    with pytest.raises(RuntimeError), transaction() as cur:
        cur.execute(
            "INSERT INTO vendors (external_id, name) VALUES (%s, %s)", ("RB-1", "Rollback Co")
        )
        cur.execute(
            "INSERT INTO audit_log (action, entity, entity_id) VALUES ('t', 'vendor', 'RB-1')"
        )
        raise RuntimeError("fail after writes")
    assert _count("SELECT COUNT(*) AS n FROM vendors WHERE external_id = 'RB-1'") == 0
    assert _count("SELECT COUNT(*) AS n FROM audit_log WHERE entity_id = 'RB-1'") == 0
    assert ping() is True  # connection went back to the pool in a usable state


# --------------------------------------------------------------------------- migrate
def test_migrate_creates_all_tables_then_is_idempotent(empty_db):
    """AC-001.3/4: first run creates every table + version row; second run is a no-op."""
    expected = [name for name, _ in migrate_mod.load_statements()]
    with capture_logs():
        assert migrate() == expected
        assert migrate() == []
    assert set(expected) <= _tables()
    assert (
        _count(
            "SELECT COUNT(*) AS n FROM schema_migrations WHERE version = %s",
            (migrate_mod.SCHEMA_VERSION,),
        )
        == 1
    )


def test_migrate_compensating_rollback_on_real_db(empty_db, monkeypatch):
    """AC-001.3: a failing 3rd DDL drops the tables created earlier in the same run."""
    real = migrate_mod.load_statements()
    broken = ("broken_tbl", "CREATE TABLE IF NOT EXISTS broken_tbl (id INT NOT NULL, BOGUS)")
    monkeypatch.setattr(migrate_mod, "load_statements", lambda: [real[0], real[1], broken])
    with capture_logs(), pytest.raises(Exception):  # noqa: B017 - any mysql error
        migrate()
    assert _tables() == set()


# --------------------------------------------------------------------------- seed
def test_seed_is_idempotent(migrated_db):
    """AC-001.5: seeding twice upserts vendors and does not duplicate documents."""
    with capture_logs():
        first = seed()
        second = seed()
    assert first == {"vendors": 5, "documents": 15}
    assert second == {"vendors": 5, "documents": 0}
    assert _count("SELECT COUNT(*) AS n FROM vendors") == 5
    assert _count("SELECT COUNT(*) AS n FROM vendor_documents") == 15
    assert _count("SELECT COUNT(*) AS n FROM audit_log WHERE action = 'document.add'") == 15


def test_seed_missing_file_uses_builtin(migrated_db, tmp_path):
    """AC-001.5: missing seed file -> built-in dataset."""
    with capture_logs() as logs:
        out = seed(str(tmp_path / "missing.json"))
    assert out["vendors"] == 5
    assert any(e["event"] == "db.seed.fallback_builtin" for e in logs)


# --------------------------------------------------------------------------- repositories
def _report(vendor_id: int, score: float, name: str = "Round Trip Co") -> RiskReport:
    return RiskReport(
        vendor_id=vendor_id,
        vendor_name=name,
        overall_risk_score=score,
        risk_level="high" if score >= 50 else "medium",
        red_flags=[RedFlag(category="financial", severity="high", description="Rising leverage")],
        financial_health=FinancialHealth(summary="Leverage up, liquidity thin."),
        risk_narrative="Synthetic narrative.",
        model="fake",
    )


def test_repositories_round_trip(migrated_db):
    vendors, docs, reports = VendorRepository(), DocumentRepository(), ReportRepository()

    vid = vendors.upsert(
        VendorCreate(
            external_id="RT-1", name="Round Trip Co", ticker="IBM:NYSE", criticality="high"
        )
    )
    assert vendors.upsert(VendorCreate(external_id="RT-1", name="Round Trip Co v2")) == vid
    v = vendors.get(vid)
    assert v is not None and v.name == "Round Trip Co v2" and v.criticality == "medium"
    assert vendors.get_by_external_id("RT-1").id == vid
    assert vendors.get(999_999) is None
    assert [x.external_id for x in vendors.list()] == ["RT-1"]
    assert vendors.list(limit=10, offset=1) == []

    d = DocumentCreate(
        doc_type="news",
        title="Outage",
        content="Feed outage of 47 minutes.",
        published_at=date(2026, 5, 6),
    )
    doc_id = docs.add(vid, d)
    assert docs.add(vid, d) == doc_id  # dedupe by sha256
    listed = docs.list_for_vendor(vid)
    assert len(listed) == 1
    assert listed[0].content_sha256 == content_sha256(d.content)
    assert listed[0].published_at == date(2026, 5, 6)
    assert (
        _count(
            "SELECT COUNT(*) AS n FROM audit_log WHERE action='document.add' AND entity_id=%s",
            (str(doc_id),),
        )
        == 1
    )

    emb = [0.1, -0.25, 0.5]
    chunks = [
        ChunkRecord(
            document_id=doc_id,
            chunk_index=i,
            text=f"chunk {i}",
            embedding=emb,
            embedding_model="hash-test",
        )
        for i in range(3)
    ]
    assert docs.save_chunks(doc_id, chunks) == 3
    got = docs.chunks_for_vendor(vid)
    assert [c.chunk_index for c in got] == [0, 1, 2]
    assert got[0].embedding == pytest.approx(emb)
    assert got[0].doc_type == "news" and got[0].title == "Outage"
    # re-chunking with fewer chunks replaces the set
    assert docs.save_chunks(doc_id, chunks[:1]) == 1
    assert len(docs.chunks_for_vendor(vid)) == 1

    assert reports.latest(vid) is None
    r1 = reports.save(vid, _report(vid, 40.0), "fake")
    r2 = reports.save(vid, _report(vid, 62.5), "fake")
    assert r2 > r1
    latest = reports.latest(vid)
    assert latest is not None and latest.id == r2 and latest.model == "fake"
    assert latest.report.overall_risk_score == 62.5
    assert latest.report.red_flags[0].description == "Rising leverage"
    assert _count("SELECT COUNT(*) AS n FROM audit_log WHERE action='report.save'") == 2


def test_document_add_for_unknown_vendor_rolls_back(migrated_db):
    """FK violation propagates and leaves no audit row behind."""
    with pytest.raises(Exception):  # noqa: B017 - mysql IntegrityError
        DocumentRepository().add(424242, DocumentCreate(title="x", content="orphan"))
    assert _count("SELECT COUNT(*) AS n FROM audit_log") == 0
