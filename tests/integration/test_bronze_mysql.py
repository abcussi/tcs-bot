"""Integration tests for the bronze layer against live MySQL (AC-005.2, AC-005.4).

Runs in a THROWAWAY database ``vendor_risk_bronzetest`` (created via root, granted to the app
user, dropped afterwards). The crawl uses real sources/fetcher over ``httpx.MockTransport`` serving
the SYNTHETIC fixtures in ``crawler/tests/fixtures/news`` — no internet. Skips if MySQL is down.
"""

from __future__ import annotations

import os
from pathlib import Path

import httpx
import pytest
from gfinance_crawler.news.fetch import PoliteFetcher
from gfinance_crawler.news.sources import GdeltDoc, HackerNews, LocalInbox
from structlog.testing import capture_logs

from tcs_bot.config import get_settings
from tcs_bot.db import migrate as migrate_mod
from tcs_bot.db import mysql as mysql_mod
from tcs_bot.db.migrate import migrate
from tcs_bot.db.mysql import transaction
from tcs_bot.medallion.bronze import BronzeRepository, run_bronze, to_bronze
from tcs_bot.medallion.models import BronzeDocument

pytestmark = pytest.mark.integration

TEST_DB = "vendor_risk_bronzetest"
FIX = Path(__file__).resolve().parents[2] / "crawler" / "tests" / "fixtures" / "news"
ROUTES = {
    "api.gdeltproject.org/api/v2/doc/doc": ("gdelt.json", "application/json"),
    "hn.algolia.com/api/v1/search": ("hn_search.json", "application/json"),
    "hn.algolia.com/api/v1/items/90000001": ("hn_item.json", "application/json"),
    "economictimes.indiatimes.com/tech/tcs-synthetic-lawsuit/articleshow/1001.cms": (
        "article_lawsuit.html",
        "text/html",
    ),
    "www.reuters.com/technology/tcs-synthetic-data-breach-2026-11-01/": (
        "article_breach.html",
        "text/html",
    ),
    "www.livemint.com/companies/tcs-synthetic-whistleblower-11.html": (
        "article_whistleblower.html",
        "text/html",
    ),
    "www.tcs.com/news/synthetic-statement": ("corporate_statement.html", "text/html"),
    "www.business-standard.com/companies/tcs-synthetic-security-review.html": (
        "article_breach.html",
        "text/html",
    ),
}


def _handler(request: httpx.Request) -> httpx.Response:
    route = ROUTES.get(f"{request.url.host}{request.url.path}")
    if route is None:
        return httpx.Response(404)
    name, ctype = route
    return httpx.Response(200, content=(FIX / name).read_bytes(), headers={"content-type": ctype})


def _fetcher() -> PoliteFetcher:
    client = httpx.Client(transport=httpx.MockTransport(_handler), follow_redirects=True)
    return PoliteFetcher(client=client, delay=0, sleep=lambda s: None)


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
    cur.execute(f"DROP DATABASE IF EXISTS `{TEST_DB}`")
    cur.execute(f"CREATE DATABASE `{TEST_DB}` CHARACTER SET utf8mb4")
    cur.execute(f"GRANT ALL PRIVILEGES ON `{TEST_DB}`.* TO '{app_user}'@'%'")
    conn.commit()
    old = os.environ.get("MYSQL_DATABASE")
    os.environ["MYSQL_DATABASE"] = TEST_DB
    get_settings.cache_clear()
    mysql_mod.reset_pool()
    try:
        with capture_logs():
            migrate()
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


@pytest.fixture
def clean(test_db):
    with transaction() as cur:
        cur.execute("DELETE FROM bronze_documents")
        cur.execute("DELETE FROM crawl_runs")
    return test_db


def _doc(i: int, **kw) -> BronzeDocument:
    url = f"https://news.test/synthetic/{i}"
    return BronzeDocument(
        url=url,
        canonical_url=url,
        depth=kw.pop("depth", 0),
        source="gdelt",
        query="TCS fine",
        title=f"SYNTHETIC item {i} — ✓ utf8mb4 😀",
        raw_content=f"SYNTHETIC FIXTURE body {i} about TCS",
        **kw,
    )


def test_migration_002_tables_and_idempotent(test_db):
    """AC-005.4: migration 002 is recorded once; re-running migrate is a no-op."""
    with capture_logs():
        assert migrate() == []
    with transaction() as cur:
        cur.execute(
            "SELECT TABLE_NAME AS t, ENGINE AS e, TABLE_COLLATION AS c "
            "FROM information_schema.TABLES WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME IN ('crawl_runs','bronze_documents')"
        )
        tables = {r["t"]: r for r in cur.fetchall()}
        cur.execute(
            "SELECT COUNT(*) AS n FROM schema_migrations WHERE version = %s",
            (migrate_mod.LATEST_VERSION,),
        )
        n = cur.fetchone()["n"]
        cur.execute(
            "SELECT INDEX_NAME AS i FROM information_schema.STATISTICS "
            "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'bronze_documents'"
        )
        indexes = {r["i"] for r in cur.fetchall()}
    assert set(tables) == {"crawl_runs", "bronze_documents"}
    assert all(t["e"] == "InnoDB" and t["c"].startswith("utf8mb4") for t in tables.values())
    assert n == 1
    assert {
        "uq_bronze_documents_url_sha",
        "ix_bronze_documents_silver",
        "ix_bronze_documents_run",
        "ix_bronze_documents_source",
    } <= indexes


def test_save_batch_dedupes_on_url_sha(clean):
    """AC-005.4: idempotent on url_sha256; returns newly inserted rows only."""
    repo = BronzeRepository()
    run_id = repo.start_run({"test": True})
    assert repo.save_batch(run_id, [_doc(1), _doc(2), _doc(1)]) == 2
    assert repo.save_batch(run_id, [_doc(2), _doc(3)]) == 1
    assert repo.save_batch(run_id, [_doc(1), _doc(2), _doc(3)]) == 0
    assert repo.count() == 3
    docs = list(repo.iter_unprocessed())
    assert docs[0].title.endswith("😀") and docs[0].run_id == run_id
    assert docs[0].fetched_at.tzinfo is not None


def test_iter_unprocessed_and_mark_processed(clean):
    repo = BronzeRepository()
    run_id = repo.start_run({})
    repo.save_batch(run_id, [_doc(i) for i in range(1, 8)])
    ids = [d.id for d in repo.iter_unprocessed(layer="silver")]
    assert ids == sorted(ids) and len(ids) == 7
    assert [d.id for d in repo.iter_unprocessed(limit=3)] == ids[:3]
    # marking while iterating does not skip or repeat rows
    seen = []
    for d in repo.iter_unprocessed():
        seen.append(d.id)
        if d.id % 2:
            repo.mark_processed([d.id], layer="silver")
    assert seen == ids
    assert repo.mark_processed(ids[:2]) == 1  # one of them already processed
    remaining = [d.id for d in repo.iter_unprocessed()]
    assert all(i % 2 == 0 for i in remaining) and ids[1] not in remaining
    assert repo.count() == 7 and repo.count(unprocessed_layer="silver") == len(remaining)


def test_run_bronze_end_to_end_with_fixture_sources(clean, tmp_path):
    """AC-005.2/4: real sources + fetcher over a mock transport -> <= max_docs rows, run recorded,
    second run inserts nothing new."""
    (tmp_path / "memo.txt").write_text("SYNTHETIC FIXTURE inbox memo about TCS", "utf-8")
    repo = BronzeRepository()

    def sources(f: PoliteFetcher):
        return [LocalInbox(tmp_path), GdeltDoc(f), HackerNews(f)]

    f = _fetcher()
    with capture_logs():
        stats = run_bronze(
            max_docs=8,
            max_depth=2,
            queries=["TCS data breach"],
            sources=sources(f),
            fetcher=f,
            per_query_limit=5,
        )
    assert stats.run_id is not None and 0 < stats.stored <= 8
    assert repo.count() == stats.stored
    run = repo.get_run(stats.run_id)
    assert run["finished_at"] is not None and run["params"]["max_docs"] == 8
    assert run["stats"]["stored"] == stats.stored
    with transaction() as cur:
        cur.execute(
            "SELECT depth, source, run_id, url_sha256, content_sha256 FROM bronze_documents"
        )
        rows = cur.fetchall()
    assert max(r["depth"] for r in rows) <= 2
    assert all(r["run_id"] == stats.run_id for r in rows)
    assert {"local_inbox", "gdelt"} <= {r["source"] for r in rows}

    f2 = _fetcher()
    with capture_logs():
        again = run_bronze(
            max_docs=8,
            max_depth=2,
            queries=["TCS data breach"],
            sources=sources(f2),
            fetcher=f2,
            per_query_limit=5,
        )
    assert again.stored == 0 and again.duplicates >= stats.stored
    assert repo.count() == stats.stored


def test_to_bronze_round_trip(clean):
    from gfinance_crawler.news.models import RawDocument

    raw = RawDocument(
        url="https://news.test/x?utm_source=a",
        canonical_url="https://news.test/x",
        depth=1,
        parent_url="https://news.test/parent",
        source="bing_news",
        source_type="news",
        raw_content="SYNTHETIC FIXTURE",
        follow_url="https://ignored.test",
    )
    repo = BronzeRepository()
    rid = repo.start_run({})
    assert repo.save_batch(rid, [to_bronze(raw, rid)]) == 1
    (doc,) = list(repo.iter_unprocessed())
    assert doc.url == raw.url and doc.parent_url == raw.parent_url and doc.depth == 1
