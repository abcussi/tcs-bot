"""Unit tests for the bronze layer (AC-005.1, AC-005.2, AC-005.4): RawDocument -> BronzeDocument
mapping, repository SQL contract (fake cursor) and ``run_bronze`` wiring with fake sources /
fetcher / repository. No network, no MySQL."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

import pytest
from gfinance_crawler.news.models import FetchResult, RawDocument
from gfinance_crawler.news.sources import LocalInbox
from structlog.testing import capture_logs

from tcs_bot.medallion import bronze as bronze_mod
from tcs_bot.medallion.bronze import (
    BATCH_SIZE,
    BronzeRepository,
    default_sources,
    run_bronze,
    sha256_hex,
    to_bronze,
    url_sha256,
)
from tcs_bot.medallion.models import BronzeDocument, CrawlStats

ARTICLE = (
    "<html><head><title>SYNTHETIC: TCS fictional fine</title></head><body>"
    "<p>SYNTHETIC FIXTURE. Tata Consultancy Services test text.</p></body></html>"
)


def _raw(i: int = 1, **kw: Any) -> RawDocument:
    url = kw.pop("url", f"https://news.test/item/{i}")
    base: dict[str, Any] = {
        "url": url,
        "canonical_url": url,
        "depth": 0,
        "source": "fake",
        "source_type": "news",
        "query": "TCS fine",
        "title": f"SYNTHETIC: TCS item {i}",
        "raw_content": f"<item>SYNTHETIC TCS item {i}</item>",
        "content_type": "application/rss+xml",
        "follow_url": f"https://news.test/article/{i}",
    }
    base.update(kw)
    return RawDocument(**base)


# --------------------------------------------------------------------------- mapper
def test_to_bronze_maps_every_field_and_drops_follow_url():
    """AC-005.4: crawler RawDocument maps 1:1 to the shared BronzeDocument contract."""
    pub = datetime(2026, 11, 1, 6, 15, tzinfo=UTC)
    raw = _raw(7, parent_url="https://p.test", depth=1, published_at=pub, http_status=203)
    doc = to_bronze(raw, run_id=42)
    assert isinstance(doc, BronzeDocument) and doc.id is None and doc.run_id == 42
    expected = raw.model_dump()
    assert "follow_url" not in expected
    assert doc.model_dump(exclude={"id", "run_id"}) == expected
    assert set(BronzeDocument.model_fields) - {"id", "run_id"} == set(expected)


def test_url_sha_uses_canonical_url():
    a = to_bronze(_raw(1, url="https://x.test/a?utm_source=z", canonical_url="https://x.test/a"), 1)
    assert url_sha256(a) == sha256_hex("https://x.test/a")


# --------------------------------------------------------------------------- repository SQL
class FakeCursor:
    def __init__(self, existing_shas: set[str] | None = None, rows: list[list[dict]] | None = None):
        self.existing = existing_shas or set()
        self.pages = list(rows or [])
        self.executed: list[tuple[str, Any]] = []
        self._result: list[dict] = []
        self.lastrowid = 9
        self.rowcount = 0

    def execute(self, sql: str, params: Any = None) -> None:
        self.executed.append((sql, params))
        if sql.startswith("SELECT url_sha256"):
            self._result = [{"url_sha256": p} for p in params if p in self.existing]
        elif sql.startswith("SELECT id, run_id"):
            self._result = (self.pages.pop(0) if self.pages else [])[: params[1]]
        elif sql.startswith("UPDATE bronze_documents"):
            self.rowcount = len(params)
        elif sql.startswith("SELECT COUNT"):
            self._result = [{"n": 3}]

    def fetchall(self) -> list[dict]:
        return self._result

    def fetchone(self) -> dict | None:
        return self._result[0] if self._result else None


@pytest.fixture
def cursor(monkeypatch) -> FakeCursor:
    cur = FakeCursor()
    transactions: list[int] = []

    @contextmanager
    def fake_tx():
        transactions.append(1)
        yield cur

    monkeypatch.setattr(bronze_mod, "transaction", fake_tx)
    cur.transactions = transactions  # type: ignore[attr-defined]
    return cur


def test_save_batch_one_transaction_upsert_and_new_count(cursor):
    """AC-005.4: one transaction per batch; INSERT ... ON DUPLICATE KEY UPDATE on url_sha256;
    returns only newly inserted rows; in-batch duplicates collapse."""
    docs = [to_bronze(_raw(i), 5) for i in (1, 2, 3)] + [to_bronze(_raw(1), 5)]
    cursor.existing = {url_sha256(docs[1])}
    assert BronzeRepository().save_batch(5, docs) == 2
    inserts = [p for s, p in cursor.executed if s.startswith("INSERT INTO bronze_documents")]
    assert len(inserts) == 2 and len(cursor.transactions) == 1
    sql = next(s for s, _ in cursor.executed if s.startswith("INSERT INTO bronze_documents"))
    assert "ON DUPLICATE KEY UPDATE id = id" in sql
    first = inserts[0]
    assert first[0] == 5 and first[2] == url_sha256(docs[0])
    assert first[-1] == sha256_hex(docs[0].raw_content)
    assert first[-2].tzinfo is None  # naive UTC for DATETIME
    assert BronzeRepository().save_batch(5, []) == 0


def test_iter_unprocessed_keyset_pages_and_limit(cursor, monkeypatch):
    monkeypatch.setattr(bronze_mod, "_PAGE", 2)

    def row(i: int) -> dict:
        return {
            "id": i,
            "run_id": 1,
            "url": f"https://n.test/{i}",
            "canonical_url": f"https://n.test/{i}",
            "parent_url": None,
            "depth": 0,
            "source": "gdelt",
            "source_type": "news",
            "query": "q",
            "title": None,
            "raw_content": "x",
            "content_type": "text/html",
            "http_status": 200,
            "published_at": datetime(2026, 1, 1),
            "fetched_at": datetime(2026, 1, 2),
        }

    cursor.pages = [[row(1), row(2)], [row(5)]]
    docs = list(BronzeRepository().iter_unprocessed("silver"))
    assert [d.id for d in docs] == [1, 2, 5]
    assert docs[0].published_at == datetime(2026, 1, 1, tzinfo=UTC)
    selects = [(s, p) for s, p in cursor.executed if s.startswith("SELECT id, run_id")]
    assert "silver_processed_at IS NULL" in selects[0][0] and "ORDER BY id" in selects[0][0]
    assert [p for _, p in selects] == [(0, 2), (2, 2)]  # keyset: id > last id
    cursor.pages = [[row(1), row(2)], [row(3), row(4)]]
    assert [d.id for d in BronzeRepository().iter_unprocessed(limit=3)] == [1, 2, 3]
    with pytest.raises(ValueError):
        list(BronzeRepository().iter_unprocessed("gold; DROP TABLE x"))


def test_mark_processed_count_and_runs(cursor):
    repo = BronzeRepository()
    assert repo.mark_processed([3, 1, 3], layer="silver") == 2
    sql, params = cursor.executed[-1]
    assert "SET silver_processed_at = UTC_TIMESTAMP()" in sql and params == (1, 3)
    assert repo.mark_processed([]) == 0
    with pytest.raises(ValueError):
        repo.mark_processed([1], layer="bogus")
    assert repo.count() == 3
    assert repo.count(unprocessed_layer="silver") == 3
    assert "silver_processed_at IS NULL" in cursor.executed[-1][0]
    assert repo.start_run({"max_docs": 5}) == 9
    repo.finish_run(9, CrawlStats(run_id=9, stored=2))
    sql, params = cursor.executed[-1]
    assert sql.startswith("UPDATE crawl_runs") and '"stored": 2' in params[0] and params[1] == 9


# --------------------------------------------------------------------------- run_bronze wiring
class FakeRepo:
    def __init__(self, known: set[str] | None = None, fail_on_save: bool = False) -> None:
        self.known = set(known or ())
        self.rows: list[BronzeDocument] = []
        self.batches: list[int] = []
        self.runs: dict[int, dict[str, Any]] = {}
        self.fail_on_save = fail_on_save

    def start_run(self, params: dict[str, Any]) -> int:
        rid = len(self.runs) + 1
        self.runs[rid] = {"params": params, "stats": None}
        return rid

    def finish_run(self, run_id: int, stats: Any) -> None:
        self.runs[run_id]["stats"] = stats

    def save_batch(self, run_id: int | None, docs: list[BronzeDocument]) -> int:
        if self.fail_on_save:
            raise RuntimeError("mysql down")
        self.batches.append(len(docs))
        new = 0
        for d in docs:
            key = d.canonical_url
            if key in self.known:
                continue
            self.known.add(key)
            self.rows.append(d)
            new += 1
        return new


class FakeSource:
    source_type = "news"

    def __init__(self, name: str, n: int) -> None:
        self.name = name
        self.n = n
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, limit: int) -> list[RawDocument]:
        self.calls.append((query, limit))
        return [
            _raw(
                i,
                url=f"https://{self.name}.test/{query.replace(' ', '-')}/{i}",
                source=self.name,
                query=query,
                follow_url=f"https://news.test/article/{self.name}/{i}",
            )
            for i in range(min(self.n, limit))
        ]


class FakeFetcher:
    def __init__(self) -> None:
        self.urls: list[str] = []
        self.closed = False

    def fetch(self, url: str) -> FetchResult:
        self.urls.append(url)
        return FetchResult(
            url=url, final_url=url, status=200, content_type="text/html", text=ARTICLE
        )

    def close(self) -> None:
        self.closed = True


def test_run_bronze_wiring_batches_caps_and_stats():
    """AC-005.2/4: crawl -> batches of 25 -> repo; caps respected; run recorded with stats."""
    repo, fetcher = FakeRepo(), FakeFetcher()
    sources = [FakeSource("alpha", 20), FakeSource("beta", 20)]
    with capture_logs():
        stats = run_bronze(
            max_docs=60,
            max_depth=1,
            queries=["TCS fine", "TCS layoffs"],
            sources=sources,
            fetcher=fetcher,
            repo=repo,
            per_query_limit=10,
        )
    assert isinstance(stats, CrawlStats) and stats.run_id == 1
    assert stats.stored == len(repo.rows) == 60
    assert repo.batches == [BATCH_SIZE, BATCH_SIZE, 10]
    assert all(d.run_id == 1 for d in repo.rows) and max(d.depth for d in repo.rows) == 1
    assert stats.by_depth == {"0": 40, "1": 20}
    run = repo.runs[1]
    assert run["params"]["max_docs"] == 60 and run["params"]["max_depth"] == 1
    assert run["params"]["sources"] == ["alpha", "beta"]
    assert run["stats"]["stored"] == 60 and run["stats"]["yielded"] == 60
    assert not fetcher.closed  # injected fetcher is not ours to close
    assert sources[0].calls == [("TCS fine", 10), ("TCS layoffs", 10)]


def test_run_bronze_counts_rows_already_in_bronze_as_duplicates():
    repo = FakeRepo(known={"https://alpha.test/TCS-fine/0", "https://alpha.test/TCS-fine/1"})
    with capture_logs():
        stats = run_bronze(
            max_docs=5,
            max_depth=0,
            queries=["TCS fine"],
            sources=[FakeSource("alpha", 5)],
            fetcher=FakeFetcher(),
            repo=repo,
            per_query_limit=5,
        )
    assert stats.stored == 3 and stats.duplicates == 2


def test_run_bronze_clamps_to_hard_caps_and_settings(monkeypatch):
    monkeypatch.setenv("MEDALLION_MAX_DOCS", "7")
    monkeypatch.setenv("MEDALLION_MAX_DEPTH", "0")
    from tcs_bot.config import get_settings

    get_settings.cache_clear()
    repo = FakeRepo()
    with capture_logs():
        stats = run_bronze(
            queries=["TCS fine"],
            sources=[FakeSource("alpha", 50)],
            fetcher=FakeFetcher(),
            repo=repo,
            per_query_limit=50,
        )
    assert stats.stored == 7 and repo.runs[1]["params"]["max_depth"] == 0
    with capture_logs():
        run_bronze(
            max_docs=10_000,
            max_depth=9,
            queries=["q"],
            sources=[],
            fetcher=FakeFetcher(),
            repo=repo,
        )
    assert repo.runs[2]["params"]["max_docs"] == 500 and repo.runs[2]["params"]["max_depth"] == 2


def test_run_bronze_records_run_even_when_saving_fails():
    repo = FakeRepo(fail_on_save=True)
    with capture_logs(), pytest.raises(RuntimeError, match="mysql down"):
        run_bronze(
            max_docs=3,
            max_depth=0,
            queries=["TCS fine"],
            sources=[FakeSource("alpha", 5)],
            fetcher=FakeFetcher(),
            repo=repo,
        )
    stats = repo.runs[1]["stats"]
    assert stats["error"].startswith("RuntimeError") and stats["stored"] == 0


def test_default_sources_include_all_spec_sources_and_inbox(tmp_path):
    """AC-005.1: Google News, Bing, GDELT, HN, Reddit + LocalInbox."""
    names = [s.name for s in default_sources(FakeFetcher(), tmp_path)]  # type: ignore[arg-type]
    assert names == ["local_inbox", "google_news", "bing_news", "gdelt", "hackernews", "reddit"]
    assert "local_inbox" not in [s.name for s in default_sources(FakeFetcher(), None)]  # type: ignore[arg-type]


def test_run_bronze_with_inbox_only(tmp_path):
    (tmp_path / "memo.txt").write_text("SYNTHETIC FIXTURE memo about TCS", "utf-8")
    repo = FakeRepo()
    with capture_logs():
        stats = run_bronze(
            max_docs=10,
            queries=["TCS fine"],
            sources=[LocalInbox(tmp_path)],
            fetcher=FakeFetcher(),
            repo=repo,
        )
    assert stats.stored == 1 and repo.rows[0].source == "local_inbox"
    assert repo.rows[0].source_type == "other" and repo.rows[0].url.startswith("file://")
