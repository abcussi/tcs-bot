"""Typed read layer over bronze / silver / gold: facade, API and MCP tools (AC-005.12), offline.

SYNTHETIC FIXTURE data only (``tests/unit/test_gold_synthetic.py`` corpus plus a labelled
near-duplicate and three audit-only documents). Silver is an in-memory ES double
(``fake_silver_es``), gold is built into ``tmp_path`` with the in-memory gold ES double, bronze is
an in-memory reader. No network, no infrastructure.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from tcs_bot.api.app import create_app
from tcs_bot.config import get_settings
from tcs_bot.errors import DependencyUnavailable, NotFoundError
from tcs_bot.mcp_server import server
from tcs_bot.medallion.gold.pipeline import run_gold
from tcs_bot.medallion.gold.vector_store import GoldVectorStore
from tcs_bot.medallion.models import RISK_CATEGORIES, SilverRecord
from tcs_bot.medallion.read import (
    BronzeLineage,
    DocumentFilters,
    MysqlBronzeReader,
    Page,
    TcsData,
    set_tcs_data,
)
from tcs_bot.medallion.read import gold_graph as gg
from tcs_bot.medallion.silver.store import SilverStore
from tcs_bot.services.report_service import ReportService

from .fake_es import DownES, FakeES
from .fake_silver_es import DownSilverES, FakeSilverES, MissingIndexES
from .test_gold_synthetic import FakeSilverStore, synthetic_records, topic_of

HEALTHY = {"mysql": lambda: True, "redis": lambda: True, "elasticsearch": lambda: True}
SILVER_INDEX = "silver-tcs-risk"
DUP_ID = 2001  # SYNTHETIC FIXTURE: copy of the first record, folded away by gold dedupe
IRRELEVANT_IDS = (3001, 3002, 3003)
NO_BRONZE_ID = 3003  # silver document without a bronze row


def _raw(doc_id: int) -> str:
    return f"<html>SYNTHETIC FIXTURE raw body {doc_id}</html>"


def fixture_records() -> list[SilverRecord]:
    """33 SYNTHETIC FIXTURE silver records: 30 topical + 1 duplicate + 3 audit-only."""
    recs = synthetic_records()
    dup = recs[0].model_copy(
        update={
            "doc_id": DUP_ID,
            "source": "synthetic_dup",
            "url": "https://synthetic-dup.example/cyber/0",
            "clean_text": recs[0].clean_text,  # same text and title -> gold dedupe folds one away
        }
    )
    audit = [
        recs[1].model_copy(
            update={
                "doc_id": i,
                "title": f"[SYNTHETIC FIXTURE] unrelated audit-only item {i}",
                "url": f"https://synthetic-audit.example/{i}",
                "relevant": False,
                "tcs_relevance": 0.1,
                "risk_score": 5.0 + n,
                "risk_signals": [],
                "risk_categories": [],
                "published_at": None,
            }
        )
        for n, i in enumerate(IRRELEVANT_IDS)
    ]
    return [*recs, dup, *audit]


class FakeBronze:
    """In-memory ``BronzeReader``: lineage rows; the body only comes back for ``include_raw``."""

    def __init__(self, records: list[SilverRecord]) -> None:
        self.rows = {
            r.doc_id: BronzeLineage(
                id=r.doc_id,
                run_id=7,
                url=r.url,
                canonical_url=r.url,
                parent_url=r.parent_url,
                depth=r.depth,
                source=r.source,
                source_type=r.source_type,
                query="synthetic query",
                title=r.title,
                content_type="text/html",
                http_status=200,
                published_at=r.published_at,
                fetched_at=datetime(2026, 10, 1, tzinfo=UTC),
                raw_chars=len(_raw(r.doc_id)),
            )
            for r in records
            if r.doc_id != NO_BRONZE_ID
        }
        self.raw_requests: list[bool] = []

    def get(self, doc_id: int, include_raw: bool = False) -> BronzeLineage | None:
        self.raw_requests.append(include_raw)
        row = self.rows.get(doc_id)
        if row is None:
            return None
        return row.model_copy(update={"raw_content": _raw(doc_id) if include_raw else None})


@dataclass
class Env:
    data: TcsData
    gold_vs: GoldVectorStore
    silver_es: FakeSilverES
    bronze: FakeBronze
    records: list[SilverRecord]

    @property
    def relevant(self) -> list[SilverRecord]:
        return [r for r in self.records if r.relevant]

    def build_gold(self) -> dict:
        return run_gold(FakeSilverStore(self.records), self.gold_vs)


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Env]:
    monkeypatch.setenv("GOLD_DIR", str(tmp_path / "gold"))
    get_settings.cache_clear()
    gg.clear_cache()
    records = fixture_records()
    silver_es = FakeSilverES().put(*[SilverStore.to_source(r) for r in records])
    gold_vs = GoldVectorStore(es=FakeES(), prefix="gold-tcs-risk")
    bronze = FakeBronze(records)
    data = TcsData(
        silver=SilverStore(es=silver_es, index=SILVER_INDEX),  # type: ignore[arg-type]
        vector_store=gold_vs,
        bronze=bronze,
    )
    set_tcs_data(data)
    yield Env(data, gold_vs, silver_es, bronze, records)
    set_tcs_data(None)
    gg.clear_cache()
    get_settings.cache_clear()


@pytest.fixture
def built(env: Env) -> Env:
    env.build_gold()
    return env


@pytest.fixture
def client(service: ReportService) -> Iterator[TestClient]:
    with TestClient(create_app(service, health_checks=HEALTHY), raise_server_exceptions=False) as c:
        yield c


def ids(page: Any) -> list[int]:
    return [d.doc_id for d in page.items]


def listing(env: Env, **filters: Any) -> list[int]:
    """All matching doc ids (one big page), in the default order."""
    return ids(env.data.list_documents(DocumentFilters(**filters), Page(limit=100)))


def blob(r: SilverRecord) -> str:
    return f"{r.title} {r.summary} {r.clean_text}".lower()


# ------------------------------------------------------------------ list_documents
def test_list_defaults_to_relevant_only_with_total(env: Env) -> None:
    """AC-005.12: relevant docs by default, total counts every match, rows carry no body."""
    page = env.data.list_documents()
    assert page.total == 31 and page.limit == 20 and page.offset == 0 and len(page.items) == 20
    assert all(d.relevant for d in page.items)
    assert not hasattr(page.items[0], "clean_text")
    all_rows = env.data.list_documents(DocumentFilters(relevant=None), Page(limit=100))
    assert all_rows.total == 34 and set(IRRELEVANT_IDS) <= set(ids(all_rows))
    audit = env.data.list_documents(DocumentFilters(relevant=False), Page(limit=100))
    assert sorted(ids(audit)) == sorted(IRRELEVANT_IDS)
    # the list query asks ES to drop the heavy fields
    assert env.silver_es.calls[0]["source_excludes"] == ["clean_text", "entities"]


def test_list_filters_match_an_independent_oracle(env: Env) -> None:
    """AC-005.12: category/severity/risk/source/source_type/date/q filters, ANDed."""
    rel = env.relevant

    def want(pred: Any) -> list[int]:
        return sorted(r.doc_id for r in rel if pred(r))

    got = listing(env, category=["labor_hr", "legal_litigation"])
    assert sorted(got) == want(lambda r: {"labor_hr", "legal_litigation"} & set(r.risk_categories))
    assert 0 < len(got) < len(rel)
    sev = want(lambda r: any(s.severity == "high" for s in r.risk_signals))
    assert sorted(listing(env, min_severity="high")) == sev and sev
    assert sorted(listing(env, min_severity="medium")) == want(lambda r: bool(r.risk_signals))
    assert listing(env, min_severity="critical") == []
    assert sorted(listing(env, min_risk_score=70)) == want(lambda r: r.risk_score >= 70)
    assert sorted(listing(env, source=["synthetic_forum", "synthetic_wire"])) == want(
        lambda r: r.source in {"synthetic_forum", "synthetic_wire"}
    )
    assert sorted(listing(env, source_type=["community"])) == want(
        lambda r: r.source_type == "community"
    )
    lo, hi = datetime(2026, 3, 1, tzinfo=UTC), datetime(2026, 5, 31, tzinfo=UTC)
    assert sorted(listing(env, published_from=lo, published_to=hi)) == want(
        lambda r: r.published_at is not None and lo <= r.published_at <= hi
    )
    assert sorted(listing(env, q="ransomware")) == want(lambda r: "ransomware" in blob(r))
    assert sorted(listing(env, q="jury verdict")) == want(
        lambda r: "jury" in blob(r) and "verdict" in blob(r)
    )
    combo = listing(env, category=["labor_hr"], min_risk_score=40, source_type=["news"])
    assert sorted(combo) == want(
        lambda r: "labor_hr" in r.risk_categories and r.risk_score >= 40 and r.source_type == "news"
    )


def test_list_sort_and_pagination_are_stable(env: Env) -> None:
    """AC-005.12: sort by risk/date/relevance asc|desc; offset pages never overlap or skip."""
    by_risk = env.data.list_documents(page=Page(limit=100))
    scores = [d.risk_score for d in by_risk.items]
    assert scores == sorted(scores, reverse=True)
    asc = env.data.list_documents(page=Page(limit=100, order="asc"))
    assert [d.risk_score for d in asc.items] == sorted(scores)
    dated = env.data.list_documents(page=Page(limit=100, sort="published_at", order="asc"))
    stamps = [d.published_at for d in dated.items if d.published_at]
    assert stamps == sorted(stamps) and len(stamps) == 31
    rel = env.data.list_documents(page=Page(limit=5, sort="tcs_relevance"))
    assert len(rel.items) == 5
    seen: list[int] = []
    for offset in range(0, 40, 7):
        chunk = env.data.list_documents(page=Page(limit=7, offset=offset))
        assert chunk.total == 31
        seen += ids(chunk)
    assert seen == ids(by_risk) and len(set(seen)) == 31
    assert env.data.list_documents(page=Page(limit=10, offset=100)).items == []


def test_page_and_filter_input_validation() -> None:
    """AC-005.12: limits are capped and inconsistent filters are rejected."""
    with pytest.raises(ValidationError):
        Page(limit=101)
    with pytest.raises(ValidationError):
        Page(limit=0)
    with pytest.raises(ValidationError):
        Page(limit=100, offset=9_950)  # beyond the ES result window
    with pytest.raises(ValidationError):
        DocumentFilters(category=["aliens"])  # type: ignore[list-item]
    with pytest.raises(ValidationError):
        DocumentFilters(min_risk_score=101)
    with pytest.raises(ValidationError):
        DocumentFilters(published_from=datetime(2026, 2, 1), published_to=datetime(2026, 1, 1))
    assert DocumentFilters(q="   ").q is None


def test_list_by_cluster_and_community_uses_gold(built: Env) -> None:
    """AC-005.12: cluster / community filters resolve gold members; they need gold."""
    summary = json.loads((Path(get_settings().gold_dir) / "summary.json").read_text("utf-8"))
    for c in summary["clusters"]:
        page = built.data.list_documents(DocumentFilters(cluster=c["cluster"]), Page(limit=100))
        assert page.total == c["size"] == len(page.items)
    comm = summary["communities"][0]
    assert listing(built, community=comm["cluster"]) != []
    both = listing(built, cluster=summary["clusters"][0]["cluster"], community=comm["cluster"])
    assert set(both) <= set(listing(built, community=comm["cluster"]))
    assert listing(built, cluster=999) == []  # unknown id -> empty, not an error


def test_cluster_filter_without_gold_is_gold_not_built(env: Env) -> None:
    with pytest.raises(NotFoundError) as exc:
        env.data.list_documents(DocumentFilters(cluster=0))
    assert exc.value.code == "gold_not_built" and exc.value.status_code == 404


# ------------------------------------------------------------------ get_document
def test_get_document_joins_bronze_silver_gold(built: Env) -> None:
    """AC-005.12: one id -> silver (signals + evidence), bronze lineage, gold node."""
    graph = gg.load_gold_graph()
    doc_id = next(i for i in graph.nodes if i != DUP_ID and graph.nodes[i]["signals"])
    rec = next(r for r in built.records if r.doc_id == doc_id)
    d = built.data.get_document(doc_id)
    assert d.doc_id == doc_id == d.silver.doc_id == d.bronze.id  # type: ignore[union-attr]
    assert d.silver.title == rec.title and d.silver.risk_score == rec.risk_score
    assert d.silver.risk_signals[0].evidence == rec.risk_signals[0].evidence
    assert d.silver.clean_text is None  # body only on request
    b = d.bronze
    assert b is not None and b.run_id == 7 and b.http_status == 200 and b.query
    assert (b.url, b.depth, b.parent_url, b.source) == (rec.url, rec.depth, None, rec.source)
    assert b.fetched_at is not None and b.raw_content is None and b.raw_chars == len(_raw(doc_id))
    node = graph.nodes[doc_id]
    g = d.gold
    assert g is not None and g.in_gold and g.merged_into is None
    assert (g.cluster, g.community) == (node["cluster"], node["community"])
    assert g.cluster_label and g.degree == node["degree"] and g.also_seen_in == node["also_seen_in"]
    assert [list(n) for n in g.neighbors] == node["neighbors"]


def test_get_document_raw_and_text_are_opt_in(built: Env) -> None:
    """AC-005.12: raw_content / clean_text only with include_raw / include_text."""
    doc_id = 1001
    assert built.data.get_document(doc_id).bronze.raw_content is None  # type: ignore[union-attr]
    full = built.data.get_document(doc_id, include_raw=True, include_text=True)
    assert full.bronze is not None and full.bronze.raw_content == _raw(doc_id)
    assert full.silver.clean_text and built.bronze.raw_requests == [False, True]
    assert "raw_content" in full.bronze.model_dump()  # field exists, populated only on request


def test_get_document_merged_duplicate_points_at_kept_doc(built: Env) -> None:
    """AC-005.12: a document folded away by gold dedupe reports merged_into."""
    graph = gg.load_gold_graph()
    dropped = {1001, DUP_ID} - set(graph.nodes)
    assert len(dropped) == 1
    gone = dropped.pop()
    kept = next(iter({1001, DUP_ID} - {gone}))
    d = built.data.get_document(gone)
    assert d.gold is not None and not d.gold.in_gold and d.gold.merged_into == kept
    k = built.data.get_document(kept)
    assert k.gold is not None and k.gold.in_gold
    assert k.gold.merged_doc_ids == [gone] and k.gold.also_seen_in == 1


def test_get_document_without_gold_or_bronze_row(env: Env) -> None:
    """AC-005.12: gold missing -> gold is None (not an error); missing bronze row -> None."""
    d = env.data.get_document(NO_BRONZE_ID)
    assert d.silver.relevant is False and d.bronze is None and d.gold is None


def test_get_document_not_found_and_dependency_errors(env: Env) -> None:
    with pytest.raises(NotFoundError) as exc:
        env.data.get_document(424242)
    assert exc.value.code == "document_not_found"
    down = TcsData(silver=SilverStore(es=DownSilverES(), index=SILVER_INDEX), bronze=env.bronze)  # type: ignore[arg-type]
    with pytest.raises(DependencyUnavailable) as dep:
        down.get_document(1001)
    assert dep.value.code == "silver_unavailable" and dep.value.status_code == 503
    with pytest.raises(DependencyUnavailable):
        down.list_documents()
    missing = TcsData(silver=SilverStore(es=MissingIndexES(), index=SILVER_INDEX))  # type: ignore[arg-type]
    with pytest.raises(NotFoundError) as nf:
        missing.list_documents()
    assert nf.value.code == "silver_not_built"


class _Cursor:
    def __init__(self, row: dict[str, Any] | None, sink: list[tuple[str, tuple]]) -> None:
        self.row, self.sink = row, sink

    def execute(self, sql: str, params: tuple = ()) -> None:
        self.sink.append((sql, params))

    def fetchone(self) -> dict[str, Any] | None:
        return self.row


def _patch_transaction(
    monkeypatch: pytest.MonkeyPatch, row: dict[str, Any] | None, sink: list[tuple[str, tuple]]
) -> None:
    @contextmanager
    def fake_tx() -> Iterator[_Cursor]:
        yield _Cursor(row, sink)

    monkeypatch.setattr("tcs_bot.db.mysql.transaction", fake_tx)


def test_mysql_bronze_reader_never_selects_the_body_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC-005.12: SQL excludes raw_content unless include_raw; DATETIMEs come back UTC-aware."""
    row = {
        "id": 5, "run_id": 2, "url": "https://synthetic.example/a", "parent_url": None,
        "canonical_url": "https://synthetic.example/a", "depth": 1, "source": "bing_news",
        "source_type": "news", "query": "synthetic", "title": "[SYNTHETIC FIXTURE]",
        "content_type": "text/html", "http_status": 200, "raw_chars": 12,
        "published_at": datetime(2026, 1, 2, 3, 4, 5), "fetched_at": datetime(2026, 1, 3),
        "silver_processed_at": None,
    }  # fmt: skip
    sink: list[tuple[str, tuple]] = []
    _patch_transaction(monkeypatch, row, sink)
    lineage = MysqlBronzeReader().get(5)
    assert lineage is not None and lineage.raw_content is None and lineage.raw_chars == 12
    assert lineage.published_at == datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
    assert "raw_content" not in sink[0][0].replace("CHAR_LENGTH(raw_content)", "")
    assert sink[0][1] == (5,)
    with_raw = MysqlBronzeReader().get(5, include_raw=True)
    assert with_raw is not None and "raw_content" in sink[1][0].split("FROM")[0]


def test_mysql_bronze_reader_missing_row_and_mysql_down(monkeypatch: pytest.MonkeyPatch) -> None:
    from mysql.connector import errors as mysql_errors

    _patch_transaction(monkeypatch, None, [])
    assert MysqlBronzeReader().get(1) is None

    @contextmanager
    def boom() -> Iterator[None]:
        raise mysql_errors.InterfaceError("Can't connect (SYNTHETIC FIXTURE)")
        yield

    monkeypatch.setattr("tcs_bot.db.mysql.transaction", boom)
    with pytest.raises(DependencyUnavailable) as exc:
        MysqlBronzeReader().get(1)
    assert exc.value.code == "bronze_unavailable" and exc.value.status_code == 503


# ------------------------------------------------------------------ similar / search
def _cos(a: list[float], b: list[float]) -> float:
    x, y = np.asarray(a), np.asarray(b)
    return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))


def test_similar_documents_are_cosine_ordered_and_exclude_self(built: Env) -> None:
    """AC-005.12: kNN by the stored vector; scores are the cosine; same topic ranks first."""
    graph = gg.load_gold_graph()
    doc_id = next(i for i in sorted(graph.nodes) if i not in (DUP_ID,))
    res = built.data.similar_documents(doc_id, k=6)
    assert res.doc_id == res.resolved_doc_id == doc_id and res.model == "lsa_256"
    hits = res.neighbours
    assert len(hits) == 6 and doc_id not in {h.doc_id for h in hits}
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
    vecs = built.gold_vs.es.indices_by_name[built.gold_vs.docs_collection].visible
    for h in hits:
        expect = _cos(vecs[str(doc_id)]["embedding"], vecs[str(h.doc_id)]["embedding"])
        assert h.score == pytest.approx(expect, abs=1e-4)
    topic = topic_of(next(r for r in built.records if r.doc_id == doc_id))
    top = [topic_of(next(r for r in built.records if r.doc_id == h.doc_id)) for h in hits[:3]]
    assert top == [topic] * 3
    assert len(built.data.similar_documents(doc_id, k=1).neighbours) == 1


def test_similar_documents_resolves_merged_duplicates(built: Env) -> None:
    graph = gg.load_gold_graph()
    gone = ({1001, DUP_ID} - set(graph.nodes)).pop()
    kept = graph.merged_into[gone]
    res = built.data.similar_documents(gone, k=3)
    assert res.doc_id == gone and res.resolved_doc_id == kept
    assert kept not in {h.doc_id for h in res.neighbours}


def test_similar_documents_errors(env: Env) -> None:
    with pytest.raises(NotFoundError) as exc:  # no graph.json yet
        env.data.similar_documents(1001)
    assert exc.value.code == "gold_not_built"
    env.build_gold()
    with pytest.raises(NotFoundError) as nf:  # audit-only document: no vector
        env.data.similar_documents(IRRELEVANT_IDS[0])
    assert nf.value.code == "document_not_in_gold"
    down = TcsData(
        silver=env.data.silver,
        vector_store=GoldVectorStore(es=DownES(), prefix="gold-tcs-risk"),  # type: ignore[arg-type]
        bronze=env.bronze,
    )
    with pytest.raises(DependencyUnavailable) as dep:
        down.similar_documents(1001)
    assert dep.value.code == "vector_store_unavailable"
    env.gold_vs.drop()
    with pytest.raises(NotFoundError) as gone:  # files exist but the ES index was dropped
        env.data.similar_documents(1001)
    assert gone.value.code == "gold_not_built"


def test_search_filters_and_best_chunk_per_document(built: Env) -> None:
    """AC-005.12: semantic search with source / category / cluster filters, one hit per doc."""
    hits = built.data.search("ransomware breach passwords", k=5)
    assert len(hits) == 5 and len({h.doc_id for h in hits}) == 5
    assert built.data.search("   ") == []
    forum = built.data.search(
        "ransomware breach", k=10, filters=DocumentFilters(source=["synthetic_forum"])
    )
    src = {r.doc_id: r for r in built.records}
    assert forum and all(src[h.doc_id].source == "synthetic_forum" for h in forum)
    legal = built.data.search(
        "jury verdict",
        k=10,
        filters=DocumentFilters(category=["legal_litigation"], min_risk_score=0),
    )
    assert legal and all("legal_litigation" in h.risk_categories for h in legal)
    cl = gg.load_gold_graph().nodes[hits[0].doc_id]["cluster"]
    in_cluster = built.data.search("ransomware breach", k=20, filters=DocumentFilters(cluster=cl))
    assert in_cluster and {h.cluster for h in in_cluster} == {cl}


# ------------------------------------------------------------------ analytics
def test_category_breakdown_sums_match_documents(env: Env) -> None:
    """AC-005.12: all 10 categories; counts sum to the category tags of the relevant docs."""
    bd = env.data.category_breakdown()
    assert bd.relevant_only and bd.total_documents == 31
    assert {c.key for c in bd.categories} == set(RISK_CATEGORIES) and len(bd.categories) == 10
    assert sum(c.count for c in bd.categories) == sum(len(r.risk_categories) for r in env.relevant)
    counts = [c.count for c in bd.categories]
    assert counts == sorted(counts, reverse=True)
    for c in bd.categories:
        docs = [r for r in env.relevant if c.key in r.risk_categories]
        assert c.count == len(docs)
        if docs:
            assert c.max_risk_score == pytest.approx(max(r.risk_score for r in docs), abs=0.01)
            mean = sum(r.risk_score for r in docs) / len(docs)
            assert c.avg_risk_score == pytest.approx(mean, abs=0.01)
        else:
            assert c.avg_risk_score == 0 and c.max_risk_score == 0
    everything = env.data.category_breakdown(relevant_only=False)
    assert everything.total_documents == 34 and not everything.relevant_only


def test_risk_overview_aggregations(env: Env) -> None:
    """AC-005.12: totals + by category / severity / source / source type / month + top docs."""
    ov = env.data.risk_overview(top_n=4)
    rel = env.relevant
    assert ov.total_documents == 31
    assert ov.max_risk_score == pytest.approx(max(r.risk_score for r in rel), abs=0.01)
    assert ov.mean_risk_score == pytest.approx(sum(r.risk_score for r in rel) / 31, abs=0.01)
    assert {b.key: b.count for b in ov.by_category} == {
        c: n for c in RISK_CATEGORIES if (n := sum(c in r.risk_categories for r in rel))
    }
    signals = [s.severity for r in rel for s in r.risk_signals]
    assert {b.key: b.count for b in ov.by_severity} == {s: signals.count(s) for s in set(signals)}
    assert [b.key for b in ov.by_severity] == ["high", "medium"]  # most severe first
    assert sum(b.count for b in ov.by_source) == 31 == sum(b.count for b in ov.by_source_type)
    months = {r.published_at.strftime("%Y-%m") for r in rel if r.published_at}
    assert [b.key for b in ov.by_month] == sorted(months)
    assert sum(b.count for b in ov.by_month) == 31
    assert len(ov.top_documents) == 4
    top = [d.risk_score for d in ov.top_documents]
    assert top == sorted(top, reverse=True) and top[0] == pytest.approx(ov.max_risk_score, abs=0.01)
    assert env.data.risk_overview(relevant_only=False).total_documents == 34


# ------------------------------------------------------------------ clusters
def test_clusters_list_and_detail(built: Env) -> None:
    """AC-005.12: summary.json clusters / communities with member documents."""
    summary = json.loads((Path(get_settings().gold_dir) / "summary.json").read_text("utf-8"))
    clusters = built.data.list_clusters()
    assert [c.model_dump() for c in clusters] == summary["clusters"]
    assert [c.cluster for c in built.data.list_clusters("community")] == [
        c["cluster"] for c in summary["communities"]
    ]
    for c in clusters:
        detail = built.data.get_cluster(c.cluster)
        assert detail.kind == "cluster" and detail.summary == c and len(detail.members) == c.size
        risks = [m.risk_score for m in detail.members]
        assert risks == sorted(risks, reverse=True)
        assert set(c.representative_doc_ids) <= {m.doc_id for m in detail.members}
    comm = built.data.get_cluster(summary["communities"][0]["cluster"], "community")
    assert comm.kind == "community" and comm.members
    with pytest.raises(NotFoundError) as exc:
        built.data.get_cluster(999)
    assert exc.value.code == "cluster_not_found"


def test_clusters_need_gold(env: Env) -> None:
    for call in (env.data.list_clusters, lambda: env.data.get_cluster(0)):
        with pytest.raises(NotFoundError) as exc:
            call()
        assert exc.value.code == "gold_not_built"


def test_graph_cache_follows_rebuilds(env: Env) -> None:
    env.build_gold()
    first = gg.load_gold_graph()
    assert gg.load_gold_graph() is first  # cached by mtime
    graph_file = Path(get_settings().gold_dir) / "graph.json"
    data = json.loads(graph_file.read_text("utf-8"))
    data["meta"]["n_docs"] = -1
    graph_file.write_text(json.dumps(data), encoding="utf-8")
    import os

    st = graph_file.stat()
    os.utime(graph_file, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
    assert gg.load_gold_graph().meta["n_docs"] == -1


# ------------------------------------------------------------------ API
def test_api_documents_list_filters_and_envelope(built: Env, client: TestClient) -> None:
    """AC-005.12: GET /tcs/risks/documents typed page, filters, pagination, validation."""
    r = client.get("/tcs/risks/documents", params={"limit": 5, "offset": 2})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"total", "limit", "offset", "items"} and body["total"] == 31
    assert len(body["items"]) == 5 and body["offset"] == 2
    assert "clean_text" not in body["items"][0] and "raw_content" not in body["items"][0]
    r = client.get(
        "/tcs/risks/documents",
        params=[
            ("category", "labor_hr"), ("category", "esg"), ("min_severity", "medium"),
            ("source_type", "news"), ("sort", "published_at"), ("order", "asc"), ("limit", 100),
        ],
    )  # fmt: skip
    rows = r.json()["items"]
    assert r.status_code == 200 and rows
    assert all("labor_hr" in x["risk_categories"] or "esg" in x["risk_categories"] for x in rows)
    assert all(x["source_type"] == "news" for x in rows)
    stamps = [x["published_at"] for x in rows]
    assert stamps == sorted(stamps)
    assert client.get("/tcs/risks/documents", params={"relevant": "all"}).json()["total"] == 34
    assert client.get("/tcs/risks/documents", params={"relevant": "false"}).json()["total"] == 3
    assert client.get("/tcs/risks/documents", params={"q": "ransomware"}).json()["total"] > 0
    assert client.get("/tcs/risks/documents", params={"cluster": 0}).json()["total"] > 0
    r = client.get("/tcs/risks/documents", params={"published_from": "2026-03-01"})
    assert r.status_code == 200 and 0 < r.json()["total"] < 31


@pytest.mark.parametrize(
    "params",
    [
        {"limit": 101},
        {"limit": 0},
        {"offset": -1},
        {"offset": 10_001},
        {"offset": 9_990, "limit": 100},
        {"category": "aliens"},
        {"min_severity": "extreme"},
        {"min_risk_score": 101},
        {"sort": "title"},
        {"order": "sideways"},
        {"relevant": "maybe"},
        {"published_from": "2026-12-01", "published_to": "2026-01-01"},
        {"published_from": "not-a-date"},
    ],
)
def test_api_documents_rejects_bad_params(built: Env, client: TestClient, params: dict) -> None:
    r = client.get("/tcs/risks/documents", params=params)
    assert r.status_code == 422, params
    err = r.json()
    assert set(err) == {"error"} and err["error"]["code"] == "validation_error"


def test_api_document_detail_similar_and_not_found(built: Env, client: TestClient) -> None:
    """AC-005.12: /documents/{id} (+ include flags) and /documents/{id}/similar."""
    doc_id = 1002
    r = client.get(f"/tcs/risks/documents/{doc_id}")
    assert r.status_code == 200
    d = r.json()
    assert set(d) == {"doc_id", "silver", "bronze", "gold"} and d["doc_id"] == doc_id
    assert d["bronze"]["raw_content"] is None and d["silver"]["clean_text"] is None
    assert d["silver"]["risk_signals"][0]["evidence"] and d["gold"]["in_gold"] is True
    full = client.get(
        f"/tcs/risks/documents/{doc_id}", params={"include_raw": True, "include_text": True}
    ).json()
    assert full["bronze"]["raw_content"] == _raw(doc_id) and full["silver"]["clean_text"]

    r = client.get(f"/tcs/risks/documents/{doc_id}/similar", params={"k": 4})
    assert r.status_code == 200
    sim = r.json()
    assert sim["doc_id"] == doc_id and len(sim["neighbours"]) == 4
    scores = [h["score"] for h in sim["neighbours"]]
    assert scores == sorted(scores, reverse=True)

    nf = client.get("/tcs/risks/documents/987654")
    assert nf.status_code == 404 and nf.json()["error"]["code"] == "document_not_found"
    assert client.get("/tcs/risks/documents/abc").status_code == 422
    assert client.get(f"/tcs/risks/documents/{doc_id}/similar", params={"k": 51}).status_code == 422
    no_vec = client.get(f"/tcs/risks/documents/{IRRELEVANT_IDS[0]}/similar")
    assert no_vec.status_code == 404 and no_vec.json()["error"]["code"] == "document_not_in_gold"


def test_api_categories_overview_clusters(built: Env, client: TestClient) -> None:
    """AC-005.12: /categories, /overview, /clusters, /clusters/{id}."""
    cats = client.get("/tcs/risks/categories").json()
    assert cats["total_documents"] == 31 and len(cats["categories"]) == 10
    assert (
        client.get("/tcs/risks/categories", params={"relevant_only": False}).json()[
            "total_documents"
        ]
        == 34
    )
    ov = client.get("/tcs/risks/overview", params={"top_n": 3}).json()
    assert ov["total_documents"] == 31 and len(ov["top_documents"]) == 3 and ov["by_month"]
    assert client.get("/tcs/risks/overview", params={"top_n": 21}).status_code == 422
    clusters = client.get("/tcs/risks/clusters").json()
    assert clusters and {"cluster", "size", "label", "top_terms"} <= set(clusters[0])
    one = client.get(f"/tcs/risks/clusters/{clusters[0]['cluster']}").json()
    assert one["kind"] == "cluster" and len(one["members"]) == clusters[0]["size"]
    comm = client.get("/tcs/risks/clusters", params={"kind": "community"}).json()
    assert (
        comm
        and client.get("/tcs/risks/clusters/0", params={"kind": "community"}).status_code == 200
    )
    assert client.get("/tcs/risks/clusters", params={"kind": "galaxy"}).status_code == 422
    nf = client.get("/tcs/risks/clusters/999")
    assert nf.status_code == 404 and nf.json()["error"]["code"] == "cluster_not_found"


def test_api_existing_gold_routes_unchanged(built: Env, client: TestClient) -> None:
    assert client.get("/tcs/risks/summary").json()["meta"]["n_docs"] == 30
    assert len(client.get("/tcs/risks/graph").json()["nodes"]) == 30
    assert len(client.get("/tcs/risks/search", params={"q": "ransomware", "k": 3}).json()) == 3
    assert client.get("/tcs/risks/graph.html").status_code == 200


def test_api_gold_not_built_and_dependencies_down(env: Env, client: TestClient) -> None:
    """AC-005.12: missing gold -> 404 gold_not_built; ES / MySQL down -> 503 envelopes."""
    for path in (
        "/tcs/risks/clusters",
        "/tcs/risks/clusters/0",
        "/tcs/risks/documents/1001/similar",
    ):
        r = client.get(path)
        assert r.status_code == 404 and r.json()["error"]["code"] == "gold_not_built", path
    assert client.get("/tcs/risks/documents", params={"cluster": 0}).status_code == 404
    assert client.get("/tcs/risks/documents/1001").status_code == 200  # gold optional here

    es_down = TcsData(silver=SilverStore(es=DownSilverES(), index=SILVER_INDEX), bronze=env.bronze)  # type: ignore[arg-type]
    set_tcs_data(es_down)
    for path in ("/tcs/risks/documents", "/tcs/risks/documents/1001", "/tcs/risks/categories",
                 "/tcs/risks/overview"):  # fmt: skip
        r = client.get(path)
        assert r.status_code == 503 and r.json()["error"]["code"] == "silver_unavailable", path

    class DownBronze:
        def get(self, doc_id: int, include_raw: bool = False) -> BronzeLineage | None:
            raise DependencyUnavailable("MySQL down (SYNTHETIC FIXTURE)", code="bronze_unavailable")

    set_tcs_data(TcsData(silver=env.data.silver, bronze=DownBronze()))
    r = client.get("/tcs/risks/documents/1001")
    assert r.status_code == 503 and r.json()["error"]["code"] == "bronze_unavailable"


# ------------------------------------------------------------------ MCP
def test_mcp_list_get_similar_clusters(built: Env) -> None:
    """AC-005.12: MCP tools list / get document / similar / clusters."""
    page = server.list_tcs_risks(category="labor_hr", min_risk_score=30, limit=3)
    assert 0 < len(page.items) <= 3 and page.total >= len(page.items)
    assert all("labor_hr" in d.risk_categories for d in page.items)
    audit = server.list_tcs_risks(include_irrelevant=True, limit=100)
    assert audit.total == 34
    q = server.list_tcs_risks(
        query="ransomware", source_type="news", sort="published_at", order="asc"
    )
    assert q.total > 0
    doc = server.get_tcs_risk_document(page.items[0].doc_id)
    assert doc.silver.clean_text is None and doc.bronze is not None and doc.gold is not None
    assert server.get_tcs_risk_document(page.items[0].doc_id, include_text=True).silver.clean_text
    sim = server.similar_tcs_risks(page.items[0].doc_id, k=3)
    assert len(sim.neighbours) == 3
    listing_ = server.tcs_risk_clusters()
    assert listing_["kind"] == "cluster" and listing_["clusters"]
    one = server.tcs_risk_clusters(cluster_id=listing_["clusters"][0]["cluster"])
    assert one["members"] and one["summary"]["cluster"] == listing_["clusters"][0]["cluster"]
    comm = server.tcs_risk_clusters(kind="community")
    assert comm["kind"] == "community" and comm["clusters"]


def test_mcp_errors_are_tool_errors(env: Env) -> None:
    for kwargs in ({"category": "aliens"}, {"min_severity": "extreme"}, {"limit": 500},
                   {"sort": "title"}, {"published_from": "yesterday-ish"}):  # fmt: skip
        with pytest.raises(server.ToolError, match="validation_error"):
            server.list_tcs_risks(**kwargs)
    with pytest.raises(server.ToolError, match="document_not_found"):
        server.get_tcs_risk_document(424242)
    with pytest.raises(server.ToolError, match="gold_not_built"):
        server.similar_tcs_risks(1001)
    with pytest.raises(server.ToolError, match="gold_not_built"):
        server.tcs_risk_clusters()
    with pytest.raises(server.ToolError, match="validation_error"):
        server.tcs_risk_clusters(kind="galaxy")
    set_tcs_data(TcsData(silver=SilverStore(es=DownSilverES(), index=SILVER_INDEX)))  # type: ignore[arg-type]
    with pytest.raises(server.ToolError, match="silver_unavailable"):
        server.list_tcs_risks()


async def test_mcp_tools_registered_with_schemas(env: Env) -> None:
    tools = {t.name: t for t in await server.mcp.list_tools()}
    assert {
        "list_tcs_risks",
        "get_tcs_risk_document",
        "similar_tcs_risks",
        "tcs_risk_clusters",
    } <= (set(tools))
    assert {"search_tcs_risks", "tcs_risk_summary"} <= set(tools)  # existing tools kept
    props = tools["list_tcs_risks"].input_schema["properties"]
    assert {"category", "min_severity", "min_risk_score", "query", "limit", "offset"} <= set(props)
    assert {"doc_id", "include_text"} <= set(
        tools["get_tcs_risk_document"].input_schema["properties"]
    )
    assert "include_raw" not in tools["get_tcs_risk_document"].input_schema["properties"]
