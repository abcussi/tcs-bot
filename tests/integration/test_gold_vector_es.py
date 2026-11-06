"""Gold ``GoldVectorStore`` against live Elasticsearch kNN (AC-005.7, AC-005.9, AC-005.11).

Uses a throwaway index prefix ``gold-test-<uuid>`` and deletes every index it created; the corpus
is the SYNTHETIC 3-topic fixture. Never touches ``silver-tcs-risk`` or any other index. Skips
when Elasticsearch is not reachable.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from elasticsearch import Elasticsearch

from tcs_bot.config import Settings, get_settings
from tcs_bot.medallion.gold import search
from tcs_bot.medallion.gold.pipeline import run_gold
from tcs_bot.medallion.gold.vector_store import MAX_DIMS, VECTOR_FIELD, GoldVectorStore, point_id
from tests.unit.test_gold_synthetic import FakeSilverStore, synthetic_records

pytestmark = pytest.mark.integration


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Settings]:
    monkeypatch.setenv("LLM_PROVIDER", "fake")  # never call OpenAI from tests
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("GOLD_DIR", str(tmp_path / "gold"))
    get_settings.cache_clear()
    yield get_settings()
    get_settings.cache_clear()


@pytest.fixture
def live_vs(settings: Settings) -> Iterator[GoldVectorStore]:
    es = Elasticsearch(settings.elastic_url, request_timeout=10)
    try:
        if not es.ping():
            pytest.skip("Elasticsearch not reachable")
    except Exception as exc:
        pytest.skip(f"Elasticsearch not reachable: {exc}")
    prefix = f"gold-test-{uuid.uuid4().hex[:8]}"
    vs = GoldVectorStore(es=es, prefix=prefix)
    search.set_vector_store(vs)
    try:
        yield vs
    finally:
        search.set_vector_store(None)
        names = list(es.indices.get(index=f"{prefix}-*", ignore_unavailable=True))
        for name in names:  # explicit names only: wildcard deletes are refused by default
            assert name.startswith(prefix), name
            es.indices.delete(index=name, ignore_unavailable=True)
        es.close()


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_store_upsert_knn_filter_idempotent_and_cosine_order(live_vs: GoldVectorStore) -> None:
    """AC-005.7: bulk upsert, read-your-write kNN, filter, idempotent re-upsert, cosine order."""
    recs = synthetic_records()[:12]
    rng = np.random.default_rng(7)
    vecs = rng.normal(size=(12, 16))
    extra = {r.doc_id: {"cluster": i % 3} for i, r in enumerate(recs)}
    vs = live_vs.with_model("test_16")
    vs.ensure(16)
    vs.ensure(16)  # idempotent

    assert vs.upsert_docs(recs, vecs, extra) == 12
    assert vs.count("docs") == 12  # visible right after the bulk (refresh)
    q = rng.normal(size=16)
    hits = vs.search(q, k=12, level="docs")
    assert len(hits) == 12

    # cosine ordering and score == true cosine (ES score is (1 + cos) / 2)
    by_id = {r.doc_id: v for r, v in zip(recs, vecs, strict=True)}
    expected = sorted(by_id, key=lambda d: -_cos(by_id[d], q))
    assert [h.doc_id for h in hits] == expected
    for h in hits:
        assert h.score == pytest.approx(_cos(by_id[h.doc_id], q), abs=1e-3)
    assert hits == sorted(hits, key=lambda h: -h.score)

    # filter: cluster + category + min risk are applied inside the kNN
    in_cluster = vs.search(q, k=12, level="docs", cluster=1)
    assert {h.doc_id for h in in_cluster} == {r.doc_id for i, r in enumerate(recs) if i % 3 == 1}
    cat = recs[0].risk_categories[0]
    by_cat = vs.search(q, k=12, level="docs", categories=[cat], min_risk=0)
    assert by_cat and all(cat in h.risk_categories for h in by_cat)
    assert vs.search(q, k=12, level="docs", categories=["no_such_category"]) == []

    # idempotent re-upsert: same ids, no duplicates, replaced vector wins
    assert vs.upsert_docs(recs, vecs, extra) == 12 and vs.count("docs") == 12
    new_vec = rng.normal(size=16)
    vs.upsert_docs(recs[:1], new_vec[None, :], extra)
    top = vs.search(new_vec, k=1, level="docs")
    assert top[0].doc_id == recs[0].doc_id and top[0].score == pytest.approx(1.0, abs=1e-3)
    src = vs.es.get(index=vs.docs_collection, id=point_id("docs", recs[0].doc_id))["_source"]
    assert src["doc_id"] == recs[0].doc_id and VECTOR_FIELD in src

    # per-document failure: a zero vector is rejected by cosine dense_vector, the rest is written
    bad = vecs[:3].copy()
    bad[1] = 0
    vs.ensure(16, recreate=True)
    assert vs.upsert_docs(recs[:3], bad) == 2
    assert vs.last_failed_ids == {recs[1].doc_id}

    # delete helper removes docs; dimension is part of the index identity
    assert vs.delete_docs([recs[0].doc_id]) == 1 and vs.count("docs") == 1
    with pytest.raises(ValueError, match="dim"):
        vs.ensure(8)


def test_dim_limits_match_elasticsearch(live_vs: GoldVectorStore) -> None:
    """ES 8.15 dense_vector (index=true) takes up to 4096 dims: LSA 256 and OpenAI 1536 fit."""
    vs = live_vs.with_model("test_dims")
    for dim in (256, 1536, MAX_DIMS):
        vs.ensure(dim, recreate=True)
    vs.drop()
    with pytest.raises(ValueError, match="range"):
        vs.ensure(MAX_DIMS + 1)


def test_gold_run_and_search_on_live_es(settings: Settings, live_vs: GoldVectorStore) -> None:
    """AC-005.7/9: run_gold upserts into real ES indices; filtered search works end to end."""
    res = run_gold(FakeSilverStore(synthetic_records()), live_vs, settings)
    assert res["vector_store"] == "ok", res["vector_store"]
    vs = live_vs.with_model(f"lsa_{settings.lsa_dim}")
    assert vs.docs_collection.startswith(live_vs.prefix)
    assert vs.count("docs") == 30 and vs.count("chunks") == res["chunks"]

    hits = search.search_risks("ransomware breach exfiltrated passwords", k=5)
    assert len(hits) == 5 and all("/cyber/" in h.url for h in hits)
    legal = search.search_risks("court verdict", k=5, categories=["legal_litigation"], min_risk=60)
    assert legal and all(h.risk_score >= 60 for h in legal)
    assert all("legal_litigation" in h.risk_categories for h in legal)

    summary = json.loads(Path(res["paths"]["summary_json"]).read_text())
    assert summary["meta"]["vector_indices"] == [vs.docs_collection, vs.chunks_collection]
    assert summary["meta"]["vector_backend"] == "elasticsearch"

    # idempotent full rebuild
    run_gold(FakeSilverStore(synthetic_records()), live_vs, settings)
    assert vs.count("docs") == 30
