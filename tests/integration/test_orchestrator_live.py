"""Orchestrator data steps (spec 004) are idempotent on the live, already-bootstrapped database."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from vendor_risk_ai import main as main_mod
from vendor_risk_ai.db.mysql import transaction
from vendor_risk_ai.services.report_service import ReportService

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[2]
TABLES = (
    "schema_migrations",
    "vendors",
    "vendor_documents",
    "document_chunks",
    "risk_reports",
    "audit_log",
)


def _counts() -> dict[str, int]:
    out: dict[str, int] = {}
    with transaction() as cur:
        for t in TABLES:
            cur.execute(f"SELECT COUNT(*) AS n FROM {t}")
            out[t] = int(cur.fetchone()["n"])
    return out


def _es_chunk_count(live_stack: dict[str, Any]) -> int:
    es, idx = live_stack["es"], live_stack["settings"].elastic_chunks_index
    es.indices.refresh(index=idx)
    return int(es.count(index=idx)["count"])


@pytest.fixture
def spies(live_stack, monkeypatch) -> dict[str, list[Any]]:
    """Record the return values of the functions behind each orchestrator step."""
    from vendor_risk_ai.db import migrate as migrate_mod
    from vendor_risk_ai.db import seed as seed_mod

    monkeypatch.chdir(ROOT)  # SEED_FILE is relative to the repo root
    calls: dict[str, list[Any]] = {"migrate": [], "seed": [], "index": []}
    real_migrate, real_seed = migrate_mod.migrate, seed_mod.seed
    real_embed = ReportService.embed_all_pending

    def migrate_spy(*a: Any, **kw: Any) -> Any:
        calls["migrate"].append(res := real_migrate(*a, **kw))
        return res

    def seed_spy(*a: Any, **kw: Any) -> Any:
        calls["seed"].append(res := real_seed(*a, **kw))
        return res

    def embed_spy(self: ReportService, *a: Any, **kw: Any) -> int:
        calls["index"].append(res := real_embed(self, *a, **kw))
        return res

    monkeypatch.setattr(migrate_mod, "migrate", migrate_spy)
    monkeypatch.setattr(seed_mod, "seed", seed_spy)
    monkeypatch.setattr(ReportService, "embed_all_pending", embed_spy)
    return calls


def test_migrate_step_is_idempotent(spies):
    """AC-001.4, AC-004.2: `vendor-risk-ai migrate` on a migrated DB is a no-op."""
    before = _counts()
    assert main_mod.main(["migrate"]) == 0
    assert main_mod.main(["migrate"]) == 0
    assert spies["migrate"] == [[], []]
    assert _counts() == before


def test_seed_step_is_idempotent(spies):
    """AC-001.5, AC-004.2: re-seeding upserts vendors and inserts no duplicate documents."""
    before = _counts()
    assert main_mod.main(["seed"]) == 0
    assert main_mod.main(["seed"]) == 0
    assert [r["documents"] for r in spies["seed"]] == [0, 0]
    assert all(r["vendors"] == before["vendors"] for r in spies["seed"])
    after = _counts()
    assert after["vendors"] == before["vendors"]
    assert after["vendor_documents"] == before["vendor_documents"]
    assert after["document_chunks"] == before["document_chunks"]


def test_index_step_is_idempotent(spies, live_stack):
    """AC-001.7, AC-004.1, AC-004.2: `index` ensures ES indices and embeds only pending docs;
    on a fully indexed DB it embeds nothing and leaves MySQL / ES unchanged."""
    assert main_mod.main(["index"]) == 0  # may pick up leftovers once
    before, es_before = _counts(), _es_chunk_count(live_stack)
    assert main_mod.main(["index"]) == 0
    assert spies["index"][-1] == 0
    assert _counts() == before
    assert _es_chunk_count(live_stack) == es_before
    s = live_stack["settings"]
    for idx in (s.elastic_quotes_index, s.elastic_chunks_index):
        assert live_stack["es"].indices.exists(index=idx)
