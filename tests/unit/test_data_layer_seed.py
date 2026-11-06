"""Unit tests for the seed loader (AC-001.5). No database: transaction() is faked."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

from structlog.testing import capture_logs

from vendor_risk_ai.db import seed as seed_mod
from vendor_risk_ai.db.seed import BUILTIN_SEED, load_seed_data

REPO_ROOT = Path(__file__).resolve().parents[2]


def _fallback_events(logs: list[dict]) -> list[dict]:
    return [e for e in logs if e["event"] == "db.seed.fallback_builtin"]


def test_missing_file_falls_back_to_builtin(tmp_path):
    """AC-001.5: missing SEED_FILE -> built-in synthetic dataset + warning."""
    with capture_logs() as logs:
        data = load_seed_data(str(tmp_path / "does-not-exist.json"))
    assert [v["external_id"] for v in data["vendors"]] == [
        v["external_id"] for v in BUILTIN_SEED["vendors"]
    ]
    ev = _fallback_events(logs)
    assert len(ev) == 1 and ev[0]["log_level"] == "warning"


def test_invalid_json_falls_back(tmp_path):
    """AC-001.5: unreadable/invalid file -> fallback."""
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with capture_logs() as logs:
        data = load_seed_data(str(bad))
    assert len(data["vendors"]) == len(BUILTIN_SEED["vendors"])
    assert _fallback_events(logs)


def test_wrong_shape_falls_back(tmp_path):
    bad = tmp_path / "shape.json"
    bad.write_text(json.dumps({"vendors": [{"name": "no external id"}]}), encoding="utf-8")
    with capture_logs() as logs:
        data = load_seed_data(str(bad))
    assert len(data["vendors"]) == len(BUILTIN_SEED["vendors"])
    assert _fallback_events(logs)


def test_valid_file_is_used(tmp_path):
    f = tmp_path / "ok.json"
    f.write_text(
        json.dumps(
            {
                "vendors": [
                    {
                        "external_id": "T-1",
                        "name": "Test Vendor",
                        "documents": [{"title": "t", "content": "hello", "doc_type": "news"}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with capture_logs() as logs:
        data = load_seed_data(str(f))
    assert [v["external_id"] for v in data["vendors"]] == ["T-1"]
    assert not _fallback_events(logs)


def test_builtin_seed_quality():
    """Synthetic dataset: 5 vendors x 3 doc types, 150-400 words each, unique ids."""
    vendors = BUILTIN_SEED["vendors"]
    assert len(vendors) == 5
    assert len({v["external_id"] for v in vendors}) == 5
    for v in vendors:
        types = sorted(d["doc_type"] for d in v["documents"])
        assert types == ["financial_report", "news", "regulatory_filing"]
        for d in v["documents"]:
            assert 150 <= len(d["content"].split()) <= 400, d["title"]


def test_repo_seed_file_matches_builtin():
    """data/seed/vendors.json is a copy of BUILTIN_SEED (regenerate with write_seed_file)."""
    on_disk = json.loads((REPO_ROOT / "data/seed/vendors.json").read_text(encoding="utf-8"))
    assert on_disk == json.loads(json.dumps(BUILTIN_SEED))


def test_seed_counts_with_fake_transaction(monkeypatch, tmp_path):
    """seed() upserts every vendor and counts only newly inserted documents."""
    seen_docs: set[tuple[int, str]] = set()
    vendor_ids: dict[str, int] = {}
    tx_count = []

    @contextmanager
    def fake_tx():
        tx_count.append(1)
        yield object()

    def fake_upsert(cur, v):
        return vendor_ids.setdefault(v.external_id, len(vendor_ids) + 1)

    def fake_add(cur, vendor_id, doc):
        key = (vendor_id, doc.content)
        new = key not in seen_docs
        seen_docs.add(key)
        return len(seen_docs), new

    monkeypatch.setattr(seed_mod, "transaction", fake_tx)
    monkeypatch.setattr(seed_mod, "upsert_vendor_cur", fake_upsert)
    monkeypatch.setattr(seed_mod, "add_document_cur", fake_add)
    missing = str(tmp_path / "nope.json")
    with capture_logs():
        first = seed_mod.seed(missing)
        second = seed_mod.seed(missing)
    assert first == {"vendors": 5, "documents": 15}
    assert second == {"vendors": 5, "documents": 0}
    assert len(tx_count) == 2  # one transaction per seed() call
