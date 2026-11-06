"""AC-005.10: `tcs-bot medallion ...` CLI wiring and layer ordering (no infra)."""

from __future__ import annotations

import pytest

from tcs_bot import main
from tcs_bot.medallion import pipeline


@pytest.fixture
def calls(monkeypatch):
    seen: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        pipeline, "run_bronze_layer", lambda **kw: seen.append(("bronze", kw)) or {"stored": 1}
    )
    monkeypatch.setattr(pipeline, "run_silver_layer", lambda: seen.append(("silver", {})) or {})
    monkeypatch.setattr(pipeline, "run_gold_layer", lambda: seen.append(("gold", {})) or {})
    return seen


def test_run_executes_layers_in_medallion_order(calls):
    """AC-005.10"""
    assert main.main(["medallion"]) == 0
    assert [c[0] for c in calls] == ["bronze", "silver", "gold"]


def test_single_layer_and_caps_are_enforced(calls):
    """AC-005.2: max docs <= 500 and depth <= 2 even if asked for more."""
    main.main(["medallion", "bronze", "--max-docs", "9999", "--max-depth", "7"])
    assert calls == [("bronze", {"max_docs": 500, "max_depth": 2})]


def test_layers_given_out_of_order_still_run_in_order(calls):
    """AC-005.10"""
    pipeline.run_layers(["gold", "bronze"])
    assert [c[0] for c in calls] == ["bronze", "gold"]


def test_schedule_dispatch(monkeypatch):
    """AC-005.10: schedule subcommand passes the cron through."""
    got = {}
    monkeypatch.setattr(pipeline, "schedule", lambda cron, **kw: got.update(cron=cron, **kw))
    main.main(["medallion", "schedule", "--cron", "*/30 * * * *"])
    assert got["cron"] == "*/30 * * * *"
