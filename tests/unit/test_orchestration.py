"""Spec 004 — orchestrator CLI, Docker artifacts and run script (no infra, no docker calls)."""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from tcs_bot import main as main_mod

ROOT = Path(__file__).resolve().parents[2]


def _forbidden(what: str):
    def _raise(*_a, **_kw):
        raise AssertionError(f"unit test tried to run a real side effect: {what}")

    return _raise


@pytest.fixture(autouse=True)
def _no_side_effects(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hard guard: no docker compose, subprocess, waiting or server start from unit tests."""
    import uvicorn

    monkeypatch.setattr(main_mod, "_compose", _forbidden("docker compose"))
    monkeypatch.setattr(main_mod, "_wait_for", _forbidden("wait_for"))
    monkeypatch.setattr(main_mod.subprocess, "run", _forbidden("subprocess.run"))
    monkeypatch.setattr(uvicorn, "run", _forbidden("uvicorn.run"))


@pytest.fixture
def recorded(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace every step in every dispatch table of ``main`` (STEPS, STEPS_INTERNAL, ...)."""
    calls: list[str] = []
    step_funcs = [v for k, v in vars(main_mod).items() if k.startswith("step_") and callable(v)]

    def is_step(v: object) -> bool:
        return any(v is f for f in step_funcs)

    for attr, value in list(vars(main_mod).items()):
        if isinstance(value, dict) and value and all(is_step(v) for v in value.values()):
            fake = {name: (lambda n=name: calls.append(n)) for name in value}
            monkeypatch.setattr(main_mod, attr, fake)
    for func in step_funcs:
        monkeypatch.setattr(main_mod, func.__name__, _forbidden(func.__name__))
    return calls


def _normalize(calls: list[str]) -> list[str]:
    return [c.removesuffix("-bg") for c in calls]  # e.g. "crawl-bg" = crawl in background


def test_all_runs_pipeline_in_order(recorded):
    """AC-004.1: `all` = up -> migrate -> seed -> index -> crawl -> api."""
    assert main_mod.main(["all"]) == 0
    assert _normalize(recorded) == ["up", "migrate", "seed", "index", "crawl", "api"]


def test_all_no_infra_skips_compose(recorded):
    """AC-004.1: `all --no-infra` skips docker compose when services already run."""
    assert main_mod.main(["all", "--no-infra"]) == 0
    assert _normalize(recorded) == ["migrate", "seed", "index", "crawl", "api"]


@pytest.mark.parametrize(
    "step", ["up", "down", "migrate", "seed", "index", "crawl", "api", "mcp", "status"]
)
def test_each_subcommand_dispatches_single_step(recorded, step):
    """AC-004.2: every documented subcommand exists and runs only that step."""
    assert main_mod.main([step]) == 0
    assert recorded == [step]


def test_unknown_subcommand_rejected(recorded):
    """AC-004.2: unknown subcommands are rejected by argparse."""
    with pytest.raises(SystemExit) as exc:
        main_mod.main(["bogus"])
    assert exc.value.code == 2
    assert recorded == []


def test_crawl_step_is_non_fatal(monkeypatch):
    """AC-004.1: a crawler failure does not abort the pipeline."""
    import gfinance_crawler.run as crawler_run

    def boom() -> dict:
        raise RuntimeError("network blocked")

    monkeypatch.setattr(crawler_run, "run_once", boom)
    main_mod.step_crawl()  # must not raise


def test_dockerfiles_multistage_and_cron():
    """AC-004.3: API Dockerfile is a multi-stage uv build; crawler image runs supercronic."""
    api = (ROOT / "Dockerfile").read_text()
    assert len(re.findall(r"^FROM ", api, re.M)) >= 2
    assert "uv sync" in api and "--frozen" in api
    assert "HEALTHCHECK" in api
    crawler = (ROOT / "crawler" / "Dockerfile").read_text()
    assert "supercronic" in crawler
    assert (ROOT / "crawler" / "cron" / "crontab").is_file()


def test_compose_services_have_healthchecks():
    """AC-004.3: docker-compose.yml defines infra with healthchecks; app waits on them."""
    yaml = pytest.importorskip("yaml")
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    services = compose["services"]
    for name in ("mysql", "redis", "elasticsearch", "api", "crawler"):
        assert name in services
    for name in ("mysql", "redis", "elasticsearch"):
        assert "healthcheck" in services[name]
    deps = services["api"]["depends_on"]
    for name in ("mysql", "redis", "elasticsearch"):
        assert deps[name]["condition"] == "service_healthy"


def test_run_script_wraps_cli():
    """AC-004.4: scripts/run.sh is an executable thin wrapper over the CLI."""
    script = ROOT / "scripts" / "run.sh"
    assert script.is_file() and os.access(script, os.X_OK)
    text = script.read_text()
    assert text.startswith("#!")
    assert "uv run tcs-bot" in text
    assert "set -euo pipefail" in text
