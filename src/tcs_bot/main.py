"""Orchestrator: one entrypoint that boots and runs the whole system (spec 004).

uv run tcs-bot all        # infra up -> migrate -> seed -> index -> crawl -> API
uv run tcs-bot <step>   # up|down|status|migrate|seed|index|crawl|api|mcp
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

from tcs_bot.config import get_settings
from tcs_bot.logging_setup import configure_logging, get_logger

log = get_logger(__name__)
ROOT = Path(__file__).resolve().parents[2]
INFRA_SERVICES = ["mysql", "redis", "elasticsearch", "qdrant"]


# ---------------------------------------------------------------- infra
def _compose(*args: str) -> subprocess.CompletedProcess[str]:
    if not shutil.which("docker"):
        raise SystemExit("docker CLI not found — install Docker (see README)")
    cmd = ["docker", "compose", *args]
    log.info("exec", cmd=" ".join(cmd))
    return subprocess.run(cmd, cwd=ROOT, check=True, text=True, capture_output=False)


def _wait_for(name: str, probe: Callable[[], bool], timeout: float = 180.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if probe():
                log.info("ready", service=name)
                return
        except Exception:  # noqa: BLE001 - probing, any failure means "not yet"
            pass
        time.sleep(2)
    raise SystemExit(f"{name} not ready after {timeout:.0f}s")


def step_up() -> None:
    _compose("up", "-d", *INFRA_SERVICES)
    from tcs_bot.db import mysql
    from tcs_bot.search import elastic

    def redis_ok() -> bool:
        import redis

        return bool(redis.Redis.from_url(get_settings().redis_url).ping())

    _wait_for("mysql", lambda: (mysql.reset_pool(), mysql.ping())[1])
    _wait_for("redis", redis_ok)
    _wait_for("elasticsearch", elastic.ping)


def step_down() -> None:
    _compose("down")


def step_status() -> None:
    _compose("ps")


# ---------------------------------------------------------------- data
def step_migrate() -> None:
    from tcs_bot.db.migrate import migrate

    log.info("migrate", created=migrate())


def step_seed() -> None:
    from tcs_bot.db.seed import seed

    log.info("seed", **seed(get_settings().seed_file))


def step_index() -> None:
    from tcs_bot.search.elastic import ensure_indices
    from tcs_bot.services.report_service import build_default_service

    ensure_indices()
    log.info("index", embedded_documents=build_default_service().embed_all_pending())


def step_crawl() -> None:
    """Non-fatal: market data is enrichment, the app works without it."""
    try:
        from gfinance_crawler.run import run_once

        log.info("crawl", **run_once())
    except Exception as exc:  # noqa: BLE001
        log.warning("crawl_failed", error=str(exc))


def step_crawl_background() -> None:
    """Used by `all`: an unreachable Google Finance can take minutes of retries,
    so the first crawl runs in a daemon thread and the API starts immediately."""
    import threading

    threading.Thread(target=step_crawl, name="initial-crawl", daemon=True).start()


# ---------------------------------------------------------------- servers
def step_api() -> None:
    import uvicorn

    s = get_settings()
    uvicorn.run("tcs_bot.api.app:create_app", factory=True, host=s.api_host, port=s.api_port)


def step_mcp() -> None:
    from tcs_bot.mcp_server.server import main as mcp_main

    mcp_main([])


def step_medallion(
    layer: str, max_docs: int | None, max_depth: int | None, cron: str | None
) -> int:
    """TCS risk medallion pipeline (spec 005): bronze (MySQL) -> silver (ES) -> gold (Qdrant)."""
    import json

    from tcs_bot.medallion import pipeline

    if layer == "schedule":
        pipeline.schedule(cron or pipeline.DEFAULT_CRON, max_docs=max_docs, max_depth=max_depth)
        return 0
    layers = list(pipeline.LAYERS) if layer == "run" else [layer]
    result = pipeline.run_layers(layers, max_docs=max_docs, max_depth=max_depth)
    print(json.dumps(result, indent=2, default=str))
    return 0


STEPS: dict[str, Callable[[], None]] = {
    "up": step_up,
    "down": step_down,
    "status": step_status,
    "migrate": step_migrate,
    "seed": step_seed,
    "index": step_index,
    "crawl": step_crawl,
    "api": step_api,
    "mcp": step_mcp,
}
PIPELINE = ["up", "migrate", "seed", "index", "crawl-bg", "api"]
STEPS_INTERNAL: dict[str, Callable[[], None]] = {**STEPS, "crawl-bg": step_crawl_background}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tcs-bot",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("command", choices=["all", "medallion", *STEPS], help="step to run")
    parser.add_argument(
        "layer",
        nargs="?",
        choices=["run", "bronze", "silver", "gold", "schedule"],
        default="run",
        help="with 'medallion': which layer(s) to run (default: run = bronze->silver->gold)",
    )
    parser.add_argument(
        "--no-infra",
        action="store_true",
        help="with 'all': skip docker compose (services already running)",
    )
    parser.add_argument("--max-docs", type=int, default=None, help="medallion: cap (<=500)")
    parser.add_argument("--max-depth", type=int, default=None, help="medallion: crawl depth (<=2)")
    parser.add_argument("--cron", default=None, help="medallion schedule: crontab expression")
    args = parser.parse_args(argv)
    configure_logging(get_settings().log_level)

    if args.command == "medallion":
        return step_medallion(args.layer, args.max_docs, args.max_depth, args.cron)

    steps = PIPELINE if args.command == "all" else [args.command]
    if args.command == "all" and args.no_infra:
        steps = [s for s in steps if s != "up"]
    for name in steps:
        log.info("step", name=name)
        STEPS_INTERNAL[name]()
    return 0


def cli() -> None:
    sys.exit(main())


if __name__ == "__main__":
    cli()
