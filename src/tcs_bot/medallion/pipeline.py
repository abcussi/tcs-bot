"""Medallion orchestration: bronze -> silver -> gold, once or on a cron schedule (AC-005.10)."""

from __future__ import annotations

import time
from typing import Any

from tcs_bot.config import get_settings
from tcs_bot.logging_setup import get_logger

log = get_logger(__name__)
LAYERS = ("bronze", "silver", "gold")
DEFAULT_CRON = "0 */6 * * *"  # every 6 hours


def run_bronze_layer(max_docs: int | None = None, max_depth: int | None = None) -> dict[str, Any]:
    from tcs_bot.medallion.bronze import run_bronze

    return run_bronze(max_docs=max_docs, max_depth=max_depth).model_dump()


def run_silver_layer() -> dict[str, Any]:
    from tcs_bot.medallion.silver.pipeline import run_silver

    return run_silver()


def run_gold_layer() -> dict[str, Any]:
    from tcs_bot.medallion.gold.pipeline import run_gold

    return run_gold()


def run_layers(
    layers: tuple[str, ...] | list[str] = LAYERS,
    max_docs: int | None = None,
    max_depth: int | None = None,
) -> dict[str, Any]:
    """Run the requested layers in medallion order; each layer reads what the previous persisted."""
    s = get_settings()
    max_docs = min(max_docs or s.medallion_max_docs, 500)
    max_depth = min(max_depth if max_depth is not None else s.medallion_max_depth, 2)
    results: dict[str, Any] = {}
    for layer in LAYERS:
        if layer not in layers:
            continue
        t0 = time.monotonic()
        if layer == "bronze":
            results[layer] = run_bronze_layer(max_docs=max_docs, max_depth=max_depth)
        elif layer == "silver":
            results[layer] = run_silver_layer()
        else:
            results[layer] = run_gold_layer()
        log.info("medallion.layer_done", layer=layer, seconds=round(time.monotonic() - t0, 2))
    return results


def schedule(cron: str = DEFAULT_CRON, **kwargs: Any) -> None:
    """Block forever, running the full pipeline on `cron` (and once at start-up)."""
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger

    def job() -> None:
        try:
            log.info(
                "medallion.scheduled_run", **{k: str(v) for k, v in run_layers(**kwargs).items()}
            )
        except Exception as exc:  # noqa: BLE001 - a failed run must not kill the scheduler
            log.error("medallion.scheduled_run_failed", error=str(exc))

    sched = BlockingScheduler(timezone="UTC")
    sched.add_job(job, CronTrigger.from_crontab(cron, timezone="UTC"), max_instances=1)
    job()
    sched.start()
