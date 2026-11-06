"""FastAPI application factory (AC-003.1, AC-003.7)."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager

from fastapi import FastAPI

from tcs_bot.api.errors import register_error_handlers
from tcs_bot.api.routes import router
from tcs_bot.logging_setup import get_logger
from tcs_bot.services.report_service import ReportService

log = get_logger(__name__)

HealthChecks = Mapping[str, Callable[[], bool]]

OPENAPI_TAGS = [
    {"name": "health", "description": "Liveness and dependency status"},
    {"name": "vendors", "description": "Third-party vendor master data"},
    {"name": "documents", "description": "Ingest vendor financial reports, news and filings"},
    {"name": "reports", "description": "AI-generated structured vendor risk reports"},
]


def default_health_checks() -> dict[str, Callable[[], bool]]:
    """Real dependency probes; imported lazily so the app starts even if a module is broken."""

    def mysql() -> bool:
        from tcs_bot.db.mysql import ping

        return ping()

    def redis() -> bool:
        from tcs_bot.cache.cache import get_cache

        cache = get_cache()
        ping = getattr(cache, "ping", None)
        if callable(ping):
            return bool(ping())
        import redis as redis_lib

        from tcs_bot.config import get_settings

        client = redis_lib.Redis.from_url(
            get_settings().redis_url, socket_connect_timeout=1, socket_timeout=1
        )
        return bool(client.ping())

    def elasticsearch() -> bool:
        from tcs_bot.search.elastic import ping

        return ping()

    return {"mysql": mysql, "redis": redis, "elasticsearch": elasticsearch}


def create_app(
    service: ReportService | None = None, health_checks: HealthChecks | None = None
) -> FastAPI:
    """Build the app. ``service`` / ``health_checks`` are injectable for tests; when omitted the
    real service is built lazily at startup and the real dependency probes are used."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if app.state.service is None:
            try:
                from tcs_bot.services.report_service import build_default_service

                app.state.service = build_default_service()
            except Exception as exc:  # keep /health alive; routes answer 503
                log.error("service_build_failed", error_type=type(exc).__name__, exc_info=exc)
        yield

    app = FastAPI(
        title="Vendor Risk AI",
        version="0.1.0",
        description="AI-generated third-party (vendor) risk reports for capital-markets firms.",
        openapi_tags=OPENAPI_TAGS,
        lifespan=lifespan,
    )
    app.state.service = service
    resolved: dict[str, HealthChecks] = {}

    def health_checks_factory() -> HealthChecks:
        if "checks" not in resolved:
            resolved["checks"] = (
                health_checks if health_checks is not None else default_health_checks()
            )
        return resolved["checks"]

    app.state.health_checks_factory = health_checks_factory
    register_error_handlers(app)
    app.include_router(router)
    return app
