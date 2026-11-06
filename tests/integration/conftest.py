"""Shared fixtures for integration tests that run against the live stack (shared ``vendor_risk``
database, Redis db 0, the real ES indices). Every fixture skips cleanly if a service is down and
every test cleans up the rows / documents / cache keys it creates."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from vendor_risk_ai.config import get_settings


def _reset_singletons() -> None:
    from vendor_risk_ai.cache.cache import reset_cache
    from vendor_risk_ai.db.mysql import reset_pool
    from vendor_risk_ai.search.elastic import reset_es

    reset_pool()
    reset_es()
    reset_cache()


@pytest.fixture(scope="module")
def live_stack() -> Iterator[dict[str, Any]]:
    """Point settings at the live stack with the offline LLM; skip if any service is down."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("LLM_PROVIDER", "fake")  # never call OpenAI from tests
        mp.setenv("OPENAI_API_KEY", "")
        mp.delenv("MYSQL_DATABASE", raising=False)  # the shared, migrated + seeded schema
        get_settings.cache_clear()
        _reset_singletons()

        from vendor_risk_ai.db.mysql import ping as mysql_ping
        from vendor_risk_ai.search.elastic import get_es
        from vendor_risk_ai.search.elastic import ping as es_ping

        if not mysql_ping():
            pytest.skip("MySQL not reachable")
        if not es_ping():
            pytest.skip("Elasticsearch not reachable")
        import redis

        s = get_settings()
        r = redis.Redis.from_url(s.redis_url, socket_connect_timeout=1)
        try:
            r.ping()
        except Exception as exc:
            pytest.skip(f"Redis not reachable: {exc}")
        try:
            yield {"settings": s, "redis": r, "es": get_es()}
        finally:
            r.close()
            _reset_singletons()
    get_settings.cache_clear()
