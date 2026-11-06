"""Unit tests for TwoTierCache (AC-001.6). No network: fake/broken Redis clients only."""

from __future__ import annotations

import redis
from structlog.testing import capture_logs

from tcs_bot.cache import cache as cache_mod
from tcs_bot.cache.cache import TwoTierCache


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def monotonic(self) -> float:
        return self.now


class FakeRedis:
    """Minimal in-memory stand-in for redis.Redis (get/set/delete/pipeline/ping)."""

    def __init__(self) -> None:
        self.store: dict[str, bytes] = {}
        self.calls = 0

    def get(self, k: str) -> bytes | None:
        self.calls += 1
        return self.store.get(k)

    def set(self, k: str, v: str, ex: int | None = None) -> bool:
        self.calls += 1
        self.store[k] = v.encode()
        return True

    def delete(self, k: str) -> int:
        self.calls += 1
        return int(self.store.pop(k, None) is not None)

    def ping(self) -> bool:
        return True

    def pipeline(self) -> FakePipeline:
        return FakePipeline(self)


class FakePipeline:
    def __init__(self, r: FakeRedis) -> None:
        self.r = r
        self.ops: list[tuple[str, str]] = []

    def get(self, k: str) -> None:
        self.ops.append(("get", k))

    def pttl(self, k: str) -> None:
        self.ops.append(("pttl", k))

    def execute(self) -> list:
        out: list = []
        for op, k in self.ops:
            out.append(self.r.get(k) if op == "get" else (60_000 if k in self.r.store else -2))
        return out


class BrokenRedis:
    def __init__(self) -> None:
        self.calls = 0

    def _boom(self, *a, **kw):
        self.calls += 1
        raise redis.exceptions.ConnectionError("Connection refused")

    get = set = delete = ping = _boom

    def pipeline(self):
        outer = self

        class P:
            def get(self, k):
                pass

            def pttl(self, k):
                pass

            def execute(self):
                return outer._boom()

        return P()


def test_l1_ttl_expiry(monkeypatch):
    """AC-001.6: L1 entries expire after their TTL."""
    clock = FakeClock()
    monkeypatch.setattr(cache_mod, "time", clock)
    c = TwoTierCache(redis_url=None, ttl=10, maxsize=8)
    c.set("k", {"a": 1})
    assert c.get("k") == {"a": 1}
    clock.now += 9.9
    assert c.get("k") == {"a": 1}
    clock.now += 0.2
    assert c.get("k") is None
    c.set("short", 1, ttl=1)
    clock.now += 1.01
    assert c.get("short") is None


def test_l1_lru_eviction():
    """AC-001.6: L1 evicts least-recently-used entries beyond maxsize."""
    c = TwoTierCache(redis_url=None, ttl=60, maxsize=2)
    c.set("a", 1)
    c.set("b", 2)
    assert c.get("a") == 1  # touch a -> b is LRU
    c.set("c", 3)
    assert c.get("b") is None
    assert c.get("a") == 1
    assert c.get("c") == 3
    assert len(c.l1) == 2


def test_values_are_json_copies_and_none_not_cached():
    c = TwoTierCache(redis_url=None, ttl=60, maxsize=8)
    v = {"x": [1, 2]}
    c.set("k", v)
    v["x"].append(3)
    assert c.get("k") == {"x": [1, 2]}
    c.set("n", None)
    assert c.get("n") is None


def test_get_or_set_calls_factory_once():
    c = TwoTierCache(redis_url=None, ttl=60, maxsize=8)
    calls = []

    def factory():
        calls.append(1)
        return {"score": 42.0}

    assert c.get_or_set("r", factory) == {"score": 42.0}
    assert c.get_or_set("r", factory) == {"score": 42.0}
    assert len(calls) == 1
    c.delete("r")
    c.get_or_set("r", factory)
    assert len(calls) == 2


def test_l2_shared_between_instances():
    """L2 hit on a cold L1 repopulates L1."""
    r = FakeRedis()
    a = TwoTierCache(redis_url=None, ttl=60, maxsize=8, redis_client=r)
    b = TwoTierCache(redis_url=None, ttl=60, maxsize=8, redis_client=r)
    a.set("shared", {"v": 1})
    assert "vra:shared" in r.store
    assert b.get("shared") == {"v": 1}
    assert b.l1.get("shared") is not None
    b.delete("shared")
    assert "vra:shared" not in r.store


def test_redis_failure_falls_back_to_l1_without_raising():
    """AC-001.6: Redis outage never raises to callers; L1 still serves; logged once."""
    r = BrokenRedis()
    c = TwoTierCache(redis_url=None, ttl=60, maxsize=8, redis_client=r)
    with capture_logs() as logs:
        c.set("k", {"ok": True})
        assert c.get("k") == {"ok": True}  # L1 hit
        assert c.get("missing") is None
        c.delete("k")
        assert c.get_or_set("k2", lambda: 5) == 5
        assert c.ping() is False
    warnings = [e for e in logs if e["event"] == "cache.redis_unavailable"]
    assert len(warnings) == 1
    # circuit breaker: after the first failure Redis is skipped
    assert r.calls == 2  # the failed set() + explicit ping()


def test_unreachable_redis_url_is_graceful():
    """A real client pointed at a closed port must degrade, not raise."""
    c = TwoTierCache(redis_url="redis://127.0.0.1:1/0", ttl=60, maxsize=8)
    with capture_logs():
        c.set("k", [1, 2])
        assert c.get("k") == [1, 2]


def test_get_cache_singleton():
    cache_mod.reset_cache()
    try:
        assert cache_mod.get_cache() is cache_mod.get_cache()
    finally:
        cache_mod.reset_cache()
