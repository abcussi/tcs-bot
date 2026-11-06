"""Two-tier cache (AC-001.6): L1 in-process TTL+LRU, L2 Redis with graceful fallback.

* Values must be JSON-serializable (Pydantic models are dumped with ``model_dump(mode="json")``,
  dates as ISO strings). Both tiers store the JSON text, so callers always get a fresh copy.
* ``None`` is never cached as a hit: ``get`` returns ``None`` for "missing".
* Any Redis error is caught and logged (first failure, then at most once per
  ``_LOG_EVERY_S``); the cache then serves from L1 only and skips Redis for ``_BACKOFF_S``
  seconds (circuit breaker) so callers never pay repeated connect timeouts.
"""

from __future__ import annotations

import json
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel

from tcs_bot.config import get_settings
from tcs_bot.logging_setup import get_logger

log = get_logger(__name__)

_BACKOFF_S = 30.0
_LOG_EVERY_S = 60.0


def _json_default(o: Any) -> Any:
    if isinstance(o, BaseModel):
        return o.model_dump(mode="json")
    if isinstance(o, datetime | date):
        return o.isoformat()
    if isinstance(o, Decimal):
        return float(o)
    if isinstance(o, set | frozenset | tuple):
        return list(o)
    raise TypeError(f"value of type {type(o).__name__} is not JSON serializable")


def _encode(value: Any) -> str:
    return json.dumps(value, default=_json_default, separators=(",", ":"))


class _LocalTTLCache:
    """Thread-safe LRU with per-entry monotonic expiry."""

    def __init__(self, maxsize: int) -> None:
        self.maxsize = max(1, int(maxsize))
        self._data: OrderedDict[str, tuple[float, str]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> str | None:
        now = time.monotonic()
        with self._lock:
            item = self._data.get(key)
            if item is None:
                return None
            expires_at, raw = item
            if expires_at <= now:
                del self._data[key]
                return None
            self._data.move_to_end(key)
            return raw

    def set(self, key: str, raw: str, ttl: float) -> None:
        if ttl <= 0:
            return
        with self._lock:
            self._data[key] = (time.monotonic() + ttl, raw)
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)

    def delete(self, key: str) -> None:
        with self._lock:
            self._data.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


class TwoTierCache:
    def __init__(
        self,
        redis_url: str | None,
        ttl: int,
        maxsize: int,
        *,
        key_prefix: str = "vra:",
        redis_client: Any | None = None,
    ) -> None:
        """``redis_url=None`` (and no ``redis_client``) means L1-only.

        ``redis_client`` lets tests inject a fake/broken client.
        """
        self.ttl = int(ttl)
        self.key_prefix = key_prefix
        self.l1 = _LocalTTLCache(maxsize)
        self._redis: Any | None = redis_client
        self._down_until = 0.0
        self._last_log = float("-inf")
        self._state_lock = threading.Lock()
        if self._redis is None and redis_url:
            try:
                import redis

                self._redis = redis.Redis.from_url(
                    redis_url,
                    socket_connect_timeout=0.5,
                    socket_timeout=0.5,
                    health_check_interval=30,
                )
            except Exception as exc:  # bad URL etc. -> L1 only
                self._redis_failed("init", exc)
                self._redis = None

    # ------------------------------------------------------------------ redis plumbing
    @property
    def l2_enabled(self) -> bool:
        return self._redis is not None

    def _l2(self) -> Any | None:
        if self._redis is None or time.monotonic() < self._down_until:
            return None
        return self._redis

    def _redis_failed(self, op: str, exc: BaseException) -> None:
        now = time.monotonic()
        with self._state_lock:
            self._down_until = now + _BACKOFF_S
            should_log = now - self._last_log >= _LOG_EVERY_S
            if should_log:
                self._last_log = now
        if should_log:
            log.warning(
                "cache.redis_unavailable",
                op=op,
                error=f"{type(exc).__name__}: {exc}"[:200],
                fallback="l1_only",
                retry_in_s=_BACKOFF_S,
            )

    def _k(self, key: str) -> str:
        return f"{self.key_prefix}{key}"

    # ------------------------------------------------------------------ public API
    def get(self, key: str) -> Any | None:
        raw = self.l1.get(key)
        if raw is None:
            client = self._l2()
            if client is not None:
                try:
                    pipe = client.pipeline()
                    pipe.get(self._k(key))
                    pipe.pttl(self._k(key))
                    l2_raw, pttl = pipe.execute()
                except Exception as exc:
                    self._redis_failed("get", exc)
                    l2_raw, pttl = None, None
                if l2_raw is not None:
                    raw = l2_raw.decode("utf-8") if isinstance(l2_raw, bytes) else str(l2_raw)
                    remaining = pttl / 1000 if isinstance(pttl, int) and pttl > 0 else self.ttl
                    self.l1.set(key, raw, min(self.ttl, remaining))
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except ValueError:
            self.delete(key)
            return None

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        if value is None:
            return
        ttl = self.ttl if ttl is None else int(ttl)
        raw = _encode(value)  # raises TypeError for non-serializable values (caller bug)
        self.l1.set(key, raw, ttl)
        client = self._l2()
        if client is not None and ttl > 0:
            try:
                client.set(self._k(key), raw, ex=ttl)
            except Exception as exc:
                self._redis_failed("set", exc)

    def delete(self, key: str) -> None:
        self.l1.delete(key)
        client = self._l2()
        if client is not None:
            try:
                client.delete(self._k(key))
            except Exception as exc:
                self._redis_failed("delete", exc)

    def get_or_set(self, key: str, factory: Callable[[], Any], ttl: int | None = None) -> Any:
        value = self.get(key)
        if value is not None:
            return value
        value = factory()
        if value is None:
            return None
        self.set(key, value, ttl)
        # Return the JSON round-tripped form so hits and misses look identical to callers.
        return json.loads(_encode(value))

    def clear_local(self) -> None:
        self.l1.clear()

    def ping(self) -> bool:
        """True if Redis answers; never raises."""
        if self._redis is None:
            return False
        try:
            return bool(self._redis.ping())
        except Exception as exc:
            self._redis_failed("ping", exc)
            return False


_cache: TwoTierCache | None = None
_cache_lock = threading.Lock()


def get_cache() -> TwoTierCache:
    """Process-wide cache built from settings (``REDIS_URL`` empty -> L1 only)."""
    global _cache
    if _cache is None:
        with _cache_lock:
            if _cache is None:
                s = get_settings()
                _cache = TwoTierCache(
                    redis_url=s.redis_url or None,
                    ttl=s.cache_ttl_seconds,
                    maxsize=s.local_cache_maxsize,
                )
    return _cache


def reset_cache() -> None:
    """Drop the singleton (tests / config reload)."""
    global _cache
    with _cache_lock:
        _cache = None
