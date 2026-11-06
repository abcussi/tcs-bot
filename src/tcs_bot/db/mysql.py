"""MySQL connection pool and transaction helper (AC-001.1, AC-001.2).

All MySQL access in the app goes through :func:`transaction` (constitution rule 5): it borrows a
pooled connection, yields a buffered *dictionary* cursor, commits on success, rolls back on any
exception (re-raised) and always returns the connection to the pool.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from typing import Any

from mysql.connector import errors as mysql_errors
from mysql.connector.cursor import MySQLCursorDict
from mysql.connector.pooling import MySQLConnectionPool

from tcs_bot.config import get_settings
from tcs_bot.logging_setup import get_logger

log = get_logger(__name__)

_pool: MySQLConnectionPool | None = None
_pool_lock = threading.Lock()

#: How long ``transaction()`` waits for a free pooled connection before giving up.
POOL_ACQUIRE_TIMEOUT_S = 10.0


def get_pool() -> MySQLConnectionPool:
    """Return the process-wide pool, creating it lazily (thread-safe, double-checked)."""
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                s = get_settings()
                _pool = MySQLConnectionPool(
                    pool_name=s.mysql_pool_name,
                    pool_size=s.mysql_pool_size,
                    pool_reset_session=True,
                    host=s.mysql_host,
                    port=s.mysql_port,
                    user=s.mysql_user,
                    password=s.mysql_password.get_secret_value(),
                    database=s.mysql_database,
                    autocommit=False,
                    charset="utf8mb4",
                    collation="utf8mb4_unicode_ci",
                    connection_timeout=10,
                )
                log.info(
                    "mysql.pool.created",
                    pool=s.mysql_pool_name,
                    size=s.mysql_pool_size,
                    host=s.mysql_host,
                    database=s.mysql_database,
                )
    return _pool


def reset_pool() -> None:
    """Drop the singleton pool (closing idle connections). Used by tests / config reloads."""
    global _pool
    with _pool_lock:
        pool, _pool = _pool, None
    if pool is not None:
        try:
            pool._remove_connections()  # noqa: SLF001 - no public close() on the pool
        except Exception as exc:  # pragma: no cover - best effort
            log.warning("mysql.pool.reset_error", error=str(exc))


def _acquire(timeout: float = POOL_ACQUIRE_TIMEOUT_S) -> Any:
    """Borrow a connection, retrying: mysql-connector raises PoolError at once when exhausted."""
    pool = get_pool()
    deadline = time.monotonic() + timeout
    delay = 0.01
    while True:
        try:
            return pool.get_connection()
        except mysql_errors.PoolError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 0.2)


@contextmanager
def transaction() -> Iterator[MySQLCursorDict]:
    """Run a unit of work in one MySQL transaction.

    Yields a buffered dictionary cursor. Commits when the block exits normally; on *any*
    exception rolls back and re-raises. The cursor is always closed and the connection is
    always returned to the pool.
    """
    conn = _acquire()
    cursor = None
    try:
        cursor = conn.cursor(dictionary=True, buffered=True)
        yield cursor
        conn.commit()
    except BaseException:
        try:
            conn.rollback()
        except Exception as exc:  # connection may be dead; still return it
            log.warning("mysql.rollback_failed", error=str(exc))
        raise
    finally:
        if cursor is not None:
            with suppress(Exception):
                cursor.close()
        try:
            conn.close()  # PooledMySQLConnection.close() returns it to the pool
        except Exception as exc:  # pragma: no cover - best effort
            log.warning("mysql.return_to_pool_failed", error=str(exc))


def ping() -> bool:
    """True when a pooled connection can run ``SELECT 1``; never raises."""
    try:
        with transaction() as cur:
            cur.execute("SELECT 1 AS ok")
            row = cur.fetchone()
        return bool(row and row.get("ok") == 1)
    except Exception as exc:
        log.warning("mysql.ping_failed", error=str(exc))
        return False
