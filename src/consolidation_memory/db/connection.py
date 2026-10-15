"""Thread-local SQLite connection pooling."""

from __future__ import annotations

import logging
import sqlite3
import threading
from contextlib import contextmanager, suppress
from pathlib import Path

from consolidation_memory.config import get_config as _get_config

logger = logging.getLogger(__name__)

_local = threading.local()
_all_connections: list[sqlite3.Connection] = []
_conn_list_lock = threading.Lock()

def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _close_and_untrack_connection(conn: sqlite3.Connection) -> None:
    """Close a cached connection and remove it from global tracking."""
    # conn.close() raises only sqlite3.Error, so suppressing exactly that keeps a
    # failed close from skipping the untrack below while masking nothing broader.
    with suppress(sqlite3.Error):
        conn.close()
    # The connection may already be untracked by a concurrent reset.
    with _conn_list_lock, suppress(ValueError):
        _all_connections.remove(conn)


def _get_cached_connection() -> sqlite3.Connection:
    """Return a thread-local cached connection. Creates one if needed."""
    current_db_path = str(_get_config().DB_PATH)
    conn: sqlite3.Connection | None = getattr(_local, "conn", None)
    cached_db_path: str | None = getattr(_local, "db_path", None)
    if conn is not None:
        # Project switching updates config paths at runtime; if the DB path changed,
        # discard the stale thread-local connection and open a new one.
        if cached_db_path and cached_db_path != current_db_path:
            _close_and_untrack_connection(conn)
            _local.conn = None
            _local.db_path = None
            _local.conn_depth = 0
            conn = None
        else:
            try:
                conn.execute("SELECT 1")
                return conn
            except (sqlite3.ProgrammingError, sqlite3.OperationalError):
                _close_and_untrack_connection(conn)
                _local.conn = None
                _local.db_path = None
                _local.conn_depth = 0
                conn = None

    _ensure_parent(_get_config().DB_PATH)
    # Connections are still thread-local for normal use, but we allow
    # cross-thread close during global teardown in tests/shutdown.
    conn = sqlite3.connect(current_db_path, timeout=10, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    _local.conn = conn
    _local.db_path = current_db_path
    with _conn_list_lock:
        _all_connections.append(conn)
    return conn


def close_thread_local_connection() -> None:
    """Close only the current thread's cached connection, if present."""
    conn: sqlite3.Connection | None = getattr(_local, "conn", None)
    if conn is not None:
        _close_and_untrack_connection(conn)
    _local.conn = None
    _local.db_path = None
    _local.conn_depth = 0


def close_all_connections() -> None:
    """Close all thread-local connections. Call during shutdown or test teardown."""
    with _conn_list_lock:
        for conn in _all_connections:
            # conn.close() raises only sqlite3.Error (e.g. unfinalized statements).
            # Suppressing exactly that lets the remaining connections still be
            # closed and the tracking list still cleared; masks nothing broader.
            with suppress(sqlite3.Error):
                conn.close()
        _all_connections.clear()
    # Also clear this thread's cached reference
    _local.conn = None
    _local.db_path = None
    _local.conn_depth = 0
    # Reset FTS5 availability cache (new DB may or may not have FTS5)
    from consolidation_memory.db.episodes import _reset_fts5_cache

    _reset_fts5_cache()


@contextmanager
def get_connection():
    """Yield a thread-local SQLite connection with commit/rollback.

    Re-entrant: nested calls share the same connection and only the
    outermost context manager commits or rolls back.
    """
    conn = _get_cached_connection()
    depth = int(getattr(_local, "conn_depth", 0))
    _local.conn_depth = depth + 1
    try:
        yield conn
        if depth == 0:
            conn.commit()
    except Exception:
        if depth == 0:
            # Rollback is itself best-effort; the original error is re-raised.
            with suppress(sqlite3.Error):
                conn.rollback()
        raise
    finally:
        _local.conn_depth = depth

