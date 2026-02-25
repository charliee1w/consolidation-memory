"""SQLite database for episode metadata and knowledge tracking.

Uses WAL mode for concurrent read (MCP server) / write (consolidation script).
Thread-local connection caching avoids per-operation open/close overhead.
Includes schema versioning with automatic migration.
"""

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from consolidation_memory.config import DB_PATH

_local = threading.local()
_all_connections: list[sqlite3.Connection] = []  # Track all thread-local connections for cleanup
_conn_list_lock = threading.Lock()

# ── Schema versioning ────────────────────────────────────────────────────────

CURRENT_SCHEMA_VERSION = 3

# Future migrations go here: version -> list of SQL statements
MIGRATIONS: dict[int, list[str]] = {
    2: [
        "ALTER TABLE episodes ADD COLUMN consolidation_attempts INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE episodes ADD COLUMN last_consolidation_attempt TEXT",
    ],
    3: [
        """CREATE TABLE IF NOT EXISTS consolidation_metrics (
            id                  TEXT PRIMARY KEY,
            run_id              TEXT NOT NULL,
            timestamp           TEXT NOT NULL,
            clusters_succeeded  INTEGER NOT NULL DEFAULT 0,
            clusters_failed     INTEGER NOT NULL DEFAULT 0,
            avg_confidence      REAL NOT NULL DEFAULT 0.0,
            episodes_processed  INTEGER NOT NULL DEFAULT 0,
            duration_seconds    REAL NOT NULL DEFAULT 0.0,
            api_calls           INTEGER NOT NULL DEFAULT 0,
            topics_created      INTEGER NOT NULL DEFAULT 0,
            topics_updated      INTEGER NOT NULL DEFAULT 0,
            episodes_pruned     INTEGER NOT NULL DEFAULT 0
        )""",
        "CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON consolidation_metrics(timestamp)",
    ],
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _get_cached_connection() -> sqlite3.Connection:
    """Return a thread-local cached connection. Creates one if needed."""
    conn = getattr(_local, "conn", None)
    if conn is not None:
        try:
            conn.execute("SELECT 1")
            return conn
        except (sqlite3.ProgrammingError, sqlite3.OperationalError):
            _local.conn = None

    _ensure_parent(DB_PATH)
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    _local.conn = conn
    with _conn_list_lock:
        _all_connections.append(conn)
    return conn


def close_all_connections() -> None:
    """Close all thread-local connections. Call during shutdown or test teardown."""
    with _conn_list_lock:
        for conn in _all_connections:
            try:
                conn.close()
            except Exception:
                pass
        _all_connections.clear()
    # Also clear this thread's cached reference
    _local.conn = None


@contextmanager
def get_connection():
    conn = _get_cached_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def ensure_schema() -> None:
    with get_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS episodes (
                id              TEXT PRIMARY KEY,
                created_at      TEXT NOT NULL,
                updated_at      TEXT NOT NULL,
                content         TEXT NOT NULL,
                content_type    TEXT NOT NULL DEFAULT 'exchange',
                tags            TEXT NOT NULL DEFAULT '[]',
                surprise_score  REAL NOT NULL DEFAULT 0.5,
                access_count    INTEGER NOT NULL DEFAULT 0,
                source_session  TEXT,
                consolidated    INTEGER NOT NULL DEFAULT 0,
                consolidated_at TEXT,
                consolidated_to TEXT,
                deleted         INTEGER NOT NULL DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_episodes_consolidated
                ON episodes(consolidated);
            CREATE INDEX IF NOT EXISTS idx_episodes_created
                ON episodes(created_at);
            CREATE INDEX IF NOT EXISTS idx_episodes_type
                ON episodes(content_type);
            CREATE INDEX IF NOT EXISTS idx_episodes_deleted
                ON episodes(deleted);

            CREATE TABLE IF NOT EXISTS knowledge_topics (
                id              TEXT PRIMARY KEY,
                filename        TEXT NOT NULL UNIQUE,
                title           TEXT NOT NULL,
                summary         TEXT NOT NULL,
                created_at      TEXT NOT NULL,
                updated_at      TEXT NOT NULL,
                source_episodes TEXT NOT NULL DEFAULT '[]',
                fact_count      INTEGER NOT NULL DEFAULT 0,
                access_count    INTEGER NOT NULL DEFAULT 0,
                confidence      REAL NOT NULL DEFAULT 0.8
            );

            CREATE INDEX IF NOT EXISTS idx_knowledge_filename
                ON knowledge_topics(filename);

            CREATE TABLE IF NOT EXISTS consolidation_runs (
                id                  TEXT PRIMARY KEY,
                started_at          TEXT NOT NULL,
                completed_at        TEXT,
                episodes_processed  INTEGER NOT NULL DEFAULT 0,
                clusters_formed     INTEGER NOT NULL DEFAULT 0,
                topics_created      INTEGER NOT NULL DEFAULT 0,
                topics_updated      INTEGER NOT NULL DEFAULT 0,
                episodes_pruned     INTEGER NOT NULL DEFAULT 0,
                status              TEXT NOT NULL DEFAULT 'running',
                error_message       TEXT
            );

            CREATE TABLE IF NOT EXISTS schema_version (
                version     INTEGER NOT NULL,
                applied_at  TEXT NOT NULL
            );
        """)

        # Check and apply migrations
        _check_and_migrate(conn)


def _check_and_migrate(conn: sqlite3.Connection) -> None:
    """Check current schema version and apply pending migrations.

    Each version's migration runs inside a SAVEPOINT so a crash mid-migration
    leaves the database in the last fully-applied version rather than a
    half-migrated state.
    """
    row = conn.execute(
        "SELECT MAX(version) as v FROM schema_version"
    ).fetchone()
    current = row["v"] if row and row["v"] is not None else 0

    if current == 0:
        # First time: apply all migrations then record current version.
        # The base CREATE TABLE statements only define the v1 schema;
        # columns and tables added in later migrations must still be executed.
        for version in range(2, CURRENT_SCHEMA_VERSION + 1):
            _apply_migration(conn, version)
        conn.execute(
            "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
            (CURRENT_SCHEMA_VERSION, _now()),
        )
        return

    if current >= CURRENT_SCHEMA_VERSION:
        return

    # Apply pending migrations, each in its own savepoint
    for version in range(current + 1, CURRENT_SCHEMA_VERSION + 1):
        _apply_migration(conn, version)
        conn.execute(
            "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
            (version, _now()),
        )


def _apply_migration(conn: sqlite3.Connection, version: int) -> None:
    """Apply a single migration version inside a SAVEPOINT for atomicity."""
    if version not in MIGRATIONS:
        return
    savepoint = f"migration_v{version}"
    conn.execute(f"SAVEPOINT {savepoint}")
    try:
        for sql in MIGRATIONS[version]:
            conn.execute(sql)
        conn.execute(f"RELEASE SAVEPOINT {savepoint}")
    except Exception:
        conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
        raise


# ── Episode CRUD ─────────────────────────────────────────────────────────────

def insert_episode(
    content: str,
    content_type: str = "exchange",
    tags: list[str] | None = None,
    surprise_score: float = 0.5,
    source_session: str | None = None,
) -> str:
    episode_id = str(uuid.uuid4())
    now = _now()
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO episodes
               (id, created_at, updated_at, content, content_type, tags,
                surprise_score, source_session)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (episode_id, now, now, content, content_type,
             json.dumps(tags or []), surprise_score, source_session),
        )
    return episode_id


def get_episode(episode_id: str) -> dict[str, Any] | None:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM episodes WHERE id = ? AND deleted = 0",
            (episode_id,),
        ).fetchone()
    return dict(row) if row else None


def get_episodes_batch(episode_ids: list[str]) -> dict[str, dict[str, Any]]:
    """Fetch multiple episodes in a single query. Returns {id: episode_dict}."""
    if not episode_ids:
        return {}
    with get_connection() as conn:
        placeholders = ",".join("?" for _ in episode_ids)
        rows = conn.execute(
            f"SELECT * FROM episodes WHERE id IN ({placeholders}) AND deleted = 0",
            episode_ids,
        ).fetchall()
    return {row["id"]: dict(row) for row in rows}


def get_unconsolidated_episodes(limit: int = 200, max_attempts: int = 5) -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT * FROM episodes
               WHERE consolidated = 0 AND deleted = 0 AND consolidation_attempts < ?
               ORDER BY created_at DESC LIMIT ?""",
            (max_attempts, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def increment_access(episode_ids: list[str]) -> None:
    if not episode_ids:
        return
    with get_connection() as conn:
        placeholders = ",".join("?" for _ in episode_ids)
        conn.execute(
            f"""UPDATE episodes SET access_count = access_count + 1,
                updated_at = ? WHERE id IN ({placeholders})""",
            [_now()] + episode_ids,
        )


def mark_consolidated(episode_ids: list[str], topic_filename: str) -> None:
    if not episode_ids:
        return
    now = _now()
    with get_connection() as conn:
        placeholders = ",".join("?" for _ in episode_ids)
        conn.execute(
            f"""UPDATE episodes SET consolidated = 1,
                consolidated_at = ?, consolidated_to = ?, updated_at = ?
                WHERE id IN ({placeholders})""",
            [now, topic_filename, now] + episode_ids,
        )


def mark_pruned(episode_ids: list[str]) -> None:
    if not episode_ids:
        return
    now = _now()
    with get_connection() as conn:
        placeholders = ",".join("?" for _ in episode_ids)
        conn.execute(
            f"""UPDATE episodes SET consolidated = 2, updated_at = ?
                WHERE id IN ({placeholders})""",
            [now] + episode_ids,
        )


def soft_delete_episode(episode_id: str) -> bool:
    with get_connection() as conn:
        cursor = conn.execute(
            "UPDATE episodes SET deleted = 1, updated_at = ? WHERE id = ? AND deleted = 0",
            (_now(), episode_id),
        )
    return cursor.rowcount > 0


def get_prunable_episodes(days: int = 30) -> list[dict[str, Any]]:
    """Episodes that are consolidated and older than `days`."""
    from datetime import timedelta
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT * FROM episodes
               WHERE consolidated = 1 AND consolidated_at < ? AND deleted = 0""",
            (cutoff,),
        ).fetchall()
    return [dict(r) for r in rows]


# ── Knowledge Topic CRUD ────────────────────────────────────────────────────

def upsert_knowledge_topic(
    filename: str,
    title: str,
    summary: str,
    source_episodes: list[str],
    fact_count: int = 0,
    confidence: float = 0.8,
) -> str:
    now = _now()
    with get_connection() as conn:
        existing = conn.execute(
            "SELECT id FROM knowledge_topics WHERE filename = ?", (filename,)
        ).fetchone()

        if existing:
            topic_id = existing["id"]
            old_row = conn.execute(
                "SELECT source_episodes FROM knowledge_topics WHERE id = ?",
                (topic_id,),
            ).fetchone()
            old_sources = json.loads(old_row["source_episodes"])
            merged = list(set(old_sources + source_episodes))
            conn.execute(
                """UPDATE knowledge_topics
                   SET title = ?, summary = ?, updated_at = ?,
                       source_episodes = ?, fact_count = ?, confidence = ?
                   WHERE id = ?""",
                (title, summary, now, json.dumps(merged),
                 fact_count, confidence, topic_id),
            )
        else:
            topic_id = str(uuid.uuid4())
            conn.execute(
                """INSERT INTO knowledge_topics
                   (id, filename, title, summary, created_at, updated_at,
                    source_episodes, fact_count, confidence)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (topic_id, filename, title, summary, now, now,
                 json.dumps(source_episodes), fact_count, confidence),
            )
    return topic_id


def get_all_knowledge_topics() -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM knowledge_topics ORDER BY updated_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def increment_topic_access(filenames: list[str]) -> None:
    if not filenames:
        return
    with get_connection() as conn:
        placeholders = ",".join("?" for _ in filenames)
        conn.execute(
            f"""UPDATE knowledge_topics SET access_count = access_count + 1,
                updated_at = ? WHERE filename IN ({placeholders})""",
            [_now()] + filenames,
        )


# ── Consolidation Run Tracking ──────────────────────────────────────────────

def start_consolidation_run() -> str:
    run_id = str(uuid.uuid4())
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO consolidation_runs (id, started_at) VALUES (?, ?)",
            (run_id, _now()),
        )
    return run_id


def complete_consolidation_run(
    run_id: str,
    episodes_processed: int = 0,
    clusters_formed: int = 0,
    topics_created: int = 0,
    topics_updated: int = 0,
    episodes_pruned: int = 0,
    status: str = "completed",
    error_message: str | None = None,
) -> None:
    with get_connection() as conn:
        conn.execute(
            """UPDATE consolidation_runs
               SET completed_at = ?, episodes_processed = ?,
                   clusters_formed = ?, topics_created = ?,
                   topics_updated = ?, episodes_pruned = ?,
                   status = ?, error_message = ?
               WHERE id = ?""",
            (_now(), episodes_processed, clusters_formed, topics_created,
             topics_updated, episodes_pruned, status, error_message, run_id),
        )


def get_last_consolidation_run() -> dict[str, Any] | None:
    with get_connection() as conn:
        row = conn.execute(
            """SELECT * FROM consolidation_runs
               ORDER BY started_at DESC LIMIT 1"""
        ).fetchone()
    return dict(row) if row else None


# ── Export / Bulk queries ────────────────────────────────────────────────────

def get_all_episodes(include_deleted: bool = False) -> list[dict[str, Any]]:
    """Return all episodes for export."""
    with get_connection() as conn:
        if include_deleted:
            rows = conn.execute(
                "SELECT * FROM episodes ORDER BY created_at"
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM episodes WHERE deleted = 0 ORDER BY created_at"
            ).fetchall()
    return [dict(r) for r in rows]


def get_all_active_episodes() -> list[dict[str, Any]]:
    """Return all non-deleted, non-pruned episodes for surprise adjustment."""
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT id, surprise_score, access_count, created_at, updated_at, consolidated
               FROM episodes WHERE deleted = 0 AND consolidated != 2"""
        ).fetchall()
    return [dict(r) for r in rows]


def update_surprise_scores(updates: list[tuple[float, str]]) -> None:
    """Batch update surprise scores. updates = [(new_score, episode_id), ...]"""
    if not updates:
        return
    now = _now()
    with get_connection() as conn:
        conn.executemany(
            "UPDATE episodes SET surprise_score = ?, updated_at = ? WHERE id = ?",
            [(score, now, eid) for score, eid in updates],
        )


# ── Stats ────────────────────────────────────────────────────────────────────

def get_stats() -> dict[str, Any]:
    with get_connection() as conn:
        ep_counts = conn.execute(
            """SELECT
                 COUNT(*) FILTER (WHERE deleted = 0) as total,
                 COUNT(*) FILTER (WHERE consolidated = 0 AND deleted = 0) as pending,
                 COUNT(*) FILTER (WHERE consolidated = 1 AND deleted = 0) as consolidated,
                 COUNT(*) FILTER (WHERE consolidated = 2 OR deleted = 1) as pruned
               FROM episodes"""
        ).fetchone()
        kt_counts = conn.execute(
            """SELECT COUNT(*) as total_topics,
                      COALESCE(SUM(fact_count), 0) as total_facts
               FROM knowledge_topics"""
        ).fetchone()

    return {
        "episodic_buffer": {
            "total": ep_counts["total"],
            "pending_consolidation": ep_counts["pending"],
            "consolidated": ep_counts["consolidated"],
            "pruned": ep_counts["pruned"],
        },
        "knowledge_base": {
            "total_topics": kt_counts["total_topics"],
            "total_facts": kt_counts["total_facts"],
        },
    }


# ── Consolidation Attempt Tracking ────────────────────────────────────────────

def increment_consolidation_attempts(episode_ids: list[str]) -> None:
    """Record a failed consolidation attempt for the given episodes."""
    if not episode_ids:
        return
    now = _now()
    with get_connection() as conn:
        placeholders = ",".join("?" for _ in episode_ids)
        conn.execute(
            f"UPDATE episodes SET consolidation_attempts = consolidation_attempts + 1, "
            f"last_consolidation_attempt = ? WHERE id IN ({placeholders})",
            [now] + episode_ids,
        )


def insert_consolidation_metrics(
    run_id: str,
    clusters_succeeded: int,
    clusters_failed: int,
    avg_confidence: float,
    episodes_processed: int,
    duration_seconds: float,
    api_calls: int,
    topics_created: int,
    topics_updated: int,
    episodes_pruned: int,
) -> str:
    """Insert a consolidation run metrics record."""
    metric_id = str(uuid.uuid4())
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO consolidation_metrics "
            "(id, run_id, timestamp, clusters_succeeded, clusters_failed, "
            "avg_confidence, episodes_processed, duration_seconds, api_calls, "
            "topics_created, topics_updated, episodes_pruned) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (metric_id, run_id, _now(), clusters_succeeded, clusters_failed,
             avg_confidence, episodes_processed, duration_seconds, api_calls,
             topics_created, topics_updated, episodes_pruned),
        )
    return metric_id


def get_consolidation_metrics(limit: int = 20) -> list[dict[str, Any]]:
    """Retrieve recent consolidation metrics, newest first."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM consolidation_metrics ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]
