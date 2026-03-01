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

from consolidation_memory.config import get_config as _get_config
from consolidation_memory.types import StatsDict

_local = threading.local()
_all_connections: list[sqlite3.Connection] = []  # Track all thread-local connections for cleanup
_conn_list_lock = threading.Lock()

# ── Schema versioning ────────────────────────────────────────────────────────

CURRENT_SCHEMA_VERSION = 6

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
    4: [
        "CREATE INDEX IF NOT EXISTS idx_episodes_consolidation_attempts ON episodes(consolidation_attempts)",
    ],
    5: [
        """CREATE TABLE IF NOT EXISTS knowledge_records (
            id              TEXT PRIMARY KEY,
            topic_id        TEXT NOT NULL,
            record_type     TEXT NOT NULL,
            content         TEXT NOT NULL,
            embedding_text  TEXT NOT NULL,
            source_episodes TEXT NOT NULL DEFAULT '[]',
            confidence      REAL NOT NULL DEFAULT 0.8,
            created_at      TEXT NOT NULL,
            updated_at      TEXT NOT NULL,
            access_count    INTEGER NOT NULL DEFAULT 0,
            deleted         INTEGER NOT NULL DEFAULT 0,
            FOREIGN KEY (topic_id) REFERENCES knowledge_topics(id)
        )""",
        "CREATE INDEX IF NOT EXISTS idx_records_topic ON knowledge_records(topic_id)",
        "CREATE INDEX IF NOT EXISTS idx_records_type ON knowledge_records(record_type)",
        "CREATE INDEX IF NOT EXISTS idx_records_deleted ON knowledge_records(deleted)",
    ],
    6: [
        "ALTER TABLE knowledge_records ADD COLUMN valid_from TEXT",
        "ALTER TABLE knowledge_records ADD COLUMN valid_until TEXT",
        "CREATE INDEX IF NOT EXISTS idx_records_valid_until ON knowledge_records(valid_until)",
    ],
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _get_cached_connection() -> sqlite3.Connection:
    """Return a thread-local cached connection. Creates one if needed."""
    conn: sqlite3.Connection | None = getattr(_local, "conn", None)
    if conn is not None:
        try:
            conn.execute("SELECT 1")
            return conn
        except (sqlite3.ProgrammingError, sqlite3.OperationalError):
            _local.conn = None

    _ensure_parent(_get_config().DB_PATH)
    conn = sqlite3.connect(str(_get_config().DB_PATH), timeout=10)
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
        # Individual execute() calls instead of executescript() — the latter
        # implicitly COMMITs before running, breaking transaction atomicity.
        conn.execute("""CREATE TABLE IF NOT EXISTS episodes (
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
            )""")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_episodes_consolidated ON episodes(consolidated)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_episodes_created ON episodes(created_at)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_episodes_type ON episodes(content_type)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_episodes_deleted ON episodes(deleted)"
        )
        conn.execute("""CREATE TABLE IF NOT EXISTS knowledge_topics (
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
            )""")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_knowledge_filename ON knowledge_topics(filename)"
        )
        conn.execute("""CREATE TABLE IF NOT EXISTS consolidation_runs (
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
            )""")
        conn.execute("""CREATE TABLE IF NOT EXISTS schema_version (
                version     INTEGER NOT NULL,
                applied_at  TEXT NOT NULL
            )""")

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
    episode_id: str | None = None,
) -> str:
    if episode_id is None:
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
    return bool(cursor.rowcount and cursor.rowcount > 0)


def hard_delete_episode(episode_id: str) -> bool:
    """Permanently delete an episode from the database.

    Used for rollback when FAISS add fails — soft-delete would leave an orphan
    that dedup checks still find.
    """
    with get_connection() as conn:
        cursor = conn.execute(
            "DELETE FROM episodes WHERE id = ?", (episode_id,)
        )
    return bool(cursor.rowcount and cursor.rowcount > 0)


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
            "SELECT id, source_episodes FROM knowledge_topics WHERE filename = ?",
            (filename,),
        ).fetchone()

        if existing:
            topic_id: str = str(existing["id"])
            old_sources = json.loads(existing["source_episodes"])
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
            try:
                conn.execute(
                    """INSERT INTO knowledge_topics
                       (id, filename, title, summary, created_at, updated_at,
                        source_episodes, fact_count, confidence)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (topic_id, filename, title, summary, now, now,
                     json.dumps(source_episodes), fact_count, confidence),
                )
            except sqlite3.IntegrityError:
                # Race: concurrent insert won — fall back to update
                existing = conn.execute(
                    "SELECT id, source_episodes FROM knowledge_topics WHERE filename = ?",
                    (filename,),
                ).fetchone()
                if existing is None:
                    raise
                topic_id = str(existing["id"])
                old_sources = json.loads(existing["source_episodes"])
                merged = list(set(old_sources + source_episodes))
                conn.execute(
                    """UPDATE knowledge_topics
                       SET title = ?, summary = ?, updated_at = ?,
                           source_episodes = ?, fact_count = ?, confidence = ?
                       WHERE id = ?""",
                    (title, summary, now, json.dumps(merged),
                     fact_count, confidence, topic_id),
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


# ── Knowledge Record CRUD ─────────────────────────────────────────────────

def insert_knowledge_records(
    topic_id: str,
    records: list[dict[str, Any]],
    source_episodes: list[str] | None = None,
) -> list[str]:
    """Insert multiple knowledge records for a topic.

    Each record dict must have: record_type, content (JSON-serializable dict),
    embedding_text. Optional: confidence, valid_from.

    Returns list of inserted record IDs.
    """
    if not records:
        return []
    now = _now()
    src = json.dumps(source_episodes or [])
    ids: list[str] = []
    with get_connection() as conn:
        for rec in records:
            rec_id = str(uuid.uuid4())
            content = rec["content"] if isinstance(rec["content"], str) else json.dumps(rec["content"])
            valid_from = rec.get("valid_from")
            conn.execute(
                """INSERT INTO knowledge_records
                   (id, topic_id, record_type, content, embedding_text,
                    source_episodes, confidence, created_at, updated_at, valid_from)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (rec_id, topic_id, rec["record_type"], content,
                 rec["embedding_text"], src, rec.get("confidence", 0.8),
                 now, now, valid_from),
            )
            ids.append(rec_id)
    return ids


def expire_record(record_id: str, valid_until: str | None = None) -> bool:
    """Set valid_until on a record, marking it as temporally superseded.

    Unlike soft-delete, expired records retain their content and can be
    retrieved with include_expired=True for historical queries.

    Returns True if a record was updated.
    """
    ts = valid_until or _now()
    with get_connection() as conn:
        cursor = conn.execute(
            "UPDATE knowledge_records SET valid_until = ?, updated_at = ? WHERE id = ? AND deleted = 0",
            (ts, _now(), record_id),
        )
    return bool(cursor.rowcount and cursor.rowcount > 0)


def get_all_active_records(include_expired: bool = False) -> list[dict[str, Any]]:
    """Return all non-deleted knowledge records.

    Args:
        include_expired: If False (default), exclude records where
            valid_until is set and in the past.
    """
    now = _now()
    with get_connection() as conn:
        if include_expired:
            rows = conn.execute(
                """SELECT kr.*, kt.filename as topic_filename, kt.title as topic_title
                   FROM knowledge_records kr
                   JOIN knowledge_topics kt ON kr.topic_id = kt.id
                   WHERE kr.deleted = 0
                   ORDER BY kr.updated_at DESC"""
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT kr.*, kt.filename as topic_filename, kt.title as topic_title
                   FROM knowledge_records kr
                   JOIN knowledge_topics kt ON kr.topic_id = kt.id
                   WHERE kr.deleted = 0
                     AND (kr.valid_until IS NULL OR kr.valid_until > ?)
                   ORDER BY kr.updated_at DESC""",
                (now,),
            ).fetchall()
    return [dict(r) for r in rows]


def get_records_by_topic(topic_id: str, include_expired: bool = False) -> list[dict[str, Any]]:
    """Return all active records for a specific topic.

    Args:
        include_expired: If False (default), exclude temporally expired records.
    """
    with get_connection() as conn:
        if include_expired:
            rows = conn.execute(
                "SELECT * FROM knowledge_records WHERE topic_id = ? AND deleted = 0",
                (topic_id,),
            ).fetchall()
        else:
            now = _now()
            rows = conn.execute(
                """SELECT * FROM knowledge_records WHERE topic_id = ? AND deleted = 0
                   AND (valid_until IS NULL OR valid_until > ?)""",
                (topic_id, now),
            ).fetchall()
    return [dict(r) for r in rows]


def soft_delete_records_by_topic(topic_id: str) -> int:
    """Soft-delete all records for a topic. Returns count of affected rows."""
    with get_connection() as conn:
        cursor = conn.execute(
            "UPDATE knowledge_records SET deleted = 1, updated_at = ? WHERE topic_id = ? AND deleted = 0",
            (_now(), topic_id),
        )
    return int(cursor.rowcount)


def soft_delete_records_by_ids(record_ids: list[str]) -> int:
    """Soft-delete specific records by their IDs. Returns count of affected rows."""
    if not record_ids:
        return 0
    now = _now()
    with get_connection() as conn:
        placeholders = ",".join("?" for _ in record_ids)
        cursor = conn.execute(
            f"UPDATE knowledge_records SET deleted = 1, updated_at = ? WHERE id IN ({placeholders}) AND deleted = 0",
            [now] + record_ids,
        )
    return int(cursor.rowcount)


def increment_record_access(record_ids: list[str]) -> None:
    """Increment access count for the given records."""
    if not record_ids:
        return
    with get_connection() as conn:
        placeholders = ",".join("?" for _ in record_ids)
        conn.execute(
            f"""UPDATE knowledge_records SET access_count = access_count + 1,
                updated_at = ? WHERE id IN ({placeholders})""",
            [_now()] + record_ids,
        )


def get_record_count(include_expired: bool = False) -> int:
    """Return count of active (non-deleted) knowledge records."""
    with get_connection() as conn:
        if include_expired:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM knowledge_records WHERE deleted = 0"
            ).fetchone()
        else:
            now = _now()
            row = conn.execute(
                """SELECT COUNT(*) as cnt FROM knowledge_records
                   WHERE deleted = 0 AND (valid_until IS NULL OR valid_until > ?)""",
                (now,),
            ).fetchone()
    return row["cnt"] if row else 0


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

def get_stats() -> StatsDict:
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
        rec_counts = conn.execute(
            """SELECT
                 COUNT(*) FILTER (WHERE deleted = 0) as total_records,
                 COUNT(*) FILTER (WHERE deleted = 0 AND record_type = 'fact') as facts,
                 COUNT(*) FILTER (WHERE deleted = 0 AND record_type = 'solution') as solutions,
                 COUNT(*) FILTER (WHERE deleted = 0 AND record_type = 'preference') as preferences,
                 COUNT(*) FILTER (WHERE deleted = 0 AND record_type = 'procedure') as procedures
               FROM knowledge_records"""
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
            "total_records": rec_counts["total_records"],
            "records_by_type": {
                "facts": rec_counts["facts"],
                "solutions": rec_counts["solutions"],
                "preferences": rec_counts["preferences"],
                "procedures": rec_counts["procedures"],
            },
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


def reset_stale_consolidation_attempts(max_attempts: int = 5, stale_hours: int = 24) -> int:
    """Reset consolidation_attempts for episodes stuck at max that haven't been
    retried recently. This allows episodes to be reconsolidated after the LLM
    backend recovers from an outage.

    Returns:
        Number of episodes reset.
    """
    from datetime import timedelta
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=stale_hours)).isoformat()
    with get_connection() as conn:
        cursor = conn.execute(
            """UPDATE episodes SET consolidation_attempts = 0, last_consolidation_attempt = NULL
               WHERE consolidation_attempts >= ? AND deleted = 0 AND consolidated = 0
               AND (last_consolidation_attempt IS NULL OR last_consolidation_attempt < ?)""",
            (max_attempts, cutoff),
        )
    return int(cursor.rowcount)


def get_median_access_count() -> float:
    """Compute the median access_count across all active episodes using SQL."""
    with get_connection() as conn:
        row = conn.execute(
            """SELECT access_count FROM episodes
               WHERE deleted = 0 AND consolidated != 2
               ORDER BY access_count
               LIMIT 1 OFFSET (
                   SELECT COUNT(*) / 2 FROM episodes
                   WHERE deleted = 0 AND consolidated != 2
               )"""
        ).fetchone()
    return float(row["access_count"]) if row else 0.0


def get_active_episodes_paginated(offset: int = 0, limit: int = 1000) -> list[dict[str, Any]]:
    """Return a page of non-deleted, non-pruned episodes for surprise adjustment."""
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT id, surprise_score, access_count, created_at, updated_at, consolidated
               FROM episodes WHERE deleted = 0 AND consolidated != 2
               ORDER BY id LIMIT ? OFFSET ?""",
            (limit, offset),
        ).fetchall()
    return [dict(r) for r in rows]


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


def search_episodes(
    query: str | None = None,
    content_types: list[str] | None = None,
    tags: list[str] | None = None,
    after: str | None = None,
    before: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Keyword/metadata search over episodes. No embeddings required.

    Args:
        query: Text substring to search in episode content (case-insensitive).
        content_types: Filter to specific content types.
        tags: Filter to episodes with at least one matching tag.
        after: Only episodes created after this ISO date.
        before: Only episodes created before this ISO date.
        limit: Max results.

    Returns:
        List of episode dicts, ordered by created_at descending.
    """
    conditions: list[str] = ["deleted = 0"]
    params: list[Any] = []

    if query:
        # Escape LIKE wildcards in user input to prevent unintended pattern matching
        escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        conditions.append("content LIKE ? ESCAPE '\\'")
        params.append(f"%{escaped}%")

    if content_types:
        placeholders = ",".join("?" for _ in content_types)
        conditions.append(f"content_type IN ({placeholders})")
        params.extend(content_types)

    if after:
        conditions.append("created_at > ?")
        params.append(after)

    if before:
        conditions.append("created_at < ?")
        params.append(before)

    where = " AND ".join(conditions)
    # Over-fetch when tag filtering is needed, since tags are stored as JSON
    # and filtered in Python after the SQL query.
    fetch_limit = limit * 5 if tags else limit
    sql = f"SELECT * FROM episodes WHERE {where} ORDER BY created_at DESC LIMIT ?"
    params.append(fetch_limit)

    with get_connection() as conn:
        rows = conn.execute(sql, params).fetchall()

    results = []
    for row in rows:
        ep = dict(row)
        # Tag filtering in Python since tags are stored as JSON array string
        if tags:
            ep_tags = json.loads(ep["tags"]) if isinstance(ep["tags"], str) else ep["tags"]
            if not set(tags).intersection(ep_tags):
                continue
        results.append(ep)
        if len(results) >= limit:
            break

    return results


def get_consolidation_metrics(limit: int = 20) -> list[dict[str, Any]]:
    """Retrieve recent consolidation metrics, newest first."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM consolidation_metrics ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]
