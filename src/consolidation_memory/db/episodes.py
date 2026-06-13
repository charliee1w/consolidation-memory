"""Episode CRUD, FTS5, pruning/protection, and access counts."""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from consolidation_memory.db._helpers import _normalize_id_tokens, _now
from consolidation_memory.db.connection import get_connection
from consolidation_memory.db.scope import _apply_scope_filters, _coerce_scope_row

logger = logging.getLogger(__name__)

_fts5_available: bool | None = None
_fts5_lock = threading.Lock()
_FTS5_OPERATORS = {"AND", "OR", "NOT", "NEAR"}

def insert_episode(
    content: str,
    content_type: str = "exchange",
    tags: list[str] | None = None,
    surprise_score: float = 0.5,
    source_session: str | None = None,
    scope: Mapping[str, Any] | None = None,
    episode_id: str | None = None,
    *,
    created_at: str | None = None,
    updated_at: str | None = None,
    access_count: int | None = None,
    consolidated: int | None = None,
    consolidated_at: str | None = None,
    consolidated_to: str | None = None,
    deleted: int | None = None,
    consolidation_attempts: int | None = None,
    last_consolidation_attempt: str | None = None,
    protected: int | None = None,
    indexed: int | bool | None = None,
    conn: sqlite3.Connection | None = None,
) -> str:
    if episode_id is None:
        episode_id = str(uuid.uuid4())
    created_ts = created_at or _now()
    updated_ts = updated_at or created_ts
    scope_row = _coerce_scope_row(scope)
    def _insert(active_conn: sqlite3.Connection) -> None:
        active_conn.execute(
            """INSERT INTO episodes
               (id, created_at, updated_at, content, content_type, tags,
                surprise_score, indexed, access_count, source_session,
                consolidated, consolidated_at, consolidated_to, deleted,
                consolidation_attempts, last_consolidation_attempt, protected,
                namespace_slug, namespace_sharing_mode,
                app_client_name, app_client_type, app_client_provider, app_client_external_key,
                agent_name, agent_external_key,
                session_external_key, session_kind,
                project_slug, project_display_name, project_root_uri,
                project_repo_remote, project_default_branch)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                episode_id,
                created_ts,
                updated_ts,
                content,
                content_type,
                json.dumps(tags or []),
                surprise_score,
                1 if indexed is None else int(bool(indexed)),
                0 if access_count is None else int(access_count),
                source_session,
                0 if consolidated is None else int(consolidated),
                consolidated_at,
                consolidated_to,
                0 if deleted is None else int(deleted),
                0 if consolidation_attempts is None else int(consolidation_attempts),
                last_consolidation_attempt,
                0 if protected is None else int(protected),
             scope_row["namespace_slug"], scope_row["namespace_sharing_mode"],
             scope_row["app_client_name"], scope_row["app_client_type"],
             scope_row["app_client_provider"], scope_row["app_client_external_key"],
             scope_row["agent_name"], scope_row["agent_external_key"],
             scope_row["session_external_key"], scope_row["session_kind"],
             scope_row["project_slug"], scope_row["project_display_name"],
             scope_row["project_root_uri"], scope_row["project_repo_remote"],
             scope_row["project_default_branch"],
            ),
        )
        # FTS insert within the same transaction for atomicity
        fts_insert(episode_id, content)
    if conn is None:
        with get_connection() as managed_conn:
            _insert(managed_conn)
    else:
        _insert(conn)
    return episode_id


def get_episode(
    episode_id: str,
    scope: Mapping[str, Any] | None = None,
    *,
    include_unindexed: bool = False,
) -> dict[str, Any] | None:
    conditions = ["id = ?", "deleted = 0"]
    if not include_unindexed:
        conditions.append("indexed = 1")
    params: list[Any] = [episode_id]
    _apply_scope_filters(conditions, params, scope)
    where_clause = " AND ".join(conditions)
    with get_connection() as conn:
        row = conn.execute(
            f"SELECT * FROM episodes WHERE {where_clause}",
            params,
        ).fetchone()
    return dict(row) if row else None


def get_episodes_batch(
    episode_ids: list[str],
    *,
    include_unindexed: bool = False,
) -> dict[str, dict[str, Any]]:
    """Fetch multiple episodes in a single query. Returns {id: episode_dict}."""
    if not episode_ids:
        return {}
    conditions = ["id IN ({placeholders})", "deleted = 0"]
    if not include_unindexed:
        conditions.append("indexed = 1")
    with get_connection() as conn:
        placeholders = ",".join("?" for _ in episode_ids)
        where_clause = " AND ".join(condition.format(placeholders=placeholders) for condition in conditions)
        rows = conn.execute(
            f"SELECT * FROM episodes WHERE {where_clause}",
            episode_ids,
        ).fetchall()
    return {row["id"]: dict(row) for row in rows}


def get_unindexed_episodes(limit: int = 200) -> list[dict[str, Any]]:
    """Return non-deleted episodes whose vectors are not yet marked durable."""
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT * FROM episodes
               WHERE deleted = 0 AND indexed = 0
               ORDER BY created_at ASC
               LIMIT ?""",
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def mark_episode_indexed(
    episode_ids: Sequence[str],
    *,
    indexed: bool = True,
) -> int:
    """Mark episodes visible once their vectors are durably persisted."""
    if not episode_ids:
        return 0
    placeholders = ",".join("?" for _ in episode_ids)
    with get_connection() as conn:
        cursor = conn.execute(
            f"""UPDATE episodes
                SET indexed = ?
                WHERE id IN ({placeholders})""",
            [1 if indexed else 0, *episode_ids],
        )
    return int(cursor.rowcount or 0)


def get_existing_episode_ids(
    episode_ids: Sequence[str],
    *,
    include_deleted: bool = False,
    scope: Mapping[str, Any] | None = None,
) -> set[str]:
    """Return the subset of provided episode IDs that currently exist in SQLite."""
    if not episode_ids:
        return set()
    placeholders = ",".join("?" for _ in episode_ids)
    conditions = [f"id IN ({placeholders})"]
    params: list[Any] = list(episode_ids)
    if not include_deleted:
        conditions.append("deleted = 0")
    _apply_scope_filters(conditions, params, scope)
    where_clause = " AND ".join(conditions)
    with get_connection() as conn:
        rows = conn.execute(
            f"SELECT id FROM episodes WHERE {where_clause}",
            params,
        ).fetchall()
    return {str(row["id"]) for row in rows}


def get_existing_record_ids(
    record_ids: Sequence[str],
    *,
    include_deleted: bool = False,
    scope: Mapping[str, Any] | None = None,
) -> set[str]:
    """Return the subset of provided record IDs that currently exist in SQLite."""
    if not record_ids:
        return set()
    placeholders = ",".join("?" for _ in record_ids)
    conditions = [f"id IN ({placeholders})"]
    params: list[Any] = list(record_ids)
    if not include_deleted:
        conditions.append("deleted = 0")
    _apply_scope_filters(conditions, params, scope)
    where_clause = " AND ".join(conditions)
    with get_connection() as conn:
        rows = conn.execute(
            f"SELECT id FROM knowledge_records WHERE {where_clause}",
            params,
        ).fetchall()
    return {str(row["id"]) for row in rows}


def get_unconsolidated_episodes(
    limit: int = 200,
    max_attempts: int = 5,
    priority_episode_ids: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    priority_ids = _normalize_id_tokens(priority_episode_ids)
    with get_connection() as conn:
        if priority_ids:
            priority_placeholders = ",".join("?" for _ in priority_ids)
            rows = conn.execute(
                f"""SELECT * FROM episodes
                   WHERE consolidated = 0 AND deleted = 0 AND indexed = 1
                     AND consolidation_attempts < ?
                   ORDER BY
                     CASE WHEN id IN ({priority_placeholders}) THEN 0 ELSE 1 END,
                     created_at DESC
                   LIMIT ?""",
                [max_attempts, *priority_ids, limit],
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT * FROM episodes
                   WHERE consolidated = 0 AND deleted = 0 AND indexed = 1
                     AND consolidation_attempts < ?
                   ORDER BY created_at DESC LIMIT ?""",
                (max_attempts, limit),
            ).fetchall()
    return [dict(r) for r in rows]


def increment_access(episode_ids: list[str]) -> None:
    if not episode_ids:
        return
    with get_connection() as conn:
        placeholders = ",".join("?" for _ in episode_ids)
        query = f"""UPDATE episodes SET access_count = access_count + 1,
            updated_at = ? WHERE id IN ({placeholders})"""
        conn.execute(
            query,
            [_now()] + episode_ids,
        )


def mark_consolidated(episode_ids: list[str], topic_filename: str) -> None:
    if not episode_ids:
        return
    now = _now()
    with get_connection() as conn:
        placeholders = ",".join("?" for _ in episode_ids)
        query = f"""UPDATE episodes SET consolidated = 1,
            consolidated_at = ?, consolidated_to = ?, updated_at = ?
            WHERE id IN ({placeholders})"""
        conn.execute(
            query,
            [now, topic_filename, now] + episode_ids,
        )


def mark_pruned(episode_ids: list[str]) -> None:
    if not episode_ids:
        return
    now = _now()
    with get_connection() as conn:
        placeholders = ",".join("?" for _ in episode_ids)
        query = f"""UPDATE episodes SET consolidated = 2, updated_at = ?
            WHERE id IN ({placeholders}) AND consolidated = 1"""
        conn.execute(
            query,
            [now] + episode_ids,
        )


def soft_delete_episode(
    episode_id: str,
    scope: Mapping[str, Any] | None = None,
) -> bool:
    conditions = ["id = ?", "deleted = 0", "indexed = 1"]
    params: list[Any] = [episode_id]
    _apply_scope_filters(conditions, params, scope)
    where_clause = " AND ".join(conditions)
    now = _now()
    with get_connection() as conn:
        cursor = conn.execute(
            f"UPDATE episodes SET deleted = 1, updated_at = ? WHERE {where_clause}",
            [now, *params],
        )
        deleted = bool(cursor.rowcount and cursor.rowcount > 0)
        if deleted:
            fts_delete(episode_id)
    return deleted


def restore_soft_deleted_episode(
    episode_id: str,
    scope: Mapping[str, Any] | None = None,
) -> bool:
    """Restore a soft-deleted episode.

    Used as a compensating action when vector tombstoning fails during forget().
    """
    conditions = ["id = ?", "deleted = 1", "indexed = 1"]
    params: list[Any] = [episode_id]
    _apply_scope_filters(conditions, params, scope)
    where_clause = " AND ".join(conditions)
    now = _now()

    content: str | None = None
    restored = False
    with get_connection() as conn:
        row = conn.execute(
            f"SELECT content FROM episodes WHERE {where_clause}",
            params,
        ).fetchone()
        if row is None:
            return False
        content = str(row["content"])
        cursor = conn.execute(
            f"UPDATE episodes SET deleted = 0, updated_at = ? WHERE {where_clause}",
            [now, *params],
        )
        restored = bool(cursor.rowcount and cursor.rowcount > 0)

    if restored and content is not None:
        fts_insert(episode_id, content)
    return restored


def hard_delete_episode(episode_id: str) -> bool:
    """Permanently delete an episode from the database.

    Used for rollback when FAISS add fails — soft-delete would leave an orphan
    that dedup checks still find.
    """
    with get_connection() as conn:
        cursor = conn.execute(
            "DELETE FROM episodes WHERE id = ?", (episode_id,)
        )
        deleted = bool(cursor.rowcount and cursor.rowcount > 0)
        if deleted:
            fts_delete(episode_id)
    return deleted


# ── FTS5 Full-Text Search ─────────────────────────────────────────────────

def fts_available() -> bool:
    """Check if the FTS5 virtual table exists. Result is cached."""
    global _fts5_available
    if _fts5_available is not None:
        return _fts5_available
    with _fts5_lock:
        if _fts5_available is not None:
            return _fts5_available
        try:
            with get_connection() as conn:
                row = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='episodes_fts'"
                ).fetchone()
                _fts5_available = row is not None
        except Exception:
            _fts5_available = False
    return _fts5_available


def _reset_fts5_cache() -> None:
    """Reset the FTS5 availability cache. Used by tests."""
    global _fts5_available
    with _fts5_lock:
        _fts5_available = None


def fts_insert(episode_id: str, content: str) -> None:
    """Insert an episode into the FTS5 index."""
    if not fts_available():
        return
    try:
        with get_connection() as conn:
            conn.execute(
                "INSERT INTO episodes_fts(episode_id, content) VALUES (?, ?)",
                (episode_id, content),
            )
    except Exception as e:
        logger.warning("FTS5 insert failed for %s: %s", episode_id, e)


def fts_delete(episode_id: str) -> None:
    """Delete an episode from the FTS5 index."""
    if not fts_available():
        return
    try:
        with get_connection() as conn:
            conn.execute(
                "DELETE FROM episodes_fts WHERE episode_id = ?",
                (episode_id,),
            )
    except Exception as e:
        logger.warning("FTS5 delete failed for %s: %s", episode_id, e)


_FTS5_OPERATORS = {"AND", "OR", "NOT", "NEAR"}


def _sanitize_fts_query(query: str) -> str:
    """Sanitize a query for FTS5 MATCH.

    Splits into terms, strips non-word characters, drops single-char tokens,
    double-quotes FTS5 reserved operators so they're treated as literals,
    and joins with OR so documents matching any term are returned.
    """
    terms = re.findall(r'\w+', query)
    terms = [t for t in terms if len(t) > 1]
    if not terms:
        return ""
    # Double-quote terms that match FTS5 operators so they're treated as literals
    safe_terms = [f'"{t}"' if t.upper() in _FTS5_OPERATORS else t for t in terms]
    return " OR ".join(safe_terms)


def fts_search(query: str, limit: int = 50) -> list[tuple[str, float]]:
    """BM25 keyword search over episodes.

    Returns list of (episode_id, normalized_bm25_score) sorted by relevance.
    Normalization: score = raw / (raw + 1.0) where raw = -bm25().
    """
    if not fts_available():
        return []
    sanitized = _sanitize_fts_query(query)
    if not sanitized:
        return []
    try:
        with get_connection() as conn:
            rows = conn.execute(
                """SELECT episode_id, bm25(episodes_fts) as rank
                   FROM episodes_fts
                   WHERE episodes_fts MATCH ?
                   ORDER BY rank
                   LIMIT ?""",
                (sanitized, limit),
            ).fetchall()
        results = []
        for row in rows:
            raw = -row["rank"]  # bm25() returns negative values; negate for positive
            normalized = raw / (raw + 1.0) if raw > 0 else 0.0
            results.append((row["episode_id"], normalized))
        return results
    except Exception as e:
        logger.warning("FTS5 search failed: %s", e)
        return []


def fts_rebuild() -> None:
    """Rebuild the FTS5 index from the episodes table."""
    if not fts_available():
        return
    with get_connection() as conn:
        conn.execute("DELETE FROM episodes_fts")
        conn.execute(
            """INSERT INTO episodes_fts(episode_id, content)
               SELECT id, content FROM episodes WHERE deleted = 0"""
        )


def get_prunable_episodes(
    days: int = 30,
    scope: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Episodes that are consolidated and older than `days`, excluding protected ones."""
    from datetime import timedelta
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    conditions = [
        "consolidated = 1",
        "consolidated_at < ?",
        "deleted = 0",
        "indexed = 1",
        "protected = 0",
    ]
    params: list[Any] = [cutoff]
    _apply_scope_filters(conditions, params, scope)
    where_clause = " AND ".join(conditions)
    with get_connection() as conn:
        rows = conn.execute(
            f"SELECT * FROM episodes WHERE {where_clause}",
            params,
        ).fetchall()
    return [dict(r) for r in rows]


def protect_episode(
    episode_id: str,
    scope: Mapping[str, Any] | None = None,
) -> bool:
    """Mark an episode as protected from pruning. Returns True if found."""
    now = _now()
    conditions = ["id = ?", "deleted = 0"]
    params: list[Any] = [episode_id]
    _apply_scope_filters(conditions, params, scope)
    where_clause = " AND ".join(conditions)
    with get_connection() as conn:
        cursor = conn.execute(
            f"UPDATE episodes SET protected = 1, updated_at = ? WHERE {where_clause}",
            [now, *params],
        )
    return bool(cursor.rowcount and cursor.rowcount > 0)


def protect_by_tag(
    tag: str,
    scope: Mapping[str, Any] | None = None,
) -> int:
    """Mark all episodes with a given tag as protected. Returns count updated."""
    now = _now()
    # Tags are stored as JSON arrays, use LIKE with the tag value.
    # Escape LIKE wildcards in the tag to prevent injection (e.g. tag="%" matching all).
    escaped = tag.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    pattern = f'%"{escaped}"%'
    conditions = [
        "tags LIKE ? ESCAPE '\\'",
        "deleted = 0",
        "indexed = 1",
        "protected = 0",
    ]
    params: list[Any] = [pattern]
    _apply_scope_filters(conditions, params, scope)
    where_clause = " AND ".join(conditions)
    with get_connection() as conn:
        cursor = conn.execute(
            "UPDATE episodes SET protected = 1, updated_at = ? "
            f"WHERE {where_clause}",
            [now, *params],
        )
    return cursor.rowcount or 0


def get_low_confidence_records(
    threshold: float = 0.5,
    scope: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return active records below the confidence threshold."""
    now = _now()
    conditions = [
        "kr.deleted = 0",
        "(kr.valid_from IS NULL OR julianday(kr.valid_from) <= julianday(?))",
        "(kr.valid_until IS NULL OR julianday(kr.valid_until) > julianday(?))",
        "kr.confidence < ?",
    ]
    params: list[Any] = [now, now, threshold]
    _apply_scope_filters(conditions, params, scope, table_alias="kr")
    where_clause = " AND ".join(conditions)
    with get_connection() as conn:
        rows = conn.execute(
            f"""SELECT kr.*, kt.filename as topic_filename, kt.title as topic_title
               FROM knowledge_records kr
               JOIN knowledge_topics kt ON kr.topic_id = kt.id
               WHERE {where_clause}
               ORDER BY kr.confidence ASC""",
            params,
        ).fetchall()
    return [dict(r) for r in rows]


def count_protected_episodes(scope: Mapping[str, Any] | None = None) -> int:
    """Count episodes marked protected from pruning."""
    conditions = ["protected = 1", "deleted = 0"]
    params: list[Any] = []
    _apply_scope_filters(conditions, params, scope)
    where_clause = " AND ".join(conditions)
    with get_connection() as conn:
        row = conn.execute(
            f"SELECT COUNT(*) as c FROM episodes WHERE {where_clause}",
            params,
        ).fetchone()
    return int(row["c"]) if row else 0


