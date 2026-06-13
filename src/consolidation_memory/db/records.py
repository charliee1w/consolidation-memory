"""Knowledge record CRUD, temporal queries, contradictions, tag cooccurrence."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Mapping

from consolidation_memory.db._helpers import _normalize_utc_timestamp, _now
from consolidation_memory.db.connection import get_connection
from consolidation_memory.db.scope import _apply_scope_filters, _coerce_scope_row
from consolidation_memory.utils import parse_json_list

def insert_knowledge_records(
    topic_id: str,
    records: list[dict[str, Any]],
    source_episodes: list[str] | None = None,
    scope: Mapping[str, Any] | None = None,
    conn: sqlite3.Connection | None = None,
) -> list[str]:
    """Insert multiple knowledge records for a topic.

    Each record dict must have: record_type, content (JSON-serializable dict),
    embedding_text. Optional: confidence, valid_from.

    Returns list of inserted record IDs.
    """
    if not records:
        return []
    now = _now()
    ids: list[str] = []

    def _coerce_int_column(value: object | None, default: int = 0) -> int:
        if value is None:
            return default
        if isinstance(value, (int, float, bool)):
            return int(value)
        if isinstance(value, (str, bytes, bytearray)):
            try:
                return int(value)
            except ValueError:
                return default
        try:
            return int(str(value))
        except (TypeError, ValueError):
            return default

    def _insert_rows(active_conn: sqlite3.Connection) -> None:
        for rec in records:
            rec_id = str(rec.get("id") or uuid.uuid4())
            content = rec["content"] if isinstance(rec["content"], str) else json.dumps(rec["content"])
            valid_from = rec.get("valid_from")
            valid_until = rec.get("valid_until")
            created_ts = str(rec.get("created_at") or now)
            updated_ts = str(rec.get("updated_at") or created_ts)
            access_count_value = _coerce_int_column(rec.get("access_count"))
            deleted_value = _coerce_int_column(rec.get("deleted"))
            record_scope_row = _coerce_scope_row(
                rec["scope"] if isinstance(rec.get("scope"), Mapping) else scope
            )
            record_source_episodes_raw = rec.get("source_episodes", source_episodes or [])
            if isinstance(record_source_episodes_raw, str):
                record_source_episodes = parse_json_list(record_source_episodes_raw)
            else:
                record_source_episodes = list(record_source_episodes_raw or [])
            active_conn.execute(
                """INSERT INTO knowledge_records
                   (id, topic_id, record_type, content, embedding_text,
                    source_episodes, confidence, created_at, updated_at, access_count,
                    deleted, valid_from, valid_until,
                    namespace_slug, namespace_sharing_mode,
                    app_client_name, app_client_type, app_client_provider, app_client_external_key,
                    agent_name, agent_external_key,
                    session_external_key, session_kind,
                    project_slug, project_display_name, project_root_uri,
                    project_repo_remote, project_default_branch)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    rec_id,
                    topic_id,
                    rec["record_type"],
                    content,
                    rec["embedding_text"],
                    json.dumps(record_source_episodes),
                    rec.get("confidence", 0.8),
                    created_ts,
                    updated_ts,
                    access_count_value,
                    deleted_value,
                    valid_from,
                    valid_until,
                    record_scope_row["namespace_slug"],
                    record_scope_row["namespace_sharing_mode"],
                    record_scope_row["app_client_name"],
                    record_scope_row["app_client_type"],
                    record_scope_row["app_client_provider"],
                    record_scope_row["app_client_external_key"],
                    record_scope_row["agent_name"],
                    record_scope_row["agent_external_key"],
                    record_scope_row["session_external_key"],
                    record_scope_row["session_kind"],
                    record_scope_row["project_slug"],
                    record_scope_row["project_display_name"],
                    record_scope_row["project_root_uri"],
                    record_scope_row["project_repo_remote"],
                    record_scope_row["project_default_branch"],
                ),
            )
            ids.append(rec_id)

    if conn is None:
        with get_connection() as managed_conn:
            _insert_rows(managed_conn)
    else:
        _insert_rows(conn)
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


def insert_contradiction(
    topic_id: str | None,
    old_record_id: str | None,
    new_record_id: str | None,
    old_content: str,
    new_content: str,
    resolution: str = "expired_old",
    reason: str | None = None,
) -> str:
    """Log a detected contradiction between knowledge records."""
    contradiction_id = str(uuid.uuid4())
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO contradiction_log
               (id, topic_id, old_record_id, new_record_id,
                old_content, new_content, resolution, reason, detected_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (contradiction_id, topic_id, old_record_id, new_record_id,
             old_content, new_content, resolution, reason, _now()),
        )
    return contradiction_id


def get_contradictions(
    topic_id: str | None = None,
    limit: int = 50,
    scope: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Retrieve logged contradictions, optionally filtered by topic and scope.

    Args:
        topic_id: If provided, filter to contradictions for this topic.
        limit: Max results (default 50).
        scope: Optional scope filter applied to joined knowledge topics.

    Returns:
        List of contradiction dicts, newest first.
    """
    conditions: list[str] = []
    params: list[Any] = []
    if topic_id:
        conditions.append("cl.topic_id = ?")
        params.append(topic_id)
    if scope:
        conditions.append("cl.topic_id IS NOT NULL")
        _apply_scope_filters(conditions, params, scope, table_alias="kt")
    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    params.append(limit)
    with get_connection() as conn:
        rows = conn.execute(
            f"""SELECT cl.*, kt.title as topic_title, kt.filename as topic_filename
               FROM contradiction_log cl
               LEFT JOIN knowledge_topics kt ON cl.topic_id = kt.id
               {where_clause}
               ORDER BY cl.detected_at DESC LIMIT ?""",
            params,
        ).fetchall()
    return [dict(r) for r in rows]


def get_recently_contradicted_topic_ids(
    days: int = 30,
    *,
    scope: Mapping[str, Any] | None = None,
) -> set[str]:
    """Return topic IDs that have had contradictions detected within the last N days."""
    from datetime import timedelta
    cutoff_dt = datetime.now(timezone.utc) - timedelta(days=days)
    cutoff_str = cutoff_dt.isoformat()
    conditions = ["cl.topic_id IS NOT NULL", "cl.detected_at >= ?"]
    params: list[Any] = [cutoff_str]
    if scope:
        _apply_scope_filters(conditions, params, scope, table_alias="kt")
    where_clause = " AND ".join(conditions)
    join_clause = ""
    if scope:
        join_clause = " JOIN knowledge_topics kt ON cl.topic_id = kt.id"
    with get_connection() as conn:
        rows = conn.execute(
            f"""SELECT DISTINCT cl.topic_id AS topic_id
               FROM contradiction_log cl
               {join_clause}
               WHERE {where_clause}""",
            params,
        ).fetchall()
    return {row["topic_id"] for row in rows}


def count_contradictions_since(since: str) -> int:
    """Return contradiction count with detected_at >= since."""
    with get_connection() as conn:
        row = conn.execute(
            """SELECT COUNT(*) AS c
               FROM contradiction_log
               WHERE detected_at >= ?""",
            (since,),
        ).fetchone()
    return int(row["c"]) if row else 0


def get_outcome_failure_rate_since(since: str) -> dict[str, int | float]:
    """Return failure/total outcome counts and failure rate since observed_at."""
    with get_connection() as conn:
        row = conn.execute(
            """SELECT
                   COUNT(*) AS total_count,
                   COALESCE(
                       SUM(CASE WHEN outcome_type = 'failure' THEN 1 ELSE 0 END),
                       0
                   ) AS failure_count
               FROM action_outcomes
               WHERE observed_at >= ?""",
            (since,),
        ).fetchone()
    total_count = int(row["total_count"]) if row else 0
    failure_count = int(row["failure_count"]) if row else 0
    failure_rate = failure_count / total_count if total_count > 0 else 0.0
    return {
        "failure_count": failure_count,
        "total_count": total_count,
        "failure_rate": failure_rate,
    }


def get_failure_linked_episode_ids_since(since: str) -> frozenset[str]:
    """Return episode IDs linked to failed outcomes via outcome or claim provenance."""
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT DISTINCT aos.source_episode_id AS episode_id
               FROM action_outcome_sources aos
               JOIN action_outcomes ao ON ao.id = aos.outcome_id
               WHERE ao.outcome_type = 'failure'
                 AND ao.observed_at >= ?
                 AND aos.source_episode_id IS NOT NULL
               UNION
               SELECT DISTINCT cs.source_episode_id AS episode_id
               FROM action_outcome_sources aos
               JOIN action_outcomes ao ON ao.id = aos.outcome_id
               JOIN claim_sources cs ON cs.claim_id = aos.source_claim_id
               WHERE ao.outcome_type = 'failure'
                 AND ao.observed_at >= ?
                 AND cs.source_episode_id IS NOT NULL""",
            (since, since),
        ).fetchall()
    return frozenset(
        str(row["episode_id"])
        for row in rows
        if row and row["episode_id"]
    )


def update_tag_cooccurrence(tags: list[str]) -> None:
    """Update co-occurrence counts for all tag pairs in a set.

    Maintains the invariant tag_a < tag_b for consistent pair ordering.
    """
    if len(tags) < 2:
        return
    now = _now()
    # Generate unique sorted pairs
    unique_tags = sorted(set(tags))
    pairs = []
    for i in range(len(unique_tags)):
        for j in range(i + 1, len(unique_tags)):
            pairs.append((unique_tags[i], unique_tags[j]))

    with get_connection() as conn:
        for tag_a, tag_b in pairs:
            conn.execute(
                """INSERT INTO tag_cooccurrence (tag_a, tag_b, count, last_seen)
                   VALUES (?, ?, 1, ?)
                   ON CONFLICT(tag_a, tag_b)
                   DO UPDATE SET count = count + 1, last_seen = ?""",
                (tag_a, tag_b, now, now),
            )


def get_cooccurring_tags(tags: list[str], min_count: int = 2) -> dict[str, int]:
    """Find tags that frequently co-occur with the given tags.

    Args:
        tags: Tags to find co-occurrences for.
        min_count: Minimum co-occurrence count to include.

    Returns:
        Dict mapping co-occurring tag to total co-occurrence count.
    """
    if not tags:
        return {}
    with get_connection() as conn:
        placeholders_a = ",".join("?" for _ in tags)
        placeholders_b = ",".join("?" for _ in tags)
        query = f"""SELECT tag_b as tag, SUM(count) as total
            FROM tag_cooccurrence
            WHERE tag_a IN ({placeholders_a}) AND count >= ?
            GROUP BY tag_b
            UNION ALL
            SELECT tag_a as tag, SUM(count) as total
            FROM tag_cooccurrence
            WHERE tag_b IN ({placeholders_b}) AND count >= ?
            GROUP BY tag_a"""
        rows = conn.execute(
            query,
            [*tags, min_count, *tags, min_count],
        ).fetchall()

    # Aggregate results and exclude the input tags themselves
    tag_set = set(tags)
    result: dict[str, int] = {}
    for row in rows:
        tag = row["tag"]
        if tag not in tag_set:
            result[tag] = result.get(tag, 0) + row["total"]
    return result


def get_tag_pairs_in_set(tags: list[str], min_count: int = 2) -> list[tuple[str, str, int]]:
    """Find co-occurrence pairs where both tags are within the given set.

    Used to discover intra-candidate tag clusters during recall.

    Returns:
        List of (tag_a, tag_b, count) tuples.
    """
    if len(tags) < 2:
        return []
    with get_connection() as conn:
        placeholders_a = ",".join("?" for _ in tags)
        placeholders_b = ",".join("?" for _ in tags)
        query = f"""SELECT tag_a, tag_b, count FROM tag_cooccurrence
            WHERE tag_a IN ({placeholders_a})
              AND tag_b IN ({placeholders_b})
              AND count >= ?"""
        rows = conn.execute(
            query,
            [*tags, *tags, min_count],
        ).fetchall()
    return [(row["tag_a"], row["tag_b"], row["count"]) for row in rows]


def get_all_active_records(
    include_expired: bool = False,
    scope: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return all non-deleted knowledge records.

    Args:
        include_expired: If False (default), exclude records where
            the validity window does not include the current time.
    """
    now = _now()
    base_conditions: list[str] = ["kr.deleted = 0"]
    base_params: list[Any] = []
    _apply_scope_filters(base_conditions, base_params, scope, table_alias="kr")
    with get_connection() as conn:
        if include_expired:
            where_clause = " AND ".join(base_conditions)
            query = f"""SELECT kr.*, kt.filename as topic_filename, kt.title as topic_title
               FROM knowledge_records kr
               JOIN knowledge_topics kt ON kr.topic_id = kt.id
               WHERE {where_clause}
               ORDER BY kr.updated_at DESC"""
            rows = conn.execute(
                query,
                base_params,
            ).fetchall()
        else:
            timed_conditions = [
                *base_conditions,
                "(kr.valid_from IS NULL OR julianday(kr.valid_from) <= julianday(?))",
                "(kr.valid_until IS NULL OR julianday(kr.valid_until) > julianday(?))",
            ]
            timed_params = [*base_params, now, now]
            where_clause = " AND ".join(timed_conditions)
            query = f"""SELECT kr.*, kt.filename as topic_filename, kt.title as topic_title
               FROM knowledge_records kr
               JOIN knowledge_topics kt ON kr.topic_id = kt.id
               WHERE {where_clause}
               ORDER BY kr.updated_at DESC"""
            rows = conn.execute(
                query,
                timed_params,
            ).fetchall()
    return [dict(r) for r in rows]


def get_records_as_of(
    as_of: str,
    scope: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return knowledge records as they existed at a specific point in time.

    Returns non-deleted records that were created on or before ``as_of`` and
    had not yet been superseded (expired) at that time.  This enables
    "what did I believe about X at time T?" queries.

    A record is considered valid at time T when:
    - ``created_at <= T`` (the record existed)
    - ``valid_from IS NULL OR valid_from <= T`` (it had become visible)
    - ``valid_until IS NULL OR valid_until > T`` (not yet superseded)

    Args:
        as_of: ISO 8601 datetime string representing the point in time.
    """
    as_of_utc = _normalize_utc_timestamp(as_of)
    conditions: list[str] = [
        "kr.deleted = 0",
        "julianday(kr.created_at) <= julianday(?)",
        "(kr.valid_from IS NULL OR julianday(kr.valid_from) <= julianday(?))",
        "(kr.valid_until IS NULL OR julianday(kr.valid_until) > julianday(?))",
    ]
    params: list[Any] = [as_of_utc, as_of_utc, as_of_utc]
    _apply_scope_filters(conditions, params, scope, table_alias="kr")
    where_clause = " AND ".join(conditions)
    query = f"""SELECT kr.*, kt.filename as topic_filename, kt.title as topic_title
       FROM knowledge_records kr
       JOIN knowledge_topics kt ON kr.topic_id = kt.id
       WHERE {where_clause}
       ORDER BY kr.updated_at DESC"""
    with get_connection() as conn:
        rows = conn.execute(
            query,
            params,
        ).fetchall()
    return [dict(r) for r in rows]


def get_records_by_topic(topic_id: str, include_expired: bool = False) -> list[dict[str, Any]]:
    """Return all active records for a specific topic.

    Args:
        include_expired: If False (default), exclude records outside the current
            validity window.
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
                   AND (valid_from IS NULL OR julianday(valid_from) <= julianday(?))
                   AND (valid_until IS NULL OR julianday(valid_until) > julianday(?))""",
                (topic_id, now, now),
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
        query = f"""UPDATE knowledge_records SET access_count = access_count + 1,
            updated_at = ? WHERE id IN ({placeholders})"""
        conn.execute(
            query,
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
                   WHERE deleted = 0
                     AND (valid_from IS NULL OR julianday(valid_from) <= julianday(?))
                     AND (valid_until IS NULL OR julianday(valid_until) > julianday(?))""",
                (now, now),
            ).fetchone()
    return row["cnt"] if row else 0


# -- Claim Graph CRUD ---------------------------------------------------------
