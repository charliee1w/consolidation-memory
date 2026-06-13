"""Consolidation scheduler, runs, metrics, and attempt tracking."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, cast

import sqlite3

from consolidation_memory.config import get_config as _get_config
from consolidation_memory.db._helpers import _now
from consolidation_memory.db.connection import get_connection
from consolidation_memory.db.scope import _apply_scope_filters
from consolidation_memory.types import (
    RUN_STATUS_COMPLETED,
    RUN_STATUS_FAILED,
    RUN_STATUS_RUNNING,
    RunStatus,
)
from consolidation_memory.utils import parse_datetime, parse_json_list

logger = logging.getLogger(__name__)

_SCHEDULER_ROW_ID = "global"

def _serialize_trigger_breakdown(breakdown: Mapping[str, object] | None) -> str | None:
    if breakdown is None:
        return None
    return json.dumps(dict(breakdown), separators=(",", ":"), sort_keys=True)


def _deserialize_trigger_breakdown(raw: object) -> dict[str, object] | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None

def _default_stale_consolidation_timeout_seconds(max_duration_seconds: float) -> float:
    """Return stale-run timeout with a grace window above configured max duration."""
    safe_max_duration = max(float(max_duration_seconds), 1.0)
    # Give the worker a 5-minute grace period while still recovering quickly
    # after process crashes. Keep a sane lower bound for very small durations.
    return max(600.0, safe_max_duration + 300.0)


def reconcile_stale_consolidation_state(
    *,
    stale_timeout_seconds: float | None = None,
    as_of: str | datetime | None = None,
) -> dict[str, Any]:
    """Recover stale running consolidation state after interrupted execution.

    Marks long-running orphaned consolidation runs as failed and clears stale
    scheduler leases/status so automatic scheduling can continue.
    """
    cfg = _get_config()
    timeout_seconds = (
        _default_stale_consolidation_timeout_seconds(cfg.CONSOLIDATION_MAX_DURATION)
        if stale_timeout_seconds is None
        else max(float(stale_timeout_seconds), 1.0)
    )

    if isinstance(as_of, str):
        as_of_dt = parse_datetime(as_of)
    elif isinstance(as_of, datetime):
        as_of_dt = as_of if as_of.tzinfo is not None else as_of.replace(tzinfo=timezone.utc)
    else:
        as_of_dt = datetime.now(timezone.utc)
    as_of_dt = as_of_dt.astimezone(timezone.utc)

    as_of_iso = as_of_dt.isoformat()
    cutoff_iso = (as_of_dt - timedelta(seconds=timeout_seconds)).isoformat()
    stale_run_message = (
        "Recovered stale running consolidation run after exceeding timeout "
        f"({int(timeout_seconds)}s)."
    )
    stale_scheduler_message = (
        "Recovered stale running scheduler state after exceeding timeout "
        f"({int(timeout_seconds)}s)."
    )

    stale_run_ids: list[str] = []
    scheduler_recovered = False

    with get_connection() as conn:
        _ensure_consolidation_scheduler_row(conn)

        stale_rows = conn.execute(
            """SELECT id
               FROM consolidation_runs
               WHERE status = ?
                 AND completed_at IS NULL
                 AND julianday(started_at) <= julianday(?)""",
            (RUN_STATUS_RUNNING, cutoff_iso),
        ).fetchall()
        stale_run_ids = [str(row["id"]) for row in stale_rows]

        if stale_run_ids:
            conn.executemany(
                """UPDATE consolidation_runs
                    SET status = ?,
                        completed_at = ?,
                        error_message = COALESCE(error_message, ?)
                    WHERE id = ?""",
                [
                    (RUN_STATUS_FAILED, as_of_iso, stale_run_message, stale_run_id)
                    for stale_run_id in stale_run_ids
                ],
            )

        scheduler_cursor = conn.execute(
            """UPDATE consolidation_scheduler
               SET last_run_completed_at = COALESCE(last_run_completed_at, ?),
                   last_status = ?,
                   last_error = COALESCE(last_error, ?),
                   next_due_at = ?,
                   lease_owner = NULL,
                   lease_expires_at = NULL,
                   updated_at = ?
               WHERE id = ?
                 AND last_status = ?
                 AND last_run_started_at IS NOT NULL
                 AND julianday(last_run_started_at) <= julianday(?)
                 AND (
                   lease_owner IS NULL
                   OR lease_expires_at IS NULL
                   OR julianday(lease_expires_at) <= julianday(?)
                 )""",
            (
                as_of_iso,
                RUN_STATUS_FAILED,
                stale_scheduler_message,
                as_of_iso,
                as_of_iso,
                _SCHEDULER_ROW_ID,
                RUN_STATUS_RUNNING,
                cutoff_iso,
                as_of_iso,
            ),
        )
        scheduler_recovered = bool(scheduler_cursor.rowcount and scheduler_cursor.rowcount > 0)

    if stale_run_ids or scheduler_recovered:
        logger.warning(
            "Recovered stale consolidation state (stale_runs=%d, scheduler_recovered=%s, timeout_seconds=%.0f)",
            len(stale_run_ids),
            scheduler_recovered,
            timeout_seconds,
        )

    return {
        "stale_timeout_seconds": timeout_seconds,
        "stale_runs_marked_failed": len(stale_run_ids),
        "stale_run_ids": stale_run_ids,
        "scheduler_state_recovered": scheduler_recovered,
        "cutoff": cutoff_iso,
        "as_of": as_of_iso,
    }


def _ensure_consolidation_scheduler_row(conn: sqlite3.Connection) -> sqlite3.Row:
    """Ensure the singleton scheduler row exists and return it."""
    row = conn.execute(
        "SELECT * FROM consolidation_scheduler WHERE id = ?",
        (_SCHEDULER_ROW_ID,),
    ).fetchone()
    if row is not None:
        return cast(sqlite3.Row, row)

    now = _now()
    conn.execute(
        """INSERT INTO consolidation_scheduler
           (id, last_status, next_due_at, updated_at)
           VALUES (?, 'idle', ?, ?)""",
        (_SCHEDULER_ROW_ID, now, now),
    )
    created = conn.execute(
        "SELECT * FROM consolidation_scheduler WHERE id = ?",
        (_SCHEDULER_ROW_ID,),
    ).fetchone()
    if created is None:
        raise RuntimeError("Failed to initialize consolidation scheduler state")
    return cast(sqlite3.Row, created)


def get_consolidation_scheduler_state() -> dict[str, Any]:
    """Return persisted scheduler state for automatic consolidation."""
    reconcile_stale_consolidation_state()
    with get_connection() as conn:
        row = _ensure_consolidation_scheduler_row(conn)
    return dict(row)


def try_acquire_consolidation_lease(owner: str, lease_seconds: float) -> bool:
    """Try to acquire the scheduler lease for a consolidation worker.

    Returns True when the lease is acquired, False when held by another owner.
    """
    owner_token = owner.strip()
    if not owner_token:
        raise ValueError("owner must be non-empty")

    lease_ttl = max(float(lease_seconds), 1.0)
    now_dt = datetime.now(timezone.utc)
    now = now_dt.isoformat()
    lease_expires_at = (now_dt + timedelta(seconds=lease_ttl)).isoformat()

    with get_connection() as conn:
        _ensure_consolidation_scheduler_row(conn)
        cursor = conn.execute(
            """UPDATE consolidation_scheduler
               SET lease_owner = ?, lease_expires_at = ?, updated_at = ?
               WHERE id = ?
                 AND (
                   lease_owner IS NULL
                   OR lease_expires_at IS NULL
                   OR julianday(lease_expires_at) <= julianday(?)
                   OR lease_owner = ?
                 )""",
            (
                owner_token,
                lease_expires_at,
                now,
                _SCHEDULER_ROW_ID,
                now,
                owner_token,
            ),
        )
    return bool(cursor.rowcount and cursor.rowcount > 0)


def mark_consolidation_scheduler_started(
    owner: str,
    *,
    trigger_reason: str,
    utility_score: float | None = None,
    trigger_breakdown: Mapping[str, object] | None = None,
    started_at: str | None = None,
) -> None:
    """Persist scheduler state when a consolidation run starts."""
    owner_token = owner.strip()
    if not owner_token:
        raise ValueError("owner must be non-empty")

    ts = started_at or _now()
    score_value = float(utility_score) if utility_score is not None else None
    breakdown_json = _serialize_trigger_breakdown(trigger_breakdown)
    with get_connection() as conn:
        _ensure_consolidation_scheduler_row(conn)
        conn.execute(
            """UPDATE consolidation_scheduler
               SET last_run_started_at = ?,
                   last_status = ?,
                   last_error = NULL,
                   last_trigger = ?,
                   last_utility_score = ?,
                   last_trigger_breakdown = ?,
                   updated_at = ?
               WHERE id = ?
                 AND (lease_owner = ? OR lease_owner IS NULL)""",
            (
                ts,
                RUN_STATUS_RUNNING,
                trigger_reason,
                score_value,
                breakdown_json,
                ts,
                _SCHEDULER_ROW_ID,
                owner_token,
            ),
        )


def mark_consolidation_scheduler_finished(
    owner: str,
    *,
    status: RunStatus,
    interval_hours: float,
    error_message: str | None = None,
    completed_at: str | None = None,
) -> None:
    """Persist scheduler state and release the lease after a run completes."""
    owner_token = owner.strip()
    if not owner_token:
        raise ValueError("owner must be non-empty")

    completed_ts = completed_at or _now()
    completed_dt = parse_datetime(completed_ts)
    next_due = (completed_dt + timedelta(hours=max(float(interval_hours), 0.01))).isoformat()

    with get_connection() as conn:
        _ensure_consolidation_scheduler_row(conn)
        conn.execute(
            """UPDATE consolidation_scheduler
               SET last_run_completed_at = ?,
                   last_status = ?,
                   last_error = ?,
                   next_due_at = ?,
                   lease_owner = NULL,
                   lease_expires_at = NULL,
                   updated_at = ?
               WHERE id = ?
                 AND (lease_owner = ? OR lease_owner IS NULL)""",
            (
                completed_ts,
                status,
                error_message,
                next_due,
                completed_ts,
                _SCHEDULER_ROW_ID,
                owner_token,
            ),
        )


def release_consolidation_lease(owner: str) -> None:
    """Release the scheduler lease without mutating run outcome fields."""
    owner_token = owner.strip()
    if not owner_token:
        return
    now = _now()
    with get_connection() as conn:
        _ensure_consolidation_scheduler_row(conn)
        conn.execute(
            """UPDATE consolidation_scheduler
               SET lease_owner = NULL,
                   lease_expires_at = NULL,
                   updated_at = ?
               WHERE id = ?
                 AND lease_owner = ?""",
            (now, _SCHEDULER_ROW_ID, owner_token),
        )

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
    status: RunStatus = RUN_STATUS_COMPLETED,
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
    reconcile_stale_consolidation_state()
    with get_connection() as conn:
        row = conn.execute(
            """SELECT * FROM consolidation_runs
               ORDER BY started_at DESC LIMIT 1"""
        ).fetchone()
    return dict(row) if row else None


def get_recent_consolidation_runs(limit: int = 5) -> list[dict[str, Any]]:
    """Return recent consolidation runs as activity summaries."""
    reconcile_stale_consolidation_state()
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT id, started_at, completed_at, status,
                      episodes_processed, clusters_formed,
                      topics_created, topics_updated, episodes_pruned,
                      error_message
               FROM consolidation_runs
               ORDER BY started_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]

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
    fast_path_hits: int = 0,
    llm_fallbacks: int = 0,
) -> str:
    """Insert a consolidation run metrics record."""
    metric_id = str(uuid.uuid4())
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO consolidation_metrics "
            "(id, run_id, timestamp, clusters_succeeded, clusters_failed, "
            "avg_confidence, episodes_processed, duration_seconds, api_calls, "
            "topics_created, topics_updated, episodes_pruned, fast_path_hits, "
            "llm_fallbacks) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                metric_id,
                run_id,
                _now(),
                clusters_succeeded,
                clusters_failed,
                avg_confidence,
                episodes_processed,
                duration_seconds,
                api_calls,
                topics_created,
                topics_updated,
                episodes_pruned,
                max(0, int(fast_path_hits)),
                max(0, int(llm_fallbacks)),
            ),
        )
    return metric_id


def search_episodes(
    query: str | None = None,
    content_types: list[str] | None = None,
    tags: list[str] | None = None,
    after: str | None = None,
    before: str | None = None,
    scope: Mapping[str, Any] | None = None,
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
    _apply_scope_filters(conditions, params, scope)

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

    if limit <= 0:
        return []

    where = " AND ".join(conditions)
    base_sql = f"SELECT * FROM episodes WHERE {where} ORDER BY created_at DESC"

    if not tags:
        with get_connection() as conn:
            rows = conn.execute(
                f"{base_sql} LIMIT ?",
                [*params, limit],
            ).fetchall()
        return [dict(row) for row in rows]

    requested_tags = set(tags)
    results: list[dict[str, Any]] = []
    offset = 0
    page_size = min(max(limit * 5, 50), 500)
    paged_sql = f"{base_sql} LIMIT ? OFFSET ?"

    with get_connection() as conn:
        while len(results) < limit:
            rows = conn.execute(
                paged_sql,
                [*params, page_size, offset],
            ).fetchall()
            if not rows:
                break

            offset += len(rows)
            for row in rows:
                ep = dict(row)
                ep_tags = parse_json_list(ep["tags"])
                if not requested_tags.intersection(ep_tags):
                    continue
                results.append(ep)
                if len(results) >= limit:
                    break

            if len(rows) < page_size:
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

