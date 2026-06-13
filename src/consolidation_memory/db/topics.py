"""Knowledge topic CRUD and storage filename helpers."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import uuid
from pathlib import PurePosixPath
from typing import Any, Mapping

from consolidation_memory.db._helpers import _now
from consolidation_memory.db.connection import get_connection
from consolidation_memory.db.scope import (
    _EXACT_SCOPE_MATCH_KEYS,
    _apply_exact_scope_filters,
    _apply_scope_filters,
    _coerce_scope_row,
)

_TOPIC_STORAGE_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")

def _topic_storage_filename(
    logical_filename: str,
    scope: Mapping[str, Any] | None = None,
) -> str:
    """Build a deterministic storage filename for a topic within one exact scope."""
    scope_row = _coerce_scope_row(scope)
    pure_name = PurePosixPath(str(logical_filename or "")).name
    suffix = "".join(PurePosixPath(pure_name).suffixes)
    stem = pure_name[: -len(suffix)] if suffix else pure_name
    cleaned_stem = _TOPIC_STORAGE_SAFE_RE.sub("_", stem).strip("._") or "topic"
    normalized_suffix = suffix or ".md"

    identity_tokens = [
        f"{key}={scope_row.get(key) or ''}"
        for key in _EXACT_SCOPE_MATCH_KEYS
    ]
    identity_payload = "|".join([*identity_tokens, f"filename={logical_filename}"])
    digest = hashlib.sha256(identity_payload.encode("utf-8")).hexdigest()[:12]
    return f"{cleaned_stem}__{digest}{normalized_suffix}"


def topic_storage_filename(topic: Mapping[str, Any]) -> str:
    """Return the unique on-disk filename for a knowledge topic row."""
    storage = topic.get("storage_filename")
    if isinstance(storage, str) and storage.strip():
        return storage.strip()
    filename = topic.get("filename")
    if isinstance(filename, str):
        return filename
    raise ValueError("Knowledge topic row is missing filename metadata")

def upsert_knowledge_topic(
    filename: str,
    title: str,
    summary: str,
    source_episodes: list[str],
    fact_count: int = 0,
    confidence: float = 0.8,
    scope: Mapping[str, Any] | None = None,
    topic_id: str | None = None,
    *,
    created_at: str | None = None,
    updated_at: str | None = None,
    access_count: int | None = None,
) -> str:
    now = _now()
    updated_ts = updated_at or now
    scope_row = _coerce_scope_row(scope)
    storage_filename = _topic_storage_filename(filename, scope_row)
    resolved_topic_id: str | None = topic_id
    with get_connection() as conn:
        existing = None
        if resolved_topic_id is not None:
            existing = conn.execute(
                "SELECT id, source_episodes, access_count, storage_filename "
                "FROM knowledge_topics WHERE id = ?",
                (resolved_topic_id,),
            ).fetchone()
        if existing is None:
            conditions = ["(filename = ? OR storage_filename = ?)"]
            params: list[Any] = [filename, filename]
            _apply_exact_scope_filters(conditions, params, scope_row)
            existing = conn.execute(
                f"""SELECT id, source_episodes, access_count, storage_filename
                    FROM knowledge_topics
                    WHERE {' AND '.join(conditions)}
                    ORDER BY updated_at DESC, id ASC
                    LIMIT 1""",
                params,
            ).fetchone()

        if existing:
            resolved_topic_id = str(existing["id"])
            old_sources = json.loads(existing["source_episodes"])
            merged = list(set(old_sources + source_episodes))
            persisted_storage_filename = str(existing["storage_filename"] or storage_filename)
            conn.execute(
                """UPDATE knowledge_topics
                   SET title = ?, summary = ?, updated_at = ?,
                       source_episodes = ?, fact_count = ?, access_count = ?, confidence = ?,
                       storage_filename = ?,
                       namespace_slug = ?, namespace_sharing_mode = ?,
                       app_client_name = ?, app_client_type = ?, app_client_provider = ?, app_client_external_key = ?,
                       agent_name = ?, agent_external_key = ?,
                       session_external_key = ?, session_kind = ?,
                       project_slug = ?, project_display_name = ?, project_root_uri = ?,
                       project_repo_remote = ?, project_default_branch = ?
                   WHERE id = ?""",
                (
                    title,
                    summary,
                    updated_ts,
                    json.dumps(merged),
                    fact_count,
                    int(existing["access_count"]) if access_count is None else int(access_count),
                    confidence,
                    persisted_storage_filename,
                 scope_row["namespace_slug"], scope_row["namespace_sharing_mode"],
                 scope_row["app_client_name"], scope_row["app_client_type"],
                 scope_row["app_client_provider"], scope_row["app_client_external_key"],
                 scope_row["agent_name"], scope_row["agent_external_key"],
                 scope_row["session_external_key"], scope_row["session_kind"],
                 scope_row["project_slug"], scope_row["project_display_name"],
                 scope_row["project_root_uri"], scope_row["project_repo_remote"],
                 scope_row["project_default_branch"], resolved_topic_id,
                ),
            )
        else:
            resolved_topic_id = resolved_topic_id or str(uuid.uuid4())
            created_ts = created_at or updated_ts
            try:
                conn.execute(
                    """INSERT INTO knowledge_topics
                       (id, filename, storage_filename, title, summary, created_at, updated_at,
                        source_episodes, fact_count, access_count, confidence,
                        namespace_slug, namespace_sharing_mode,
                        app_client_name, app_client_type, app_client_provider, app_client_external_key,
                        agent_name, agent_external_key,
                        session_external_key, session_kind,
                        project_slug, project_display_name, project_root_uri,
                        project_repo_remote, project_default_branch)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        resolved_topic_id,
                        filename,
                        storage_filename,
                        title,
                        summary,
                        created_ts,
                        updated_ts,
                        json.dumps(source_episodes),
                        fact_count,
                        0 if access_count is None else int(access_count),
                        confidence,
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
            except sqlite3.IntegrityError:
                # Race: concurrent insert won — fall back to update
                conditions = ["(filename = ? OR storage_filename = ?)"]
                params = [filename, filename]
                _apply_exact_scope_filters(conditions, params, scope_row)
                existing = conn.execute(
                    f"""SELECT id, source_episodes, access_count, storage_filename
                        FROM knowledge_topics
                        WHERE {' AND '.join(conditions)}
                        ORDER BY updated_at DESC, id ASC
                        LIMIT 1""",
                    params,
                ).fetchone()
                if existing is None:
                    raise
                resolved_topic_id = str(existing["id"])
                old_sources = json.loads(existing["source_episodes"])
                merged = list(set(old_sources + source_episodes))
                persisted_storage_filename = str(existing["storage_filename"] or storage_filename)
                conn.execute(
                    """UPDATE knowledge_topics
                       SET title = ?, summary = ?, updated_at = ?,
                           source_episodes = ?, fact_count = ?, access_count = ?, confidence = ?,
                           storage_filename = ?,
                           namespace_slug = ?, namespace_sharing_mode = ?,
                           app_client_name = ?, app_client_type = ?, app_client_provider = ?, app_client_external_key = ?,
                           agent_name = ?, agent_external_key = ?,
                           session_external_key = ?, session_kind = ?,
                           project_slug = ?, project_display_name = ?, project_root_uri = ?,
                           project_repo_remote = ?, project_default_branch = ?
                       WHERE id = ?""",
                    (
                        title,
                        summary,
                        updated_ts,
                        json.dumps(merged),
                        fact_count,
                        int(existing["access_count"]) if access_count is None else int(access_count),
                        confidence,
                        persisted_storage_filename,
                     scope_row["namespace_slug"], scope_row["namespace_sharing_mode"],
                     scope_row["app_client_name"], scope_row["app_client_type"],
                     scope_row["app_client_provider"], scope_row["app_client_external_key"],
                     scope_row["agent_name"], scope_row["agent_external_key"],
                     scope_row["session_external_key"], scope_row["session_kind"],
                     scope_row["project_slug"], scope_row["project_display_name"],
                     scope_row["project_root_uri"], scope_row["project_repo_remote"],
                     scope_row["project_default_branch"], resolved_topic_id,
                    ),
                )
    if resolved_topic_id is None:
        raise RuntimeError("Failed to resolve knowledge topic id during upsert")
    return resolved_topic_id


def get_all_knowledge_topics(
    scope: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    conditions: list[str] = []
    params: list[Any] = []
    _apply_scope_filters(conditions, params, scope)
    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    with get_connection() as conn:
        rows = conn.execute(
            f"SELECT * FROM knowledge_topics {where} ORDER BY updated_at DESC",
            params,
        ).fetchall()
    return [dict(r) for r in rows]


def get_knowledge_topic(
    filename: str,
    scope: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    conditions = ["(filename = ? OR storage_filename = ?)"]
    params: list[Any] = [filename, filename]
    _apply_scope_filters(conditions, params, scope)
    where_clause = " AND ".join(conditions)
    with get_connection() as conn:
        row = conn.execute(
            f"SELECT * FROM knowledge_topics WHERE {where_clause}",
            params,
        ).fetchone()
    return dict(row) if row else None


def get_knowledge_topics_by_name(
    filename: str,
    scope: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    conditions = ["(filename = ? OR storage_filename = ?)"]
    params: list[Any] = [filename, filename]
    _apply_scope_filters(conditions, params, scope)
    where_clause = " AND ".join(conditions)
    with get_connection() as conn:
        rows = conn.execute(
            f"""SELECT * FROM knowledge_topics
                WHERE {where_clause}
                ORDER BY updated_at DESC, id ASC""",
            params,
        ).fetchall()
    return [dict(row) for row in rows]


def increment_topic_access(
    filenames: list[str],
    *,
    scope: Mapping[str, Any] | None = None,
) -> None:
    """Increment topic access counters using legacy filename-based lookup.

    This compatibility surface is still used by call sites that only have the
    topic filename. Prefer ``increment_topic_access_by_ids`` for unambiguous
    updates when topic IDs are available.
    """
    deduped_filenames: list[str] = []
    seen: set[str] = set()
    for filename in filenames:
        token = str(filename or "").strip()
        if not token or token in seen:
            continue
        seen.add(token)
        deduped_filenames.append(token)

    if not deduped_filenames:
        return

    now = _now()
    with get_connection() as conn:
        # Keep chunks well under SQLite's default parameter limit.
        for start in range(0, len(deduped_filenames), 250):
            chunk = deduped_filenames[start:start + 250]
            placeholders = ",".join("?" for _ in chunk)
            conditions: list[str] = [
                f"(filename IN ({placeholders}) OR storage_filename IN ({placeholders}))",
            ]
            params: list[Any] = [now, *chunk, *chunk]
            _apply_scope_filters(conditions, params, scope)
            where_clause = " AND ".join(conditions)
            conn.execute(
                f"""UPDATE knowledge_topics
                    SET access_count = access_count + 1,
                        updated_at = ?
                    WHERE {where_clause}""",
                params,
            )


def increment_topic_access_by_ids(topic_ids: list[str]) -> None:
    if not topic_ids:
        return
    with get_connection() as conn:
        placeholders = ",".join("?" for _ in topic_ids)
        query = f"""UPDATE knowledge_topics SET access_count = access_count + 1,
            updated_at = ? WHERE id IN ({placeholders})"""
        conn.execute(
            query,
            [_now()] + topic_ids,
        )


