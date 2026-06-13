"""Scope coercion, filters, and policy/ACL CRUD."""

from __future__ import annotations

import sqlite3
import uuid
from typing import Any, Mapping, Sequence

from consolidation_memory.config import get_config as _get_config
from consolidation_memory.db._helpers import _now
from consolidation_memory.db.connection import get_connection

_DEFAULT_NAMESPACE_SLUG = "default"
_DEFAULT_NAMESPACE_SHARING_MODE = "private"
_DEFAULT_APP_CLIENT_NAME = "legacy_client"
_DEFAULT_APP_CLIENT_TYPE = "python_sdk"

_POLICY_SELECTOR_KEYS: tuple[str, ...] = (
    "namespace_slug",
    "project_slug",
    "app_client_name",
    "app_client_type",
    "app_client_provider",
    "app_client_external_key",
    "agent_name",
    "agent_external_key",
    "session_external_key",
    "session_kind",
)

_EXACT_SCOPE_MATCH_KEYS: tuple[str, ...] = (
    "namespace_slug",
    "namespace_sharing_mode",
    "project_slug",
    "app_client_name",
    "app_client_type",
    "app_client_provider",
    "app_client_external_key",
    "agent_name",
    "agent_external_key",
    "session_external_key",
    "session_kind",
)


def _normalize_scope_token(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _default_project_slug() -> str:
    project = _normalize_scope_token(getattr(_get_config(), "active_project", None))
    return project or "default"


def _coerce_scope_row(scope: Mapping[str, Any] | None = None) -> dict[str, str | None]:
    raw = scope or {}
    project_slug = _normalize_scope_token(raw.get("project_slug")) or _default_project_slug()
    project_display_name = _normalize_scope_token(raw.get("project_display_name")) or project_slug
    return {
        "namespace_slug": _normalize_scope_token(raw.get("namespace_slug")) or _DEFAULT_NAMESPACE_SLUG,
        "namespace_sharing_mode": (
            _normalize_scope_token(raw.get("namespace_sharing_mode"))
            or _DEFAULT_NAMESPACE_SHARING_MODE
        ),
        "app_client_name": _normalize_scope_token(raw.get("app_client_name")) or _DEFAULT_APP_CLIENT_NAME,
        "app_client_type": _normalize_scope_token(raw.get("app_client_type")) or _DEFAULT_APP_CLIENT_TYPE,
        "app_client_provider": _normalize_scope_token(raw.get("app_client_provider")),
        "app_client_external_key": _normalize_scope_token(raw.get("app_client_external_key")),
        "agent_name": _normalize_scope_token(raw.get("agent_name")),
        "agent_external_key": _normalize_scope_token(raw.get("agent_external_key")),
        "session_external_key": _normalize_scope_token(raw.get("session_external_key")),
        "session_kind": _normalize_scope_token(raw.get("session_kind")),
        "project_slug": project_slug,
        "project_display_name": project_display_name,
        "project_root_uri": _normalize_scope_token(raw.get("project_root_uri")),
        "project_repo_remote": _normalize_scope_token(raw.get("project_repo_remote")),
        "project_default_branch": _normalize_scope_token(raw.get("project_default_branch")),
    }


def _apply_exact_scope_filters(
    conditions: list[str],
    params: list[Any],
    scope: Mapping[str, Any] | None = None,
    *,
    table_alias: str = "",
) -> None:
    """Apply exact-scope matching, including explicit NULL checks."""
    if scope is None:
        return

    prefix = f"{table_alias}." if table_alias else ""
    for key in _EXACT_SCOPE_MATCH_KEYS:
        value = _normalize_scope_token(scope.get(key))
        if value is None:
            conditions.append(f"{prefix}{key} IS NULL")
        else:
            conditions.append(f"{prefix}{key} = ?")
            params.append(value)


def _apply_scope_filters(
    conditions: list[str],
    params: list[Any],
    scope: Mapping[str, Any] | None = None,
    *,
    table_alias: str = "",
) -> None:
    if not scope:
        return

    prefix = f"{table_alias}." if table_alias else ""
    filter_keys = (
        "namespace_slug",
        "project_slug",
        "app_client_name",
        "app_client_type",
        "app_client_provider",
        "app_client_external_key",
        "agent_name",
        "agent_external_key",
        "session_external_key",
        "session_kind",
    )
    for key in filter_keys:
        value = _normalize_scope_token(scope.get(key))
        if value is None:
            continue
        conditions.append(f"{prefix}{key} = ?")
        params.append(value)


def _normalize_principal_token(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("principal value must be a string")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError("principal value must be non-empty")
    return cleaned


def upsert_policy_principal(
    principal_type: str,
    principal_key: str,
    *,
    principal_id: str | None = None,
    conn: sqlite3.Connection | None = None,
) -> str:
    """Create or reuse a persisted policy principal and return its ID."""
    normalized_type = _normalize_principal_token(principal_type)
    normalized_key = _normalize_principal_token(principal_key)

    def _upsert(active_conn: sqlite3.Connection) -> str:
        existing = active_conn.execute(
            """SELECT id
               FROM policy_principals
               WHERE principal_type = ? AND principal_key = ?""",
            (normalized_type, normalized_key),
        ).fetchone()
        if existing is not None:
            return str(existing["id"])

        resolved_id = principal_id or str(uuid.uuid4())
        active_conn.execute(
            """INSERT INTO policy_principals
               (id, principal_type, principal_key, created_at)
               VALUES (?, ?, ?, ?)""",
            (resolved_id, normalized_type, normalized_key, _now()),
        )
        return resolved_id

    if conn is not None:
        return _upsert(conn)
    with get_connection() as managed_conn:
        return _upsert(managed_conn)


def upsert_access_policy(
    *,
    namespace_slug: str | None = None,
    project_slug: str | None = None,
    app_client_name: str | None = None,
    app_client_type: str | None = None,
    app_client_provider: str | None = None,
    app_client_external_key: str | None = None,
    agent_name: str | None = None,
    agent_external_key: str | None = None,
    session_external_key: str | None = None,
    session_kind: str | None = None,
    enabled: bool = True,
    policy_id: str | None = None,
    conn: sqlite3.Connection | None = None,
) -> str:
    """Insert/update an access policy row and return its ID."""
    now = _now()
    resolved_id = policy_id or str(uuid.uuid4())
    values = {
        "namespace_slug": _normalize_scope_token(namespace_slug),
        "project_slug": _normalize_scope_token(project_slug),
        "app_client_name": _normalize_scope_token(app_client_name),
        "app_client_type": _normalize_scope_token(app_client_type),
        "app_client_provider": _normalize_scope_token(app_client_provider),
        "app_client_external_key": _normalize_scope_token(app_client_external_key),
        "agent_name": _normalize_scope_token(agent_name),
        "agent_external_key": _normalize_scope_token(agent_external_key),
        "session_external_key": _normalize_scope_token(session_external_key),
        "session_kind": _normalize_scope_token(session_kind),
        "enabled": 1 if enabled else 0,
    }

    def _upsert(active_conn: sqlite3.Connection) -> str:
        active_conn.execute(
            """INSERT INTO access_policies
               (id, namespace_slug, project_slug, app_client_name, app_client_type,
                app_client_provider, app_client_external_key, agent_name, agent_external_key,
                session_external_key, session_kind, enabled, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                   namespace_slug = excluded.namespace_slug,
                   project_slug = excluded.project_slug,
                   app_client_name = excluded.app_client_name,
                   app_client_type = excluded.app_client_type,
                   app_client_provider = excluded.app_client_provider,
                   app_client_external_key = excluded.app_client_external_key,
                   agent_name = excluded.agent_name,
                   agent_external_key = excluded.agent_external_key,
                   session_external_key = excluded.session_external_key,
                   session_kind = excluded.session_kind,
                   enabled = excluded.enabled,
                   updated_at = excluded.updated_at""",
            (
                resolved_id,
                values["namespace_slug"],
                values["project_slug"],
                values["app_client_name"],
                values["app_client_type"],
                values["app_client_provider"],
                values["app_client_external_key"],
                values["agent_name"],
                values["agent_external_key"],
                values["session_external_key"],
                values["session_kind"],
                values["enabled"],
                now,
                now,
            ),
        )
        return resolved_id

    if conn is not None:
        return _upsert(conn)
    with get_connection() as managed_conn:
        return _upsert(managed_conn)


def upsert_policy_acl_entry(
    *,
    policy_id: str,
    principal_id: str,
    write_mode: str | None = None,
    read_visibility: str | None = None,
    acl_entry_id: str | None = None,
    conn: sqlite3.Connection | None = None,
) -> str:
    """Insert/update a policy ACL entry and return its ID."""
    normalized_write_mode = _normalize_scope_token(write_mode)
    normalized_read_visibility = _normalize_scope_token(read_visibility)
    if normalized_write_mode not in {None, "allow", "deny"}:
        raise ValueError("write_mode must be one of: allow, deny")
    if normalized_read_visibility not in {None, "private", "project", "namespace"}:
        raise ValueError("read_visibility must be one of: private, project, namespace")
    if normalized_write_mode is None and normalized_read_visibility is None:
        raise ValueError("ACL entry requires write_mode and/or read_visibility")

    now = _now()
    resolved_id = acl_entry_id or str(uuid.uuid4())

    def _upsert(active_conn: sqlite3.Connection) -> str:
        existing = active_conn.execute(
            """SELECT id
               FROM policy_acl_entries
               WHERE policy_id = ?
                 AND principal_id = ?
                 AND ((write_mode IS NULL AND ? IS NULL) OR write_mode = ?)
                 AND ((read_visibility IS NULL AND ? IS NULL) OR read_visibility = ?)""",
            (
                policy_id,
                principal_id,
                normalized_write_mode,
                normalized_write_mode,
                normalized_read_visibility,
                normalized_read_visibility,
            ),
        ).fetchone()
        if existing is not None:
            active_conn.execute(
                "UPDATE policy_acl_entries SET updated_at = ? WHERE id = ?",
                (now, str(existing["id"])),
            )
            return str(existing["id"])

        active_conn.execute(
            """INSERT INTO policy_acl_entries
               (id, policy_id, principal_id, write_mode, read_visibility, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                resolved_id,
                policy_id,
                principal_id,
                normalized_write_mode,
                normalized_read_visibility,
                now,
                now,
            ),
        )
        return resolved_id

    if conn is not None:
        return _upsert(conn)
    with get_connection() as managed_conn:
        return _upsert(managed_conn)


def get_matching_policy_acl_entries(
    scope: Mapping[str, Any] | None,
    principals: Sequence[tuple[str, str]],
) -> list[dict[str, Any]]:
    """Return ACL entries matching a scope envelope and principal tokens."""
    if not principals:
        return []

    scope_row = _coerce_scope_row(scope)
    scope_conditions = [
        f"(p.{key} IS NULL OR p.{key} = ?)"
        for key in _POLICY_SELECTOR_KEYS
    ]
    scope_params: list[Any] = [scope_row.get(key) for key in _POLICY_SELECTOR_KEYS]

    principal_conditions = []
    principal_params: list[Any] = []
    for principal_type, principal_key in principals:
        principal_conditions.append("(pp.principal_type = ? AND pp.principal_key = ?)")
        principal_params.append(_normalize_principal_token(principal_type))
        principal_params.append(_normalize_principal_token(principal_key))

    where = " AND ".join([
        "p.enabled = 1",
        *scope_conditions,
        f"({' OR '.join(principal_conditions)})",
    ])

    query = f"""SELECT
            pae.id AS acl_entry_id,
            pae.policy_id,
            pae.write_mode,
            pae.read_visibility,
            pp.id AS principal_id,
            pp.principal_type,
            pp.principal_key
        FROM policy_acl_entries pae
        JOIN access_policies p ON p.id = pae.policy_id
        JOIN policy_principals pp ON pp.id = pae.principal_id
        WHERE {where}
        ORDER BY p.updated_at DESC, pae.updated_at DESC, pae.id ASC"""

    with get_connection() as conn:
        rows = conn.execute(
            query,
            [*scope_params, *principal_params],
        ).fetchall()
    return [dict(row) for row in rows]

