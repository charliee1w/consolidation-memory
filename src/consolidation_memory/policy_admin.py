"""Canonical policy administration operations for all adapter surfaces."""

from __future__ import annotations

from typing import Any

from consolidation_memory.database import (
    ensure_schema,
    list_policy_admin_rows,
    upsert_access_policy,
    upsert_policy_acl_entry,
    upsert_policy_principal,
)

_WRITE_MODES = frozenset({"allow", "deny"})
_READ_VISIBILITIES = frozenset({"private", "project", "namespace"})


def _format_policy_row(row: dict[str, Any]) -> dict[str, Any]:
    principal: dict[str, str] | None = None
    if row.get("principal_type") and row.get("principal_key"):
        principal = {
            "type": str(row["principal_type"]),
            "key": str(row["principal_key"]),
        }
    return {
        "policy_id": row.get("policy_id"),
        "acl_entry_id": row.get("acl_entry_id"),
        "namespace_slug": row.get("namespace_slug"),
        "project_slug": row.get("project_slug"),
        "principal": principal,
        "write_mode": row.get("write_mode"),
        "read_visibility": row.get("read_visibility"),
        "enabled": row.get("enabled"),
        "policy_updated_at": row.get("policy_updated_at"),
        "acl_updated_at": row.get("acl_updated_at"),
    }


def list_policy_bindings() -> dict[str, Any]:
    """Return persisted access policies and ACL bindings."""
    ensure_schema()
    rows = list_policy_admin_rows()
    policies = [_format_policy_row(row) for row in rows]
    return {
        "status": "ok",
        "count": len(policies),
        "policies": policies,
    }


def grant_policy_binding(
    *,
    namespace: str | None = None,
    project: str | None = None,
    principal_type: str,
    principal_key: str,
    write_mode: str | None = None,
    read_visibility: str | None = None,
) -> dict[str, Any]:
    """Create or update a persisted policy ACL binding."""
    if write_mode is None and read_visibility is None:
        raise ValueError("Provide at least one of write_mode or read_visibility")
    if write_mode is not None and write_mode not in _WRITE_MODES:
        raise ValueError("write_mode must be one of: allow, deny")
    if read_visibility is not None and read_visibility not in _READ_VISIBILITIES:
        raise ValueError("read_visibility must be one of: private, project, namespace")

    ensure_schema()
    principal_id = upsert_policy_principal(principal_type, principal_key)
    policy_id = upsert_access_policy(namespace_slug=namespace, project_slug=project)
    acl_id = upsert_policy_acl_entry(
        policy_id=policy_id,
        principal_id=principal_id,
        write_mode=write_mode,
        read_visibility=read_visibility,
    )
    return {
        "status": "granted",
        "policy_id": policy_id,
        "principal_id": principal_id,
        "acl_entry_id": acl_id,
        "namespace": namespace,
        "project": project,
        "principal_type": principal_type,
        "principal_key": principal_key,
        "write_mode": write_mode,
        "read_visibility": read_visibility,
    }