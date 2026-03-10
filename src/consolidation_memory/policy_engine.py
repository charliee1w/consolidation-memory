"""Policy/ACL evaluation helpers for canonical scope enforcement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence, cast

from consolidation_memory.types import (
    PolicyReadVisibility,
    PolicyResolutionSource,
    PolicyScope,
    PolicyWriteMode,
    ResolvedScopeEnvelope,
)

_WRITE_MODE_VALUES: frozenset[PolicyWriteMode] = frozenset({"allow", "deny"})
_READ_VISIBILITY_VALUES: frozenset[PolicyReadVisibility] = frozenset(
    {"private", "project", "namespace"}
)
_READ_VISIBILITY_RANK: dict[PolicyReadVisibility, int] = {
    "private": 0,
    "project": 1,
    "namespace": 2,
}


@dataclass(frozen=True)
class EffectivePolicyResolution:
    """Effective policy computed from scope policy + persisted ACL entries."""

    policy: PolicyScope
    source: PolicyResolutionSource
    matched_entries: int
    conflicts: tuple[str, ...] = ()


def principal_tokens_for_scope(scope: ResolvedScopeEnvelope) -> list[tuple[str, str]]:
    """Build principal tokens used for ACL matching."""
    principals: list[tuple[str, str]] = [("any", "*")]

    principals.append(("namespace_slug", scope.namespace.slug))
    project_slug = scope.project.slug or "default"
    principals.append(("project_slug", project_slug))
    principals.append(("app_client", f"{scope.app_client.app_type}:{scope.app_client.name}"))

    app_client_external_key = scope.app_client.external_key
    if app_client_external_key:
        principals.append(("app_client_external_key", app_client_external_key))
    app_client_provider = scope.app_client.provider
    if app_client_provider:
        principals.append(("app_client_provider", app_client_provider))

    if scope.agent is not None:
        agent_external_key = scope.agent.external_key
        if agent_external_key:
            principals.append(("agent_external_key", agent_external_key))
        else:
            agent_name = scope.agent.name
            if agent_name:
                principals.append(("agent_name", agent_name))

    if scope.session is not None:
        session_external_key = scope.session.external_key
        if session_external_key:
            principals.append(("session_external_key", session_external_key))
        principals.append(("session_kind", scope.session.session_kind))

    return principals


def resolve_effective_policy(
    base_policy: PolicyScope,
    acl_rows: Sequence[Mapping[str, object]],
) -> EffectivePolicyResolution:
    """Resolve an effective policy with persisted ACL rows authoritative when present."""
    if not acl_rows:
        return EffectivePolicyResolution(
            policy=base_policy,
            source="scope_policy",
            matched_entries=0,
        )

    write_modes: list[PolicyWriteMode] = []
    read_visibilities: list[PolicyReadVisibility] = []

    for row in acl_rows:
        raw_write_mode = row.get("write_mode")
        if isinstance(raw_write_mode, str) and raw_write_mode in _WRITE_MODE_VALUES:
            write_modes.append(cast(PolicyWriteMode, raw_write_mode))

        raw_read_visibility = row.get("read_visibility")
        if (
            isinstance(raw_read_visibility, str)
            and raw_read_visibility in _READ_VISIBILITY_VALUES
        ):
            read_visibilities.append(cast(PolicyReadVisibility, raw_read_visibility))

    conflicts: list[str] = []
    effective_write_mode = base_policy.write_mode
    if write_modes:
        if "deny" in write_modes and "allow" in write_modes:
            conflicts.append("write_mode_conflict_deny_overrides_allow")
        # Conflict rule: deny overrides allow.
        effective_write_mode = "deny" if "deny" in write_modes else "allow"

    effective_read_visibility = base_policy.read_visibility
    if read_visibilities:
        if len(set(read_visibilities)) > 1:
            conflicts.append("read_visibility_conflict_most_restrictive_wins")
        # Conflict rule mirrors deny-overrides: most restrictive visibility wins.
        effective_read_visibility = min(
            read_visibilities,
            key=lambda value: _READ_VISIBILITY_RANK[value],
        )

    return EffectivePolicyResolution(
        policy=PolicyScope(
            read_visibility=effective_read_visibility,
            write_mode=effective_write_mode,
        ),
        source="persisted_acl",
        matched_entries=len(acl_rows),
        conflicts=tuple(conflicts),
    )


__all__ = [
    "EffectivePolicyResolution",
    "principal_tokens_for_scope",
    "resolve_effective_policy",
]
