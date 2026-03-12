"""Shared trust semantics for query-time filtering and payload parsing."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

_LEGACY_DEFAULT_APP_NAME = "legacy_client"
_LEGACY_DEFAULT_APP_TYPE = "python_sdk"
_LEGACY_DEFAULT_NAMESPACE = "default"
_LEGACY_DEFAULT_PROJECT = "default"


def parse_claim_payload(payload_raw: object) -> dict[str, object]:
    """Parse a claim payload from DB storage into a dict."""
    if isinstance(payload_raw, dict):
        return dict(payload_raw)
    if isinstance(payload_raw, str):
        try:
            parsed = json.loads(payload_raw)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError, ValueError):
            return {}
    return {}


def matches_scope_filter(
    row: Mapping[str, object],
    scope_filter: Mapping[str, str | None] | None,
) -> bool:
    """Return True when a row falls within the provided scope filter."""
    if not scope_filter:
        return True
    for key, expected in scope_filter.items():
        if expected is None:
            continue
        actual = row.get(key)
        if actual is None or str(actual) != expected:
            return False
    return True


def filter_claims_for_scope(
    claims: Sequence[dict[str, object]],
    scope_filter: Mapping[str, str | None] | None,
) -> list[dict[str, object]]:
    """Filter claims by scope using claim provenance source rows."""
    if not scope_filter or not claims:
        return [dict(claim) for claim in claims]

    from consolidation_memory.database import get_claim_source_scope_rows

    claim_ids = [str(claim["id"]) for claim in claims if claim.get("id")]
    if not claim_ids:
        return []

    source_rows = get_claim_source_scope_rows(claim_ids)
    allowed_ids: set[str] = set()
    for claim_id in claim_ids:
        rows = source_rows.get(claim_id, [])
        if not rows:
            if (
                scope_filter.get("namespace_slug") == _LEGACY_DEFAULT_NAMESPACE
                and scope_filter.get("project_slug") == _LEGACY_DEFAULT_PROJECT
                and scope_filter.get("app_client_name") == _LEGACY_DEFAULT_APP_NAME
                and scope_filter.get("app_client_type") == _LEGACY_DEFAULT_APP_TYPE
                and not scope_filter.get("app_client_provider")
                and not scope_filter.get("app_client_external_key")
                and not scope_filter.get("agent_name")
                and not scope_filter.get("agent_external_key")
                and not scope_filter.get("session_external_key")
                and not scope_filter.get("session_kind")
            ):
                allowed_ids.add(claim_id)
            continue
        if all(matches_scope_filter(row, scope_filter) for row in rows):
            allowed_ids.add(claim_id)

    return [dict(claim) for claim in claims if str(claim.get("id")) in allowed_ids]


__all__ = [
    "filter_claims_for_scope",
    "matches_scope_filter",
    "parse_claim_payload",
]
