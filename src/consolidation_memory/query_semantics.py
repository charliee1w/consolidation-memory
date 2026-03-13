"""Shared trust semantics for query-time filtering and payload parsing."""

from __future__ import annotations

import json
import math
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


def strategy_reuse_profile(evidence: Mapping[str, object] | None) -> dict[str, object]:
    """Build trust signals used to rank strategy claims for reuse."""
    data = dict(evidence or {})
    validation_count = max(0, int(data.get("validation_count", 0) or 0))
    success_count = max(0, int(data.get("success_count", 0) or 0))
    partial_success_count = max(0, int(data.get("partial_success_count", 0) or 0))
    failure_count = max(0, int(data.get("failure_count", 0) or 0))
    contradiction_count = max(0, int(data.get("contradiction_count", 0) or 0))
    challenged_count = max(0, int(data.get("challenged_count", 0) or 0))

    support_weight = success_count + (0.5 * partial_success_count)
    risk_weight = failure_count + contradiction_count + (0.5 * challenged_count)
    density = min(1.0, validation_count / 5.0)
    density_bonus = min(0.35, math.log1p(validation_count) * 0.12)
    support_bonus = min(0.45, support_weight * 0.1)
    risk_penalty = min(0.75, risk_weight * 0.2)
    reuse_multiplier = max(0.2, 1.0 + density_bonus + support_bonus - risk_penalty)

    if validation_count == 0:
        reusability = "unvalidated"
    elif risk_weight == 0 and support_weight >= 2:
        reusability = "validated"
    elif risk_weight > support_weight:
        reusability = "degraded"
    else:
        reusability = "mixed"

    return {
        "validation_count": validation_count,
        "success_count": success_count,
        "partial_success_count": partial_success_count,
        "failure_count": failure_count,
        "contradiction_count": contradiction_count,
        "challenged_count": challenged_count,
        "support_weight": round(support_weight, 3),
        "risk_weight": round(risk_weight, 3),
        "evidence_density": round(density, 3),
        "reuse_multiplier": round(reuse_multiplier, 3),
        "reusability": reusability,
        "last_observed_at": data.get("last_observed_at"),
    }


__all__ = [
    "filter_claims_for_scope",
    "matches_scope_filter",
    "parse_claim_payload",
    "strategy_reuse_profile",
]
