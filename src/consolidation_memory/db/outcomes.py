"""Action outcomes, sources, refs, and claim outcome evidence."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from consolidation_memory.db._helpers import (
    _derive_action_key,
    _normalize_id_tokens,
    _normalize_outcome_type,
    _normalize_utc_timestamp,
    _now,
)
from consolidation_memory.db.connection import get_connection
from consolidation_memory.db.scope import _apply_scope_filters, _coerce_scope_row


def _uuid4():
    """Resolve uuid4 via the database facade for test patch compatibility."""
    import consolidation_memory.database as database

    return database.uuid.uuid4()


def record_action_outcome(
    *,
    action_summary: str,
    outcome_type: str,
    source_claim_ids: Sequence[str] | None = None,
    source_record_ids: Sequence[str] | None = None,
    source_episode_ids: Sequence[str] | None = None,
    code_anchors: Sequence[Mapping[str, Any]] | None = None,
    issue_ids: Sequence[str] | None = None,
    pr_ids: Sequence[str] | None = None,
    action_key: str | None = None,
    summary: str | None = None,
    details: Mapping[str, Any] | str | None = None,
    confidence: float = 0.8,
    provenance: Mapping[str, Any] | str | None = None,
    observed_at: str | None = None,
    scope: Mapping[str, Any] | None = None,
    outcome_id: str | None = None,
) -> str:
    """Persist one action outcome observation and provenance links atomically."""
    action_summary_token = str(action_summary or "").strip()
    if not action_summary_token:
        raise ValueError("action_summary must not be empty")

    normalized_outcome_type = _normalize_outcome_type(outcome_type)
    claim_ids = _normalize_id_tokens(source_claim_ids)
    record_ids = _normalize_id_tokens(source_record_ids)
    episode_ids = _normalize_id_tokens(source_episode_ids)
    if not claim_ids and not record_ids and not episode_ids:
        raise ValueError(
            "At least one source_claim_id, source_record_id, or source_episode_id is required"
        )

    issue_tokens = _normalize_id_tokens(issue_ids)
    pr_tokens = _normalize_id_tokens(pr_ids)
    normalized_anchors: list[tuple[str, str]] = []
    seen_anchors: set[tuple[str, str]] = set()
    for anchor in code_anchors or ():
        anchor_type = str(anchor.get("anchor_type") or anchor.get("type") or "").strip()
        anchor_value = str(anchor.get("anchor_value") or anchor.get("value") or "").strip()
        if not anchor_type or not anchor_value:
            continue
        pair = (anchor_type, anchor_value)
        if pair in seen_anchors:
            continue
        seen_anchors.add(pair)
        normalized_anchors.append(pair)

    now = _now()
    observed_ts = _normalize_utc_timestamp(observed_at or now)
    scope_row = _coerce_scope_row(scope)
    resolved_outcome_id = str(outcome_id or _uuid4())
    resolved_action_key = (
        str(action_key).strip()
        if action_key is not None and str(action_key).strip()
        else _derive_action_key(action_summary_token)
    )
    summary_token = str(summary).strip() if summary is not None else None
    summary_token = summary_token or None
    details_text = (
        details
        if isinstance(details, str)
        else (json.dumps(details, default=str) if details is not None else None)
    )
    provenance_text = (
        provenance
        if isinstance(provenance, str)
        else (json.dumps(provenance, default=str) if provenance is not None else "{}")
    )
    confidence_value = max(0.0, min(1.0, float(confidence)))

    with get_connection() as conn:
        conn.execute(
            """INSERT INTO action_outcomes
               (id, action_key, action_summary, outcome_type, summary, details,
                confidence, provenance, observed_at, created_at, updated_at,
                namespace_slug, namespace_sharing_mode,
                app_client_name, app_client_type, app_client_provider, app_client_external_key,
                agent_name, agent_external_key,
                session_external_key, session_kind,
                project_slug, project_display_name, project_root_uri,
                project_repo_remote, project_default_branch)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                resolved_outcome_id,
                resolved_action_key,
                action_summary_token,
                normalized_outcome_type,
                summary_token,
                details_text,
                confidence_value,
                provenance_text,
                observed_ts,
                now,
                now,
                scope_row["namespace_slug"],
                scope_row["namespace_sharing_mode"],
                scope_row["app_client_name"],
                scope_row["app_client_type"],
                scope_row["app_client_provider"],
                scope_row["app_client_external_key"],
                scope_row["agent_name"],
                scope_row["agent_external_key"],
                scope_row["session_external_key"],
                scope_row["session_kind"],
                scope_row["project_slug"],
                scope_row["project_display_name"],
                scope_row["project_root_uri"],
                scope_row["project_repo_remote"],
                scope_row["project_default_branch"],
            ),
        )

        source_rows: list[tuple[str, str, str | None, str | None, str | None, str]] = []
        for claim_id in claim_ids:
            source_rows.append((str(_uuid4()), resolved_outcome_id, claim_id, None, None, now))
        for record_id in record_ids:
            source_rows.append((str(_uuid4()), resolved_outcome_id, None, record_id, None, now))
        for episode_id in episode_ids:
            source_rows.append((str(_uuid4()), resolved_outcome_id, None, None, episode_id, now))
        if source_rows:
            conn.executemany(
                """INSERT INTO action_outcome_sources
                   (id, outcome_id, source_claim_id, source_record_id, source_episode_id, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT DO NOTHING""",
                source_rows,
            )

        ref_rows: list[tuple[str, str, str, str, str, str]] = []
        for anchor_type, anchor_value in normalized_anchors:
            ref_rows.append(
                (
                    str(_uuid4()),
                    resolved_outcome_id,
                    "code_anchor",
                    anchor_type,
                    anchor_value,
                    now,
                )
            )
        for issue_id in issue_tokens:
            ref_rows.append(
                (
                    str(_uuid4()),
                    resolved_outcome_id,
                    "issue",
                    "id",
                    issue_id,
                    now,
                )
            )
        for pr_id in pr_tokens:
            ref_rows.append(
                (
                    str(_uuid4()),
                    resolved_outcome_id,
                    "pr",
                    "id",
                    pr_id,
                    now,
                )
            )
        if ref_rows:
            conn.executemany(
                """INSERT INTO action_outcome_refs
                   (id, outcome_id, ref_type, ref_key, ref_value, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT DO NOTHING""",
                ref_rows,
            )

    from consolidation_memory.db.claims import _refresh_claim_precisions

    _refresh_claim_precisions(claim_ids)
    return resolved_outcome_id


def get_action_outcomes(
    *,
    outcome_type: str | None = None,
    action_key: str | None = None,
    source_claim_id: str | None = None,
    source_record_id: str | None = None,
    source_episode_id: str | None = None,
    as_of: str | None = None,
    limit: int = 100,
    offset: int = 0,
    scope: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Browse action outcomes with optional source and temporal filters."""
    bounded_limit = max(1, int(limit))
    bounded_offset = max(0, int(offset))
    conditions: list[str] = []
    params: list[Any] = []

    join_sources = source_claim_id is not None or source_record_id is not None or source_episode_id is not None
    joins = ""
    if join_sources:
        joins = "JOIN action_outcome_sources aos ON aos.outcome_id = ao.id"

    _apply_scope_filters(conditions, params, scope, table_alias="ao")
    if outcome_type is not None:
        conditions.append("ao.outcome_type = ?")
        params.append(_normalize_outcome_type(outcome_type))
    if action_key is not None:
        action_key_token = str(action_key).strip()
        if action_key_token:
            conditions.append("ao.action_key = ?")
            params.append(action_key_token)
    if source_claim_id is not None:
        source_claim_token = str(source_claim_id).strip()
        if source_claim_token:
            conditions.append("aos.source_claim_id = ?")
            params.append(source_claim_token)
    if source_record_id is not None:
        source_record_token = str(source_record_id).strip()
        if source_record_token:
            conditions.append("aos.source_record_id = ?")
            params.append(source_record_token)
    if source_episode_id is not None:
        source_episode_token = str(source_episode_id).strip()
        if source_episode_token:
            conditions.append("aos.source_episode_id = ?")
            params.append(source_episode_token)
    if as_of is not None:
        as_of_utc = _normalize_utc_timestamp(as_of)
        conditions.append("julianday(ao.observed_at) <= julianday(?)")
        params.append(as_of_utc)

    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    with get_connection() as conn:
        rows = conn.execute(
            f"""SELECT DISTINCT ao.*
                FROM action_outcomes ao
                {joins}
                {where_clause}
                ORDER BY julianday(ao.observed_at) DESC, ao.created_at DESC, ao.id ASC
                LIMIT ? OFFSET ?""",
            [*params, bounded_limit, bounded_offset],
        ).fetchall()
    return [dict(row) for row in rows]


def get_action_outcome_sources_by_outcome_ids(
    outcome_ids: Sequence[str],
) -> list[dict[str, Any]]:
    """Return source-link rows for a set of outcome IDs."""
    if not outcome_ids:
        return []
    placeholders = ",".join("?" for _ in outcome_ids)
    with get_connection() as conn:
        rows = conn.execute(
            f"""SELECT id, outcome_id, source_claim_id, source_record_id, source_episode_id, created_at
                FROM action_outcome_sources
                WHERE outcome_id IN ({placeholders})
                ORDER BY created_at ASC, id ASC""",
            list(outcome_ids),
        ).fetchall()
    return [dict(row) for row in rows]


def get_action_outcome_refs_by_outcome_ids(
    outcome_ids: Sequence[str],
) -> list[dict[str, Any]]:
    """Return reference rows for a set of outcome IDs."""
    if not outcome_ids:
        return []
    placeholders = ",".join("?" for _ in outcome_ids)
    with get_connection() as conn:
        rows = conn.execute(
            f"""SELECT id, outcome_id, ref_type, ref_key, ref_value, created_at
                FROM action_outcome_refs
                WHERE outcome_id IN ({placeholders})
                ORDER BY created_at ASC, id ASC""",
            list(outcome_ids),
        ).fetchall()
    return [dict(row) for row in rows]


def get_claim_outcome_evidence(
    claim_ids: Sequence[str],
    *,
    as_of: str | None = None,
    scope: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Aggregate outcome + challenge evidence for claim-level trust scoring."""
    normalized_ids: list[str] = []
    seen_ids: set[str] = set()
    for claim_id in claim_ids:
        token = str(claim_id or "").strip()
        if not token or token in seen_ids:
            continue
        seen_ids.add(token)
        normalized_ids.append(token)
    if not normalized_ids:
        return {}

    placeholders = ",".join("?" for _ in normalized_ids)
    outcome_conditions = [f"aos.source_claim_id IN ({placeholders})"]
    outcome_params: list[Any] = [*normalized_ids]
    _apply_scope_filters(outcome_conditions, outcome_params, scope, table_alias="ao")
    if as_of is not None:
        as_of_utc = _normalize_utc_timestamp(as_of)
        outcome_conditions.append("julianday(ao.observed_at) <= julianday(?)")
        outcome_params.append(as_of_utc)

    outcome_where = " AND ".join(outcome_conditions)
    event_conditions = [f"claim_id IN ({placeholders})"]
    event_params: list[Any] = [*normalized_ids]
    if as_of is not None:
        as_of_utc = _normalize_utc_timestamp(as_of)
        event_conditions.append("julianday(created_at) <= julianday(?)")
        event_params.append(as_of_utc)
    event_where = " AND ".join(event_conditions)

    with get_connection() as conn:
        outcome_rows = conn.execute(
            f"""SELECT
                    aos.source_claim_id AS claim_id,
                    COUNT(*) AS validation_count,
                    SUM(CASE WHEN ao.outcome_type = 'success' THEN 1 ELSE 0 END) AS success_count,
                    SUM(CASE WHEN ao.outcome_type = 'partial_success' THEN 1 ELSE 0 END) AS partial_success_count,
                    SUM(CASE WHEN ao.outcome_type IN ('failure', 'reverted', 'superseded') THEN 1 ELSE 0 END) AS failure_count,
                    SUM(CASE WHEN ao.outcome_type = 'failure' THEN 1 ELSE 0 END) AS explicit_failure_count,
                    SUM(CASE WHEN ao.outcome_type = 'reverted' THEN 1 ELSE 0 END) AS reverted_count,
                    SUM(CASE WHEN ao.outcome_type = 'superseded' THEN 1 ELSE 0 END) AS superseded_count,
                    SUM(
                        CASE
                            WHEN ao.provenance IS NOT NULL
                                 AND TRIM(ao.provenance) NOT IN ('', '{{}}', 'null')
                                THEN 1
                            ELSE 0
                        END
                    ) AS outcomes_with_provenance_count,
                    MAX(ao.observed_at) AS last_observed_at
                FROM action_outcome_sources aos
                JOIN action_outcomes ao ON ao.id = aos.outcome_id
                WHERE {outcome_where}
                GROUP BY aos.source_claim_id""",
            outcome_params,
        ).fetchall()
        outcome_anchor_rows = conn.execute(
            f"""SELECT
                    aos.source_claim_id AS claim_id,
                    COUNT(DISTINCT aor.id) AS outcome_anchor_count
                FROM action_outcome_sources aos
                JOIN action_outcomes ao ON ao.id = aos.outcome_id
                JOIN action_outcome_refs aor ON aor.outcome_id = ao.id
                WHERE {outcome_where}
                  AND aor.ref_type = 'code_anchor'
                GROUP BY aos.source_claim_id""",
            outcome_params,
        ).fetchall()
        event_rows = conn.execute(
            f"""SELECT
                    claim_id,
                    SUM(CASE WHEN event_type = 'contradiction' THEN 1 ELSE 0 END) AS contradiction_count,
                    SUM(CASE WHEN event_type = 'challenged' THEN 1 ELSE 0 END) AS challenged_count,
                    SUM(CASE WHEN event_type = 'code_drift_detected' THEN 1 ELSE 0 END) AS drift_event_count,
                    SUM(
                        CASE
                            WHEN event_type IN ('expire', 'auto_expired_challenged') THEN 1
                            ELSE 0
                        END
                    ) AS expiry_event_count
                FROM claim_events
                WHERE {event_where}
                GROUP BY claim_id""",
            event_params,
        ).fetchall()
        source_rows = conn.execute(
            f"""SELECT
                    cs.claim_id AS claim_id,
                    COUNT(*) AS source_link_count,
                    COUNT(DISTINCT cs.source_episode_id) AS source_episode_count,
                    COUNT(DISTINCT cs.source_topic_id) AS source_topic_count,
                    COUNT(DISTINCT cs.source_record_id) AS source_record_count,
                    COUNT(DISTINCT ea.id) AS source_anchor_count
                FROM claim_sources cs
                LEFT JOIN episode_anchors ea ON ea.episode_id = cs.source_episode_id
                WHERE cs.claim_id IN ({placeholders})
                GROUP BY cs.claim_id""",
            [*normalized_ids],
        ).fetchall()

    evidence: dict[str, dict[str, Any]] = {
        claim_id: {
            "validation_count": 0,
            "success_count": 0,
            "partial_success_count": 0,
            "failure_count": 0,
            "explicit_failure_count": 0,
            "reverted_count": 0,
            "superseded_count": 0,
            "contradiction_count": 0,
            "challenged_count": 0,
            "drift_event_count": 0,
            "expiry_event_count": 0,
            "outcome_anchor_count": 0,
            "outcomes_with_provenance_count": 0,
            "source_link_count": 0,
            "source_episode_count": 0,
            "source_topic_count": 0,
            "source_record_count": 0,
            "source_anchor_count": 0,
            "last_observed_at": None,
        }
        for claim_id in normalized_ids
    }
    for row in outcome_rows:
        claim_id = str(row["claim_id"])
        evidence[claim_id] = {
            **evidence.get(claim_id, {}),
            "validation_count": int(row["validation_count"] or 0),
            "success_count": int(row["success_count"] or 0),
            "partial_success_count": int(row["partial_success_count"] or 0),
            "failure_count": int(row["failure_count"] or 0),
            "explicit_failure_count": int(row["explicit_failure_count"] or 0),
            "reverted_count": int(row["reverted_count"] or 0),
            "superseded_count": int(row["superseded_count"] or 0),
            "outcomes_with_provenance_count": int(row["outcomes_with_provenance_count"] or 0),
            "last_observed_at": row["last_observed_at"],
        }
    for row in outcome_anchor_rows:
        claim_id = str(row["claim_id"])
        evidence[claim_id] = {
            **evidence.get(claim_id, {}),
            "outcome_anchor_count": int(row["outcome_anchor_count"] or 0),
        }
    for row in event_rows:
        claim_id = str(row["claim_id"])
        evidence[claim_id] = {
            **evidence.get(claim_id, {}),
            "contradiction_count": int(row["contradiction_count"] or 0),
            "challenged_count": int(row["challenged_count"] or 0),
            "drift_event_count": int(row["drift_event_count"] or 0),
            "expiry_event_count": int(row["expiry_event_count"] or 0),
        }
    for row in source_rows:
        claim_id = str(row["claim_id"])
        evidence[claim_id] = {
            **evidence.get(claim_id, {}),
            "source_link_count": int(row["source_link_count"] or 0),
            "source_episode_count": int(row["source_episode_count"] or 0),
            "source_topic_count": int(row["source_topic_count"] or 0),
            "source_record_count": int(row["source_record_count"] or 0),
            "source_anchor_count": int(row["source_anchor_count"] or 0),
        }
    return evidence

