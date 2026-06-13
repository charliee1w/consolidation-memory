"""Bulk export queries and claim graph snapshot import."""

from __future__ import annotations

import json
import uuid
from typing import Any, Mapping, Sequence

from consolidation_memory.db._helpers import OUTCOME_TYPES, _derive_action_key, _now
from consolidation_memory.db.connection import get_connection
from consolidation_memory.db.scope import _apply_scope_filters, _coerce_scope_row

def get_all_episodes(
    include_deleted: bool = False,
    scope: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return all episodes for export."""
    conditions: list[str] = []
    params: list[Any] = []
    if not include_deleted:
        conditions.append("deleted = 0")
    _apply_scope_filters(conditions, params, scope)
    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    with get_connection() as conn:
        rows = conn.execute(
            f"SELECT * FROM episodes {where_clause} ORDER BY created_at",
            params,
        ).fetchall()
    return [dict(r) for r in rows]


def get_all_claims(
    claim_ids: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Return all claims for export."""
    conditions: list[str] = []
    params: list[Any] = []
    if claim_ids is not None:
        if not claim_ids:
            return []
        placeholders = ",".join("?" for _ in claim_ids)
        conditions.append(f"id IN ({placeholders})")
        params.extend(claim_ids)
    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT id, claim_type, canonical_text, payload, status, confidence,
                      valid_from, valid_until, created_at, updated_at
               FROM claims
               {where_clause}
               ORDER BY created_at ASC, id ASC""".format(where_clause=where_clause),
            params,
        ).fetchall()
    return [dict(r) for r in rows]


def get_all_claim_edges(
    claim_ids: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Return all claim edges for export."""
    conditions: list[str] = []
    params: list[Any] = []
    if claim_ids is not None:
        if not claim_ids:
            return []
        placeholders = ",".join("?" for _ in claim_ids)
        conditions.append(f"from_claim_id IN ({placeholders})")
        params.extend(claim_ids)
        conditions.append(f"to_claim_id IN ({placeholders})")
        params.extend(claim_ids)
    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT id, from_claim_id, to_claim_id, edge_type, confidence, details, created_at
               FROM claim_edges
               {where_clause}
               ORDER BY created_at ASC, id ASC""".format(where_clause=where_clause),
            params,
        ).fetchall()
    return [dict(r) for r in rows]


def get_all_claim_sources(
    claim_ids: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Return all claim source links for export."""
    conditions: list[str] = []
    params: list[Any] = []
    if claim_ids is not None:
        if not claim_ids:
            return []
        placeholders = ",".join("?" for _ in claim_ids)
        conditions.append(f"claim_id IN ({placeholders})")
        params.extend(claim_ids)
    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT id, claim_id, source_episode_id, source_topic_id, source_record_id, created_at
               FROM claim_sources
               {where_clause}
               ORDER BY created_at ASC, id ASC""".format(where_clause=where_clause),
            params,
        ).fetchall()
    return [dict(r) for r in rows]


def get_all_claim_events(
    claim_ids: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Return all claim lifecycle events for export."""
    conditions: list[str] = []
    params: list[Any] = []
    if claim_ids is not None:
        if not claim_ids:
            return []
        placeholders = ",".join("?" for _ in claim_ids)
        conditions.append(f"claim_id IN ({placeholders})")
        params.extend(claim_ids)
    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT id, claim_id, event_type, details, created_at
               FROM claim_events
               {where_clause}
               ORDER BY created_at ASC, id ASC""".format(where_clause=where_clause),
            params,
        ).fetchall()
    return [dict(r) for r in rows]


def get_all_episode_anchors(
    episode_ids: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Return all episode anchors for export."""
    conditions: list[str] = []
    params: list[Any] = []
    if episode_ids is not None:
        if not episode_ids:
            return []
        placeholders = ",".join("?" for _ in episode_ids)
        conditions.append(f"episode_id IN ({placeholders})")
        params.extend(episode_ids)
    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT id, episode_id, anchor_type, anchor_value, created_at
               FROM episode_anchors
               {where_clause}
               ORDER BY created_at ASC, id ASC""".format(where_clause=where_clause),
            params,
        ).fetchall()
    return [dict(r) for r in rows]


def get_all_action_outcomes(
    outcome_ids: Sequence[str] | None = None,
    scope: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return all action outcome observations for export."""
    conditions: list[str] = []
    params: list[Any] = []
    if outcome_ids is not None:
        if not outcome_ids:
            return []
        placeholders = ",".join("?" for _ in outcome_ids)
        conditions.append(f"id IN ({placeholders})")
        params.extend(outcome_ids)
    _apply_scope_filters(conditions, params, scope)
    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT id, action_key, action_summary, outcome_type, summary, details,
                      confidence, provenance, observed_at, created_at, updated_at,
                      namespace_slug, namespace_sharing_mode,
                      app_client_name, app_client_type, app_client_provider, app_client_external_key,
                      agent_name, agent_external_key,
                      session_external_key, session_kind,
                      project_slug, project_display_name, project_root_uri,
                      project_repo_remote, project_default_branch
               FROM action_outcomes
               {where_clause}
               ORDER BY observed_at ASC, id ASC""".format(where_clause=where_clause),
            params,
        ).fetchall()
    return [dict(row) for row in rows]


def get_all_action_outcome_sources(
    outcome_ids: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Return all action outcome source links for export."""
    conditions: list[str] = []
    params: list[Any] = []
    if outcome_ids is not None:
        if not outcome_ids:
            return []
        placeholders = ",".join("?" for _ in outcome_ids)
        conditions.append(f"outcome_id IN ({placeholders})")
        params.extend(outcome_ids)
    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT id, outcome_id, source_claim_id, source_record_id, source_episode_id, created_at
               FROM action_outcome_sources
               {where_clause}
               ORDER BY created_at ASC, id ASC""".format(where_clause=where_clause),
            params,
        ).fetchall()
    return [dict(row) for row in rows]


def get_all_action_outcome_refs(
    outcome_ids: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Return all action outcome references (anchors/issues/PRs) for export."""
    conditions: list[str] = []
    params: list[Any] = []
    if outcome_ids is not None:
        if not outcome_ids:
            return []
        placeholders = ",".join("?" for _ in outcome_ids)
        conditions.append(f"outcome_id IN ({placeholders})")
        params.extend(outcome_ids)
    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT id, outcome_id, ref_type, ref_key, ref_value, created_at
               FROM action_outcome_refs
               {where_clause}
               ORDER BY created_at ASC, id ASC""".format(where_clause=where_clause),
            params,
        ).fetchall()
    return [dict(row) for row in rows]


def import_claim_graph_snapshot(
    *,
    claims: Sequence[Mapping[str, Any]] = (),
    claim_edges: Sequence[Mapping[str, Any]] = (),
    claim_sources: Sequence[Mapping[str, Any]] = (),
    claim_events: Sequence[Mapping[str, Any]] = (),
    episode_anchors: Sequence[Mapping[str, Any]] = (),
    action_outcomes: Sequence[Mapping[str, Any]] = (),
    action_outcome_sources: Sequence[Mapping[str, Any]] = (),
    action_outcome_refs: Sequence[Mapping[str, Any]] = (),
) -> dict[str, int]:
    """Import claim graph and action outcome entities from an export snapshot.

    Existing rows are preserved or updated by primary key:
    - claims: upsert by claim ID
    - edges/sources/events/anchors/outcome links: insert with conflict-ignore by unique key/ID
    """
    now = _now()
    imported = {
        "claims": 0,
        "claim_edges": 0,
        "claim_sources": 0,
        "claim_events": 0,
        "episode_anchors": 0,
        "action_outcomes": 0,
        "action_outcome_sources": 0,
        "action_outcome_refs": 0,
    }

    with get_connection() as conn:
        for claim in claims:
            claim_id = str(claim.get("id") or "").strip()
            if not claim_id:
                continue
            payload_raw = claim.get("payload")
            payload_text = payload_raw if isinstance(payload_raw, str) else json.dumps(payload_raw or {})
            claim_confidence_raw = claim.get("confidence", 0.8)
            claim_confidence = 0.8 if claim_confidence_raw is None else float(claim_confidence_raw)

            created_at = str(claim.get("created_at") or now)
            updated_at = str(claim.get("updated_at") or created_at)
            valid_from = str(claim.get("valid_from") or created_at)
            valid_until_raw = claim.get("valid_until")
            valid_until = str(valid_until_raw) if valid_until_raw is not None else None

            conn.execute(
                """INSERT INTO claims
                   (id, claim_type, canonical_text, payload, status, confidence,
                    valid_from, valid_until, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET
                       claim_type = excluded.claim_type,
                       canonical_text = excluded.canonical_text,
                       payload = excluded.payload,
                       status = excluded.status,
                       confidence = excluded.confidence,
                       valid_from = excluded.valid_from,
                       valid_until = excluded.valid_until,
                       updated_at = excluded.updated_at""",
                (
                    claim_id,
                    str(claim.get("claim_type") or "fact"),
                    str(claim.get("canonical_text") or ""),
                    payload_text,
                    str(claim.get("status") or "active"),
                    claim_confidence,
                    valid_from,
                    valid_until,
                    created_at,
                    updated_at,
                ),
            )
            imported["claims"] += 1

        for edge in claim_edges:
            edge_id = str(edge.get("id") or uuid.uuid4())
            from_claim_id = str(edge.get("from_claim_id") or "").strip()
            to_claim_id = str(edge.get("to_claim_id") or "").strip()
            edge_type = str(edge.get("edge_type") or "").strip()
            if not from_claim_id or not to_claim_id or not edge_type:
                continue

            details_raw = edge.get("details")
            details_text = details_raw if isinstance(details_raw, str) else (
                json.dumps(details_raw) if details_raw is not None else None
            )
            created_at = str(edge.get("created_at") or now)
            edge_confidence_raw = edge.get("confidence", 1.0)
            edge_confidence = 1.0 if edge_confidence_raw is None else float(edge_confidence_raw)

            cursor = conn.execute(
                """INSERT INTO claim_edges
                   (id, from_claim_id, to_claim_id, edge_type, confidence, details, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT DO NOTHING""",
                (
                    edge_id,
                    from_claim_id,
                    to_claim_id,
                    edge_type,
                    edge_confidence,
                    details_text,
                    created_at,
                ),
            )
            if cursor.rowcount and cursor.rowcount > 0:
                imported["claim_edges"] += 1

        for source in claim_sources:
            source_id = str(source.get("id") or uuid.uuid4())
            claim_id = str(source.get("claim_id") or "").strip()
            if not claim_id:
                continue

            source_episode_id = source.get("source_episode_id")
            source_topic_id = source.get("source_topic_id")
            source_record_id = source.get("source_record_id")
            created_at = str(source.get("created_at") or now)

            cursor = conn.execute(
                """INSERT INTO claim_sources
                   (id, claim_id, source_episode_id, source_topic_id, source_record_id, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT DO NOTHING""",
                (
                    source_id,
                    claim_id,
                    str(source_episode_id) if source_episode_id is not None else None,
                    str(source_topic_id) if source_topic_id is not None else None,
                    str(source_record_id) if source_record_id is not None else None,
                    created_at,
                ),
            )
            if cursor.rowcount and cursor.rowcount > 0:
                imported["claim_sources"] += 1

        for event in claim_events:
            event_id = str(event.get("id") or uuid.uuid4())
            claim_id = str(event.get("claim_id") or "").strip()
            event_type = str(event.get("event_type") or "").strip()
            if not claim_id or not event_type:
                continue

            details_raw = event.get("details")
            details_text = details_raw if isinstance(details_raw, str) else (
                json.dumps(details_raw) if details_raw is not None else None
            )
            created_at = str(event.get("created_at") or now)

            cursor = conn.execute(
                """INSERT INTO claim_events
                   (id, claim_id, event_type, details, created_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT DO NOTHING""",
                (event_id, claim_id, event_type, details_text, created_at),
            )
            if cursor.rowcount and cursor.rowcount > 0:
                imported["claim_events"] += 1

        for anchor in episode_anchors:
            anchor_id = str(anchor.get("id") or uuid.uuid4())
            episode_id = str(anchor.get("episode_id") or "").strip()
            anchor_type = str(anchor.get("anchor_type") or "").strip()
            anchor_value = str(anchor.get("anchor_value") or "").strip()
            if not episode_id or not anchor_type or not anchor_value:
                continue
            created_at = str(anchor.get("created_at") or now)

            cursor = conn.execute(
                """INSERT INTO episode_anchors
                   (id, episode_id, anchor_type, anchor_value, created_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT DO NOTHING""",
                (anchor_id, episode_id, anchor_type, anchor_value, created_at),
            )
            if cursor.rowcount and cursor.rowcount > 0:
                imported["episode_anchors"] += 1

        for outcome in action_outcomes:
            outcome_id = str(outcome.get("id") or uuid.uuid4())
            action_summary = str(outcome.get("action_summary") or "").strip()
            outcome_type = str(outcome.get("outcome_type") or "").strip().lower()
            if not action_summary or not outcome_type:
                continue
            if outcome_type not in OUTCOME_TYPES:
                continue

            action_key = str(outcome.get("action_key") or "").strip() or _derive_action_key(action_summary)
            summary = outcome.get("summary")
            summary_text = str(summary).strip() if summary is not None else None
            summary_text = summary_text or None
            details_raw = outcome.get("details")
            details_text = details_raw if isinstance(details_raw, str) else (
                json.dumps(details_raw, default=str) if details_raw is not None else None
            )
            provenance_raw = outcome.get("provenance")
            provenance_text = provenance_raw if isinstance(provenance_raw, str) else (
                json.dumps(provenance_raw, default=str) if provenance_raw is not None else "{}"
            )
            confidence_raw = outcome.get("confidence", 0.8)
            confidence = 0.8 if confidence_raw is None else float(confidence_raw)
            observed_at = str(outcome.get("observed_at") or now)
            created_at = str(outcome.get("created_at") or observed_at)
            updated_at = str(outcome.get("updated_at") or created_at)
            scope_row = _coerce_scope_row(outcome)

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
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET
                       action_key = excluded.action_key,
                       action_summary = excluded.action_summary,
                       outcome_type = excluded.outcome_type,
                       summary = excluded.summary,
                       details = excluded.details,
                       confidence = excluded.confidence,
                       provenance = excluded.provenance,
                       observed_at = excluded.observed_at,
                       updated_at = excluded.updated_at,
                       namespace_slug = excluded.namespace_slug,
                       namespace_sharing_mode = excluded.namespace_sharing_mode,
                       app_client_name = excluded.app_client_name,
                       app_client_type = excluded.app_client_type,
                       app_client_provider = excluded.app_client_provider,
                       app_client_external_key = excluded.app_client_external_key,
                       agent_name = excluded.agent_name,
                       agent_external_key = excluded.agent_external_key,
                       session_external_key = excluded.session_external_key,
                       session_kind = excluded.session_kind,
                       project_slug = excluded.project_slug,
                       project_display_name = excluded.project_display_name,
                       project_root_uri = excluded.project_root_uri,
                       project_repo_remote = excluded.project_repo_remote,
                       project_default_branch = excluded.project_default_branch""",
                (
                    outcome_id,
                    action_key,
                    action_summary,
                    outcome_type,
                    summary_text,
                    details_text,
                    confidence,
                    provenance_text,
                    observed_at,
                    created_at,
                    updated_at,
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
            imported["action_outcomes"] += 1

        for source in action_outcome_sources:
            source_id = str(source.get("id") or uuid.uuid4())
            outcome_id = str(source.get("outcome_id") or "").strip()
            if not outcome_id:
                continue
            source_claim_id_raw = source.get("source_claim_id")
            source_record_id_raw = source.get("source_record_id")
            source_episode_id_raw = source.get("source_episode_id")
            if source_claim_id_raw is None and source_record_id_raw is None and source_episode_id_raw is None:
                continue
            created_at = str(source.get("created_at") or now)

            cursor = conn.execute(
                """INSERT INTO action_outcome_sources
                   (id, outcome_id, source_claim_id, source_record_id, source_episode_id, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT DO NOTHING""",
                (
                    source_id,
                    outcome_id,
                    str(source_claim_id_raw) if source_claim_id_raw is not None else None,
                    str(source_record_id_raw) if source_record_id_raw is not None else None,
                    str(source_episode_id_raw) if source_episode_id_raw is not None else None,
                    created_at,
                ),
            )
            if cursor.rowcount and cursor.rowcount > 0:
                imported["action_outcome_sources"] += 1

        for ref in action_outcome_refs:
            ref_id = str(ref.get("id") or uuid.uuid4())
            outcome_id = str(ref.get("outcome_id") or "").strip()
            ref_type = str(ref.get("ref_type") or "").strip()
            ref_key = str(ref.get("ref_key") or "").strip()
            ref_value = str(ref.get("ref_value") or "").strip()
            if not outcome_id or not ref_type or not ref_key or not ref_value:
                continue
            created_at = str(ref.get("created_at") or now)

            cursor = conn.execute(
                """INSERT INTO action_outcome_refs
                   (id, outcome_id, ref_type, ref_key, ref_value, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT DO NOTHING""",
                (ref_id, outcome_id, ref_type, ref_key, ref_value, created_at),
            )
            if cursor.rowcount and cursor.rowcount > 0:
                imported["action_outcome_refs"] += 1

    return imported


def get_all_active_episodes() -> list[dict[str, Any]]:
    """Return all non-deleted, non-pruned episodes for surprise adjustment."""
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT id, surprise_score, access_count, created_at, updated_at, consolidated
               FROM episodes WHERE deleted = 0 AND consolidated != 2"""
        ).fetchall()
    return [dict(r) for r in rows]


def update_surprise_scores(updates: list[tuple[float, str]]) -> None:
    """Batch update surprise scores. updates = [(new_score, episode_id), ...]"""
    if not updates:
        return
    now = _now()
    with get_connection() as conn:
        conn.executemany(
            "UPDATE episodes SET surprise_score = ?, updated_at = ? WHERE id = ?",
            [(score, now, eid) for score, eid in updates],
        )
