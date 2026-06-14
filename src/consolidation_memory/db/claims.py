"""Claims graph, precision, trust stats, anchors, and challenge flow."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import timedelta
from typing import Any, Mapping, Sequence

from consolidation_memory.db._helpers import _normalize_utc_timestamp, _now
from consolidation_memory.db.connection import get_connection
from consolidation_memory.db.scope import _apply_scope_filters
from consolidation_memory.utils import parse_datetime

logger = logging.getLogger(__name__)

_DEFAULT_CLAIM_PRECISION = 1.0
_PRECISION_REFRESH_EVENT_TYPES = frozenset({
    "contradiction",
    "challenged",
    "code_drift_detected",
})

def claim_exists(claim_id: str) -> bool:
    """Return True when a claim row already exists."""
    with get_connection() as conn:
        row = conn.execute("SELECT 1 FROM claims WHERE id = ? LIMIT 1", (claim_id,)).fetchone()
    return row is not None


def _bound_claim_precision(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


_PRECISION_REFRESH_EVENT_TYPES = frozenset({
    "contradiction",
    "challenged",
    "code_drift_detected",
})


def update_claim_precision(claim_id: str, precision: float) -> None:
    """Persist a bounded precision value for one claim."""
    token = str(claim_id or "").strip()
    if not token:
        return
    bounded = _bound_claim_precision(precision)
    with get_connection() as conn:
        conn.execute(
            "UPDATE claims SET precision = ?, updated_at = ? WHERE id = ?",
            (bounded, _now(), token),
        )


def recompute_claim_precision(claim_id: str) -> float | None:
    """Recompute and persist claim precision from aggregated trust evidence."""
    from consolidation_memory.query_semantics import claim_precision_from_evidence

    token = str(claim_id or "").strip()
    if not token:
        return None

    with get_connection() as conn:
        row = conn.execute(
            "SELECT status FROM claims WHERE id = ?",
            (token,),
        ).fetchone()
    if row is None:
        return None

    from consolidation_memory.db.outcomes import get_claim_outcome_evidence

    evidence = get_claim_outcome_evidence([token]).get(token, {})
    new_precision = claim_precision_from_evidence(
        evidence,
        claim_status=str(row["status"]),
    )
    update_claim_precision(token, new_precision)
    return new_precision


def _refresh_claim_precisions(claim_ids: Sequence[str]) -> None:
    """Recompute precision for each unique claim ID."""
    for claim_id in dict.fromkeys(str(claim_id or "").strip() for claim_id in claim_ids):
        if claim_id:
            recompute_claim_precision(claim_id)


def upsert_claim(
    claim_id: str,
    claim_type: str,
    canonical_text: str,
    payload: dict[str, Any] | str | None = None,
    status: str = "active",
    confidence: float = 0.8,
    valid_from: str | None = None,
    valid_until: str | None = None,
    precision: float | None = None,
) -> str:
    """Insert or update a claim row by ID.

    When ``precision`` is omitted on conflict, the existing stored precision is
    preserved so consolidation refreshes do not reset trust adjustments.
    """
    now = _now()
    payload_text = payload if isinstance(payload, str) else json.dumps(payload or {})
    valid_from_ts = valid_from or now
    insert_precision = (
        _DEFAULT_CLAIM_PRECISION
        if precision is None
        else _bound_claim_precision(precision)
    )
    update_sets = [
        "claim_type = excluded.claim_type",
        "canonical_text = excluded.canonical_text",
        "payload = excluded.payload",
        "status = excluded.status",
        "confidence = excluded.confidence",
        "valid_from = excluded.valid_from",
        "valid_until = excluded.valid_until",
        "updated_at = excluded.updated_at",
    ]
    if precision is not None:
        update_sets.append("precision = excluded.precision")

    with get_connection() as conn:
        conn.execute(
            f"""INSERT INTO claims
               (id, claim_type, canonical_text, payload, status, confidence,
                valid_from, valid_until, created_at, updated_at, precision)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                   {", ".join(update_sets)}""",
            (
                claim_id,
                claim_type,
                canonical_text,
                payload_text,
                status,
                confidence,
                valid_from_ts,
                valid_until,
                now,
                now,
                insert_precision,
            ),
        )
    return claim_id


def get_claims_by_ids(claim_ids: Sequence[str]) -> list[dict[str, Any]]:
    """Return claim rows for the provided IDs."""
    normalized = [str(claim_id).strip() for claim_id in claim_ids if str(claim_id).strip()]
    if not normalized:
        return []
    placeholders = ",".join("?" for _ in normalized)
    with get_connection() as conn:
        rows = conn.execute(
            f"SELECT * FROM claims WHERE id IN ({placeholders})",
            normalized,
        ).fetchall()
    return [dict(row) for row in rows]


def get_contradicting_partner_claim_ids(
    claim_ids: Sequence[str],
    *,
    partner_statuses: frozenset[str] | None = None,
) -> dict[str, set[str]]:
    """Map claim IDs to contradicting partner claim IDs via claim_edges."""
    normalized = sorted({str(claim_id).strip() for claim_id in claim_ids if str(claim_id).strip()})
    if not normalized:
        return {}

    placeholders = ",".join("?" for _ in normalized)
    with get_connection() as conn:
        rows = conn.execute(
            f"""SELECT from_claim_id, to_claim_id
                FROM claim_edges
                WHERE edge_type = 'contradicts'
                  AND (from_claim_id IN ({placeholders})
                       OR to_claim_id IN ({placeholders}))""",
            [*normalized, *normalized],
        ).fetchall()

    partners: dict[str, set[str]] = {claim_id: set() for claim_id in normalized}
    for row in rows:
        from_id = str(row["from_claim_id"])
        to_id = str(row["to_claim_id"])
        if from_id in partners:
            partners[from_id].add(to_id)
        if to_id in partners:
            partners[to_id].add(from_id)

    if not partner_statuses:
        return partners

    all_partner_ids = sorted({pid for partner_set in partners.values() for pid in partner_set})
    if not all_partner_ids:
        return partners

    status_by_id = {
        str(row["id"]): str(row.get("status") or "active")
        for row in get_claims_by_ids(all_partner_ids)
    }
    for claim_id, partner_set in partners.items():
        partners[claim_id] = {
            partner_id
            for partner_id in partner_set
            if status_by_id.get(partner_id, "active") in partner_statuses
        }
    return partners


def get_active_claims(
    claim_type: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """Return currently active claims, optionally filtered by type."""
    now = _now()
    bounded_limit = max(1, int(limit))
    bounded_offset = max(0, int(offset))
    conditions = [
        "status = ?",
        "julianday(valid_from) <= julianday(?)",
        "(valid_until IS NULL OR julianday(valid_until) > julianday(?))",
    ]
    params: list[Any] = ["active", now, now]

    if claim_type:
        conditions.append("claim_type = ?")
        params.append(claim_type)

    where = " AND ".join(conditions)
    params.extend([bounded_limit, bounded_offset])
    query = f"""SELECT * FROM claims
        WHERE {where}
        ORDER BY updated_at DESC, id ASC
        LIMIT ? OFFSET ?"""

    with get_connection() as conn:
        rows = conn.execute(
            query,
            params,
        ).fetchall()
    return [dict(r) for r in rows]


def get_claims_as_of(
    as_of: str,
    claim_type: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """Return claims valid at a specific point in time."""
    as_of_utc = _normalize_utc_timestamp(as_of)
    bounded_limit = max(1, int(limit))
    bounded_offset = max(0, int(offset))
    conditions = [
        "julianday(valid_from) <= julianday(?)",
        "(valid_until IS NULL OR julianday(valid_until) > julianday(?))",
    ]
    params: list[Any] = [as_of_utc, as_of_utc]

    if claim_type:
        conditions.append("claim_type = ?")
        params.append(claim_type)

    where = " AND ".join(conditions)
    params.extend([bounded_limit, bounded_offset])
    query = f"""SELECT c.*,
                   CASE
                       WHEN c.status = 'challenged'
                            AND COALESCE(ch.first_challenged_at, julianday(c.updated_at)) > julianday(?)
                           THEN 'active'
                       WHEN c.status = 'challenged'
                           THEN 'challenged'
                       WHEN c.status = 'expired'
                            AND ch.first_challenged_at IS NOT NULL
                            AND ch.first_challenged_at <= julianday(?)
                           THEN 'challenged'
                       WHEN c.status = 'expired'
                           THEN 'active'
                       ELSE c.status
                   END AS snapshot_status
            FROM claims c
            LEFT JOIN (
                SELECT claim_id, MIN(julianday(created_at)) AS first_challenged_at
                FROM claim_events
                WHERE event_type = 'challenged'
                GROUP BY claim_id
            ) ch ON ch.claim_id = c.id
            WHERE {where}
            ORDER BY c.updated_at DESC, c.id ASC
            LIMIT ? OFFSET ?"""

    with get_connection() as conn:
        rows = conn.execute(
            query,
            [as_of_utc, as_of_utc, *params],
        ).fetchall()
    claims = [dict(r) for r in rows]
    for claim in claims:
        claim["status"] = claim.pop("snapshot_status")
    return claims


def get_existing_claim_ids(claim_ids: Sequence[str]) -> set[str]:
    """Return the subset of provided claim IDs that currently exist in SQLite."""
    if not claim_ids:
        return set()
    placeholders = ",".join("?" for _ in claim_ids)
    with get_connection() as conn:
        rows = conn.execute(
            f"SELECT id FROM claims WHERE id IN ({placeholders})",
            list(claim_ids),
        ).fetchall()
    return {str(row["id"]) for row in rows}


def _scoped_claim_ids(scope: Mapping[str, Any] | None) -> set[str] | None:
    """Return claim IDs visible within scope, or None when unscoped."""
    if not scope:
        return None
    from consolidation_memory.query_semantics import filter_claims_for_scope

    from consolidation_memory.db.export import get_all_claims

    scoped = filter_claims_for_scope(get_all_claims(), scope)
    return {str(claim["id"]) for claim in scoped if claim.get("id")}


def _empty_claim_trust_stats() -> dict[str, float | int]:
    return {
        "total_claims": 0,
        "currently_valid_claims": 0,
        "active_claims": 0,
        "challenged_claims": 0,
        "claims_with_sources": 0,
        "claims_without_sources": 0,
        "claims_with_anchors": 0,
        "source_coverage_ratio": 0.0,
        "anchor_coverage_ratio": 0.0,
    }


def get_claim_trust_stats(
    as_of: str | None = None,
    *,
    scope: Mapping[str, Any] | None = None,
) -> dict[str, float | int]:
    """Return trust-oriented claim coverage stats for the current snapshot."""
    scoped_ids = _scoped_claim_ids(scope)
    if scoped_ids is not None and not scoped_ids:
        return _empty_claim_trust_stats()

    as_of_utc = _normalize_utc_timestamp(as_of or _now())
    scope_clause = ""
    scope_params: list[Any] = []
    if scoped_ids is not None:
        placeholders = ",".join("?" for _ in scoped_ids)
        scope_clause = f" AND id IN ({placeholders})"
        scope_params = list(scoped_ids)

    with get_connection() as conn:
        row = conn.execute(
            f"""WITH valid_claims AS (
                   SELECT id, status
                   FROM claims
                   WHERE julianday(valid_from) <= julianday(?)
                     AND (valid_until IS NULL OR julianday(valid_until) > julianday(?))
                     {scope_clause}
               ),
               sourced_claims AS (
                   SELECT DISTINCT claim_id
                   FROM claim_sources
               ),
               anchored_claims AS (
                   SELECT DISTINCT cs.claim_id
                   FROM claim_sources cs
                   JOIN episode_anchors ea ON ea.episode_id = cs.source_episode_id
               )
               SELECT
                   COUNT(*) AS currently_valid_claims,
                   COALESCE(SUM(CASE WHEN vc.status = 'active' THEN 1 ELSE 0 END), 0) AS active_claims,
                   COALESCE(SUM(CASE WHEN vc.status = 'challenged' THEN 1 ELSE 0 END), 0) AS challenged_claims,
                   COALESCE(SUM(CASE WHEN sc.claim_id IS NOT NULL THEN 1 ELSE 0 END), 0) AS claims_with_sources,
                   COALESCE(SUM(CASE WHEN ac.claim_id IS NOT NULL THEN 1 ELSE 0 END), 0) AS claims_with_anchors
               FROM valid_claims vc
               LEFT JOIN sourced_claims sc ON sc.claim_id = vc.id
               LEFT JOIN anchored_claims ac ON ac.claim_id = vc.id""",
            (as_of_utc, as_of_utc, *scope_params),
        ).fetchone()
        total_sql = "SELECT COUNT(*) AS total_claims FROM claims"
        total_params: list[Any] = []
        if scoped_ids is not None:
            placeholders = ",".join("?" for _ in scoped_ids)
            total_sql += f" WHERE id IN ({placeholders})"
            total_params = list(scoped_ids)
        total_row = conn.execute(total_sql, total_params).fetchone()

    currently_valid_claims = int(row["currently_valid_claims"]) if row else 0
    active_claims = int(row["active_claims"]) if row else 0
    challenged_claims = int(row["challenged_claims"]) if row else 0
    claims_with_sources = int(row["claims_with_sources"]) if row else 0
    claims_with_anchors = int(row["claims_with_anchors"]) if row else 0

    source_coverage_ratio = (
        round(claims_with_sources / currently_valid_claims, 3)
        if currently_valid_claims
        else 0.0
    )
    anchor_coverage_ratio = (
        round(claims_with_anchors / claims_with_sources, 3)
        if claims_with_sources
        else 0.0
    )

    total_claims = int(total_row["total_claims"]) if total_row else 0
    return {
        "total_claims": total_claims,
        "currently_valid_claims": currently_valid_claims,
        "active_claims": active_claims,
        "challenged_claims": challenged_claims,
        "claims_with_sources": claims_with_sources,
        "claims_without_sources": max(0, currently_valid_claims - claims_with_sources),
        "claims_with_anchors": claims_with_anchors,
        "source_coverage_ratio": source_coverage_ratio,
        "anchor_coverage_ratio": anchor_coverage_ratio,
    }


def get_claim_source_scope_rows(
    claim_ids: Sequence[str],
) -> dict[str, list[dict[str, Any]]]:
    """Return scope metadata for each claim based on its provenance sources."""
    if not claim_ids:
        return {}
    placeholders = ",".join("?" for _ in claim_ids)
    query = f"""SELECT
                cs.claim_id,
                COALESCE(e.namespace_slug, kr.namespace_slug, kt.namespace_slug) AS namespace_slug,
                COALESCE(e.project_slug, kr.project_slug, kt.project_slug) AS project_slug,
                COALESCE(e.app_client_name, kr.app_client_name, kt.app_client_name) AS app_client_name,
                COALESCE(e.app_client_type, kr.app_client_type, kt.app_client_type) AS app_client_type,
                COALESCE(e.app_client_provider, kr.app_client_provider, kt.app_client_provider) AS app_client_provider,
                COALESCE(e.app_client_external_key, kr.app_client_external_key, kt.app_client_external_key) AS app_client_external_key,
                COALESCE(e.agent_name, kr.agent_name, kt.agent_name) AS agent_name,
                COALESCE(e.agent_external_key, kr.agent_external_key, kt.agent_external_key) AS agent_external_key,
                COALESCE(e.session_external_key, kr.session_external_key, kt.session_external_key) AS session_external_key,
                COALESCE(e.session_kind, kr.session_kind, kt.session_kind) AS session_kind
            FROM claim_sources cs
            LEFT JOIN episodes e ON cs.source_episode_id = e.id
            LEFT JOIN knowledge_records kr ON cs.source_record_id = kr.id
            LEFT JOIN knowledge_topics kt ON cs.source_topic_id = kt.id
            WHERE cs.claim_id IN ({placeholders})"""
    with get_connection() as conn:
        rows = conn.execute(
            query,
            list(claim_ids),
        ).fetchall()

    grouped: dict[str, list[dict[str, Any]]] = {str(cid): [] for cid in claim_ids}
    for row in rows:
        grouped[str(row["claim_id"])].append(dict(row))
    return grouped


def expire_claim(claim_id: str, valid_until: str | None = None) -> bool:
    """Expire a claim by setting status=expired and valid_until."""
    ts = valid_until or _now()
    with get_connection() as conn:
        cursor = conn.execute(
            """UPDATE claims
               SET status = 'expired',
                   valid_until = CASE
                       WHEN valid_until IS NULL OR valid_until > ? THEN ?
                       ELSE valid_until
                   END,
                   updated_at = ?
               WHERE id = ?""",
            (ts, ts, _now(), claim_id),
        )
    return bool(cursor.rowcount and cursor.rowcount > 0)


def detach_claim_sources_for_episode(episode_id: str) -> list[str]:
    """Remove episode provenance while preserving any remaining record/topic source."""
    episode_token = str(episode_id or "").strip()
    if not episode_token:
        return []

    with get_connection() as conn:
        rows = conn.execute(
            """SELECT id, claim_id, source_topic_id, source_record_id
               FROM claim_sources
               WHERE source_episode_id = ?""",
            (episode_token,),
        ).fetchall()
        if not rows:
            return []

        delete_ids: list[str] = []
        update_ids: list[str] = []
        claim_ids: list[str] = []
        seen_claim_ids: set[str] = set()
        for row in rows:
            claim_id = str(row["claim_id"])
            if claim_id not in seen_claim_ids:
                seen_claim_ids.add(claim_id)
                claim_ids.append(claim_id)
            if row["source_topic_id"] is not None or row["source_record_id"] is not None:
                update_ids.append(str(row["id"]))
            else:
                delete_ids.append(str(row["id"]))

        if update_ids:
            placeholders = ",".join("?" for _ in update_ids)
            conn.execute(
                f"""UPDATE claim_sources
                    SET source_episode_id = NULL
                    WHERE id IN ({placeholders})""",
                update_ids,
            )
        if delete_ids:
            placeholders = ",".join("?" for _ in delete_ids)
            conn.execute(
                f"DELETE FROM claim_sources WHERE id IN ({placeholders})",
                delete_ids,
            )

    return claim_ids


def remove_claim_sources_for_records(record_ids: Sequence[str]) -> list[str]:
    """Remove claim source rows tied to superseded knowledge record IDs."""
    normalized_ids: list[str] = []
    seen_ids: set[str] = set()
    for record_id in record_ids:
        token = str(record_id or "").strip()
        if not token or token in seen_ids:
            continue
        seen_ids.add(token)
        normalized_ids.append(token)
    if not normalized_ids:
        return []

    placeholders = ",".join("?" for _ in normalized_ids)
    with get_connection() as conn:
        rows = conn.execute(
            f"""SELECT DISTINCT claim_id
                FROM claim_sources
                WHERE source_record_id IN ({placeholders})
                ORDER BY claim_id ASC""",
            normalized_ids,
        ).fetchall()
        if not rows:
            return []
        conn.execute(
            f"DELETE FROM claim_sources WHERE source_record_id IN ({placeholders})",
            normalized_ids,
        )

    return [str(row["claim_id"]) for row in rows]


def remove_topic_only_claim_sources(topic_id: str) -> list[str]:
    """Remove legacy topic-only claim source rows for a corrected topic snapshot."""
    topic_token = str(topic_id or "").strip()
    if not topic_token:
        return []

    with get_connection() as conn:
        rows = conn.execute(
            """SELECT DISTINCT claim_id
               FROM claim_sources
               WHERE source_topic_id = ?
                 AND source_record_id IS NULL
                 AND source_episode_id IS NULL
               ORDER BY claim_id ASC""",
            (topic_token,),
        ).fetchall()
        if not rows:
            return []
        conn.execute(
            """DELETE FROM claim_sources
               WHERE source_topic_id = ?
                 AND source_record_id IS NULL
                 AND source_episode_id IS NULL""",
            (topic_token,),
        )

    return [str(row["claim_id"]) for row in rows]


def expire_claims_without_sources(
    claim_ids: Sequence[str],
    *,
    valid_until: str | None = None,
    reason: str,
    details: Mapping[str, Any] | None = None,
) -> list[str]:
    """Expire currently-valid claims that no longer have any provenance rows."""
    normalized_ids: list[str] = []
    seen_ids: set[str] = set()
    for claim_id in claim_ids:
        token = str(claim_id or "").strip()
        if not token or token in seen_ids:
            continue
        seen_ids.add(token)
        normalized_ids.append(token)
    if not normalized_ids:
        return []

    expired_at = _normalize_utc_timestamp(valid_until or _now())
    placeholders = ",".join("?" for _ in normalized_ids)
    with get_connection() as conn:
        rows = conn.execute(
            f"""SELECT c.id
                  FROM claims c
                  LEFT JOIN claim_sources cs ON cs.claim_id = c.id
                 WHERE c.id IN ({placeholders})
                   AND julianday(c.valid_from) <= julianday(?)
                   AND (c.valid_until IS NULL OR julianday(c.valid_until) > julianday(?))
              GROUP BY c.id
                HAVING COUNT(cs.id) = 0
              ORDER BY c.id ASC""",
            [*normalized_ids, expired_at, expired_at],
        ).fetchall()
        orphaned_ids = [str(row["id"]) for row in rows]
        if not orphaned_ids:
            return []

        orphan_placeholders = ",".join("?" for _ in orphaned_ids)
        conn.execute(
            f"""UPDATE claims
                   SET status = 'expired',
                       valid_until = CASE
                           WHEN valid_until IS NULL OR julianday(valid_until) > julianday(?) THEN ?
                           ELSE valid_until
                       END,
                       updated_at = ?
                 WHERE id IN ({orphan_placeholders})
                   AND julianday(valid_from) <= julianday(?)
                   AND (valid_until IS NULL OR julianday(valid_until) > julianday(?))""",
            [expired_at, expired_at, expired_at, *orphaned_ids, expired_at, expired_at],
        )

        event_details = dict(details or {})
        event_details["reason"] = reason
        event_details["expired_at"] = expired_at
        conn.executemany(
            """INSERT INTO claim_events
               (id, claim_id, event_type, details, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            [
                (
                    str(uuid.uuid4()),
                    claim_id,
                    "expire",
                    json.dumps(event_details, default=str),
                    expired_at,
                )
                for claim_id in orphaned_ids
            ],
        )

    return orphaned_ids


def count_active_challenged_claims(
    as_of: str | None = None,
    *,
    scope: Mapping[str, Any] | None = None,
) -> int:
    """Return count of challenged claims that are still temporally valid."""
    scoped_ids = _scoped_claim_ids(scope)
    if scoped_ids is not None and not scoped_ids:
        return 0

    ts = _normalize_utc_timestamp(as_of or _now())
    scope_clause = ""
    scope_params: list[Any] = []
    if scoped_ids is not None:
        placeholders = ",".join("?" for _ in scoped_ids)
        scope_clause = f" AND id IN ({placeholders})"
        scope_params = list(scoped_ids)

    with get_connection() as conn:
        row = conn.execute(
            f"""SELECT COUNT(*) AS c
               FROM claims
               WHERE status = 'challenged'
                 AND julianday(valid_from) <= julianday(?)
                 AND (valid_until IS NULL OR julianday(valid_until) > julianday(?))
                 {scope_clause}""",
            (ts, ts, *scope_params),
        ).fetchone()
    return int(row["c"]) if row else 0


def auto_expire_stale_challenged_claims(
    *,
    max_age_hours: float,
    max_claims: int = 200,
    as_of: str | None = None,
) -> dict[str, Any]:
    """Expire stale challenged claims and record audit events.

    Claims remain in `challenged` status until an explicit resolution. To prevent
    unbounded challenged backlogs, this helper expires challenged claims whose
    earliest challenge event is older than the configured age threshold.
    """
    age_hours = max(float(max_age_hours), 0.0)
    if age_hours <= 0:
        raise ValueError("max_age_hours must be > 0")

    limit = max(int(max_claims), 1)
    as_of_utc = _normalize_utc_timestamp(as_of or _now())
    as_of_dt = parse_datetime(as_of_utc)
    cutoff = (as_of_dt - timedelta(hours=age_hours)).isoformat()

    with get_connection() as conn:
        rows = conn.execute(
            """SELECT c.id
                 FROM claims c
                WHERE c.status = 'challenged'
                  AND julianday(c.valid_from) <= julianday(?)
                  AND (c.valid_until IS NULL OR julianday(c.valid_until) > julianday(?))
                  AND julianday(
                        COALESCE(
                            (
                                SELECT MIN(ce.created_at)
                                  FROM claim_events ce
                                 WHERE ce.claim_id = c.id
                                   AND ce.event_type = 'challenged'
                            ),
                            c.updated_at
                        )
                      ) <= julianday(?)
                ORDER BY c.updated_at ASC, c.id ASC
                LIMIT ?""",
            (as_of_utc, as_of_utc, cutoff, limit),
        ).fetchall()
        stale_ids = [row["id"] for row in rows]
        if not stale_ids:
            return {
                "as_of": as_of_utc,
                "cutoff": cutoff,
                "max_age_hours": age_hours,
                "max_claims": limit,
                "expired_count": 0,
                "expired_claim_ids": [],
            }

        placeholders = ",".join("?" for _ in stale_ids)
        conn.execute(
            f"""UPDATE claims
                    SET status = 'expired',
                        valid_until = CASE
                            WHEN valid_until IS NULL OR julianday(valid_until) > julianday(?) THEN ?
                            ELSE valid_until
                        END,
                        updated_at = ?
                  WHERE id IN ({placeholders})
                    AND status = 'challenged'""",
            [as_of_utc, as_of_utc, as_of_utc, *stale_ids],
        )

        details_payload = json.dumps(
            {
                "policy": "auto_expire_stale_challenged",
                "max_age_hours": age_hours,
                "cutoff": cutoff,
                "expired_at": as_of_utc,
            },
            default=str,
        )
        conn.executemany(
            """INSERT INTO claim_events
               (id, claim_id, event_type, details, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            [
                (
                    str(uuid.uuid4()),
                    claim_id,
                    "auto_expired_challenged",
                    details_payload,
                    as_of_utc,
                )
                for claim_id in stale_ids
            ],
        )

    return {
        "as_of": as_of_utc,
        "cutoff": cutoff,
        "max_age_hours": age_hours,
        "max_claims": limit,
        "expired_count": len(stale_ids),
        "expired_claim_ids": stale_ids,
    }


def insert_claim_edge(
    from_claim_id: str,
    to_claim_id: str,
    edge_type: str,
    confidence: float = 1.0,
    details: dict[str, Any] | str | None = None,
    edge_id: str | None = None,
) -> str:
    """Insert an edge between two claims."""
    eid = edge_id or str(uuid.uuid4())
    details_text = details if isinstance(details, str) else (
        json.dumps(details) if details is not None else None
    )
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO claim_edges
               (id, from_claim_id, to_claim_id, edge_type, confidence, details, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (eid, from_claim_id, to_claim_id, edge_type, confidence, details_text, _now()),
        )
    return eid


def insert_claim_sources(claim_id: str, sources: list[dict[str, Any]]) -> list[str]:
    """Insert source links for a claim. Returns inserted source IDs."""
    if not sources:
        return []

    now = _now()
    source_ids: list[str] = []
    with get_connection() as conn:
        for source in sources:
            source_id = str(source.get("id") or uuid.uuid4())
            source_episode_id = source.get("source_episode_id") or source.get("episode_id")
            source_topic_id = source.get("source_topic_id") or source.get("topic_id")
            source_record_id = source.get("source_record_id") or source.get("record_id")

            conn.execute(
                """INSERT INTO claim_sources
                   (id, claim_id, source_episode_id, source_topic_id, source_record_id, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    source_id,
                    claim_id,
                    source_episode_id,
                    source_topic_id,
                    source_record_id,
                    now,
                ),
            )
            source_ids.append(source_id)
    return source_ids


def insert_claim_event(
    claim_id: str,
    event_type: str,
    details: dict[str, Any] | str | None = None,
    event_id: str | None = None,
    created_at: str | None = None,
) -> str:
    """Insert a claim lifecycle event."""
    eid = event_id or str(uuid.uuid4())
    details_text = details if isinstance(details, str) else (
        json.dumps(details) if details is not None else None
    )
    ts = created_at or _now()
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO claim_events
               (id, claim_id, event_type, details, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (eid, claim_id, event_type, details_text, ts),
        )
    if event_type in _PRECISION_REFRESH_EVENT_TYPES:
        recompute_claim_precision(claim_id)
    return eid


def insert_claim_events(events: Sequence[Mapping[str, Any]]) -> list[str]:
    """Insert multiple claim lifecycle events in a single transaction."""
    if not events:
        return []

    now = _now()
    normalized_rows: list[tuple[str, str, str, str | None, str]] = []
    inserted_ids: list[str] = []
    for event in events:
        claim_id = str(event.get("claim_id") or "").strip()
        event_type = str(event.get("event_type") or "").strip()
        if not claim_id or not event_type:
            continue

        event_id = str(event.get("id") or uuid.uuid4())
        details_raw = event.get("details")
        details_text = (
            details_raw
            if isinstance(details_raw, str)
            else (json.dumps(details_raw) if details_raw is not None else None)
        )
        created_at_raw = event.get("created_at")
        created_at = str(created_at_raw) if created_at_raw is not None else now

        normalized_rows.append((event_id, claim_id, event_type, details_text, created_at))
        inserted_ids.append(event_id)

    if not normalized_rows:
        return []

    with get_connection() as conn:
        conn.executemany(
            """INSERT INTO claim_events
               (id, claim_id, event_type, details, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            normalized_rows,
        )

    refresh_ids = [
        claim_id
        for _, claim_id, event_type, _, _ in normalized_rows
        if event_type in _PRECISION_REFRESH_EVENT_TYPES
    ]
    _refresh_claim_precisions(refresh_ids)
    return inserted_ids


def insert_episode_anchors(episode_id: str, anchors: Sequence[Mapping[str, Any]]) -> list[str]:
    """Insert anchors for an episode. Duplicate anchors are ignored."""
    if not anchors:
        return []

    normalized: list[tuple[str, str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for anchor in anchors:
        anchor_type = str(anchor.get("anchor_type") or anchor.get("type") or "").strip()
        anchor_value = str(anchor.get("anchor_value") or anchor.get("value") or "").strip()
        if not anchor_type or not anchor_value:
            continue

        pair = (anchor_type, anchor_value)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)

        anchor_id = str(
            anchor.get("id")
            or uuid.uuid5(uuid.NAMESPACE_URL, f"{episode_id}:{anchor_type}:{anchor_value}")
        )
        normalized.append((anchor_id, anchor_type, anchor_value))

    if not normalized:
        return []

    inserted_ids: list[str] = []
    with get_connection() as conn:
        for anchor_id, anchor_type, anchor_value in normalized:
            cursor = conn.execute(
                """INSERT INTO episode_anchors
                   (id, episode_id, anchor_type, anchor_value, created_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(episode_id, anchor_type, anchor_value) DO NOTHING""",
                (anchor_id, episode_id, anchor_type, anchor_value, _now()),
            )
            if cursor.rowcount and cursor.rowcount > 0:
                inserted_ids.append(anchor_id)
    return inserted_ids


def get_claims_by_anchor(
    anchor_type: str | None = None,
    anchor_value: str | None = None,
    include_expired: bool = False,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Fetch claims linked to episodes matching the provided anchor filter."""
    if not anchor_type and not anchor_value:
        return []

    conditions: list[str] = []
    params: list[Any] = []
    if anchor_type:
        conditions.append("ea.anchor_type = ?")
        params.append(anchor_type)
    if anchor_value:
        conditions.append("ea.anchor_value = ?")
        params.append(anchor_value)
    if not include_expired:
        now = _now()
        conditions.append("julianday(c.valid_from) <= julianday(?)")
        conditions.append("(c.valid_until IS NULL OR julianday(c.valid_until) > julianday(?))")
        params.extend([now, now])

    where = " AND ".join(conditions)
    params.append(limit)
    query = f"""SELECT DISTINCT
                c.id, c.claim_type, c.canonical_text, c.payload, c.status,
                c.confidence, c.valid_from, c.valid_until, c.created_at, c.updated_at
            FROM claims c
            JOIN claim_sources cs ON cs.claim_id = c.id
            JOIN episode_anchors ea ON ea.episode_id = cs.source_episode_id
            WHERE {where}
            ORDER BY c.updated_at DESC, c.id ASC
            LIMIT ?"""

    with get_connection() as conn:
        rows = conn.execute(
            query,
            params,
        ).fetchall()
    return [dict(r) for r in rows]


def get_claims_by_anchor_values(
    anchor_type: str,
    anchor_values: Sequence[str],
    include_expired: bool = False,
    scope: Mapping[str, Any] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Fetch claims linked to episodes matching any anchor value for one anchor type."""
    anchor_type_token = str(anchor_type or "").strip()
    if not anchor_type_token:
        return []

    deduped_values: list[str] = []
    seen_values: set[str] = set()
    for value in anchor_values:
        token = str(value or "").strip()
        if not token or token in seen_values:
            continue
        seen_values.add(token)
        deduped_values.append(token)

    if not deduped_values:
        return []

    max_results = limit if limit is None else max(1, int(limit))
    params_prefix: list[Any] = [anchor_type_token]
    common_conditions = ["ea.anchor_type = ?"]
    _apply_scope_filters(common_conditions, params_prefix, scope, table_alias="e")
    if not include_expired:
        now = _now()
        common_conditions.extend(
            [
                "julianday(c.valid_from) <= julianday(?)",
                "(c.valid_until IS NULL OR julianday(c.valid_until) > julianday(?))",
            ]
        )
        params_prefix.extend([now, now])

    rows: list[dict[str, Any]] = []
    remaining = max_results
    # Keep chunks well under SQLite's default parameter limit.
    with get_connection() as conn:
        for start in range(0, len(deduped_values), 250):
            if remaining is not None and remaining <= 0:
                break
            chunk = deduped_values[start:start + 250]
            placeholders = ",".join("?" for _ in chunk)
            where = " AND ".join([*common_conditions, f"ea.anchor_value IN ({placeholders})"])
            query_params: list[Any] = [*params_prefix, *chunk]
            sql = f"""SELECT DISTINCT
                        c.id, c.claim_type, c.canonical_text, c.payload, c.status,
                        c.confidence, c.valid_from, c.valid_until, c.created_at, c.updated_at,
                        ea.anchor_value
                    FROM claims c
                    JOIN claim_sources cs ON cs.claim_id = c.id
                    JOIN episode_anchors ea ON ea.episode_id = cs.source_episode_id
                    JOIN episodes e ON e.id = ea.episode_id
                    WHERE {where}
                    ORDER BY c.updated_at DESC, c.id ASC"""
            if remaining is not None:
                sql += " LIMIT ?"
                query_params.append(remaining)

            chunk_rows = conn.execute(sql, query_params).fetchall()

            mapped_chunk = [dict(row) for row in chunk_rows]
            rows.extend(mapped_chunk)
            if remaining is not None:
                remaining -= len(mapped_chunk)

    return rows


def mark_claims_challenged_by_ids(
    claim_ids: Sequence[str],
    challenged_at: str | None = None,
) -> list[str]:
    """Mark active claims as challenged for a known set of impacted claim IDs."""
    deduped_ids: list[str] = []
    seen_ids: set[str] = set()
    for claim_id in claim_ids:
        token = str(claim_id or "").strip()
        if not token or token in seen_ids:
            continue
        seen_ids.add(token)
        deduped_ids.append(token)
    if not deduped_ids:
        return []

    challenged_ts = _normalize_utc_timestamp(challenged_at or _now())
    challenged_ids: list[str] = []
    with get_connection() as conn:
        for start in range(0, len(deduped_ids), 250):
            chunk = deduped_ids[start:start + 250]
            placeholders = ",".join("?" for _ in chunk)
            active_rows = conn.execute(
                f"""SELECT id
                    FROM claims
                    WHERE id IN ({placeholders})
                      AND status = 'active'
                      AND julianday(valid_from) <= julianday(?)
                      AND (valid_until IS NULL OR julianday(valid_until) > julianday(?))
                    ORDER BY id ASC""",
                [*chunk, challenged_ts, challenged_ts],
            ).fetchall()
            active_ids = [row["id"] for row in active_rows]
            if not active_ids:
                continue

            active_placeholders = ",".join("?" for _ in active_ids)
            conn.execute(
                f"""UPDATE claims
                    SET status = 'challenged', updated_at = ?
                    WHERE id IN ({active_placeholders})
                      AND status = 'active'""",
                [challenged_ts, *active_ids],
            )
            conn.executemany(
                """INSERT INTO claim_events
                   (id, claim_id, event_type, details, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                [
                    (
                        str(uuid.uuid4()),
                        claim_id,
                        "challenged",
                        json.dumps({"challenged_at": challenged_ts}),
                        challenged_ts,
                    )
                    for claim_id in active_ids
                ],
            )
            challenged_ids.extend(active_ids)

    _refresh_claim_precisions(challenged_ids)
    return sorted(challenged_ids)


def mark_claims_challenged_by_anchors(
    anchors: list[dict[str, Any]],
    challenged_at: str | None = None,
) -> list[str]:
    """Mark active claims as challenged when linked episode anchors match."""
    if not anchors:
        return []

    anchor_pairs: list[tuple[str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for anchor in anchors:
        anchor_type = str(anchor.get("anchor_type") or anchor.get("type") or "").strip()
        anchor_value = str(anchor.get("anchor_value") or anchor.get("value") or "").strip()
        if not anchor_type or not anchor_value:
            continue
        pair = (anchor_type, anchor_value)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        anchor_pairs.append(pair)

    if not anchor_pairs:
        return []

    challenged_ts = _normalize_utc_timestamp(challenged_at or _now())
    anchor_clauses = []
    anchor_params: list[Any] = []
    for anchor_type, anchor_value in anchor_pairs:
        anchor_clauses.append("(ea.anchor_type = ? AND ea.anchor_value = ?)")
        anchor_params.extend([anchor_type, anchor_value])
    select_query = f"""SELECT DISTINCT c.id
            FROM claims c
            JOIN claim_sources cs ON cs.claim_id = c.id
            JOIN episode_anchors ea ON ea.episode_id = cs.source_episode_id
            WHERE c.status = 'active'
              AND julianday(c.valid_from) <= julianday(?)
              AND (c.valid_until IS NULL OR julianday(c.valid_until) > julianday(?))
              AND ({' OR '.join(anchor_clauses)})
            ORDER BY c.id ASC"""

    with get_connection() as conn:
        rows = conn.execute(
            select_query,
            [challenged_ts, challenged_ts, *anchor_params],
        ).fetchall()

        claim_ids = [row["id"] for row in rows]
        if not claim_ids:
            return []

        placeholders = ",".join("?" for _ in claim_ids)
        update_query = f"""UPDATE claims
            SET status = 'challenged', updated_at = ?
            WHERE id IN ({placeholders})
              AND status = 'active'"""
        conn.execute(
            update_query,
            [challenged_ts, *claim_ids],
        )
        # Record the transition so temporal as_of queries can reconstruct the prior active state.
        conn.executemany(
            """INSERT INTO claim_events
               (id, claim_id, event_type, details, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            [
                (
                    str(uuid.uuid4()),
                    claim_id,
                    "challenged",
                    json.dumps({"challenged_at": challenged_ts}),
                    challenged_ts,
                )
                for claim_id in claim_ids
            ],
        )

    _refresh_claim_precisions(claim_ids)
    return claim_ids


# -- Consolidation Run Tracking -----------------------------------------------

_SCHEDULER_ROW_ID = "global"


