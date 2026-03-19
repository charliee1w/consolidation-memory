"""Shared trust semantics for query-time filtering and payload parsing."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone

from consolidation_memory.utils import parse_datetime

_LEGACY_DEFAULT_APP_NAME = "legacy_client"
_LEGACY_DEFAULT_APP_TYPE = "python_sdk"
_LEGACY_DEFAULT_NAMESPACE = "default"
_LEGACY_DEFAULT_PROJECT = "default"


def _coerce_int_metric(value: object, *, default: int = 0) -> int:
    """Best-effort coercion for metric counters loaded from flexible payloads."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return default
        return int(value)
    if isinstance(value, str):
        token = value.strip()
        if not token:
            return default
        try:
            return int(token)
        except ValueError:
            try:
                parsed = float(token)
            except ValueError:
                return default
            if not math.isfinite(parsed):
                return default
            return int(parsed)
    return default


def coerce_numeric_float(value: object, *, default: float) -> float:
    """Best-effort float conversion that fails closed to a default."""
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        parsed = float(value)
        return parsed if math.isfinite(parsed) else default
    if isinstance(value, str):
        token = value.strip()
        if not token:
            return default
        try:
            parsed = float(token)
        except ValueError:
            return default
        return parsed if math.isfinite(parsed) else default
    return default


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


def _normalize_claim_status(value: object) -> str:
    token = str(value or "").strip().lower()
    if token in {"active", "challenged", "expired"}:
        return token
    return "active"


def _coarse_reliability_score(score: float) -> int:
    bounded = max(0.0, min(100.0, score))
    return int(max(0.0, min(100.0, 5.0 * round(bounded / 5.0))))


def _reliability_band(score: int) -> str:
    if score >= 75:
        return "high"
    if score >= 55:
        return "moderate"
    if score >= 35:
        return "guarded"
    return "low"


def _reliability_recommendation(score: int, status: str) -> str:
    if status == "expired" or score < 35:
        return "avoid_reuse"
    if status == "challenged" or score < 55:
        return "reuse_with_caution"
    if score >= 75:
        return "reusable"
    return "conditionally_reusable"


def _coerce_utc_datetime(value: object) -> datetime | None:
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        token = value.strip()
        if not token:
            return None
        try:
            dt = parse_datetime(token)
        except ValueError:
            return None
    else:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def claim_reliability_profile(
    evidence: Mapping[str, object] | None,
    *,
    claim_status: str | None = None,
    as_of: str | None = None,
    claim_updated_at: str | None = None,
) -> dict[str, object]:
    """Build a deterministic, inspectable reliability score for claim reuse."""
    data = dict(evidence or {})
    status = _normalize_claim_status(claim_status)

    validation_count = max(0, _coerce_int_metric(data.get("validation_count", 0)))
    success_count = max(0, _coerce_int_metric(data.get("success_count", 0)))
    partial_success_count = max(0, _coerce_int_metric(data.get("partial_success_count", 0)))
    failure_count = max(0, _coerce_int_metric(data.get("failure_count", 0)))
    explicit_failure_count = max(0, _coerce_int_metric(data.get("explicit_failure_count", 0)))
    reverted_count = max(0, _coerce_int_metric(data.get("reverted_count", 0)))
    superseded_count = max(0, _coerce_int_metric(data.get("superseded_count", 0)))

    contradiction_count = max(0, _coerce_int_metric(data.get("contradiction_count", 0)))
    challenged_count = max(0, _coerce_int_metric(data.get("challenged_count", 0)))
    drift_event_count = max(0, _coerce_int_metric(data.get("drift_event_count", 0)))
    expiry_event_count = max(0, _coerce_int_metric(data.get("expiry_event_count", 0)))

    source_link_count = max(0, _coerce_int_metric(data.get("source_link_count", 0)))
    source_episode_count = max(0, _coerce_int_metric(data.get("source_episode_count", 0)))
    source_topic_count = max(0, _coerce_int_metric(data.get("source_topic_count", 0)))
    source_record_count = max(0, _coerce_int_metric(data.get("source_record_count", 0)))
    source_anchor_count = max(0, _coerce_int_metric(data.get("source_anchor_count", 0)))
    outcome_anchor_count = max(0, _coerce_int_metric(data.get("outcome_anchor_count", 0)))
    outcomes_with_provenance_count = max(
        0,
        _coerce_int_metric(data.get("outcomes_with_provenance_count", 0)),
    )

    negative_outcomes = max(
        failure_count,
        explicit_failure_count + reverted_count + superseded_count,
    )
    weighted_success = success_count + (0.5 * partial_success_count)
    total_outcomes = success_count + partial_success_count + negative_outcomes
    success_ratio = (weighted_success / total_outcomes) if total_outcomes else None

    source_type_count = int(source_episode_count > 0) + int(source_topic_count > 0) + int(
        source_record_count > 0
    )
    code_anchor_count = source_anchor_count + outcome_anchor_count

    reference_dt = _coerce_utc_datetime(as_of) or datetime.now(timezone.utc)
    observed_dt = _coerce_utc_datetime(data.get("last_observed_at"))
    if observed_dt is None:
        observed_dt = _coerce_utc_datetime(claim_updated_at)

    adjustments: list[dict[str, object]] = []
    score = 50.0

    if validation_count >= 5:
        points = 15
        reason = "5+ linked outcomes provide strong support"
    elif validation_count >= 2:
        points = 8
        reason = "2-4 linked outcomes provide moderate support"
    elif validation_count == 1:
        points = 4
        reason = "single linked outcome provides minimal support"
    else:
        points = -10
        reason = "no linked outcomes"
    score += points
    adjustments.append({"signal": "supporting_outcomes", "points": points, "reason": reason})

    if success_ratio is None:
        points = -4
        reason = "success ratio unknown without outcomes"
    elif success_ratio >= 0.8:
        points = 12
        reason = "success ratio >= 0.8"
    elif success_ratio >= 0.6:
        points = 6
        reason = "success ratio >= 0.6"
    elif success_ratio >= 0.4:
        points = 0
        reason = "success ratio between 0.4 and 0.6"
    else:
        points = -8
        reason = "success ratio < 0.4"
    score += points
    adjustments.append({"signal": "success_ratio", "points": points, "reason": reason})

    failure_penalty = min(12, explicit_failure_count * 4)
    reversion_penalty = min(12, (reverted_count + superseded_count) * 6)
    points = -(failure_penalty + reversion_penalty)
    score += points
    adjustments.append(
        {
            "signal": "failures_and_reversions",
            "points": points,
            "reason": (
                f"{explicit_failure_count} failures, {reverted_count} reverts, "
                f"{superseded_count} superseded"
            ),
        }
    )

    contradiction_penalty = min(24, contradiction_count * 8)
    points = -contradiction_penalty
    score += points
    adjustments.append(
        {
            "signal": "contradictions",
            "points": points,
            "reason": f"{contradiction_count} contradiction events",
        }
    )

    drift_penalty = min(18, drift_event_count * 6)
    challenged_penalty = min(9, challenged_count * 3)
    status_penalty = 15 if status == "challenged" else (20 if status == "expired" else 0)
    points = -(drift_penalty + challenged_penalty + status_penalty)
    score += points
    adjustments.append(
        {
            "signal": "drift_and_challenge_state",
            "points": points,
            "reason": (
                f"status={status}, drift_events={drift_event_count}, "
                f"challenged_events={challenged_count}"
            ),
        }
    )

    if observed_dt is None:
        points = 0 if validation_count == 0 else -2
        reason = "no observed_at timestamp"
    else:
        age_days = max(0.0, (reference_dt - observed_dt).total_seconds() / 86400.0)
        if age_days <= 14:
            points = 8
            reason = "outcomes observed within 14 days"
        elif age_days <= 60:
            points = 4
            reason = "outcomes observed within 60 days"
        elif age_days <= 180:
            points = 0
            reason = "outcomes older than 60 days"
        else:
            points = -6
            reason = "outcomes older than 180 days"
    score += points
    adjustments.append({"signal": "recency", "points": points, "reason": reason})

    if source_link_count >= 3:
        source_points = 4
        source_reason = "3+ provenance links"
    elif source_link_count >= 1:
        source_points = 2
        source_reason = "at least one provenance link"
    else:
        source_points = -6
        source_reason = "no provenance links"

    if source_type_count >= 3:
        source_type_points = 3
        type_reason = "episode/topic/record provenance present"
    elif source_type_count >= 2:
        source_type_points = 2
        type_reason = "multiple provenance source types present"
    elif source_type_count == 1:
        source_type_points = 1
        type_reason = "single provenance source type present"
    else:
        source_type_points = -4
        type_reason = "no typed provenance source"

    provenance_ref_points = 2 if outcomes_with_provenance_count > 0 else 0
    points = source_points + source_type_points + provenance_ref_points
    score += points
    adjustments.append(
        {
            "signal": "provenance_richness",
            "points": points,
            "reason": (
                f"{source_reason}; {type_reason}; "
                f"{outcomes_with_provenance_count} outcomes with provenance payload"
            ),
        }
    )

    if code_anchor_count >= 3:
        points = 4
        reason = "3+ code anchors linked to evidence"
    elif code_anchor_count >= 1:
        points = 2
        reason = "at least one code anchor linked to evidence"
    else:
        points = -3
        reason = "no code-anchor support"
    score += points
    adjustments.append({"signal": "code_anchor_support", "points": points, "reason": reason})

    expiry_penalty = min(10, expiry_event_count * 5)
    points = -expiry_penalty
    score += points
    adjustments.append(
        {
            "signal": "expiry_history",
            "points": points,
            "reason": f"{expiry_event_count} expiry events",
        }
    )

    reliability_score = _coarse_reliability_score(score)
    ranking_multiplier = round(0.2 + ((reliability_score / 100.0) * 1.2), 3)
    recommendation = _reliability_recommendation(reliability_score, status)

    return {
        "score": reliability_score,
        "band": _reliability_band(reliability_score),
        "ranking_multiplier": ranking_multiplier,
        "recommendation": recommendation,
        "inputs": {
            "status": status,
            "validation_count": validation_count,
            "success_count": success_count,
            "partial_success_count": partial_success_count,
            "failure_count": failure_count,
            "explicit_failure_count": explicit_failure_count,
            "reverted_count": reverted_count,
            "superseded_count": superseded_count,
            "contradiction_count": contradiction_count,
            "challenged_count": challenged_count,
            "drift_event_count": drift_event_count,
            "expiry_event_count": expiry_event_count,
            "source_link_count": source_link_count,
            "source_episode_count": source_episode_count,
            "source_topic_count": source_topic_count,
            "source_record_count": source_record_count,
            "source_anchor_count": source_anchor_count,
            "outcome_anchor_count": outcome_anchor_count,
            "outcomes_with_provenance_count": outcomes_with_provenance_count,
            "last_observed_at": data.get("last_observed_at"),
            "success_ratio": round(success_ratio, 3) if success_ratio is not None else None,
            "total_outcomes": total_outcomes,
        },
        "adjustments": adjustments,
    }


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
    reliability = claim_reliability_profile(
        data,
        claim_status=str(data.get("claim_status", "active")),
    )

    validation_count = max(0, _coerce_int_metric(data.get("validation_count", 0)))
    success_count = max(0, _coerce_int_metric(data.get("success_count", 0)))
    partial_success_count = max(0, _coerce_int_metric(data.get("partial_success_count", 0)))
    failure_count = max(0, _coerce_int_metric(data.get("failure_count", 0)))
    contradiction_count = max(0, _coerce_int_metric(data.get("contradiction_count", 0)))
    challenged_count = max(0, _coerce_int_metric(data.get("challenged_count", 0)))

    support_weight = success_count + (0.5 * partial_success_count)
    risk_weight = failure_count + contradiction_count + (0.5 * challenged_count)
    density = min(1.0, validation_count / 5.0)
    density_bonus = min(0.35, math.log1p(validation_count) * 0.12)
    support_bonus = min(0.45, support_weight * 0.1)
    risk_penalty = min(0.75, risk_weight * 0.2)
    legacy_reuse_multiplier = max(0.2, 1.0 + density_bonus + support_bonus - risk_penalty)
    reliability_multiplier = coerce_numeric_float(
        reliability.get("ranking_multiplier"),
        default=legacy_reuse_multiplier,
    )
    reuse_multiplier = max(0.2, min(1.5, round((legacy_reuse_multiplier + reliability_multiplier) / 2.0, 3)))

    if validation_count == 0:
        reusability = "unvalidated"
    elif (
        risk_weight == 0
        and support_weight >= 2
        and str(reliability.get("recommendation")) not in {"avoid_reuse", "reuse_with_caution"}
    ):
        reusability = "validated"
    elif (
        risk_weight > support_weight
        or str(reliability.get("recommendation")) == "avoid_reuse"
    ):
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
        "reliability_score": reliability.get("score"),
        "reliability_band": reliability.get("band"),
        "reliability_recommendation": reliability.get("recommendation"),
        "reliability": reliability,
    }


def _clamp_unit(value: object, *, default: float) -> float:
    parsed = coerce_numeric_float(value, default=default)
    return max(0.0, min(1.0, parsed))


def _temporal_validity_score(
    *,
    valid_from: object,
    valid_until: object,
    as_of: str | None,
) -> float:
    reference_dt = _coerce_utc_datetime(as_of) or datetime.now(timezone.utc)
    valid_from_dt = _coerce_utc_datetime(valid_from)
    valid_until_dt = _coerce_utc_datetime(valid_until)

    if valid_from_dt is not None and reference_dt < valid_from_dt:
        return 0.05
    if valid_until_dt is not None and reference_dt >= valid_until_dt:
        return 0.05
    if valid_until_dt is not None:
        remaining_days = (valid_until_dt - reference_dt).total_seconds() / 86400.0
        if remaining_days <= 1:
            return 0.8
        if remaining_days <= 7:
            return 0.9
    return 1.0


def _outcome_support_score(evidence: Mapping[str, object]) -> float:
    validation_count = max(0, _coerce_int_metric(evidence.get("validation_count", 0)))
    success_count = max(0, _coerce_int_metric(evidence.get("success_count", 0)))
    partial_success_count = max(0, _coerce_int_metric(evidence.get("partial_success_count", 0)))
    failure_count = max(0, _coerce_int_metric(evidence.get("failure_count", 0)))
    explicit_failure_count = max(0, _coerce_int_metric(evidence.get("explicit_failure_count", 0)))
    reverted_count = max(0, _coerce_int_metric(evidence.get("reverted_count", 0)))
    superseded_count = max(0, _coerce_int_metric(evidence.get("superseded_count", 0)))

    weighted_success = success_count + (0.5 * partial_success_count)
    negative_outcomes = max(
        failure_count,
        explicit_failure_count + reverted_count + superseded_count,
    )
    if validation_count == 0:
        return 0.35

    success_ratio = weighted_success / max(1.0, weighted_success + negative_outcomes)
    support_density = min(1.0, validation_count / 5.0)
    score = 0.35 + (success_ratio * 0.45) + (support_density * 0.2)
    if negative_outcomes > weighted_success:
        score -= min(0.2, (negative_outcomes - weighted_success) * 0.05)
    return max(0.2, min(1.0, score))


def _drift_challenge_penalty(
    evidence: Mapping[str, object],
    *,
    claim_status: str,
) -> float:
    drift_event_count = max(0, _coerce_int_metric(evidence.get("drift_event_count", 0)))
    challenged_count = max(0, _coerce_int_metric(evidence.get("challenged_count", 0)))
    contradiction_count = max(0, _coerce_int_metric(evidence.get("contradiction_count", 0)))

    status_penalty = 0.0
    if claim_status == "challenged":
        status_penalty = 0.22
    elif claim_status == "expired":
        status_penalty = 0.35

    event_penalty = min(
        0.45,
        (drift_event_count * 0.07)
        + (challenged_count * 0.05)
        + (contradiction_count * 0.03),
    )
    return max(0.25, 1.0 - (status_penalty + event_penalty))


def claim_query_rank_profile(
    *,
    semantic_similarity: float,
    keyword_relevance: float,
    phrase_match: float,
    confidence: float,
    reliability: Mapping[str, object] | None,
    evidence: Mapping[str, object] | None,
    claim_status: str | None,
    valid_from: object,
    valid_until: object,
    as_of: str | None,
    semantic_weight: float,
    keyword_weight: float,
    strategy_reuse_multiplier: float | None = None,
) -> dict[str, object]:
    """Compose an explainable, trust-aware ranking score for claim retrieval."""
    reliability_payload = dict(reliability or {})
    evidence_payload = dict(evidence or {})
    normalized_status = _normalize_claim_status(claim_status)

    semantic = _clamp_unit(semantic_similarity, default=0.0)
    keyword = _clamp_unit(keyword_relevance, default=0.0)
    phrase = _clamp_unit(phrase_match, default=0.0)
    lexical_relevance = max(keyword, phrase * 0.85)

    bounded_confidence = max(0.0, min(1.0, confidence))
    confidence_factor = 0.5 + (0.5 * bounded_confidence)
    base_relevance = (
        (semantic * semantic_weight) + (lexical_relevance * keyword_weight)
    ) * confidence_factor

    reliability_score = _clamp_unit(
        coerce_numeric_float(reliability_payload.get("score"), default=50.0) / 100.0,
        default=0.5,
    )
    reliability_multiplier = max(
        0.2,
        min(
            1.5,
            coerce_numeric_float(reliability_payload.get("ranking_multiplier"), default=1.0),
        ),
    )
    temporal_validity = _temporal_validity_score(
        valid_from=valid_from,
        valid_until=valid_until,
        as_of=as_of,
    )
    outcome_support = _outcome_support_score(evidence_payload)
    drift_challenge_penalty = _drift_challenge_penalty(
        evidence_payload,
        claim_status=normalized_status,
    )

    weights = {
        "base_relevance": 0.55,
        "reliability": 0.2,
        "temporal_validity": 0.1,
        "outcome_support": 0.15,
    }
    composite_score = (
        (base_relevance * weights["base_relevance"])
        + (reliability_score * weights["reliability"])
        + (temporal_validity * weights["temporal_validity"])
        + (outcome_support * weights["outcome_support"])
    )
    composite_score *= reliability_multiplier
    composite_score *= drift_challenge_penalty

    strategy_multiplier = 1.0
    if strategy_reuse_multiplier is not None:
        strategy_multiplier = max(0.2, min(1.5, strategy_reuse_multiplier))
        composite_score *= strategy_multiplier

    return {
        "score": round(max(0.0, composite_score), 3),
        "weights": weights,
        "components": {
            "semantic_similarity": round(semantic, 3),
            "keyword_relevance": round(keyword, 3),
            "phrase_match": round(phrase, 3),
            "lexical_relevance": round(lexical_relevance, 3),
            "base_relevance": round(base_relevance, 3),
            "confidence_factor": round(confidence_factor, 3),
            "reliability_score": round(reliability_score, 3),
            "reliability_multiplier": round(reliability_multiplier, 3),
            "temporal_validity": round(temporal_validity, 3),
            "outcome_support": round(outcome_support, 3),
            "drift_challenge_penalty": round(drift_challenge_penalty, 3),
            "strategy_multiplier": round(strategy_multiplier, 3),
        },
    }


__all__ = [
    "claim_reliability_profile",
    "claim_query_rank_profile",
    "coerce_numeric_float",
    "filter_claims_for_scope",
    "matches_scope_filter",
    "parse_claim_payload",
    "strategy_reuse_profile",
]
