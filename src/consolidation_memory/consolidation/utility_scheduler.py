"""Utility scoring for adaptive consolidation scheduling."""

from __future__ import annotations

from typing import Mapping, TypedDict

UTILITY_WEIGHT_KEYS = (
    "unconsolidated_backlog",
    "recall_miss_fallback",
    "contradiction_spike",
    "challenged_claim_backlog",
    "outcome_failure_rate",
)


class UtilityScoreBreakdown(TypedDict):
    """Computed utility score with normalized components."""

    score: float
    normalized_signals: dict[str, float]
    weighted_components: dict[str, float]


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def compute_utility_score(
    *,
    unconsolidated_backlog: int,
    recall_miss_count: int,
    recall_fallback_count: int,
    contradiction_count: int,
    challenged_claim_backlog: int,
    outcome_failure_rate: float,
    weights: Mapping[str, float],
    backlog_target: int,
    recall_signal_target: int,
    contradiction_target: int,
    challenged_claim_target: int,
) -> UtilityScoreBreakdown:
    """Compute deterministic utility score from scheduling signals."""
    safe_backlog_target = max(1, backlog_target)
    safe_recall_target = max(1, recall_signal_target)
    safe_contradiction_target = max(1, contradiction_target)
    safe_challenged_target = max(1, challenged_claim_target)

    # Count fallback events as stronger misses.
    recall_signal_raw = max(0, recall_miss_count) + 2 * max(0, recall_fallback_count)

    normalized = {
        "unconsolidated_backlog": _clamp01(max(0, unconsolidated_backlog) / safe_backlog_target),
        "recall_miss_fallback": _clamp01(recall_signal_raw / safe_recall_target),
        "contradiction_spike": _clamp01(max(0, contradiction_count) / safe_contradiction_target),
        "challenged_claim_backlog": _clamp01(max(0, challenged_claim_backlog) / safe_challenged_target),
        "outcome_failure_rate": _clamp01(max(0.0, outcome_failure_rate)),
    }

    components = {
        key: round(float(weights.get(key, 0.0)) * normalized[key], 6)
        for key in UTILITY_WEIGHT_KEYS
    }
    score = round(sum(components.values()), 6)

    return {
        "score": score,
        "normalized_signals": normalized,
        "weighted_components": components,
    }


__all__ = ["UTILITY_WEIGHT_KEYS", "UtilityScoreBreakdown", "compute_utility_score"]
