"""Hypothesis competition helpers — keep competing claims with lowered precision."""

from __future__ import annotations

from consolidation_memory.claim_graph import normalize_claim_precision
from consolidation_memory.database import update_claim_precision

_COMPETING_HYPOTHESIS_WARNING = "Competing hypothesis (contradicted peer still visible)"


def competing_hypothesis_precision(current: float | None, factor: float) -> float:
    """Lower persisted precision for a claim entering hypothesis competition."""
    base = normalize_claim_precision(current)
    bounded_factor = max(0.0, min(1.0, float(factor)))
    return normalize_claim_precision(base * bounded_factor)


def apply_competing_hypothesis_precision(
    claim_id: str,
    *,
    current_precision: float | None,
    factor: float,
) -> float:
    """Persist lowered precision for a competing hypothesis claim."""
    lowered = competing_hypothesis_precision(current_precision, factor)
    update_claim_precision(claim_id, lowered)
    return lowered


__all__ = [
    "_COMPETING_HYPOTHESIS_WARNING",
    "apply_competing_hypothesis_precision",
    "competing_hypothesis_precision",
]