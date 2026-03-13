"""Tests for shared query-time semantic helpers."""

from __future__ import annotations

import pytest

from consolidation_memory.query_semantics import (
    coerce_numeric_float,
    strategy_reuse_profile,
)


def test_strategy_reuse_profile_coerces_invalid_metric_values_to_safe_defaults():
    profile = strategy_reuse_profile(
        {
            "validation_count": object(),
            "success_count": "3",
            "partial_success_count": "2.0",
            "failure_count": "bad",
            "contradiction_count": None,
            "challenged_count": float("inf"),
        }
    )

    assert profile["validation_count"] == 0
    assert profile["success_count"] == 3
    assert profile["partial_success_count"] == 2
    assert profile["failure_count"] == 0
    assert profile["contradiction_count"] == 0
    assert profile["challenged_count"] == 0
    assert profile["reuse_multiplier"] >= 0.2


def test_coerce_numeric_float_rejects_invalid_or_non_finite_values():
    assert coerce_numeric_float("1.25", default=1.0) == pytest.approx(1.25)
    assert coerce_numeric_float(2, default=1.0) == pytest.approx(2.0)
    assert coerce_numeric_float("nan", default=1.0) == pytest.approx(1.0)
    assert coerce_numeric_float(float("inf"), default=1.0) == pytest.approx(1.0)
    assert coerce_numeric_float(object(), default=1.0) == pytest.approx(1.0)
