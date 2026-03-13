"""Tests for shared query-time semantic helpers."""

from __future__ import annotations

import pytest

from consolidation_memory.query_semantics import (
    claim_reliability_profile,
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


def test_claim_reliability_high_support_scores_above_low_support():
    high_support = claim_reliability_profile(
        {
            "validation_count": 6,
            "success_count": 5,
            "partial_success_count": 1,
            "failure_count": 0,
            "explicit_failure_count": 0,
            "reverted_count": 0,
            "superseded_count": 0,
            "source_link_count": 3,
            "source_episode_count": 1,
            "source_topic_count": 1,
            "source_record_count": 1,
            "source_anchor_count": 2,
            "outcome_anchor_count": 1,
            "outcomes_with_provenance_count": 2,
            "last_observed_at": "2026-03-12T00:00:00+00:00",
        },
        claim_status="active",
        as_of="2026-03-13T00:00:00+00:00",
    )
    low_support = claim_reliability_profile(
        {
            "validation_count": 0,
            "success_count": 0,
            "partial_success_count": 0,
            "failure_count": 0,
            "source_link_count": 0,
            "source_episode_count": 0,
            "source_topic_count": 0,
            "source_record_count": 0,
            "source_anchor_count": 0,
            "outcome_anchor_count": 0,
            "outcomes_with_provenance_count": 0,
            "last_observed_at": None,
        },
        claim_status="active",
        as_of="2026-03-13T00:00:00+00:00",
    )

    assert high_support["score"] > low_support["score"]


def test_claim_reliability_failures_and_reversions_lower_trust():
    baseline = claim_reliability_profile(
        {
            "validation_count": 4,
            "success_count": 3,
            "partial_success_count": 1,
            "failure_count": 0,
            "explicit_failure_count": 0,
            "reverted_count": 0,
            "superseded_count": 0,
            "source_link_count": 2,
            "source_episode_count": 1,
            "source_topic_count": 1,
            "source_record_count": 0,
            "source_anchor_count": 1,
            "outcome_anchor_count": 1,
            "outcomes_with_provenance_count": 1,
            "last_observed_at": "2026-03-12T00:00:00+00:00",
        },
        claim_status="active",
        as_of="2026-03-13T00:00:00+00:00",
    )
    degraded = claim_reliability_profile(
        {
            "validation_count": 4,
            "success_count": 1,
            "partial_success_count": 0,
            "failure_count": 3,
            "explicit_failure_count": 2,
            "reverted_count": 1,
            "superseded_count": 1,
            "source_link_count": 2,
            "source_episode_count": 1,
            "source_topic_count": 1,
            "source_record_count": 0,
            "source_anchor_count": 1,
            "outcome_anchor_count": 1,
            "outcomes_with_provenance_count": 1,
            "last_observed_at": "2026-03-12T00:00:00+00:00",
        },
        claim_status="active",
        as_of="2026-03-13T00:00:00+00:00",
    )

    assert degraded["score"] < baseline["score"]


def test_claim_reliability_penalizes_challenged_and_drifted_items():
    stable = claim_reliability_profile(
        {
            "validation_count": 3,
            "success_count": 3,
            "source_link_count": 2,
            "source_episode_count": 1,
            "source_topic_count": 1,
            "source_record_count": 0,
            "source_anchor_count": 1,
            "outcome_anchor_count": 1,
            "outcomes_with_provenance_count": 1,
            "last_observed_at": "2026-03-12T00:00:00+00:00",
        },
        claim_status="active",
        as_of="2026-03-13T00:00:00+00:00",
    )
    challenged = claim_reliability_profile(
        {
            "validation_count": 3,
            "success_count": 3,
            "challenged_count": 1,
            "drift_event_count": 2,
            "source_link_count": 2,
            "source_episode_count": 1,
            "source_topic_count": 1,
            "source_record_count": 0,
            "source_anchor_count": 1,
            "outcome_anchor_count": 1,
            "outcomes_with_provenance_count": 1,
            "last_observed_at": "2026-03-12T00:00:00+00:00",
        },
        claim_status="challenged",
        as_of="2026-03-13T00:00:00+00:00",
    )

    assert challenged["score"] < stable["score"]
    assert challenged["recommendation"] in {"reuse_with_caution", "avoid_reuse"}


def test_unsupported_strategy_profiles_rank_lower_for_reuse():
    supported = strategy_reuse_profile(
        {
            "validation_count": 3,
            "success_count": 3,
            "partial_success_count": 0,
            "failure_count": 0,
            "source_link_count": 2,
            "source_episode_count": 1,
            "source_topic_count": 1,
            "source_record_count": 0,
            "source_anchor_count": 1,
            "outcome_anchor_count": 1,
            "outcomes_with_provenance_count": 1,
            "last_observed_at": "2026-03-12T00:00:00+00:00",
            "claim_status": "active",
        }
    )
    unsupported = strategy_reuse_profile(
        {
            "validation_count": 0,
            "success_count": 0,
            "partial_success_count": 0,
            "failure_count": 0,
            "source_link_count": 0,
            "source_episode_count": 0,
            "source_topic_count": 0,
            "source_record_count": 0,
            "source_anchor_count": 0,
            "outcome_anchor_count": 0,
            "outcomes_with_provenance_count": 0,
            "last_observed_at": None,
            "claim_status": "active",
        }
    )

    assert supported["reuse_multiplier"] > unsupported["reuse_multiplier"]


def test_claim_reliability_explanation_payload_is_stable_and_ordered():
    profile = claim_reliability_profile(
        {
            "validation_count": 2,
            "success_count": 1,
            "partial_success_count": 1,
            "failure_count": 0,
            "source_link_count": 1,
            "source_episode_count": 1,
            "source_topic_count": 0,
            "source_record_count": 0,
            "source_anchor_count": 1,
            "outcome_anchor_count": 0,
            "outcomes_with_provenance_count": 1,
            "last_observed_at": "2026-03-10T00:00:00+00:00",
        },
        claim_status="active",
        as_of="2026-03-13T00:00:00+00:00",
    )

    assert set(profile.keys()) == {"score", "band", "ranking_multiplier", "recommendation", "inputs", "adjustments"}
    assert [adj["signal"] for adj in profile["adjustments"]] == [
        "supporting_outcomes",
        "success_ratio",
        "failures_and_reversions",
        "contradictions",
        "drift_and_challenge_state",
        "recency",
        "provenance_richness",
        "code_anchor_support",
        "expiry_history",
    ]
    assert profile["inputs"]["validation_count"] == 2
    assert profile["inputs"]["last_observed_at"] == "2026-03-10T00:00:00+00:00"
