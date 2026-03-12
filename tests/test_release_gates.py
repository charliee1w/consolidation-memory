"""Tests for release gate evaluation helpers."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone

from consolidation_memory.release_gates import evaluate_release_gates


def _base_novelty_results() -> dict:
    return {
        "benchmark": "novelty_eval",
        "run_id": "novelty_eval_full_abc123",
        "mode": "full",
        "generated_at": "2026-03-06T00:00:00+00:00",
        "sections": {
            "belief_freshness_after_code_drift": {
                "aligned_metric_section": "1) Belief Freshness After Code Drift",
                "thresholds": {"freshness_after_drift_gte": 0.97},
                "measured": {"freshness_after_drift": 1.0},
                "pass": True,
            },
            "contradiction_evolution": {
                "aligned_metric_section": "2) Contradiction Resolution Latency",
                "thresholds": {"median_latency_seconds_lte": 90.0},
                "measured": {"median_latency_seconds": 42.0},
                "pass": True,
            },
        },
        "overall_pass": True,
    }


def test_release_gates_pass_with_fresh_full_evidence():
    results = _base_novelty_results()
    report = evaluate_release_gates(
        novelty_results=results,
        max_age_days=7,
        required_mode="full",
        scope_alignment_pass=True,
        scope_alignment_note="matched wedge use-case",
        now=datetime(2026, 3, 7, tzinfo=timezone.utc),
    )

    assert report["overall_pass"] is True
    assert all(bool(gate["pass"]) for gate in report["gates"].values())
    assert report["evidence"]["benchmark_run_id"] == "novelty_eval_full_abc123"


def test_release_gates_fail_on_stale_evidence():
    results = _base_novelty_results()
    report = evaluate_release_gates(
        novelty_results=results,
        max_age_days=7,
        required_mode="full",
        scope_alignment_pass=True,
        scope_alignment_note="matched wedge use-case",
        now=datetime(2026, 3, 20, tzinfo=timezone.utc),
    )

    assert report["overall_pass"] is False
    assert report["gates"]["evidence_recency_gate"]["pass"] is False


def test_release_gates_fail_on_wrong_mode():
    results = _base_novelty_results()
    results["mode"] = "quick"
    report = evaluate_release_gates(
        novelty_results=results,
        max_age_days=7,
        required_mode="full",
        scope_alignment_pass=True,
        scope_alignment_note="matched wedge use-case",
        now=datetime(2026, 3, 7, tzinfo=timezone.utc),
    )

    assert report["overall_pass"] is False
    assert report["gates"]["metric_threshold_gate"]["pass"] is False


def test_release_gates_fail_on_missing_section_fields():
    results = _base_novelty_results()
    broken = deepcopy(results)
    del broken["sections"]["belief_freshness_after_code_drift"]["measured"]

    report = evaluate_release_gates(
        novelty_results=broken,
        max_age_days=7,
        required_mode="full",
        scope_alignment_pass=True,
        scope_alignment_note="matched wedge use-case",
        now=datetime(2026, 3, 7, tzinfo=timezone.utc),
    )

    assert report["overall_pass"] is False
    assert report["gates"]["evidence_completeness_gate"]["pass"] is False


def test_release_gates_fail_closed_on_string_booleans():
    results = _base_novelty_results()
    results["overall_pass"] = "false"
    results["sections"]["belief_freshness_after_code_drift"]["pass"] = "false"

    report = evaluate_release_gates(
        novelty_results=results,
        max_age_days=7,
        required_mode="full",
        scope_alignment_pass=True,
        scope_alignment_note="matched wedge use-case",
        now=datetime(2026, 3, 7, tzinfo=timezone.utc),
    )

    assert report["overall_pass"] is False
    assert report["gates"]["metric_threshold_gate"]["pass"] is False
    assert any("must be a boolean" in error for error in report["errors"])
