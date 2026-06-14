"""Regression tests for live-memory benchmark helpers."""

from __future__ import annotations

from benchmarks.real_world_eval import (
    _claim_query_from_payload,
    _drift_challenge_rate,
    _episode_recall_hit,
    _problem_query_from_content,
)


def test_problem_query_strips_fix_section():
    content = (
        "CI failed on novelty gates after threshold drift. "
        "Fix: update docs/NOVELTY_METRICS.md and rerun pytest."
    )
    query = _problem_query_from_content(content)
    assert "Fix:" not in query
    assert "CI failed on novelty gates" in query


def test_claim_query_prefers_problem_field():
    query = _claim_query_from_payload(
        {"problem": "MCP recall timeout under load", "fix": "reload faiss signal"}
    )
    assert query == "MCP recall timeout under load"


def test_drift_challenge_rate_counts_already_challenged_impacts():
    drift = {
        "impacted_claim_ids": ["claim-active", "claim-challenged"],
        "challenged_claim_ids": ["claim-active"],
        "impacts": [
            {"claim_id": "claim-active", "new_status": "challenged"},
            {"claim_id": "claim-challenged", "new_status": "challenged"},
        ],
        "checked_anchors": [{"anchor_type": "path", "anchor_value": "src/app.py"}],
    }
    rate, challenged_outcomes, newly_challenged, impacted = _drift_challenge_rate(drift)
    assert rate == 1.0
    assert challenged_outcomes == 2
    assert newly_challenged == 1
    assert impacted == 2


def test_episode_recall_hit_accepts_claim_provenance_link():
    class _Recall:
        episodes = []
        claims = [{"sources": [{"source_episode_id": "ep-1"}]}]
        records = []

    assert _episode_recall_hit(_Recall(), "ep-1") is True


def test_ci_mode_fixture_eval_passes():
    from benchmarks.real_world_eval import run_eval

    results = run_eval("ci")
    assert results["data_source"] == "ci_fixture_project"
    assert results["overall_pass"] is True