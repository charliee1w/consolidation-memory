"""Regression tests for coding-agent benchmark harness."""

from __future__ import annotations

import tempfile
from pathlib import Path

from benchmarks.novelty_eval import _local_embedding_patches
from benchmarks.coding_agent_eval import (
    _reset_eval_environment,
    evaluate_contradiction_visibility,
    evaluate_debug_solution_pipeline,
    evaluate_outcome_informed_ranking,
    evaluate_scope_isolation_under_recall,
    evaluate_stale_fix_suppression_after_drift,
    run_eval,
)


def test_debug_solution_pipeline_passes_quick_sample():
    local_tmp_base = Path.cwd() / ".tmp_coding_agent_eval_runtime"
    local_tmp_base.mkdir(parents=True, exist_ok=True)
    tmp_root = Path(tempfile.mkdtemp(prefix="coding_agent_eval_test_", dir=str(local_tmp_base)))

    _reset_eval_environment(tmp_root)
    with _local_embedding_patches():
        result = evaluate_debug_solution_pipeline(scenario_count=2)

    assert result["pass"] is True
    assert result["measured"]["solution_recall_hit_rate"] == 1.0


def test_scope_isolation_blocks_cross_namespace_recall():
    local_tmp_base = Path.cwd() / ".tmp_coding_agent_eval_runtime"
    local_tmp_base.mkdir(parents=True, exist_ok=True)
    tmp_root = Path(tempfile.mkdtemp(prefix="coding_agent_eval_test_", dir=str(local_tmp_base)))

    _reset_eval_environment(tmp_root)
    with _local_embedding_patches():
        result = evaluate_scope_isolation_under_recall(scope_pairs=2)

    assert result["pass"] is True
    assert result["measured"]["scope_leak_rate"] == 0.0


def test_contradiction_visibility_surfaces_warnings():
    local_tmp_base = Path.cwd() / ".tmp_coding_agent_eval_runtime"
    local_tmp_base.mkdir(parents=True, exist_ok=True)
    tmp_root = Path(tempfile.mkdtemp(prefix="coding_agent_eval_test_", dir=str(local_tmp_base)))

    _reset_eval_environment(tmp_root)
    with _local_embedding_patches():
        result = evaluate_contradiction_visibility(scenario_count=2)

    assert result["pass"] is True
    assert result["measured"]["contradiction_visibility_rate"] == 1.0


def test_outcome_informed_ranking_prefers_backed_claim():
    local_tmp_base = Path.cwd() / ".tmp_coding_agent_eval_runtime"
    local_tmp_base.mkdir(parents=True, exist_ok=True)
    tmp_root = Path(tempfile.mkdtemp(prefix="coding_agent_eval_test_", dir=str(local_tmp_base)))

    _reset_eval_environment(tmp_root)
    with _local_embedding_patches():
        result = evaluate_outcome_informed_ranking(pair_count=2)

    assert result["pass"] is True
    assert result["measured"]["outcome_ranking_win_rate"] == 1.0


def test_stale_fix_suppression_after_drift_quick_sample():
    local_tmp_base = Path.cwd() / ".tmp_coding_agent_eval_runtime"
    local_tmp_base.mkdir(parents=True, exist_ok=True)
    tmp_root = Path(tempfile.mkdtemp(prefix="coding_agent_eval_test_", dir=str(local_tmp_base)))

    _reset_eval_environment(tmp_root)
    with _local_embedding_patches():
        result = evaluate_stale_fix_suppression_after_drift(tmp_root=tmp_root, scenario_count=2)

    assert result["pass"] is True
    assert result["measured"]["stale_fix_leak_rate"] == 0.0


def test_run_eval_quick_mode_overall_pass():
    result = run_eval(mode="quick")
    assert result["benchmark"] == "coding_agent_eval"
    assert result["mode"] == "quick"
    assert result["overall_pass"] is True
    assert len(result["sections"]) == 5