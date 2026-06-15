"""Tests for browser UI operational helpers."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from consolidation_memory.ui_ops import (
    build_ops_overview,
    load_metrics_for_ui,
    summarize_metrics_report,
)


class TestSummarizeMetricsReport:
    @pytest.fixture
    def sample_report(self) -> dict:
        path = Path("benchmarks/results/real_world_eval_full.json")
        if not path.is_file():
            pytest.skip("full eval report not present")
        return json.loads(path.read_text(encoding="utf-8"))

    def test_summarize_includes_all_sections(self, sample_report):
        summary = summarize_metrics_report(sample_report)
        assert summary["benchmark"] == "real_world_eval"
        assert summary["overall_pass"] is True
        keys = [s["key"] for s in summary["sections"]]
        assert "live_solution_recall_at_5" in keys
        assert "memory_health_snapshot" in keys
        assert len(summary["sections"]) == 6

    def test_recall_section_has_pct_and_detail(self, sample_report):
        summary = summarize_metrics_report(sample_report)
        recall = next(s for s in summary["sections"] if s["key"] == "live_solution_recall_at_5")
        assert recall["value_pct"] is not None
        assert recall["value_pct"] > 0
        assert recall["detail"] == "98/120"
        assert recall["pass"] is True

    def test_minimal_report(self):
        report = {
            "benchmark": "real_world_eval",
            "overall_pass": False,
            "sections": {
                "live_claim_recall_at_5": {
                    "aligned_metric_section": "Claim recall",
                    "measured": {"live_claim_recall_at_5": 0.5, "recall_hits": 4, "cases_evaluated": 8},
                    "thresholds": {"live_claim_recall_at_5_gte": 0.4},
                    "pass": True,
                }
            },
        }
        summary = summarize_metrics_report(report)
        assert len(summary["sections"]) == 1
        assert summary["sections"][0]["value_label"] == "50.0%"


class TestBuildOpsOverview:
    def test_overview_includes_warnings_and_fix_actions(self):
        fake_warnings = [
            {
                "id": "embedding_cache_cold",
                "severity": "warning",
                "message": "Record embedding cache is still cold",
                "fix_action": "warmup",
                "fix_label": "Warm caches",
            },
            {
                "id": "consolidation_backlog",
                "severity": "warning",
                "message": "10 of 50 episodes await consolidation",
                "fix_action": "consolidate",
                "fix_label": "Run consolidation",
            },
        ]
        with patch("consolidation_memory.ui_ops.collect_ops_warnings", return_value=fake_warnings):
            with patch("consolidation_memory.ui_ops.assess_setup_status", return_value={"needs_setup": False}):
                with patch("consolidation_memory.ui_ops.build_health_snapshot", return_value=("ok", "Ready")):
                    overview = build_ops_overview()

        assert overview["health"] == "warning"
        assert len(overview["warnings"]) == 2
        actions = {a["action"] for a in overview["fix_actions"]}
        assert actions == {"warmup", "consolidate"}


class TestLoadMetricsForUi:
    def test_loads_bundled_metrics_when_no_user_report(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            "consolidation_memory.ui_ops._user_metrics_candidates",
            lambda: [tmp_path / "missing.json"],
        )
        result = load_metrics_for_ui()
        assert result.get("source") in ("bundled", "unavailable")
        if result["source"] == "bundled":
            assert isinstance(result.get("sections"), list)
            assert result.get("overall_pass") is not None