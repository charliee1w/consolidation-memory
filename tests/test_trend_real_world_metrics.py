"""Tests for scripts/trend_real_world_metrics.py."""

from __future__ import annotations

import json
import subprocess  # nosec B404
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "trend_real_world_metrics.py"


def _run_trend(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # nosec B603
        [sys.executable, str(SCRIPT), *args],
        cwd=str(ROOT),
        check=False,
        capture_output=True,
        text=True,
    )


class TestTrendRealWorldMetrics:
    def test_trend_against_bundled_baseline(self):
        report = ROOT / "benchmarks" / "results" / "real_world_eval_full.json"
        if not report.is_file():
            import pytest

            pytest.skip("full eval report not present")

        result = _run_trend(str(report), "--baseline", "bundled", "--json")
        assert result.returncode == 0, result.stderr
        payload = json.loads(result.stdout)
        assert payload["baseline"] == "bundled:published_metrics.json"
        assert len(payload["rows"]) == 6
        assert "section" in payload["rows"][0]

    def test_trend_markdown_output(self):
        report = ROOT / "benchmarks" / "results" / "real_world_eval_full.json"
        if not report.is_file():
            import pytest

            pytest.skip("full eval report not present")

        result = _run_trend(str(report), str(report))
        assert result.returncode == 0, result.stderr
        assert "# real_world_eval trend" in result.stdout
        assert "| Section |" in result.stdout