#!/usr/bin/env python3
"""Compare two real_world_eval JSON reports and print a trend table.

Usage:
    python scripts/trend_real_world_metrics.py \\
        benchmarks/results/real_world_eval_full.json \\
        benchmarks/results/real_world_eval_full_prior.json

    python scripts/trend_real_world_metrics.py --baseline bundled
"""

from __future__ import annotations

import argparse
import importlib.resources
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from consolidation_memory.ui_ops import summarize_metrics_report  # noqa: E402

_SECTION_ORDER = (
    "live_solution_recall_at_5",
    "live_claim_recall_at_5",
    "challenged_claim_suppression",
    "live_provenance_coverage_on_recall",
    "live_drift_response",
    "memory_health_snapshot",
)


def _load_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _bundled_report() -> dict[str, Any]:
    bundled = importlib.resources.files("consolidation_memory.web") / "published_metrics.json"
    return json.loads(bundled.read_text(encoding="utf-8"))


def _sections_by_key(report: dict[str, Any]) -> dict[str, dict[str, object]]:
    sections = report.get("sections")
    if isinstance(sections, list):
        return {str(section["key"]): section for section in sections if isinstance(section, dict)}
    summary = summarize_metrics_report(report)
    return {str(section["key"]): section for section in summary.get("sections", [])}


def _delta_label(before: float | None, after: float | None) -> str:
    if before is None or after is None:
        return "—"
    delta = after - before
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.1f}pp"


def build_trend_table(
    current: dict[str, Any],
    baseline: dict[str, Any],
) -> list[dict[str, str]]:
    current_sections = _sections_by_key(current)
    baseline_sections = _sections_by_key(baseline)
    rows: list[dict[str, str]] = []

    for key in _SECTION_ORDER:
        cur = current_sections.get(key, {})
        base = baseline_sections.get(key, {})
        title = str(cur.get("title") or base.get("title") or key)
        cur_pct = cur.get("value_pct")
        base_pct = base.get("value_pct")
        cur_label = str(cur.get("value_label") or "—")
        base_label = str(base.get("value_label") or "—")
        cur_pass = "yes" if cur.get("pass") else "no"
        rows.append({
            "section": title,
            "baseline": base_label,
            "current": cur_label,
            "delta": _delta_label(
                float(base_pct) if isinstance(base_pct, (int, float)) else None,
                float(cur_pct) if isinstance(cur_pct, (int, float)) else None,
            ),
            "pass": cur_pass,
            "detail": str(cur.get("detail") or ""),
        })
    return rows


def render_markdown(rows: list[dict[str, str]], *, current_path: str, baseline_path: str) -> str:
    lines = [
        "# real_world_eval trend",
        "",
        f"- **Current:** `{current_path}`",
        f"- **Baseline:** `{baseline_path}`",
        "",
        "| Section | Baseline | Current | Δ | Pass | Detail |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['section']} | {row['baseline']} | {row['current']} | "
            f"{row['delta']} | {row['pass']} | {row['detail']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Trend real_world_eval metrics vs a baseline report")
    parser.add_argument("current", nargs="?", help="Path to the current full eval JSON report")
    parser.add_argument("baseline", nargs="?", help="Path to the baseline eval JSON report")
    parser.add_argument(
        "--baseline",
        dest="baseline_mode",
        choices=["bundled"],
        help="Use the packaged published_metrics.json baseline instead of a file path",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = parser.parse_args()

    if args.current is None:
        default = ROOT / "benchmarks" / "results" / "real_world_eval_full.json"
        if not default.is_file():
            parser.error("current report path required when benchmarks/results/real_world_eval_full.json is missing")
        current_path = default
    else:
        current_path = Path(args.current)

    if args.baseline_mode == "bundled":
        baseline_report = _bundled_report()
        baseline_label = "bundled:published_metrics.json"
    elif args.baseline is None:
        parser.error("baseline report path or --baseline bundled is required")
    else:
        baseline_path = Path(args.baseline)
        baseline_report = _load_report(baseline_path)
        baseline_label = str(baseline_path)

    current_report = _load_report(current_path)
    rows = build_trend_table(current_report, baseline_report)

    if args.json:
        payload = {
            "current": str(current_path),
            "baseline": baseline_label,
            "rows": rows,
            "overall_pass": bool(current_report.get("overall_pass")),
        }
        print(json.dumps(payload, indent=2))
    else:
        print(render_markdown(rows, current_path=str(current_path), baseline_path=baseline_label), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())