"""Operational health, fix-it flows, and metrics for the browser UI."""

from __future__ import annotations

import importlib.resources
import json
from pathlib import Path
from typing import Any, cast

from consolidation_memory import __version__
from consolidation_memory.desktop_backend import build_health_snapshot
from consolidation_memory.dashboard_data import DashboardData
from consolidation_memory.setup_service import assess_setup_status
from consolidation_memory.types import EpisodicBufferStats, HealthStatus


_METRIC_SECTION_ORDER = (
    "live_solution_recall_at_5",
    "live_claim_recall_at_5",
    "challenged_claim_suppression",
    "live_provenance_coverage_on_recall",
    "live_drift_response",
    "memory_health_snapshot",
)


def _severity_for_issue(message: str) -> str:
    lowered = message.lower()
    if "unreachable" in lowered or "failed" in lowered:
        return "error"
    if "stale" in lowered or "ago" in lowered or "backlog" in lowered:
        return "warning"
    return "info"


def _fix_action_for_issue(message: str) -> str | None:
    lowered = message.lower()
    if "unreachable" in lowered:
        return None
    if "consolidation" in lowered and ("ago" in lowered or "failed" in lowered):
        return "consolidate"
    if "tombstone" in lowered:
        return "reindex"
    if "consistency" in lowered:
        return "consolidate"
    return None


def collect_ops_warnings() -> list[dict[str, object]]:
    """Build actionable warnings from lightweight status and cache probes."""
    from consolidation_memory.client import MemoryClient
    from consolidation_memory.tool_adapter import recall_knowledge_cache_ready

    warnings: list[dict[str, object]] = []

    if not recall_knowledge_cache_ready():
        warnings.append({
            "id": "embedding_cache_cold",
            "severity": "warning",
            "message": "Record embedding cache is still cold; recall may omit knowledge until warmed.",
            "fix_action": "warmup",
            "fix_label": "Warm caches",
        })

    client = MemoryClient(auto_consolidate=False)
    try:
        status = client.status(lightweight=True)
    finally:
        client.close()

    from consolidation_memory.daemon_service import daemon_status

    daemon = daemon_status()
    if not daemon.get("running"):
        warnings.append({
            "id": "maintenance_daemon_stopped",
            "severity": "warning",
            "message": (
                "Background maintenance daemon is not running; "
                "episodes may not consolidate until you start it or run consolidation manually."
            ),
            "fix_action": "daemon_install",
            "fix_label": "Enable background maintenance",
        })

    health = cast(HealthStatus, status.health or {})
    for issue in health.get("issues", []):
        if not isinstance(issue, str) or not issue.strip():
            continue
        fix_action = _fix_action_for_issue(issue)
        warnings.append({
            "id": f"health_{len(warnings)}",
            "severity": _severity_for_issue(issue),
            "message": issue,
            "fix_action": fix_action,
            "fix_label": {
                "consolidate": "Run consolidation",
                "reindex": "Reindex episodes",
                "warmup": "Warm caches",
                "daemon_install": "Enable background maintenance",
            }.get(fix_action or "", None),
        })

    episodic = cast(EpisodicBufferStats, status.episodic_buffer or {})
    pending = int(episodic.get("pending_consolidation", 0) or 0)
    total = int(episodic.get("total", 0) or 0)
    if total > 0:
        backlog_ratio = pending / total
        if backlog_ratio > 0.15 and not any(w.get("id") == "consolidation_backlog" for w in warnings):
            warnings.append({
                "id": "consolidation_backlog",
                "severity": "warning",
                "message": (
                    f"{pending} of {total} episodes await consolidation "
                    f"({backlog_ratio:.0%} backlog)."
                ),
                "fix_action": "consolidate",
                "fix_label": "Run consolidation",
            })

    return warnings


def build_ops_overview(data: DashboardData | None = None) -> dict[str, object]:
    """Extended overview payload for the browser UI health panel."""
    dashboard = data or DashboardData()
    stats = dashboard.get_stats()
    faiss = dashboard.get_faiss_stats()
    health, health_note = build_health_snapshot(stats.get("last_consolidation"))
    warnings = collect_ops_warnings()

    setup = assess_setup_status()
    from consolidation_memory.config import get_active_project

    worst_severity = "ok"
    for warning in warnings:
        severity = str(warning.get("severity") or "info")
        if severity == "error":
            worst_severity = "error"
            break
        if severity == "warning" and worst_severity == "ok":
            worst_severity = "warning"

    if worst_severity != "ok":
        health = worst_severity if worst_severity == "error" else "warning"
        if warnings:
            health_note = str(warnings[0].get("message") or health_note)

    from consolidation_memory.daemon_service import daemon_status

    daemon_state = daemon_status()

    return {
        "version": __version__,
        "project": get_active_project(),
        "health": health,
        "health_note": health_note,
        "needs_setup": setup.get("needs_setup", False),
        "stats": stats,
        "faiss": faiss,
        "warnings": warnings,
        "fix_actions": _unique_fix_actions(warnings),
        "maintenance_daemon": daemon_state,
    }


def _unique_fix_actions(warnings: list[dict[str, object]]) -> list[dict[str, object]]:
    seen: set[str] = set()
    actions: list[dict[str, object]] = []
    for warning in warnings:
        action = warning.get("fix_action")
        label = warning.get("fix_label")
        if not action or not label or action in seen:
            continue
        seen.add(str(action))
        actions.append({"action": action, "label": label})
    return actions


def _metric_display(section_key: str, section: dict[str, Any]) -> dict[str, object]:
    measured = section.get("measured") or {}
    thresholds = section.get("thresholds") or {}
    title = str(section.get("aligned_metric_section") or section_key)
    value_label = ""
    value_pct: float | None = None
    detail = ""

    if section_key.endswith("recall_at_5") or section_key == "challenged_claim_suppression":
        ratio_keys = (
            "live_solution_recall_at_5",
            "live_claim_recall_at_5",
            "challenged_suppression_rate",
        )
        for key in ratio_keys:
            raw = measured.get(key)
            if isinstance(raw, (int, float)) and 0 <= float(raw) <= 1:
                value_pct = float(raw) * 100.0
                break
        hits = (
            measured.get("recall_hits")
            or measured.get("suppressed_claims")
            or measured.get("suppressed_count")
        )
        total = (
            measured.get("cases_evaluated")
            or measured.get("claims_evaluated")
            or measured.get("evaluated_count")
        )
        if hits is not None and total is not None:
            detail = f"{hits}/{total}"
        value_label = f"{value_pct:.1f}%" if value_pct is not None else "—"
    elif section_key == "live_provenance_coverage_on_recall":
        ratio = measured.get("live_provenance_coverage")
        if isinstance(ratio, (int, float)):
            value_pct = float(ratio) * 100.0
            value_label = f"{value_pct:.1f}%"
        covered = measured.get("claims_with_complete_provenance")
        total = measured.get("claims_returned")
        if covered is not None and total is not None:
            detail = f"{covered}/{total}"
    elif section_key == "live_drift_response":
        rate = measured.get("drift_challenge_rate")
        if isinstance(rate, (int, float)):
            value_pct = float(rate) * 100.0
            value_label = f"{value_pct:.1f}%"
        impacted = measured.get("impacted_claim_count")
        challenged = measured.get("challenged_outcome_count")
        if impacted is not None:
            detail = f"{challenged or 0}/{impacted} challenged"
    elif section_key == "memory_health_snapshot":
        value_label = str(measured.get("health_status") or "—")
        backlog = measured.get("consolidation_backlog_ratio")
        if isinstance(backlog, (int, float)):
            value_pct = float(backlog) * 100.0
            detail = f"backlog {value_pct:.1f}%"
    else:
        value_label = "—"

    threshold_label = ""
    for key, raw in thresholds.items():
        if isinstance(raw, (int, float)) and 0 < float(raw) <= 1:
            threshold_label = f"≥ {float(raw) * 100:.0f}%"
            break
        if isinstance(raw, set):
            threshold_label = ", ".join(sorted(str(item) for item in raw))
            break
        if isinstance(raw, str) and raw:
            threshold_label = raw
            break

    return {
        "key": section_key,
        "title": title,
        "value_label": value_label,
        "value_pct": value_pct,
        "detail": detail,
        "threshold_label": threshold_label,
        "pass": bool(section.get("pass")),
    }


def summarize_metrics_report(report: dict[str, Any]) -> dict[str, object]:
    """Trim a real_world_eval JSON report for chart rendering."""
    sections_raw = report.get("sections") or {}
    sections: list[dict[str, object]] = []
    for key in _METRIC_SECTION_ORDER:
        raw = sections_raw.get(key)
        if isinstance(raw, dict):
            sections.append(_metric_display(key, raw))

    return {
        "benchmark": report.get("benchmark", "real_world_eval"),
        "mode": report.get("mode"),
        "generated_at": report.get("generated_at"),
        "data_source": report.get("data_source"),
        "overall_pass": bool(report.get("overall_pass")),
        "sections": sections,
    }


def _user_metrics_candidates() -> list[Path]:
    from consolidation_memory.config import get_config

    cfg = get_config()
    candidates = [
        cfg.DATA_DIR / "real_world_eval_full.json",
        cfg.DATA_DIR / "metrics" / "real_world_eval_full.json",
        Path("benchmarks/results/real_world_eval_full.json"),
    ]
    return candidates


def _metrics_payload_from_report(report: dict[str, Any], *, source: str) -> dict[str, object]:
    """Normalize raw eval JSON or pre-summarized bundle for the UI."""
    sections = report.get("sections")
    if isinstance(sections, list):
        payload: dict[str, object] = {
            "benchmark": report.get("benchmark", "real_world_eval"),
            "mode": report.get("mode"),
            "generated_at": report.get("generated_at"),
            "data_source": report.get("data_source"),
            "overall_pass": report.get("overall_pass"),
            "sections": sections,
            "source": source,
        }
        return payload
    summary = summarize_metrics_report(report)
    summary["source"] = source
    return summary


def load_metrics_for_ui() -> dict[str, object]:
    """Load the best available real_world_eval summary for the UI."""
    for candidate in _user_metrics_candidates():
        if candidate.is_file():
            try:
                report = json.loads(candidate.read_text(encoding="utf-8"))
                return _metrics_payload_from_report(report, source=str(candidate))
            except (OSError, json.JSONDecodeError, TypeError, AttributeError):
                continue

    bundled = importlib.resources.files("consolidation_memory.web") / "published_metrics.json"
    try:
        report = json.loads(bundled.read_text(encoding="utf-8"))
        return _metrics_payload_from_report(report, source="bundled")
    except (OSError, json.JSONDecodeError, TypeError, AttributeError):
        return {
            "benchmark": "real_world_eval",
            "overall_pass": None,
            "sections": [],
            "source": "unavailable",
            "message": "No metrics report found. Run: python -m benchmarks.real_world_eval --mode full",
        }


def build_published_metrics_bundle(full_report_path: Path) -> dict[str, object]:
    """Build a compact metrics bundle from a full eval report (maintainer helper)."""
    report = json.loads(full_report_path.read_text(encoding="utf-8"))
    trimmed = {
        "benchmark": report.get("benchmark"),
        "mode": report.get("mode"),
        "generated_at": report.get("generated_at"),
        "data_source": report.get("data_source"),
        "overall_pass": report.get("overall_pass"),
        "sections": report.get("sections", {}),
    }
    return summarize_metrics_report(trimmed)