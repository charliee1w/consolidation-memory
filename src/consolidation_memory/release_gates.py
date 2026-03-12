"""Release gate validation helpers for novelty evidence.

These checks are used by CI and release tooling to enforce the
release policy in docs/RELEASE_GATES.md with fail-closed behavior.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from consolidation_memory.utils import parse_datetime

REQUIRED_TOP_LEVEL_FIELDS = (
    "benchmark",
    "run_id",
    "mode",
    "generated_at",
    "sections",
    "overall_pass",
)

REQUIRED_SECTION_FIELDS = (
    "aligned_metric_section",
    "thresholds",
    "measured",
    "pass",
)


def _parse_required_bool(field_name: str, raw: Any) -> tuple[bool, str | None]:
    if isinstance(raw, bool):
        return raw, None
    return False, f"{field_name} must be a boolean"


def _parse_timestamp(raw: Any) -> datetime | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        parsed = parse_datetime(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _section_statuses(sections: Any) -> tuple[dict[str, bool], list[str]]:
    if not isinstance(sections, dict) or not sections:
        return {}, ["sections must be a non-empty object"]

    statuses: dict[str, bool] = {}
    errors: list[str] = []
    for name, section in sections.items():
        if not isinstance(section, dict):
            errors.append(f"section '{name}' must be an object")
            continue

        missing = [field for field in REQUIRED_SECTION_FIELDS if field not in section]
        if missing:
            errors.append(f"section '{name}' missing required fields: {', '.join(missing)}")
            continue

        section_pass, error = _parse_required_bool(f"section '{name}'.pass", section["pass"])
        if error is not None:
            errors.append(error)
            continue
        statuses[str(name)] = section_pass
    return statuses, errors


def evaluate_release_gates(
    novelty_results: dict[str, Any],
    *,
    max_age_days: int = 7,
    required_mode: str = "full",
    scope_alignment_pass: bool,
    scope_alignment_note: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Evaluate release gates against novelty benchmark evidence.

    Returns a normalized report with per-gate pass/fail signals and a
    final `overall_pass` flag.
    """

    errors: list[str] = []
    now_utc = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)

    missing_top_level = [field for field in REQUIRED_TOP_LEVEL_FIELDS if field not in novelty_results]
    if missing_top_level:
        errors.append(f"missing top-level fields: {', '.join(missing_top_level)}")

    run_id = novelty_results.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        errors.append("run_id must be a non-empty string")

    generated_at = novelty_results.get("generated_at")
    generated_dt = _parse_timestamp(generated_at)
    if generated_dt is None:
        errors.append("generated_at must be a valid ISO datetime string")

    mode = novelty_results.get("mode")
    mode_gate_pass = mode == required_mode
    if not mode_gate_pass:
        errors.append(f"mode must be '{required_mode}' for release gating")

    section_statuses, section_errors = _section_statuses(novelty_results.get("sections"))
    errors.extend(section_errors)

    overall_pass_flag, overall_pass_error = _parse_required_bool(
        "overall_pass",
        novelty_results.get("overall_pass"),
    )
    if overall_pass_error is not None:
        errors.append(overall_pass_error)
    metrics_gate_pass = bool(section_statuses) and overall_pass_flag and all(section_statuses.values())
    if not metrics_gate_pass:
        errors.append("metric threshold gate failed (overall_pass false or a section failed)")

    completeness_pass = (
        not missing_top_level
        and isinstance(run_id, str)
        and bool(run_id.strip())
        and generated_dt is not None
        and not section_errors
        and bool(section_statuses)
    )
    if not completeness_pass:
        errors.append("evidence completeness gate failed")

    recency_pass = False
    age_seconds: float | None = None
    if generated_dt is not None:
        max_age = timedelta(days=max_age_days)
        age = now_utc - generated_dt
        age_seconds = age.total_seconds()
        recency_pass = timedelta(0) <= age <= max_age
    if not recency_pass:
        errors.append(f"evidence recency gate failed (must be <= {max_age_days} days old)")

    scope_pass = bool(scope_alignment_pass)
    if not scope_pass:
        errors.append("scope alignment gate failed")

    gates = {
        "scope_alignment_gate": {
            "pass": scope_pass,
            "details": scope_alignment_note,
        },
        "metric_threshold_gate": {
            "pass": mode_gate_pass and metrics_gate_pass,
            "required_mode": required_mode,
            "actual_mode": mode,
            "overall_pass_flag": overall_pass_flag,
            "section_statuses": section_statuses,
        },
        "evidence_completeness_gate": {
            "pass": completeness_pass,
            "required_top_level_fields": list(REQUIRED_TOP_LEVEL_FIELDS),
            "required_section_fields": list(REQUIRED_SECTION_FIELDS),
        },
        "evidence_recency_gate": {
            "pass": recency_pass,
            "max_age_days": max_age_days,
            "age_seconds": age_seconds,
        },
    }

    overall = bool(
        scope_pass
        and mode_gate_pass
        and metrics_gate_pass
        and completeness_pass
        and recency_pass
    )
    return {
        "overall_pass": overall,
        "gates": gates,
        "errors": errors,
        "evidence": {
            "benchmark": novelty_results.get("benchmark"),
            "benchmark_run_id": run_id,
            "benchmark_timestamp": generated_at,
            "raw_metric_outputs": novelty_results,
            "computed_section_pass_fail": section_statuses,
            "evaluated_at": now_utc.isoformat(),
        },
    }
