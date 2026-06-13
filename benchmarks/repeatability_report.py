"""Aggregate repeated benchmark runs with stability and confidence statistics.

Runs live and synthetic harnesses multiple times, then reports mean/std/min/max,
Wilson score intervals, and bootstrap confidence bounds for headline metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_PATH = _REPO_ROOT / "src"
if str(_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(_SRC_PATH))


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def wilson_score_ci(successes: int, total: int, z: float = 1.96) -> dict[str, float]:
    if total <= 0:
        return {"low": 0.0, "high": 0.0, "point": 0.0}
    p = successes / total
    z2 = z * z
    denom = 1.0 + z2 / total
    centre = (p + z2 / (2.0 * total)) / denom
    margin = (z / denom) * math.sqrt((p * (1.0 - p) / total) + (z2 / (4.0 * total * total)))
    return {
        "point": round(p, 6),
        "low": round(max(0.0, centre - margin), 6),
        "high": round(min(1.0, centre + margin), 6),
    }


def bootstrap_ci(
    outcomes: list[bool],
    *,
    iterations: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    if not outcomes:
        return {"low": 0.0, "high": 0.0, "point": 0.0}
    rng = random.Random(seed)
    n = len(outcomes)
    rates: list[float] = []
    for _ in range(iterations):
        sample = [outcomes[rng.randrange(n)] for _ in range(n)]
        rates.append(sum(sample) / n)
    rates.sort()
    low_idx = int((alpha / 2.0) * iterations)
    high_idx = int((1.0 - alpha / 2.0) * iterations) - 1
    point = sum(outcomes) / n
    return {
        "point": round(point, 6),
        "low": round(rates[low_idx], 6),
        "high": round(rates[high_idx], 6),
    }


@dataclass(frozen=True)
class RunSummary:
    values: list[float]

    @property
    def count(self) -> int:
        return len(self.values)

    def as_dict(self) -> dict[str, Any]:
        if not self.values:
            return {"count": 0}
        return {
            "count": self.count,
            "mean": round(statistics.fmean(self.values), 6),
            "stdev": round(statistics.pstdev(self.values), 6) if self.count > 1 else 0.0,
            "min": round(min(self.values), 6),
            "max": round(max(self.values), 6),
            "values": [round(v, 6) for v in self.values],
            "stable": self.count <= 1 or statistics.pstdev(self.values) < 1e-9,
        }


def _metric_from_real_world(result: dict[str, Any], section: str, key: str) -> float | None:
    measured = result.get("sections", {}).get(section, {}).get("measured", {})
    value = measured.get(key)
    return float(value) if value is not None else None


def _metric_from_section(result: dict[str, Any], section: str, key: str) -> float | None:
    measured = result.get("sections", {}).get(section, {}).get("measured", {})
    value = measured.get(key)
    return float(value) if value is not None else None


def evaluate_solution_recall_subsample(*, seed: int, sample_size: int) -> dict[str, Any]:
    from consolidation_memory import MemoryClient
    from consolidation_memory.database import ensure_schema, get_connection
    from benchmarks.real_world_eval import (
        _embedding_backend_ready,
        _episode_recall_hit,
        _problem_query_from_content,
    )

    ready, reason = _embedding_backend_ready()
    if not ready:
        return {
            "seed": seed,
            "sample_size": sample_size,
            "evaluated": 0,
            "hits": 0,
            "rate": 0.0,
            "skipped_reason": reason,
            "scenario_hits": [],
        }

    ensure_schema()
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, content
            FROM episodes
            WHERE deleted = 0
              AND content_type = 'solution'
              AND length(content) >= 80
            ORDER BY id
            """
        ).fetchall()

    eligible = [row for row in rows if len(_problem_query_from_content(row["content"])) >= 24]
    rng = random.Random(seed)
    if len(eligible) <= sample_size:
        sample = eligible
    else:
        sample = rng.sample(eligible, sample_size)

    hits = 0
    scenario_hits: list[bool] = []
    client = MemoryClient(auto_consolidate=False)
    try:
        for row in sample:
            query = _problem_query_from_content(row["content"])
            recall = client.recall(query, n_results=10, include_knowledge=True)
            hit = _episode_recall_hit(recall, row["id"])
            scenario_hits.append(hit)
            if hit:
                hits += 1
    finally:
        client.close()

    evaluated = len(scenario_hits)
    rate = hits / evaluated if evaluated else 0.0
    return {
        "seed": seed,
        "sample_size": sample_size,
        "pool_size": len(eligible),
        "evaluated": evaluated,
        "hits": hits,
        "rate": round(rate, 6),
        "scenario_hits": scenario_hits,
    }


def _confidence_block(successes: int, total: int, outcomes: list[bool]) -> dict[str, Any]:
    return {
        "wilson_95": wilson_score_ci(successes, total),
        "bootstrap_95": bootstrap_ci(outcomes, seed=total + successes),
    }


def run_repeatability_report(
    *,
    mode: str,
    deterministic_runs: int,
    subsample_runs: int,
    subsample_size: int,
    subsample_seed_base: int,
    include_synthetic: bool,
    repo_path: Path | None,
) -> dict[str, Any]:
    from benchmarks.coding_agent_eval import run_eval as run_coding_agent_eval
    from benchmarks.novelty_eval import run_eval as run_novelty_eval
    from benchmarks.real_world_eval import run_eval as run_real_world_eval

    report_id = f"repeatability_{mode}_{uuid.uuid4().hex[:12]}"
    deterministic_results: list[dict[str, Any]] = []
    for i in range(deterministic_runs):
        deterministic_results.append(run_real_world_eval(mode=mode, repo_path=repo_path))

    solution_rates = [
        _metric_from_real_world(r, "live_solution_recall_at_5", "live_solution_recall_at_5")
        for r in deterministic_results
    ]
    claim_rates = [
        _metric_from_real_world(r, "live_claim_recall_at_5", "live_claim_recall_at_5")
        for r in deterministic_results
    ]
    suppression_rates = [
        _metric_from_real_world(r, "challenged_claim_suppression", "challenged_suppression_rate")
        for r in deterministic_results
    ]
    provenance_rates = [
        _metric_from_real_world(
            r,
            "live_provenance_coverage_on_recall",
            "live_provenance_coverage",
        )
        for r in deterministic_results
    ]
    overall_pass = [1.0 if r.get("overall_pass") else 0.0 for r in deterministic_results]

    solution_rates = [v for v in solution_rates if v is not None]
    claim_rates = [v for v in claim_rates if v is not None]
    suppression_rates = [v for v in suppression_rates if v is not None]
    provenance_rates = [v for v in provenance_rates if v is not None]

    reference = deterministic_results[-1]
    solution_scenarios = reference["sections"]["live_solution_recall_at_5"].get("scenarios", [])
    solution_outcomes = [bool(s.get("recall_hit_top5")) for s in solution_scenarios]
    solution_hits = sum(solution_outcomes)
    solution_evaluated = len(solution_outcomes)

    claim_scenarios = reference["sections"]["live_claim_recall_at_5"].get("scenarios", [])
    claim_outcomes = [bool(s.get("recall_hit_top5")) for s in claim_scenarios]
    claim_hits = sum(claim_outcomes)
    claim_evaluated = len(claim_outcomes)

    subsample_rows: list[dict[str, Any]] = []
    subsample_rates: list[float] = []
    pooled_subsample_outcomes: list[bool] = []
    for offset in range(subsample_runs):
        seed = subsample_seed_base + offset
        row = evaluate_solution_recall_subsample(seed=seed, sample_size=subsample_size)
        subsample_rows.append(
            {
                "seed": row["seed"],
                "pool_size": row.get("pool_size"),
                "evaluated": row["evaluated"],
                "hits": row["hits"],
                "rate": row["rate"],
            }
        )
        subsample_rates.append(float(row["rate"]))
        pooled_subsample_outcomes.extend(row.get("scenario_hits", []))

    synthetic: dict[str, Any] = {}
    if include_synthetic:
        coding_results = [run_coding_agent_eval(mode="quick") for _ in range(deterministic_runs)]
        novelty_results = [run_novelty_eval(mode="quick") for _ in range(deterministic_runs)]

        synthetic = {
            "coding_agent_eval_quick": {
                "runs": deterministic_runs,
                "overall_pass": RunSummary(
                    [1.0 if r.get("overall_pass") else 0.0 for r in coding_results]
                ).as_dict(),
                "solution_recall_hit_rate": RunSummary(
                    [
                        float(
                            _metric_from_section(
                                r, "debug_solution_pipeline", "solution_recall_hit_rate"
                            )
                            or 0.0
                        )
                        for r in coding_results
                    ]
                ).as_dict(),
                "stale_fix_suppression_rate": RunSummary(
                    [
                        float(
                            _metric_from_section(
                                r,
                                "stale_fix_suppression_after_drift",
                                "stale_fix_suppression_rate",
                            )
                            or 0.0
                        )
                        for r in coding_results
                    ]
                ).as_dict(),
            },
            "novelty_eval_quick": {
                "runs": deterministic_runs,
                "overall_pass": RunSummary(
                    [1.0 if r.get("overall_pass") else 0.0 for r in novelty_results]
                ).as_dict(),
            },
        }

    return {
        "benchmark": "repeatability_report",
        "report_id": report_id,
        "mode": mode,
        "generated_at": _iso_now(),
        "cloud_dependencies_used": False,
        "methodology": {
            "deterministic_runs": deterministic_runs,
            "subsample_runs": subsample_runs,
            "subsample_size": subsample_size,
            "subsample_seed_base": subsample_seed_base,
            "confidence_methods": ["wilson_95", "bootstrap_95_percentile"],
            "notes": [
                "Deterministic runs use fixed ORDER BY sampling from real_world_eval.",
                "Subsample runs draw random solution episodes from the live pool per seed.",
                "Wilson and bootstrap intervals are computed from per-case hit outcomes.",
            ],
        },
        "real_world_eval": {
            "deterministic": {
                "runs": deterministic_runs,
                "overall_pass": RunSummary(overall_pass).as_dict(),
                "live_solution_recall_at_5": RunSummary(solution_rates).as_dict(),
                "live_claim_recall_at_5": RunSummary(claim_rates).as_dict(),
                "challenged_suppression_rate": RunSummary(suppression_rates).as_dict(),
                "live_provenance_coverage": RunSummary(provenance_rates).as_dict(),
            },
            "reference_run_confidence": {
                "live_solution_recall_at_5": _confidence_block(
                    solution_hits, solution_evaluated, solution_outcomes
                ),
                "live_claim_recall_at_5": _confidence_block(
                    claim_hits, claim_evaluated, claim_outcomes
                ),
            },
            "subsampled_solution_recall": {
                "runs": subsample_runs,
                "sample_size_per_run": subsample_size,
                "rate_summary": RunSummary(subsample_rates).as_dict(),
                "pooled_confidence": _confidence_block(
                    sum(pooled_subsample_outcomes),
                    len(pooled_subsample_outcomes),
                    pooled_subsample_outcomes,
                ),
                "runs_detail": subsample_rows,
            },
        },
        "synthetic_harnesses": synthetic,
        "headline": _build_headline(
            solution_rates,
            claim_rates,
            suppression_rates,
            solution_hits,
            solution_evaluated,
            solution_outcomes,
            claim_hits,
            claim_evaluated,
            claim_outcomes,
            subsample_rates,
        ),
    }


def _build_headline(
    solution_rates: list[float],
    claim_rates: list[float],
    suppression_rates: list[float],
    solution_hits: int,
    solution_evaluated: int,
    solution_outcomes: list[bool],
    claim_hits: int,
    claim_evaluated: int,
    claim_outcomes: list[bool],
    subsample_rates: list[float],
) -> dict[str, Any]:
    sol_ci = _confidence_block(solution_hits, solution_evaluated, solution_outcomes)
    claim_ci = _confidence_block(claim_hits, claim_evaluated, claim_outcomes)
    return {
        "live_solution_recall_at_5": {
            "deterministic_mean": round(statistics.fmean(solution_rates), 6) if solution_rates else None,
            "wilson_95": sol_ci["wilson_95"],
            "bootstrap_95": sol_ci["bootstrap_95"],
            "deterministic_stable": len(solution_rates) <= 1
            or statistics.pstdev(solution_rates) < 1e-9,
        },
        "live_claim_recall_at_5": {
            "deterministic_mean": round(statistics.fmean(claim_rates), 6) if claim_rates else None,
            "wilson_95": claim_ci["wilson_95"],
            "bootstrap_95": claim_ci["bootstrap_95"],
            "deterministic_stable": len(claim_rates) <= 1
            or statistics.pstdev(claim_rates) < 1e-9,
        },
        "challenged_suppression_rate": {
            "deterministic_mean": round(statistics.fmean(suppression_rates), 6)
            if suppression_rates
            else None,
            "deterministic_stable": len(suppression_rates) <= 1
            or statistics.pstdev(suppression_rates) < 1e-9,
        },
        "subsampled_solution_recall_at_5": {
            "mean_across_seeds": round(statistics.fmean(subsample_rates), 6)
            if subsample_rates
            else None,
            "stdev_across_seeds": round(statistics.pstdev(subsample_rates), 6)
            if len(subsample_rates) > 1
            else 0.0,
        },
    }


def save_report(report: dict[str, Any], output_path: Path | None = None) -> Path:
    if output_path is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_path = Path(__file__).parent / "results" / f"repeatability_report_{stamp}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Repeated benchmark aggregation report")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--deterministic-runs", type=int, default=5)
    parser.add_argument("--subsample-runs", type=int, default=8)
    parser.add_argument("--subsample-size", type=int, default=30)
    parser.add_argument("--subsample-seed-base", type=int, default=20260613)
    parser.add_argument("--no-synthetic", action="store_true")
    parser.add_argument("--repo-path", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report = run_repeatability_report(
        mode=args.mode,
        deterministic_runs=args.deterministic_runs,
        subsample_runs=args.subsample_runs,
        subsample_size=args.subsample_size,
        subsample_seed_base=args.subsample_seed_base,
        include_synthetic=not args.no_synthetic,
        repo_path=args.repo_path,
    )
    out = save_report(report, output_path=args.output)
    print(json.dumps({"output_path": str(out), "headline": report["headline"]}, indent=2))


if __name__ == "__main__":
    main()