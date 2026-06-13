"""Coding-agent trust benchmark harness.

Measures end-to-end MemoryClient workflows that generic conversational-memory
benchmarks (for example LoCoMo) do not cover:

- debug solution ingest -> consolidate -> recall
- stale fix suppression after git drift
- scope isolation under recall
- contradiction visibility in recall warnings
- outcome-informed claim ranking
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import statistics
import sys
import tempfile
import uuid
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any
# Ensure local src/ takes precedence over any globally installed package.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_PATH = _REPO_ROOT / "src"
if str(_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(_SRC_PATH))

from benchmarks.novelty_eval import (  # noqa: E402
    _iso,
    _local_embedding_patches,
    _prepare_git_repo,
    _utc_now,
)

logger = logging.getLogger("benchmark.coding_agent_eval")


@dataclass(frozen=True)
class EvalModeConfig:
    debug_scenarios: int
    drift_scenarios: int
    scope_pairs: int
    contradiction_scenarios: int
    outcome_pairs: int


MODE_CONFIG: dict[str, EvalModeConfig] = {
    "quick": EvalModeConfig(
        debug_scenarios=3,
        drift_scenarios=4,
        scope_pairs=4,
        contradiction_scenarios=4,
        outcome_pairs=3,
    ),
    "full": EvalModeConfig(
        debug_scenarios=12,
        drift_scenarios=20,
        scope_pairs=16,
        contradiction_scenarios=12,
        outcome_pairs=10,
    ),
}


def _reset_eval_environment(tmp_root: Path) -> None:
    from consolidation_memory import database, record_cache, topic_cache
    from consolidation_memory.backends import reset_backends
    from consolidation_memory.config import get_config, reset_config
    from consolidation_memory.database import ensure_schema

    database.close_all_connections()
    reset_backends()
    topic_cache.invalidate()
    record_cache.invalidate()

    data_root = tmp_root / "data"
    reset_config(
        _base_data_dir=data_root,
        active_project="coding_agent_eval",
        EMBEDDING_BACKEND="fastembed",
        LLM_BACKEND="disabled",
        CONSOLIDATION_AUTO_RUN=False,
        CONSOLIDATION_FAST_PATH_ENABLED=True,
        CONSOLIDATION_MIN_CLUSTER_SIZE=1,
        EMBEDDING_DIMENSION=384,
        RENDER_MARKDOWN=False,
        CONTRADICTION_LLM_ENABLED=False,
    )
    cfg = get_config()
    for d in [cfg.DATA_DIR, cfg.KNOWLEDGE_DIR, cfg.CONSOLIDATION_LOG_DIR, cfg.LOG_DIR, cfg.BACKUP_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    ensure_schema()


def _consolidation_succeeded(report: dict[str, Any]) -> bool:
    status = report.get("status")
    if status in {"error", "failed", "already_running"}:
        return False
    if status in {"nothing_to_consolidate", "completed", "created"}:
        return True
    return bool(report.get("run_id")) and int(report.get("clusters_failed", 0) or 0) == 0


def _recall_contains_debug_solution(recall: Any, rel_path: str, fix_token: str) -> bool:
    for record in recall.records:
        if record.get("record_type") != "solution":
            continue
        content = record.get("content", {})
        if not isinstance(content, dict):
            continue
        context = str(content.get("context", ""))
        fix = str(content.get("fix", ""))
        if rel_path in context and fix_token in fix:
            return True

    for topic in recall.knowledge:
        blob = f"{topic.get('title', '')} {topic.get('content', '')}".lower()
        if rel_path.lower() in blob and fix_token.lower() in blob:
            return True

    for claim in recall.claims:
        payload = claim.get("payload", {})
        if not isinstance(payload, dict):
            payload = {}
        context = str(payload.get("context", ""))
        fix = str(payload.get("fix", ""))
        canonical = str(claim.get("canonical_text", ""))
        if rel_path in context and fix_token in fix:
            return True
        if rel_path in canonical and fix_token.lower() in canonical.lower():
            return True

    for episode in recall.episodes:
        content = str(episode.get("content", ""))
        if rel_path in content and fix_token in content:
            return True

    return False


def evaluate_debug_solution_pipeline(scenario_count: int) -> dict[str, Any]:
    from consolidation_memory.client import MemoryClient

    hits = 0
    scenario_rows: list[dict[str, Any]] = []

    with _local_embedding_patches():
        client = MemoryClient(auto_consolidate=False)
        try:
            for i in range(scenario_count):
                rel_path = f"src/module_{i:03d}.py"
                fix_token = f"TIMEOUT_{i}"
                content = (
                    f"Tests fail in {rel_path} when request timeout hits. "
                    f"Fix: set {fix_token}=30 in config and rerun pytest {rel_path}"
                )
                stored = client.store(content, content_type="solution", tags=["debug", f"mod-{i}"])
                assert stored.status == "stored"

                report = client.consolidate()
                assert _consolidation_succeeded(report)

                recall = client.recall(
                    f"request timeout failure in module {i}",
                    n_results=10,
                    include_knowledge=True,
                )
                hit = _recall_contains_debug_solution(recall, rel_path, fix_token)
                if hit:
                    hits += 1

                scenario_rows.append(
                    {
                        "scenario_id": f"debug_{i:03d}",
                        "rel_path": rel_path,
                        "fix_token": fix_token,
                        "recall_hit": hit,
                        "records_returned": len(recall.records),
                        "claims_returned": len(recall.claims),
                    }
                )
        finally:
            client.close()

    hit_rate = hits / scenario_count if scenario_count else 0.0
    thresholds = {"solution_recall_hit_rate_gte": 0.90}
    passed = hit_rate >= thresholds["solution_recall_hit_rate_gte"]

    return {
        "aligned_metric_section": "1) Debug Solution Ingest -> Consolidate -> Recall",
        "scenario_count": scenario_count,
        "formula": "SolutionRecallHitRate = scenarios_with_solution_in_recall / total_scenarios",
        "thresholds": thresholds,
        "measured": {
            "scenarios_with_solution_in_recall": hits,
            "total_scenarios": scenario_count,
            "solution_recall_hit_rate": round(hit_rate, 6),
        },
        "pass": passed,
        "scenarios": scenario_rows,
    }


def evaluate_stale_fix_suppression_after_drift(
    tmp_root: Path,
    scenario_count: int,
) -> dict[str, Any]:
    from consolidation_memory.client import MemoryClient

    suppressed = 0
    leaked = 0
    scenario_rows: list[dict[str, Any]] = []

    with _local_embedding_patches():
        client = MemoryClient(auto_consolidate=False)
        try:
            for i in range(scenario_count):
                repo_dir = tmp_root / "repos" / f"drift_{i:03d}"
                rel_path = f"src/service_{i:03d}.py"
                fix_token = f"PATCH_{i}"
                _prepare_git_repo(
                    repo_dir=repo_dir,
                    rel_path=rel_path,
                    initial_content=f"TIMEOUT = {i}\n",
                )

                content = (
                    f"Intermittent failures in {rel_path}. "
                    f"Fix: apply {fix_token} and rerun pytest {rel_path}"
                )
                stored = client.store(content, content_type="solution", tags=["drift", f"svc-{i}"])
                assert stored.status == "stored"
                client.consolidate()

                pre = client.search_claims(f"failures in service {i}", claim_type="solution", limit=20)
                pre_ids = {claim["id"] for claim in pre.claims}

                (repo_dir / rel_path).write_text(f"TIMEOUT = {i + 99}\n", encoding="utf-8")
                drift = client.detect_drift(repo_path=str(repo_dir))

                post = client.recall(
                    f"intermittent failures in {rel_path}",
                    n_results=10,
                    include_knowledge=True,
                )
                post_claim_ids = {claim["id"] for claim in post.claims}
                impacted = set(drift.get("impacted_claim_ids", []))
                challenged = set(drift.get("challenged_claim_ids", []))

                leaked_ids = {
                    claim_id
                    for claim_id in post_claim_ids
                    if claim_id in impacted and claim_id in pre_ids
                }
                leak = bool(leaked_ids)
                if leak:
                    leaked += 1
                else:
                    suppressed += 1

                scenario_rows.append(
                    {
                        "scenario_id": f"stale_{i:03d}",
                        "rel_path": rel_path,
                        "impacted_claim_ids": sorted(impacted),
                        "challenged_claim_ids": sorted(challenged),
                        "leaked_active_claim_ids": sorted(leaked_ids),
                        "suppressed": not leak,
                    }
                )
        finally:
            client.close()

    suppression_rate = suppressed / scenario_count if scenario_count else 0.0
    leak_rate = leaked / scenario_count if scenario_count else 0.0
    thresholds = {
        "stale_fix_suppression_rate_gte": 0.95,
        "stale_fix_leak_rate_lte": 0.05,
    }
    passed = (
        suppression_rate >= thresholds["stale_fix_suppression_rate_gte"]
        and leak_rate <= thresholds["stale_fix_leak_rate_lte"]
    )

    return {
        "aligned_metric_section": "2) Stale Fix Suppression After Drift",
        "scenario_count": scenario_count,
        "formula": "StaleFixSuppressionRate = drift_scenarios_without_leaked_claims / total_scenarios",
        "thresholds": thresholds,
        "measured": {
            "suppressed_scenarios": suppressed,
            "leaked_scenarios": leaked,
            "total_scenarios": scenario_count,
            "stale_fix_suppression_rate": round(suppression_rate, 6),
            "stale_fix_leak_rate": round(leak_rate, 6),
        },
        "pass": passed,
        "scenarios": scenario_rows,
    }


def evaluate_scope_isolation_under_recall(scope_pairs: int) -> dict[str, Any]:
    from consolidation_memory.client import MemoryClient

    isolated = 0
    leaked = 0
    scenario_rows: list[dict[str, Any]] = []

    with _local_embedding_patches():
        client = MemoryClient(auto_consolidate=False)
        try:
            for i in range(scope_pairs):
                team_a = f"team-a-{i:03d}"
                team_b = f"team-b-{i:03d}"
                secret_b = f"team-b-secret-token-{i:03d}"
                client.store_with_scope(
                    content=secret_b,
                    content_type="fact",
                    scope={
                        "namespace": {"slug": team_b, "sharing_mode": "private"},
                        "app_client": {"name": "agent-b", "app_type": "mcp"},
                        "project": {"slug": f"repo-{i:03d}"},
                    },
                )
                recall_a = client.recall_with_scope(
                    query="secret token",
                    include_knowledge=False,
                    scope={
                        "namespace": {"slug": team_a, "sharing_mode": "private"},
                        "app_client": {"name": "agent-a", "app_type": "mcp"},
                        "project": {"slug": f"repo-{i:03d}"},
                    },
                    n_results=10,
                )
                contents = [ep["content"] for ep in recall_a.episodes]
                leak = any(secret_b in content for content in contents)
                if leak:
                    leaked += 1
                else:
                    isolated += 1

                scenario_rows.append(
                    {
                        "scenario_id": f"scope_{i:03d}",
                        "viewer_namespace": team_a,
                        "owner_namespace": team_b,
                        "leaked": leak,
                    }
                )
        finally:
            client.close()

    isolation_rate = isolated / scope_pairs if scope_pairs else 0.0
    leak_rate = leaked / scope_pairs if scope_pairs else 0.0
    thresholds = {
        "scope_isolation_rate_gte": 1.0,
        "scope_leak_rate_lte": 0.0,
    }
    passed = (
        isolation_rate >= thresholds["scope_isolation_rate_gte"]
        and leak_rate <= thresholds["scope_leak_rate_lte"]
    )

    return {
        "aligned_metric_section": "3) Scope Isolation Under Recall",
        "scenario_count": scope_pairs,
        "formula": "ScopeIsolationRate = recall_scenarios_without_cross_namespace_leaks / total_scenarios",
        "thresholds": thresholds,
        "measured": {
            "isolated_scenarios": isolated,
            "leaked_scenarios": leaked,
            "total_scenarios": scope_pairs,
            "scope_isolation_rate": round(isolation_rate, 6),
            "scope_leak_rate": round(leak_rate, 6),
        },
        "pass": passed,
        "scenarios": scenario_rows,
    }


def evaluate_contradiction_visibility(scenario_count: int) -> dict[str, Any]:
    from consolidation_memory.client import MemoryClient
    from consolidation_memory.database import (
        insert_claim_event,
        insert_claim_sources,
        insert_episode,
        upsert_claim,
    )

    visible = 0
    scenario_rows: list[dict[str, Any]] = []
    now = _utc_now()

    with _local_embedding_patches():
        client = MemoryClient(auto_consolidate=False)
        try:
            for i in range(scenario_count):
                rel_path = f"src/auth_{i:03d}.py"
                old_claim = f"contradiction-old-{i:03d}"
                new_claim = f"contradiction-new-{i:03d}"
                episode_id = insert_episode(
                    content=f"Conflicting auth fixes for {rel_path}",
                    content_type="solution",
                )
                upsert_claim(
                    claim_id=old_claim,
                    claim_type="solution",
                    canonical_text=f"{rel_path} uses legacy JWT validation",
                    payload={
                        "type": "solution",
                        "problem": f"auth failures in {rel_path}",
                        "fix": "use legacy JWT middleware",
                        "context": rel_path,
                    },
                    status="active",
                    valid_from=_iso(now - timedelta(days=1)),
                )
                upsert_claim(
                    claim_id=new_claim,
                    claim_type="solution",
                    canonical_text=f"{rel_path} requires bearer token rotation",
                    payload={
                        "type": "solution",
                        "problem": f"auth failures in {rel_path}",
                        "fix": "rotate bearer tokens every hour",
                        "context": rel_path,
                    },
                    status="active",
                    valid_from=_iso(now),
                )
                insert_claim_sources(old_claim, [{"source_episode_id": episode_id}])
                insert_claim_sources(new_claim, [{"source_episode_id": episode_id}])
                insert_claim_event(
                    old_claim,
                    event_type="contradiction",
                    details={"paired_claim_id": new_claim, "scenario": i},
                    created_at=_iso(now),
                )

                recall = client.recall(
                    f"auth failures in {rel_path}",
                    n_results=10,
                    include_knowledge=True,
                )
                warning_text = " ".join(recall.warnings).lower()
                claim_uncertainty = any(
                    "contradict" in str(claim.get("uncertainty", "")).lower()
                    for claim in recall.claims
                )
                surfaced = "contradict" in warning_text or claim_uncertainty
                if surfaced:
                    visible += 1

                scenario_rows.append(
                    {
                        "scenario_id": f"contradiction_{i:03d}",
                        "old_claim_id": old_claim,
                        "new_claim_id": new_claim,
                        "surfaced_in_recall": surfaced,
                        "warnings": recall.warnings,
                    }
                )
        finally:
            client.close()

    visibility_rate = visible / scenario_count if scenario_count else 0.0
    thresholds = {"contradiction_visibility_rate_gte": 0.95}
    passed = visibility_rate >= thresholds["contradiction_visibility_rate_gte"]

    return {
        "aligned_metric_section": "4) Contradiction Visibility On Recall",
        "scenario_count": scenario_count,
        "formula": "ContradictionVisibilityRate = scenarios_with_contradiction_signals / total_scenarios",
        "thresholds": thresholds,
        "measured": {
            "visible_scenarios": visible,
            "total_scenarios": scenario_count,
            "contradiction_visibility_rate": round(visibility_rate, 6),
        },
        "pass": passed,
        "scenarios": scenario_rows,
    }


def evaluate_outcome_informed_ranking(pair_count: int) -> dict[str, Any]:
    from consolidation_memory.client import MemoryClient
    from consolidation_memory.database import insert_claim_sources, insert_episode, upsert_claim

    wins = 0
    reliability_deltas: list[float] = []
    scenario_rows: list[dict[str, Any]] = []

    with _local_embedding_patches():
        client = MemoryClient(auto_consolidate=False)
        try:
            for i in range(pair_count):
                topic = f"deploy retry backoff service {i:03d}"
                backed_id = f"outcome-backed-{i:03d}"
                plain_id = f"outcome-plain-{i:03d}"
                episode_id = insert_episode(
                    content=f"Deploy failures for service {i:03d}",
                    content_type="solution",
                )
                canonical = f"Use retry backoff when deploy fails for service {i:03d}"
                upsert_claim(
                    claim_id=backed_id,
                    claim_type="solution",
                    canonical_text=canonical,
                    payload={
                        "type": "solution",
                        "problem": f"deploy failures service {i:03d}",
                        "fix": "retry with exponential backoff",
                        "context": f"src/deploy_{i:03d}.py",
                    },
                    status="active",
                    confidence=0.7,
                    valid_from="2026-01-01T00:00:00+00:00",
                )
                upsert_claim(
                    claim_id=plain_id,
                    claim_type="solution",
                    canonical_text=f"Use retry backoff when deploy fails for service {i:03d} (alt)",
                    payload={
                        "type": "solution",
                        "problem": f"deploy failures service {i:03d}",
                        "fix": "retry with linear backoff",
                        "context": f"src/deploy_{i:03d}.py",
                    },
                    status="active",
                    confidence=0.7,
                    valid_from="2026-01-01T00:00:00+00:00",
                )
                insert_claim_sources(backed_id, [{"source_episode_id": episode_id}])
                insert_claim_sources(plain_id, [{"source_episode_id": episode_id}])

                recorded = client.record_outcome(
                    action_summary=f"retry deploy for service {i:03d}",
                    outcome_type="success",
                    source_claim_ids=[backed_id],
                    source_episode_ids=[episode_id],
                    code_anchors=[{"anchor_type": "path", "anchor_value": f"src/deploy_{i:03d}.py"}],
                )
                assert recorded.status == "recorded"

                result = client.search_claims(topic, claim_type="solution", limit=10)
                claim_positions = {claim["id"]: idx for idx, claim in enumerate(result.claims)}
                backed_pos = claim_positions.get(backed_id)
                plain_pos = claim_positions.get(plain_id)

                backed_rel = next(
                    (claim.get("reliability", {}) for claim in result.claims if claim["id"] == backed_id),
                    {},
                )
                plain_rel = next(
                    (claim.get("reliability", {}) for claim in result.claims if claim["id"] == plain_id),
                    {},
                )
                backed_score = float(backed_rel.get("score", 0.0) or 0.0)
                plain_score = float(plain_rel.get("score", 0.0) or 0.0)
                delta = backed_score - plain_score
                reliability_deltas.append(delta)

                ranked_ahead = (
                    backed_pos is not None
                    and plain_pos is not None
                    and backed_pos < plain_pos
                )
                reliability_win = backed_score > plain_score
                win = ranked_ahead or reliability_win
                if win:
                    wins += 1

                scenario_rows.append(
                    {
                        "scenario_id": f"outcome_{i:03d}",
                        "backed_claim_id": backed_id,
                        "plain_claim_id": plain_id,
                        "backed_position": backed_pos,
                        "plain_position": plain_pos,
                        "backed_reliability_score": backed_score,
                        "plain_reliability_score": plain_score,
                        "reliability_delta": round(delta, 6),
                        "outcome_ranking_win": win,
                    }
                )
        finally:
            client.close()

    win_rate = wins / pair_count if pair_count else 0.0
    median_delta = statistics.median(reliability_deltas) if reliability_deltas else 0.0
    thresholds = {
        "outcome_ranking_win_rate_gte": 0.80,
        "median_reliability_delta_gte": 0.0,
    }
    passed = (
        win_rate >= thresholds["outcome_ranking_win_rate_gte"]
        and median_delta >= thresholds["median_reliability_delta_gte"]
    )

    return {
        "aligned_metric_section": "5) Outcome-Informed Claim Ranking",
        "scenario_count": pair_count,
        "formula": "OutcomeRankingWinRate = scenarios_where_backed_claim_outranks_plain / total_scenarios",
        "thresholds": thresholds,
        "measured": {
            "winning_scenarios": wins,
            "total_scenarios": pair_count,
            "outcome_ranking_win_rate": round(win_rate, 6),
            "median_reliability_delta": round(float(median_delta), 6),
        },
        "pass": passed,
        "scenarios": scenario_rows,
    }


def run_eval(mode: str) -> dict[str, Any]:
    from consolidation_memory import database

    cfg = MODE_CONFIG[mode]
    local_tmp_base = Path.cwd() / ".tmp_coding_agent_eval_runtime"
    local_tmp_base.mkdir(parents=True, exist_ok=True)
    tmp_root = Path(tempfile.mkdtemp(prefix="coding_agent_eval_", dir=str(local_tmp_base)))
    try:
        _reset_eval_environment(tmp_root)
        with _local_embedding_patches():
            sections = {
                "debug_solution_pipeline": evaluate_debug_solution_pipeline(cfg.debug_scenarios),
                "stale_fix_suppression_after_drift": evaluate_stale_fix_suppression_after_drift(
                    tmp_root=tmp_root,
                    scenario_count=cfg.drift_scenarios,
                ),
                "scope_isolation_under_recall": evaluate_scope_isolation_under_recall(cfg.scope_pairs),
                "contradiction_visibility_on_recall": evaluate_contradiction_visibility(
                    cfg.contradiction_scenarios,
                ),
                "outcome_informed_claim_ranking": evaluate_outcome_informed_ranking(cfg.outcome_pairs),
            }

        overall_pass = all(bool(section["pass"]) for section in sections.values())
        return {
            "benchmark": "coding_agent_eval",
            "run_id": f"coding_agent_eval_{mode}_{uuid.uuid4().hex[:12]}",
            "mode": mode,
            "generated_at": _iso(_utc_now()),
            "cloud_dependencies_used": False,
            "aligned_metrics_doc": "docs/CODING_AGENT_METRICS.md",
            "product_wedge": "Drift-aware debugging memory for coding agents",
            "sections": sections,
            "overall_pass": overall_pass,
        }
    finally:
        database.close_all_connections()
        shutil.rmtree(tmp_root, ignore_errors=True)


def save_results(results: dict[str, Any], output_path: Path | None = None) -> Path:
    if output_path is None:
        stamp = _utc_now().strftime("%Y%m%d_%H%M%S")
        output_path = (
            Path(__file__).parent / "results" / f"coding_agent_eval_{results['mode']}_{stamp}.json"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Coding-agent trust benchmark harness")
    parser.add_argument(
        "--mode",
        choices=sorted(MODE_CONFIG.keys()),
        default="quick",
        help="Benchmark mode (default: quick)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output JSON path",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    results = run_eval(mode=args.mode)
    out = save_results(results, output_path=args.output)

    print(
        json.dumps(
            {
                "mode": results["mode"],
                "overall_pass": results["overall_pass"],
                "output_path": str(out),
                "section_passes": {
                    key: section["pass"] for key, section in results["sections"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()