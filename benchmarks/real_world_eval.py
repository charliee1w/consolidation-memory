"""Live-memory benchmark harness.

Evaluates the active consolidation-memory project against real stored episodes,
claims, and repository drift signals. Read-only: never mutates user memory.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_PATH = _REPO_ROOT / "src"
if str(_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(_SRC_PATH))

logger = logging.getLogger("benchmark.real_world_eval")


@dataclass(frozen=True)
class EvalModeConfig:
    max_solution_cases: int
    max_claim_cases: int
    max_challenged_cases: int
    max_provenance_queries: int


MODE_CONFIG: dict[str, EvalModeConfig] = {
    "ci": EvalModeConfig(
        max_solution_cases=5,
        max_claim_cases=5,
        max_challenged_cases=3,
        max_provenance_queries=5,
    ),
    "quick": EvalModeConfig(
        max_solution_cases=30,
        max_claim_cases=25,
        max_challenged_cases=20,
        max_provenance_queries=20,
    ),
    "full": EvalModeConfig(
        max_solution_cases=120,
        max_claim_cases=80,
        max_challenged_cases=60,
        max_provenance_queries=60,
    ),
}


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _problem_query_from_content(content: str) -> str:
    from consolidation_memory.episode_embedding import problem_query_from_content

    return problem_query_from_content(content)


def _claim_query_from_payload(payload: dict[str, Any]) -> str:
    problem = str(payload.get("problem", "")).strip()
    if problem:
        return problem[:180]
    subject = str(payload.get("subject", "")).strip()
    info = str(payload.get("info", "")).strip()
    if subject or info:
        return f"{subject} {info}".strip()[:180]
    return str(payload.get("fix", "")).strip()[:180]


def _parse_payload(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            loaded = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return loaded if isinstance(loaded, dict) else {}
    return {}


def _episode_recall_hit(recall: Any, episode_id: str) -> bool:
    top_episode_ids = {ep.get("id") for ep in recall.episodes[:5]}
    if episode_id in top_episode_ids:
        return True
    for claim in recall.claims[:5]:
        for source in claim.get("sources", []) or []:
            if source.get("source_episode_id") == episode_id:
                return True
    for record in recall.records[:5]:
        if episode_id in (record.get("source_episodes") or []):
            return True
    return False


def _claim_has_provenance(claim_id: str, conn: Any) -> bool:
    row = conn.execute(
        """SELECT 1
           FROM claim_sources
           WHERE claim_id = ?
             AND (
               source_episode_id IS NOT NULL
               OR source_record_id IS NOT NULL
               OR source_topic_id IS NOT NULL
             )
           LIMIT 1""",
        (claim_id,),
    ).fetchone()
    event = conn.execute(
        "SELECT 1 FROM claim_events WHERE claim_id = ? LIMIT 1",
        (claim_id,),
    ).fetchone()
    return row is not None and event is not None


def _embedding_backend_ready() -> tuple[bool, str]:
    from consolidation_memory.backends import encode_query

    try:
        vec = encode_query("real world eval preflight")
        if vec is None or len(vec) == 0:
            return False, "encode_query returned empty vector"
        return True, "ok"
    except Exception as exc:
        return False, str(exc)


def evaluate_live_solution_recall(max_cases: int) -> dict[str, Any]:
    from consolidation_memory import MemoryClient
    from consolidation_memory.database import ensure_schema, get_connection

    ready, reason = _embedding_backend_ready()
    if not ready:
        return {
            "aligned_metric_section": "1) Live Solution Recall@5",
            "formula": "LiveSolutionRecall@5 = recalled_source_in_top5 / evaluated_cases",
            "thresholds": {"live_solution_recall_at_5_gte": 0.35},
            "measured": {
                "cases_sampled": 0,
                "cases_evaluated": 0,
                "recall_hits": 0,
                "live_solution_recall_at_5": 0.0,
                "skipped_reason": f"embedding_backend_unavailable: {reason}",
            },
            "pass": False,
            "scenarios": [],
        }

    ensure_schema()
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, content, tags
            FROM episodes
            WHERE deleted = 0
              AND content_type = 'solution'
              AND length(content) >= 80
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (max_cases,),
        ).fetchall()

    hits = 0
    evaluated = 0
    scenario_rows: list[dict[str, Any]] = []
    client = MemoryClient(auto_consolidate=False)
    try:
        for row in rows:
            query = _problem_query_from_content(row["content"])
            if len(query) < 24:
                continue
            evaluated += 1
            recall = client.recall(query, n_results=10, include_knowledge=True)
            hit = _episode_recall_hit(recall, row["id"])
            if hit:
                hits += 1
            scenario_rows.append(
                {
                    "episode_id": row["id"],
                    "query": query,
                    "recall_hit_top5": hit,
                    "episodes_returned": len(recall.episodes),
                    "claims_returned": len(recall.claims),
                }
            )
    finally:
        client.close()

    hit_rate = hits / evaluated if evaluated else 0.0
    thresholds = {"live_solution_recall_at_5_gte": 0.35}
    passed = evaluated > 0 and hit_rate >= thresholds["live_solution_recall_at_5_gte"]

    return {
        "aligned_metric_section": "1) Live Solution Recall@5",
        "formula": "LiveSolutionRecall@5 = recalled_source_in_top5 / evaluated_cases",
        "thresholds": thresholds,
        "measured": {
            "cases_sampled": len(rows),
            "cases_evaluated": evaluated,
            "recall_hits": hits,
            "live_solution_recall_at_5": round(hit_rate, 6),
        },
        "pass": passed,
        "scenarios": scenario_rows,
    }


def evaluate_live_claim_recall(max_cases: int) -> dict[str, Any]:
    from consolidation_memory import MemoryClient
    from consolidation_memory.database import ensure_schema, get_connection

    ensure_schema()
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, claim_type, canonical_text, payload
            FROM claims
            WHERE status = 'active'
              AND claim_type IN ('solution', 'fact', 'procedure', 'strategy')
            ORDER BY valid_from DESC
            LIMIT ?
            """,
            (max_cases,),
        ).fetchall()

    hits = 0
    evaluated = 0
    scenario_rows: list[dict[str, Any]] = []
    client = MemoryClient(auto_consolidate=False)
    try:
        for row in rows:
            payload = _parse_payload(row["payload"])
            query = _claim_query_from_payload(payload) or str(row["canonical_text"])[:120]
            if len(query) < 16:
                continue
            evaluated += 1
            result = client.search_claims(query, claim_type=row["claim_type"], limit=10)
            top_ids = [claim["id"] for claim in result.claims[:5]]
            hit = row["id"] in top_ids
            if hit:
                hits += 1
            scenario_rows.append(
                {
                    "claim_id": row["id"],
                    "claim_type": row["claim_type"],
                    "query": query,
                    "recall_hit_top5": hit,
                    "top_position": top_ids.index(row["id"]) if hit else None,
                }
            )
    finally:
        client.close()

    hit_rate = hits / evaluated if evaluated else 0.0
    thresholds = {"live_claim_recall_at_5_gte": 0.40}
    passed = evaluated > 0 and hit_rate >= thresholds["live_claim_recall_at_5_gte"]

    return {
        "aligned_metric_section": "2) Live Claim Recall@5",
        "formula": "LiveClaimRecall@5 = target_claim_in_top5 / evaluated_claims",
        "thresholds": thresholds,
        "measured": {
            "claims_sampled": len(rows),
            "claims_evaluated": evaluated,
            "recall_hits": hits,
            "live_claim_recall_at_5": round(hit_rate, 6),
        },
        "pass": passed,
        "scenarios": scenario_rows,
    }


def evaluate_challenged_claim_suppression(max_cases: int) -> dict[str, Any]:
    from consolidation_memory import MemoryClient
    from consolidation_memory.database import ensure_schema, get_connection

    ensure_schema()
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, canonical_text, status
            FROM claims
            WHERE status = 'challenged'
            ORDER BY valid_from DESC
            LIMIT ?
            """,
            (max_cases,),
        ).fetchall()

    suppressed = 0
    evaluated = 0
    scenario_rows: list[dict[str, Any]] = []
    client = MemoryClient(auto_consolidate=False)
    try:
        for row in rows:
            query = str(row["canonical_text"])[:140]
            if len(query) < 12:
                continue
            evaluated += 1
            result = client.search_claims(query, limit=10)
            match = next((c for c in result.claims[:5] if c["id"] == row["id"]), None)
            if match is None:
                suppressed += 1
                mode = "absent_from_top5"
            else:
                uncertainty = str(match.get("uncertainty", ""))
                reliability = match.get("reliability", {}) or {}
                band = str(reliability.get("band", ""))
                if (
                    uncertainty
                    or band in {"guarded", "caution", "low"}
                    or match.get("status") == "challenged"
                ):
                    suppressed += 1
                    mode = "visible_with_trust_downgrade"
                else:
                    mode = "leaked_as_active"
            scenario_rows.append(
                {
                    "claim_id": row["id"],
                    "suppression_mode": mode,
                    "suppressed": mode != "leaked_as_active",
                }
            )
    finally:
        client.close()

    suppression_rate = suppressed / evaluated if evaluated else 0.0
    thresholds = {"challenged_suppression_rate_gte": 0.90}
    passed = evaluated > 0 and suppression_rate >= thresholds["challenged_suppression_rate_gte"]

    return {
        "aligned_metric_section": "3) Challenged Claim Suppression",
        "formula": "ChallengedSuppressionRate = challenged_claims_not_reused_as_active / evaluated_claims",
        "thresholds": thresholds,
        "measured": {
            "claims_sampled": len(rows),
            "claims_evaluated": evaluated,
            "suppressed_claims": suppressed,
            "challenged_suppression_rate": round(suppression_rate, 6),
        },
        "pass": passed,
        "scenarios": scenario_rows,
    }


def evaluate_live_provenance_on_recall(max_queries: int) -> dict[str, Any]:
    from consolidation_memory import MemoryClient
    from consolidation_memory.database import ensure_schema, get_connection

    ready, reason = _embedding_backend_ready()
    if not ready:
        return {
            "aligned_metric_section": "4) Live Provenance Coverage On Recall",
            "formula": "LiveProvenanceCoverage = claims_with_complete_provenance / claims_returned",
            "thresholds": {"live_provenance_coverage_gte": 0.85},
            "measured": {
                "queries_sampled": 0,
                "claims_returned": 0,
                "claims_with_complete_provenance": 0,
                "live_provenance_coverage": 0.0,
                "skipped_reason": f"embedding_backend_unavailable: {reason}",
            },
            "pass": False,
            "scenarios": [],
        }

    ensure_schema()
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT content
            FROM episodes
            WHERE deleted = 0
              AND content_type IN ('solution', 'fact')
              AND length(content) >= 60
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (max_queries,),
        ).fetchall()

    complete = 0
    returned = 0
    scenario_rows: list[dict[str, Any]] = []
    client = MemoryClient(auto_consolidate=False)
    try:
        for row in rows:
            query = _problem_query_from_content(row["content"])
            if len(query) < 20:
                continue
            recall = client.recall(query, n_results=8, include_knowledge=True)
            for claim in recall.claims[:5]:
                claim_id = str(claim.get("id", ""))
                if not claim_id:
                    continue
                returned += 1
                with get_connection() as provenance_conn:
                    has_provenance = _claim_has_provenance(claim_id, provenance_conn)
                if has_provenance:
                    complete += 1
                scenario_rows.append(
                    {
                        "query": query,
                        "claim_id": claim_id,
                        "provenance_complete": has_provenance,
                    }
                )
    finally:
        client.close()

    coverage = complete / returned if returned else 0.0
    thresholds = {"live_provenance_coverage_gte": 0.85}
    passed = returned > 0 and coverage >= thresholds["live_provenance_coverage_gte"]

    return {
        "aligned_metric_section": "4) Live Provenance Coverage On Recall",
        "formula": "LiveProvenanceCoverage = claims_with_complete_provenance / claims_returned",
        "thresholds": thresholds,
        "measured": {
            "queries_sampled": len(rows),
            "claims_returned": returned,
            "claims_with_complete_provenance": complete,
            "live_provenance_coverage": round(coverage, 6),
        },
        "pass": passed,
        "scenarios": scenario_rows[:50],
    }


def _drift_challenge_rate(drift: dict[str, Any]) -> tuple[float, int, int, int]:
    """Return challenge rate and counts for impacted vs challenged outcomes."""
    impacts = list(drift.get("impacts", []))
    impacted_count = len(drift.get("impacted_claim_ids", []))
    newly_challenged_count = len(drift.get("challenged_claim_ids", []))
    challenged_outcome_count = sum(
        1
        for impact in impacts
        if str(impact.get("new_status", "")) == "challenged"
    )
    changed_count = len(
        [
            anchor
            for anchor in drift.get("checked_anchors", [])
            if anchor.get("anchor_type") == "path"
        ]
    )
    if impacted_count:
        challenge_rate = challenged_outcome_count / impacted_count
    else:
        challenge_rate = 1.0 if changed_count == 0 else 0.0
    return challenge_rate, challenged_outcome_count, newly_challenged_count, impacted_count


def evaluate_live_drift_response(repo_path: Path) -> dict[str, Any]:
    from consolidation_memory import MemoryClient

    client = MemoryClient(auto_consolidate=False)
    try:
        drift = client.detect_drift(repo_path=str(repo_path))
    finally:
        client.close()

    changed_paths = [
        anchor.get("anchor_value", "")
        for anchor in drift.get("checked_anchors", [])
        if anchor.get("anchor_type") == "path"
    ]
    changed_count = len(changed_paths)
    (
        challenge_rate,
        challenged_outcome_count,
        newly_challenged_count,
        impacted_count,
    ) = _drift_challenge_rate(drift)

    thresholds = {
        "changed_paths_gt": 0,
        "drift_challenge_rate_gte": 0.80,
    }
    if changed_count == 0:
        passed = True
        note = "No changed repo paths detected; drift challenge rate not required."
    elif impacted_count == 0:
        passed = True
        note = "Changed repo paths detected but no anchored claims were impacted."
    else:
        passed = challenge_rate >= thresholds["drift_challenge_rate_gte"]
        note = "Drift challenge evaluated against currently changed repo paths."

    return {
        "aligned_metric_section": "5) Live Drift Response On Changed Repo Paths",
        "formula": (
            "DriftChallengeRate = impacted_claims_with_challenged_status / impacted_claims "
            "(when paths changed)"
        ),
        "thresholds": thresholds,
        "measured": {
            "repo_path": str(repo_path.resolve()),
            "changed_path_count": changed_count,
            "impacted_claim_count": impacted_count,
            "challenged_outcome_count": challenged_outcome_count,
            "newly_challenged_claim_count": newly_challenged_count,
            "drift_challenge_rate": round(challenge_rate, 6),
            "note": note,
        },
        "pass": passed,
        "changed_paths_sample": changed_paths[:20],
    }


def evaluate_memory_health_snapshot() -> dict[str, Any]:
    from consolidation_memory import MemoryClient

    client = MemoryClient(auto_consolidate=False)
    try:
        status = client.status(lightweight=True)
    finally:
        client.close()

    episodic = status.episodic_buffer or {}
    knowledge = status.knowledge_base or {}
    trust = status.trust_profile or {}
    health = status.health or {}

    pending = int(episodic.get("pending_consolidation", 0) or 0)
    total = int(episodic.get("total", 0) or 0)
    backlog_ratio = pending / total if total else 0.0

    thresholds = {
        "health_status_in": {"healthy", "degraded"},
        "consolidation_backlog_ratio_lte": 0.20,
    }
    passed = (
        str(health.get("status", "")) in thresholds["health_status_in"]
        and backlog_ratio <= thresholds["consolidation_backlog_ratio_lte"]
    )

    return {
        "aligned_metric_section": "6) Live Memory Health Snapshot",
        "thresholds": thresholds,
        "measured": {
            "health_status": health.get("status"),
            "health_issues": health.get("issues", []),
            "episode_total": total,
            "pending_consolidation": pending,
            "consolidation_backlog_ratio": round(backlog_ratio, 6),
            "topic_count": knowledge.get("total_topics"),
            "record_count": knowledge.get("total_records"),
            "trust_profile": trust,
        },
        "pass": passed,
    }


def _reset_ci_environment(tmp_root: Path) -> None:
    from consolidation_memory import claim_cache, database, record_cache, topic_cache
    from consolidation_memory.backends import reset_backends
    from consolidation_memory.config import get_config, reset_config
    from consolidation_memory.database import ensure_schema

    database.close_all_connections()
    reset_backends()
    topic_cache.invalidate()
    record_cache.invalidate()
    claim_cache.invalidate()

    data_root = tmp_root / "data"
    reset_config(
        _base_data_dir=data_root,
        active_project="real_world_eval_ci",
        EMBEDDING_BACKEND="fastembed",
        LLM_BACKEND="disabled",
        CONSOLIDATION_AUTO_RUN=False,
        EMBEDDING_DIMENSION=384,
    )
    cfg = get_config()
    for directory in [
        cfg.DATA_DIR,
        cfg.KNOWLEDGE_DIR,
        cfg.CONSOLIDATION_LOG_DIR,
        cfg.LOG_DIR,
        cfg.BACKUP_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)
    ensure_schema()


def _seed_ci_corpus() -> None:
    from consolidation_memory import MemoryClient
    from consolidation_memory.database import (
        insert_claim_event,
        insert_claim_sources,
        mark_consolidated,
        upsert_claim,
    )

    solutions = [
        (
            "Problem: pytest fails on Windows path separators in MCP recall tests.\n"
            "Fix: normalize scope paths before anchor extraction in server.py.\n"
            "Context: path:src/consolidation_memory/server.py"
        ),
        (
            "Problem: REST recall times out when record cache is cold on first call.\n"
            "Fix: defer knowledge until record_cache is warm and inject recall deadlines.\n"
            "Context: path:src/consolidation_memory/rest.py"
        ),
        (
            "Problem: policy ACL rows exist but operators cannot inspect them easily.\n"
            "Fix: add consolidation-memory policy list and policy grant CLI commands.\n"
            "Context: path:src/consolidation_memory/cli.py"
        ),
    ]
    client = MemoryClient(auto_consolidate=False)
    episode_ids: list[str] = []
    try:
        for content in solutions:
            result = client.store(content=content, content_type="solution", tags=["ci-seed"])
            if result.id:
                episode_ids.append(result.id)
    finally:
        client.close()

    if episode_ids:
        mark_consolidated(episode_ids, "ci-seed-topic.md")

    upsert_claim(
        claim_id="ci-claim-active-a",
        claim_type="solution",
        canonical_text="normalize scope paths before anchor extraction",
        payload={"problem": "pytest fails on Windows path separators", "fix": "normalize paths"},
        status="active",
        valid_from="2026-06-01T00:00:00+00:00",
    )
    upsert_claim(
        claim_id="ci-claim-active-b",
        claim_type="fact",
        canonical_text="defer knowledge until record cache is warm",
        payload={"subject": "recall", "info": "defer knowledge until record cache is warm"},
        status="active",
        valid_from="2026-06-01T00:00:00+00:00",
    )
    upsert_claim(
        claim_id="ci-claim-challenged",
        claim_type="procedure",
        canonical_text="legacy deploy script still referenced in docs",
        payload={"trigger": "deploy", "steps": "run legacy deploy script"},
        status="challenged",
        valid_from="2026-06-01T00:00:00+00:00",
    )
    if episode_ids:
        insert_claim_sources("ci-claim-active-a", [{"source_episode_id": episode_ids[0]}])
        insert_claim_event(
            claim_id="ci-claim-active-a",
            event_type="created",
            details={"source": "real_world_eval_ci"},
            created_at="2026-06-01T00:00:00+00:00",
        )
    insert_claim_event(
        claim_id="ci-claim-challenged",
        event_type="challenged",
        details={"reason": "drift"},
        created_at="2026-06-02T00:00:00+00:00",
    )


def run_eval(mode: str, repo_path: Path | None = None) -> dict[str, Any]:
    cfg = MODE_CONFIG[mode]
    repo = repo_path or _REPO_ROOT
    data_source = "live_active_project"
    tmp_root: Path | None = None

    if mode == "ci":
        from consolidation_memory import database

        local_tmp_base = Path.cwd() / ".tmp_real_world_eval_runtime"
        local_tmp_base.mkdir(parents=True, exist_ok=True)
        tmp_root = Path(tempfile.mkdtemp(prefix="real_world_eval_ci_", dir=str(local_tmp_base)))
        data_source = "ci_fixture_project"
        try:
            _reset_ci_environment(tmp_root)
            _seed_ci_corpus()
            sections = {
                "live_solution_recall_at_5": evaluate_live_solution_recall(cfg.max_solution_cases),
                "live_claim_recall_at_5": evaluate_live_claim_recall(cfg.max_claim_cases),
                "challenged_claim_suppression": evaluate_challenged_claim_suppression(
                    cfg.max_challenged_cases
                ),
                "live_provenance_coverage_on_recall": evaluate_live_provenance_on_recall(
                    cfg.max_provenance_queries
                ),
                "live_drift_response": evaluate_live_drift_response(repo),
                "memory_health_snapshot": evaluate_memory_health_snapshot(),
            }
        finally:
            database.close_all_connections()
            shutil.rmtree(tmp_root, ignore_errors=True)
    else:
        sections = {
            "live_solution_recall_at_5": evaluate_live_solution_recall(cfg.max_solution_cases),
            "live_claim_recall_at_5": evaluate_live_claim_recall(cfg.max_claim_cases),
            "challenged_claim_suppression": evaluate_challenged_claim_suppression(
                cfg.max_challenged_cases
            ),
            "live_provenance_coverage_on_recall": evaluate_live_provenance_on_recall(
                cfg.max_provenance_queries
            ),
            "live_drift_response": evaluate_live_drift_response(repo),
            "memory_health_snapshot": evaluate_memory_health_snapshot(),
        }

    overall_pass = all(bool(section["pass"]) for section in sections.values())

    return {
        "benchmark": "real_world_eval",
        "run_id": f"real_world_eval_{mode}_{uuid.uuid4().hex[:12]}",
        "mode": mode,
        "generated_at": _iso_now(),
        "data_source": data_source,
        "cloud_dependencies_used": False,
        "aligned_metrics_doc": "docs/REAL_WORLD_METRICS.md",
        "sections": sections,
        "overall_pass": overall_pass,
    }


def save_results(results: dict[str, Any], output_path: Path | None = None) -> Path:
    if output_path is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_path = Path(__file__).parent / "results" / f"real_world_eval_{results['mode']}_{stamp}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Live-memory real-world benchmark harness")
    parser.add_argument("--mode", choices=sorted(MODE_CONFIG.keys()), default="quick")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--repo-path", type=Path, default=None, help="Repo path for drift checks")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    results = run_eval(mode=args.mode, repo_path=args.repo_path)
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
                "headline_metrics": {
                    key: section.get("measured", {})
                    for key, section in results["sections"].items()
                },
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()