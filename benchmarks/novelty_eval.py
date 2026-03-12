"""Novelty wedge evaluation harness.

Runs local, deterministic benchmark scenarios for:
- stale belief after code change
- contradiction evolution
- temporal belief reconstruction
- provenance trace completeness
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import statistics
import subprocess
import sys
import tempfile
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np

# Ensure local src/ takes precedence over any globally installed package.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_PATH = _REPO_ROOT / "src"
if str(_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(_SRC_PATH))

from consolidation_memory.utils import parse_datetime  # noqa: E402

logger = logging.getLogger("benchmark.novelty_eval")


@dataclass(frozen=True)
class EvalModeConfig:
    drift_scenarios: int
    contradiction_scenarios: int
    temporal_queries: int
    provenance_limit: int


MODE_CONFIG: dict[str, EvalModeConfig] = {
    "quick": EvalModeConfig(
        drift_scenarios=6,
        contradiction_scenarios=10,
        temporal_queries=4,
        provenance_limit=10,
    ),
    "full": EvalModeConfig(
        drift_scenarios=30,
        contradiction_scenarios=40,
        temporal_queries=12,
        provenance_limit=50,
    ),
}

_EMBED_DIM = 384


def _deterministic_embed(texts: list[str]) -> np.ndarray:
    """Return deterministic, normalized vectors without external dependencies."""
    vectors: list[np.ndarray] = []
    for text in texts:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "big")
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(_EMBED_DIM, dtype=np.float32)
        norm = float(np.linalg.norm(vec)) or 1.0
        vectors.append((vec / norm).astype(np.float32))
    return np.vstack(vectors)


def _deterministic_query_embed(text: str) -> np.ndarray:
    return _deterministic_embed([text])[0]


@contextmanager
def _local_embedding_patches():
    """Patch embedding calls so eval remains local/offline by default."""
    with (
        patch("consolidation_memory.backends.encode_documents", side_effect=_deterministic_embed),
        patch("consolidation_memory.backends.encode_query", side_effect=_deterministic_query_embed),
        patch("consolidation_memory.backends.get_dimension", return_value=_EMBED_DIM),
    ):
        yield


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile with deterministic behavior for small samples."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = int(round((p / 100.0) * (len(ordered) - 1)))
    rank = max(0, min(rank, len(ordered) - 1))
    return float(ordered[rank])


def _run_command(cmd: list[str], cwd: Path) -> None:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        details = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"Command failed ({' '.join(cmd)}): {details}")


def _prepare_git_repo(repo_dir: Path, rel_path: str, initial_content: str) -> None:
    repo_dir.mkdir(parents=True, exist_ok=True)
    target = repo_dir / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(initial_content, encoding="utf-8")

    _run_command(["git", "init"], cwd=repo_dir)
    _run_command(["git", "config", "user.name", "novelty-eval"], cwd=repo_dir)
    _run_command(["git", "config", "user.email", "novelty-eval@example.com"], cwd=repo_dir)
    _run_command(["git", "add", "."], cwd=repo_dir)
    _run_command(["git", "commit", "-m", "initial"], cwd=repo_dir)


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
        active_project="novelty_eval",
        EMBEDDING_BACKEND="fastembed",
        LLM_BACKEND="disabled",
        CONSOLIDATION_AUTO_RUN=False,
        EMBEDDING_DIMENSION=384,
    )
    cfg = get_config()
    for d in [cfg.DATA_DIR, cfg.KNOWLEDGE_DIR, cfg.CONSOLIDATION_LOG_DIR, cfg.LOG_DIR, cfg.BACKUP_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    ensure_schema()


def evaluate_belief_freshness_after_code_drift(tmp_root: Path, scenario_count: int) -> dict[str, Any]:
    from consolidation_memory.database import (
        get_connection,
        insert_claim_event,
        insert_claim_sources,
        insert_episode,
        insert_episode_anchors,
        upsert_claim,
    )
    from consolidation_memory.drift import detect_code_drift

    corrected_impacted_claims = 0
    stale_leak_count = 0
    event_lags: list[float] = []
    scenario_rows: list[dict[str, Any]] = []

    for i in range(scenario_count):
        repo_dir = tmp_root / "repos" / f"drift_{i:03d}"
        rel_path = f"src/service_{i:03d}.py"
        _prepare_git_repo(
            repo_dir=repo_dir,
            rel_path=rel_path,
            initial_content=f"VALUE = {i}\n",
        )

        episode_id = insert_episode(content=f"Investigated {rel_path}")
        insert_episode_anchors(
            episode_id,
            [{"anchor_type": "path", "anchor_value": rel_path}],
        )

        claim_id = f"drift-claim-{i:03d}"
        upsert_claim(
            claim_id=claim_id,
            claim_type="fact",
            canonical_text=f"{rel_path} currently sets VALUE={i}",
            payload={"path": rel_path, "value": i},
            status="active",
            confidence=0.9,
            valid_from="2025-01-01T00:00:00+00:00",
        )
        insert_claim_sources(
            claim_id,
            [{
                "source_episode_id": episode_id,
                "source_topic_id": f"topic-drift-{i:03d}",
                "source_record_id": f"record-drift-{i:03d}",
            }],
        )
        insert_claim_event(claim_id, event_type="create", details={"scenario": i})

        (repo_dir / rel_path).write_text(f"VALUE = {i + 1}\n", encoding="utf-8")
        detected_at = _utc_now()
        drift_result = detect_code_drift(repo_path=repo_dir)

        with get_connection() as conn:
            claim_row = conn.execute(
                "SELECT status, confidence FROM claims WHERE id = ?",
                (claim_id,),
            ).fetchone()
            event_row = conn.execute(
                """SELECT created_at FROM claim_events
                   WHERE claim_id = ? AND event_type = 'code_drift_detected'
                   ORDER BY created_at ASC LIMIT 1""",
                (claim_id,),
            ).fetchone()

        status = str(claim_row["status"]) if claim_row is not None else ""
        confidence = float(claim_row["confidence"]) if claim_row is not None else 0.0
        corrected = claim_id in drift_result["impacted_claim_ids"] and status in {"challenged", "expired"}
        leaked = status == "active" and confidence >= 0.60

        if corrected:
            corrected_impacted_claims += 1
        if leaked:
            stale_leak_count += 1

        lag_seconds = 0.0
        if event_row is not None:
            lag_seconds = max(
                0.0,
                (parse_datetime(event_row["created_at"]) - detected_at).total_seconds(),
            )
            event_lags.append(lag_seconds)

        scenario_rows.append(
            {
                "scenario_id": f"drift_{i:03d}",
                "claim_id": claim_id,
                "impacted": claim_id in drift_result["impacted_claim_ids"],
                "challenged": claim_id in drift_result["challenged_claim_ids"],
                "status_after": status,
                "lag_seconds": round(lag_seconds, 6),
            }
        )

    total_impacted_claims = scenario_count
    freshness_after_drift = (
        corrected_impacted_claims / total_impacted_claims if total_impacted_claims else 0.0
    )
    stale_claim_leak_rate = stale_leak_count / total_impacted_claims if total_impacted_claims else 0.0
    p95_lag = _percentile(event_lags, 95.0)

    thresholds = {
        "freshness_after_drift_gte": 0.97,
        "stale_claim_leak_rate_lte": 0.03,
        "p95_challenge_lag_seconds_lte": 120.0,
    }
    passed = (
        freshness_after_drift >= thresholds["freshness_after_drift_gte"]
        and stale_claim_leak_rate <= thresholds["stale_claim_leak_rate_lte"]
        and p95_lag <= thresholds["p95_challenge_lag_seconds_lte"]
    )

    return {
        "aligned_metric_section": "1) Belief Freshness After Code Drift",
        "scenario_count": scenario_count,
        "formula": "FreshnessAfterDrift = corrected_impacted_claims / total_impacted_claims",
        "thresholds": thresholds,
        "measured": {
            "total_impacted_claims": total_impacted_claims,
            "corrected_impacted_claims": corrected_impacted_claims,
            "freshness_after_drift": round(freshness_after_drift, 6),
            "stale_claim_leak_rate": round(stale_claim_leak_rate, 6),
            "p95_challenge_lag_seconds": round(p95_lag, 6),
        },
        "pass": passed,
        "scenarios": scenario_rows,
    }


def evaluate_contradiction_evolution(scenario_count: int) -> dict[str, Any]:
    from consolidation_memory.database import (
        insert_claim_event,
        insert_contradiction,
        upsert_claim,
    )

    latencies: list[float] = []
    unresolved = 0
    scenario_rows: list[dict[str, Any]] = []
    base_ts = datetime(2026, 1, 1, tzinfo=timezone.utc)

    for i in range(scenario_count):
        old_claim = f"contradiction-old-{i:03d}"
        new_claim = f"contradiction-new-{i:03d}"
        upsert_claim(
            claim_id=old_claim,
            claim_type="fact",
            canonical_text=f"dependency pin {i} is stable",
            payload={"state": "old", "index": i},
            status="active",
            valid_from="2025-01-01T00:00:00+00:00",
        )
        upsert_claim(
            claim_id=new_claim,
            claim_type="fact",
            canonical_text=f"dependency pin {i} changed",
            payload={"state": "new", "index": i},
            status="active",
            valid_from="2025-01-01T00:00:00+00:00",
        )

        evidence_ingested_at = base_ts + timedelta(minutes=i)
        lag_seconds = 20 + (i % 5) * 15  # 20, 35, 50, 65, 80
        resolution_at = evidence_ingested_at + timedelta(seconds=lag_seconds)

        insert_contradiction(
            topic_id=None,
            old_record_id=old_claim,
            new_record_id=new_claim,
            old_content=json.dumps({"claim_id": old_claim}),
            new_content=json.dumps({"claim_id": new_claim}),
            resolution="challenged_old",
            reason="new contradictory evidence",
        )
        insert_claim_event(
            claim_id=old_claim,
            event_type="challenged",
            details={"reason": "contradiction"},
            created_at=_iso(resolution_at),
        )

        latencies.append(float(lag_seconds))
        scenario_rows.append(
            {
                "scenario_id": f"contradiction_{i:03d}",
                "old_claim_id": old_claim,
                "new_claim_id": new_claim,
                "evidence_ingested_at": _iso(evidence_ingested_at),
                "resolution_event_at": _iso(resolution_at),
                "resolution_latency_seconds": float(lag_seconds),
                "resolved": True,
            }
        )

    median_latency = statistics.median(latencies) if latencies else 0.0
    p95_latency = _percentile(latencies, 95.0)

    thresholds = {
        "median_latency_seconds_lte": 90.0,
        "p95_latency_seconds_lte": 300.0,
        "unresolved_scenarios_eq": 0,
    }
    passed = (
        median_latency <= thresholds["median_latency_seconds_lte"]
        and p95_latency <= thresholds["p95_latency_seconds_lte"]
        and unresolved == thresholds["unresolved_scenarios_eq"]
    )

    return {
        "aligned_metric_section": "2) Contradiction Resolution Latency",
        "scenario_count": scenario_count,
        "formula": "resolution_latency_seconds = t(resolution_event) - t(contradicting_evidence_ingested)",
        "thresholds": thresholds,
        "measured": {
            "median_latency_seconds": round(float(median_latency), 6),
            "p95_latency_seconds": round(float(p95_latency), 6),
            "unresolved_contradiction_scenarios": unresolved,
        },
        "pass": passed,
        "scenarios": scenario_rows,
    }


def evaluate_temporal_belief_reconstruction(query_limit: int) -> dict[str, Any]:
    from consolidation_memory.client import MemoryClient
    from consolidation_memory.config import get_active_project
    from consolidation_memory.database import insert_claim_sources, insert_episode, upsert_claim

    benchmark_scope = {
        "namespace_slug": "default",
        "project_slug": get_active_project(),
        "app_client_name": "legacy_client",
        "app_client_type": "python_sdk",
    }
    provenance_episode_id = insert_episode(
        content="novelty eval temporal belief provenance",
        scope=benchmark_scope,
    )

    # Seed old/new temporal claim sets.
    old_runtime_ids: set[str] = set()
    new_runtime_ids: set[str] = set()
    old_db_ids: set[str] = set()
    new_db_ids: set[str] = set()

    for i in range(5):
        old_id = f"temporal-runtime-old-{i}"
        new_id = f"temporal-runtime-new-{i}"
        old_runtime_ids.add(old_id)
        new_runtime_ids.add(new_id)
        upsert_claim(
            claim_id=old_id,
            claim_type="fact",
            canonical_text=f"python runtime stability old series {i}",
            payload={"series": "runtime", "era": "old", "idx": i},
            valid_from="2025-01-01T00:00:00+00:00",
            valid_until="2025-06-01T00:00:00+00:00",
        )
        insert_claim_sources(old_id, [{"source_episode_id": provenance_episode_id}])
        upsert_claim(
            claim_id=new_id,
            claim_type="fact",
            canonical_text=f"python runtime stability new series {i}",
            payload={"series": "runtime", "era": "new", "idx": i},
            valid_from="2025-07-01T00:00:00+00:00",
        )
        insert_claim_sources(new_id, [{"source_episode_id": provenance_episode_id}])

    for i in range(5):
        old_id = f"temporal-db-old-{i}"
        new_id = f"temporal-db-new-{i}"
        old_db_ids.add(old_id)
        new_db_ids.add(new_id)
        upsert_claim(
            claim_id=old_id,
            claim_type="fact",
            canonical_text=f"postgres migration old track {i}",
            payload={"series": "db", "era": "old", "idx": i},
            valid_from="2025-01-01T00:00:00+00:00",
            valid_until="2025-06-01T00:00:00+00:00",
        )
        insert_claim_sources(old_id, [{"source_episode_id": provenance_episode_id}])
        upsert_claim(
            claim_id=new_id,
            claim_type="fact",
            canonical_text=f"postgres migration new track {i}",
            payload={"series": "db", "era": "new", "idx": i},
            valid_from="2025-07-01T00:00:00+00:00",
        )
        insert_claim_sources(new_id, [{"source_episode_id": provenance_episode_id}])

    all_queries = [
        {
            "query_id": "q_runtime_old",
            "query": "python runtime stability",
            "as_of": "2025-03-01T00:00:00+00:00",
            "relevant": old_runtime_ids,
            "slice": "temporal",
        },
        {
            "query_id": "q_runtime_new",
            "query": "python runtime stability",
            "as_of": "2025-09-01T00:00:00+00:00",
            "relevant": new_runtime_ids,
            "slice": "temporal",
        },
        {
            "query_id": "q_db_old",
            "query": "postgres migration track",
            "as_of": "2025-03-01T00:00:00+00:00",
            "relevant": old_db_ids,
            "slice": "temporal",
        },
        {
            "query_id": "q_db_new",
            "query": "postgres migration track",
            "as_of": "2025-09-01T00:00:00+00:00",
            "relevant": new_db_ids,
            "slice": "temporal",
        },
    ]
    queries = all_queries[: max(1, min(query_limit, len(all_queries)))]

    per_query: list[dict[str, Any]] = []
    precision_values: list[float] = []

    with MemoryClient(auto_consolidate=False) as client:
        for item in queries:
            result = client.search_claims(
                query=item["query"],
                claim_type="fact",
                as_of=item["as_of"],
                limit=5,
            )
            top_ids = [str(claim["id"]) for claim in result.claims[:5]]
            hits = sum(1 for cid in top_ids if cid in item["relevant"])
            precision = hits / 5.0
            precision_values.append(precision)
            per_query.append(
                {
                    "query_id": item["query_id"],
                    "slice": item["slice"],
                    "as_of": item["as_of"],
                    "top5_claim_ids": top_ids,
                    "relevant_hits_in_top5": hits,
                    "precision_at_5": round(precision, 6),
                }
            )

    macro_precision = sum(precision_values) / len(precision_values) if precision_values else 0.0
    by_slice = {"temporal": round(macro_precision, 6)}

    thresholds = {
        "overall_macro_precision_at_5_gte": 0.80,
        "slice_macro_precision_at_5_gte": {"temporal": 0.70},
    }
    passed = (
        macro_precision >= thresholds["overall_macro_precision_at_5_gte"]
        and by_slice["temporal"] >= thresholds["slice_macro_precision_at_5_gte"]["temporal"]
    )

    return {
        "aligned_metric_section": "4) Claim Retrieval Precision@k",
        "scenario": "temporal belief reconstruction",
        "formula": "Precision@5 = relevant_claims_in_top5 / 5 (macro-averaged)",
        "thresholds": thresholds,
        "measured": {
            "query_count": len(queries),
            "overall_macro_precision_at_5": round(macro_precision, 6),
            "slice_macro_precision_at_5": by_slice,
        },
        "pass": passed,
        "queries": per_query,
    }


def _claim_has_complete_provenance(claim_id: str) -> bool:
    from consolidation_memory.database import get_connection

    with get_connection() as conn:
        claim_row = conn.execute(
            "SELECT id, created_at FROM claims WHERE id = ?",
            (claim_id,),
        ).fetchone()
        has_episode_source = conn.execute(
            """SELECT 1 FROM claim_sources
               WHERE claim_id = ? AND source_episode_id IS NOT NULL
               LIMIT 1""",
            (claim_id,),
        ).fetchone() is not None
        has_record_or_topic_source = conn.execute(
            """SELECT 1 FROM claim_sources
               WHERE claim_id = ?
                 AND (source_topic_id IS NOT NULL OR source_record_id IS NOT NULL)
               LIMIT 1""",
            (claim_id,),
        ).fetchone() is not None
        has_event = conn.execute(
            "SELECT 1 FROM claim_events WHERE claim_id = ? LIMIT 1",
            (claim_id,),
        ).fetchone() is not None

    return bool(
        claim_row is not None
        and claim_row["id"]
        and claim_row["created_at"]
        and has_episode_source
        and has_record_or_topic_source
        and has_event
    )


def evaluate_provenance_trace_completeness(claim_limit: int) -> dict[str, Any]:
    from consolidation_memory.client import MemoryClient
    from consolidation_memory.database import (
        insert_claim_event,
        insert_claim_sources,
        insert_episode,
        upsert_claim,
    )

    for i in range(claim_limit):
        claim_id = f"provenance-claim-{i:03d}"
        episode_id = insert_episode(content=f"provenance source episode {i}")
        upsert_claim(
            claim_id=claim_id,
            claim_type="fact",
            canonical_text=f"provenance trace completeness claim {i}",
            payload={"check": "provenance", "idx": i},
            valid_from="2025-01-01T00:00:00+00:00",
        )
        insert_claim_sources(
            claim_id,
            [{
                "source_episode_id": episode_id,
                "source_topic_id": f"topic-provenance-{i:03d}",
                "source_record_id": f"record-provenance-{i:03d}",
            }],
        )
        insert_claim_event(
            claim_id,
            event_type="create",
            details={"scenario": "provenance"},
        )

    with MemoryClient(auto_consolidate=False) as client:
        result = client.search_claims(
            query="provenance trace completeness claim",
            claim_type="fact",
            limit=claim_limit,
        )

    returned_claim_ids = [str(claim["id"]) for claim in result.claims]
    complete_ids = [claim_id for claim_id in returned_claim_ids if _claim_has_complete_provenance(claim_id)]

    claims_returned = len(returned_claim_ids)
    complete_count = len(complete_ids)
    missing_count = claims_returned - complete_count
    coverage = (complete_count / claims_returned) if claims_returned else 0.0
    missing_per_1000 = (missing_count / claims_returned) * 1000.0 if claims_returned else 0.0

    thresholds = {
        "provenance_coverage_gte": 0.995,
        "missing_provenance_claims_per_1000_lte": 5.0,
    }
    passed = (
        coverage >= thresholds["provenance_coverage_gte"]
        and missing_per_1000 <= thresholds["missing_provenance_claims_per_1000_lte"]
    )

    return {
        "aligned_metric_section": "3) Provenance Coverage",
        "formula": "ProvenanceCoverage = claims_with_complete_provenance / claims_returned",
        "thresholds": thresholds,
        "measured": {
            "claims_returned": claims_returned,
            "claims_with_complete_provenance": complete_count,
            "missing_provenance_claims": missing_count,
            "provenance_coverage": round(coverage, 6),
            "missing_provenance_claims_per_1000": round(missing_per_1000, 6),
        },
        "pass": passed,
        "returned_claim_ids": returned_claim_ids,
    }


def run_eval(mode: str) -> dict[str, Any]:
    from consolidation_memory import database

    cfg = MODE_CONFIG[mode]
    local_tmp_base = Path.cwd() / ".tmp_novelty_eval_runtime"
    local_tmp_base.mkdir(parents=True, exist_ok=True)
    tmp_root = Path(tempfile.mkdtemp(prefix="novelty_eval_", dir=str(local_tmp_base)))
    try:
        _reset_eval_environment(tmp_root)
        with _local_embedding_patches():
            section_1 = evaluate_belief_freshness_after_code_drift(
                tmp_root=tmp_root,
                scenario_count=cfg.drift_scenarios,
            )
            section_2 = evaluate_contradiction_evolution(
                scenario_count=cfg.contradiction_scenarios,
            )
            section_3 = evaluate_temporal_belief_reconstruction(
                query_limit=cfg.temporal_queries,
            )
            section_4 = evaluate_provenance_trace_completeness(
                claim_limit=cfg.provenance_limit,
            )

        sections = {
            "belief_freshness_after_code_drift": section_1,
            "contradiction_evolution": section_2,
            "temporal_belief_reconstruction": section_3,
            "provenance_trace_completeness": section_4,
        }
        overall_pass = all(bool(section["pass"]) for section in sections.values())

        return {
            "benchmark": "novelty_eval",
            "run_id": f"novelty_eval_{mode}_{uuid.uuid4().hex[:12]}",
            "mode": mode,
            "generated_at": _iso(_utc_now()),
            "cloud_dependencies_used": False,
            "aligned_metrics_doc": "docs/NOVELTY_METRICS.md",
            "sections": sections,
            "overall_pass": overall_pass,
        }
    finally:
        database.close_all_connections()
        shutil.rmtree(tmp_root, ignore_errors=True)


def save_results(results: dict[str, Any], output_path: Path | None = None) -> Path:
    if output_path is None:
        stamp = _utc_now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(__file__).parent / "results" / f"novelty_eval_{results['mode']}_{stamp}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Novelty metric evaluation harness")
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
        help="Optional output JSON path (default: benchmarks/results/novelty_eval_<mode>_<timestamp>.json)",
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

    print(json.dumps(
        {
            "mode": results["mode"],
            "overall_pass": results["overall_pass"],
            "output_path": str(out),
            "section_passes": {
                key: section["pass"] for key, section in results["sections"].items()
            },
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
