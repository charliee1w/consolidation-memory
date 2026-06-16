#!/usr/bin/env python3
"""Demonstrate recall + drift challenge vs opaque RAG retrieval."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

from consolidation_memory import MemoryClient
from consolidation_memory.backends import reset_backends
from consolidation_memory import claim_cache, database, record_cache, topic_cache
from consolidation_memory.config import reset_config


def _isolated_client() -> MemoryClient:
    tmp = tempfile.mkdtemp(prefix="trust_vs_rag_")
    tmp_path = Path(tmp)
    for sub in ("knowledge", "knowledge/versions", "consolidation_logs", "backups"):
        (tmp_path / "projects" / "demo" / sub).mkdir(parents=True, exist_ok=True)

    database.close_all_connections()
    reset_backends()
    topic_cache.invalidate()
    record_cache.invalidate()
    claim_cache.invalidate()
    reset_config(
        _base_data_dir=tmp_path,
        active_project="demo",
        CONSOLIDATION_MIN_CLUSTER_SIZE=1,
        LLM_BACKEND="disabled",
    )
    os.environ.setdefault("LLM_BACKEND", "disabled")
    return MemoryClient(auto_consolidate=False)


def _init_git_repo(repo: Path, anchor_path: str) -> None:
    """Create a git repo with anchor_path committed, then delete it (drift)."""
    old_file = repo / anchor_path
    old_file.parent.mkdir(parents=True, exist_ok=True)
    old_file.write_text("# auth\n", encoding="utf-8")

    git_env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "demo",
        "GIT_AUTHOR_EMAIL": "demo@example.com",
        "GIT_COMMITTER_NAME": "demo",
        "GIT_COMMITTER_EMAIL": "demo@example.com",
    }
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=repo,
        check=True,
        capture_output=True,
        env=git_env,
    )
    old_file.unlink()


def main() -> None:
    solution = (
        "Problem: JWT refresh tokens ignored SameSite on login redirect. "
        "Fix: set cookie flags in src/demo_app/auth.py handle_login() before redirect."
    )
    anchor_path = "src/demo_app/auth.py"

    with _isolated_client() as mem:
        store = mem.store(
            solution,
            content_type="solution",
            tags=["trust-vs-rag", "demo"],
        )
        print(f"Stored episode: {store.id}")

        report = mem.consolidate()
        print(
            f"Consolidation: status={report.get('status')} "
            f"fast_path_hits={report.get('fast_path_hits', 0)}"
        )

        recall_before = mem.recall(
            "JWT refresh tokens SameSite login redirect",
            n_results=5,
            include_knowledge=True,
        )
        active_claims = [
            c for c in (recall_before.claims or []) if c.get("status") == "active"
        ]
        print(f"Recall before drift: {len(active_claims)} active claims in top results")

        with tempfile.TemporaryDirectory() as repo_dir:
            repo = Path(repo_dir)
            _init_git_repo(repo, anchor_path)

            drift = mem.detect_drift(base_ref="HEAD", repo_path=str(repo))
            challenged = drift.get("challenged_claim_ids") or []
            impacted = drift.get("impacted_claim_ids") or []
            print(
                f"Drift scan: impacted={len(impacted)} challenged={len(challenged)}"
            )
            for impact in (drift.get("impacts") or [])[:3]:
                print(
                    f"  - claim {impact.get('claim_id')}: "
                    f"{impact.get('previous_status')} → {impact.get('new_status')}"
                )

        recall_after = mem.recall(
            "JWT refresh tokens SameSite login redirect",
            n_results=5,
            include_knowledge=True,
            hypothesis_competition=True,
        )
        claims = recall_after.claims or []
        print("\nTop claims after drift (hypothesis_competition=True):")
        for claim in claims[:3]:
            print(
                f"  - [{claim.get('status')}] "
                f"{str(claim.get('canonical_text') or '')[:100]}…"
            )

        print("\nFull recall envelope (truncated):")
        payload = {
            "episodes": len(recall_after.episodes or []),
            "claims": len(claims),
            "knowledge": len(recall_after.knowledge or []),
        }
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()