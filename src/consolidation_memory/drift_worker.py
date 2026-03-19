"""Lean drift-detection worker used by CLI and subprocess callers.

This avoids importing the full MemoryClient stack when drift detection only
needs git, config, and database access.
"""

from __future__ import annotations

import argparse
import json
import sys

from consolidation_memory.types import DriftOutput

_DEFAULT_NAMESPACE_SLUG = "default"


def run_detect_drift_worker(
    *,
    base_ref: str | None = None,
    repo_path: str | None = None,
    project: str | None = None,
) -> DriftOutput:
    """Execute drift detection under the requested project scope."""
    from consolidation_memory.config import set_active_project
    from consolidation_memory.database import ensure_schema
    from consolidation_memory.drift import detect_code_drift

    active_project = set_active_project(project)
    ensure_schema()
    return detect_code_drift(
        base_ref=base_ref,
        repo_path=repo_path,
        scope={
            "namespace_slug": _DEFAULT_NAMESPACE_SLUG,
            "project_slug": active_project,
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m consolidation_memory.drift_worker",
        description="Run consolidation-memory drift detection in a lean worker process.",
    )
    parser.add_argument(
        "--project",
        default=None,
        help="Project namespace (default: current CONSOLIDATION_MEMORY_PROJECT or config default)",
    )
    parser.add_argument(
        "--base-ref",
        default=None,
        help="Optional git base ref for comparison (e.g. origin/main)",
    )
    parser.add_argument(
        "--repo-path",
        default=None,
        help="Repository path (default: current working directory)",
    )
    args = parser.parse_args(argv)

    try:
        result = run_detect_drift_worker(
            base_ref=args.base_ref,
            repo_path=args.repo_path,
            project=args.project,
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(json.dumps(result, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
