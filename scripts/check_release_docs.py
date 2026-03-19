#!/usr/bin/env python3
"""Guardrails for release automation documentation freshness.

Fails CI when release automation code changes without matching doc updates.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess  # nosec B404
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RELEASE_AUTOMATION_PATHS = {
    ".github/workflows/release-on-main.yml",
    "scripts/release.py",
    "scripts/release_criteria.py",
}
RELEASE_DOC_PATHS = {
    "docs/RELEASE_AUTOMATION.md",
    "README.md",
}
RELEASE_AUTOMATION_LINK = "(docs/RELEASE_AUTOMATION.md)"
REQUIRED_RELEASE_DOC_MARKERS = (
    "RELEASE_AUTOMATION_PAT",
    "workflow_dispatch",
    "[skip release]",
    "[release major|minor|patch]",
)


def _run_git(args: list[str]) -> str:
    git_executable = shutil.which("git")
    if not git_executable:
        raise RuntimeError("git executable not found in PATH")
    result = subprocess.run(  # nosec B603
        [str(Path(git_executable).resolve()), *args],
        cwd=str(ROOT),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _changed_files(*, base_ref: str, head_ref: str) -> set[str]:
    output = _run_git(
        ["diff", "--name-only", "--diff-filter=ACMR", f"{base_ref}...{head_ref}"]
    )
    return {line.strip().replace("\\", "/") for line in output.splitlines() if line.strip()}


def evaluate_release_docs_guard(
    *,
    changed_files: set[str],
    release_doc_text: str,
    readme_text: str,
) -> list[str]:
    errors: list[str] = []
    release_changed = sorted(changed_files & RELEASE_AUTOMATION_PATHS)
    if not release_changed:
        return errors

    docs_changed = sorted(changed_files & RELEASE_DOC_PATHS)
    if not docs_changed:
        errors.append(
            "Release automation files changed without docs updates. "
            "Update docs/RELEASE_AUTOMATION.md or README.md in the same change."
        )

    missing_markers = [
        marker for marker in REQUIRED_RELEASE_DOC_MARKERS if marker not in release_doc_text
    ]
    if missing_markers:
        errors.append(
            "docs/RELEASE_AUTOMATION.md is missing required markers: "
            + ", ".join(missing_markers)
        )

    if RELEASE_AUTOMATION_LINK not in readme_text:
        errors.append("README.md is missing the release automation documentation link.")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fail when release automation code changes without docs maintenance."
    )
    parser.add_argument(
        "--base-ref",
        required=True,
        help="Base git ref/sha for diff comparison.",
    )
    parser.add_argument(
        "--head-ref",
        default="HEAD",
        help="Head git ref/sha for diff comparison (default: HEAD).",
    )
    args = parser.parse_args()

    changed = _changed_files(base_ref=args.base_ref, head_ref=args.head_ref)
    release_changed = sorted(changed & RELEASE_AUTOMATION_PATHS)
    if not release_changed:
        print("No release automation file changes detected. Docs guard passed.")
        return 0

    release_doc_path = ROOT / "docs" / "RELEASE_AUTOMATION.md"
    readme_path = ROOT / "README.md"
    errors = evaluate_release_docs_guard(
        changed_files=changed,
        release_doc_text=release_doc_path.read_text(encoding="utf-8"),
        readme_text=readme_path.read_text(encoding="utf-8"),
    )
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        print("Changed release automation files:", ", ".join(release_changed))
        return 1

    print("Release docs guard passed.")
    print("Changed release automation files:", ", ".join(release_changed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
