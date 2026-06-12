#!/usr/bin/env python3
"""Update CHANGELOG.md from git history without running a full release.

Usage:
    python scripts/update_changelog.py                  # refresh ## Unreleased
    python scripts/update_changelog.py --dry-run        # preview only
    python scripts/update_changelog.py --commit       # commit changelog update
    python scripts/update_changelog.py --commit --push  # commit and push to main
"""

from __future__ import annotations

import argparse
import subprocess  # nosec B404
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHANGELOG = ROOT / "CHANGELOG.md"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.changelog_builder import upsert_unreleased_section  # noqa: E402


def run(cmd: list[str], *, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(  # nosec B603
        cmd,
        cwd=str(ROOT),
        check=check,
        capture_output=capture,
        text=True,
    )


def get_latest_tag() -> str | None:
    result = run(["git", "describe", "--tags", "--abbrev=0"], check=False, capture=True)
    if result.returncode != 0:
        return None
    tag = result.stdout.strip()
    return tag or None


def collect_commit_subjects(previous_tag: str | None) -> list[str]:
    cmd = ["git", "log", "--pretty=format:%s"]
    if previous_tag:
        cmd.append(f"{previous_tag}..HEAD")
    result = run(cmd, capture=True)
    subjects = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    subjects.reverse()
    return subjects


def main() -> None:
    parser = argparse.ArgumentParser(description="Update CHANGELOG.md Unreleased section.")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing.")
    parser.add_argument("--commit", action="store_true", help="Commit CHANGELOG.md if it changed.")
    parser.add_argument("--push", action="store_true", help="Push commit to origin/main (requires --commit).")
    args = parser.parse_args()

    if not CHANGELOG.exists():
        sys.exit(f"CHANGELOG.md not found at {CHANGELOG}")
    if args.push and not args.commit:
        sys.exit("--push requires --commit.")

    if args.commit:
        status = run(["git", "status", "--porcelain"], capture=True)
        if status.stdout.strip():
            sys.exit("Working tree is not clean. Commit or stash changes before --commit.")

    previous_tag = get_latest_tag()
    subjects = collect_commit_subjects(previous_tag)
    before = CHANGELOG.read_text(encoding="utf-8")
    after = upsert_unreleased_section(before, subjects)

    if after == before:
        print("CHANGELOG.md is already up to date.")
        return

    print("\nPlanned CHANGELOG.md update:")
    print("=" * 60)
    for line in after.splitlines()[:40]:
        print(line)
    if after.count("\n") > 40:
        print("...")
    print("=" * 60)

    if args.dry_run:
        print("\nDry run complete. No files changed.")
        return

    CHANGELOG.write_text(after, encoding="utf-8")
    print("\nUpdated CHANGELOG.md Unreleased section.")

    if not args.commit:
        return

    run(["git", "add", "CHANGELOG.md"])
    message = "chore(changelog): update Unreleased section [skip release]"
    run(["git", "commit", "-m", message])
    print(f"Committed changelog update ({message}).")

    if args.push:
        run(["git", "push", "origin", "main"])
        print("Pushed changelog update to origin/main.")


if __name__ == "__main__":
    main()