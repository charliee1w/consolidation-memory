#!/usr/bin/env python3
"""Release automation for consolidation-memory.

Usage:
    python scripts/release.py 0.4.0           # bump, test, commit, tag, push
    python scripts/release.py 0.4.0 --dry-run # show what would happen
    python scripts/release.py 0.4.0 --no-push # commit + tag but don't push
"""

import argparse
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
CHANGELOG = ROOT / "CHANGELOG.md"


def run(cmd: str, *, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    print(f"  $ {cmd}")
    return subprocess.run(cmd, shell=True, cwd=str(ROOT), check=check, capture_output=capture, text=True)


def get_current_version() -> str:
    text = PYPROJECT.read_text()
    match = re.search(r'^version\s*=\s*"(.+?)"', text, re.MULTILINE)
    if not match:
        sys.exit("Could not find version in pyproject.toml")
    return match.group(1)


def set_version(new_version: str) -> None:
    text = PYPROJECT.read_text()
    text = re.sub(r'^(version\s*=\s*)"(.+?)"', rf'\1"{new_version}"', text, count=1, flags=re.MULTILINE)
    PYPROJECT.write_text(text)


def add_changelog_header(new_version: str) -> None:
    text = CHANGELOG.read_text()
    today = date.today().isoformat()
    header = f"## {new_version} — {today}\n"

    # Insert after "# Changelog\n\n"
    marker = "# Changelog\n"
    if marker not in text:
        sys.exit("Could not find '# Changelog' header in CHANGELOG.md")

    # Check if this version already has an entry
    if f"## {new_version}" in text:
        print(f"  Changelog already has entry for {new_version}, skipping header insertion")
        return

    text = text.replace(marker, f"{marker}\n{header}\n", 1)
    CHANGELOG.write_text(text)


def main():
    parser = argparse.ArgumentParser(description="Release consolidation-memory")
    parser.add_argument("version", help="New version (e.g. 0.4.0)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen")
    parser.add_argument("--no-push", action="store_true", help="Commit + tag but don't push")
    args = parser.parse_args()

    new_version = args.version

    # Pre-flight checks
    if not PYPROJECT.exists():
        sys.exit(f"pyproject.toml not found at {PYPROJECT}")
    if not CHANGELOG.exists():
        sys.exit(f"CHANGELOG.md not found at {CHANGELOG}")

    if not re.match(r"^\d+\.\d+\.\d+$", new_version):
        sys.exit(f"Invalid version format: {new_version} (expected X.Y.Z)")

    current = get_current_version()

    print(f"\nRelease: {current} -> {new_version}")
    print(f"{'(DRY RUN)' if args.dry_run else ''}\n")

    # 1. Check working tree is clean
    result = run("git status --porcelain", capture=True)
    if result.stdout.strip():
        sys.exit("Working tree is not clean. Commit or stash changes first.")

    # 2. Pull latest
    print("\n[1/6] Pulling latest from origin...")
    if not args.dry_run:
        run("git pull --ff-only origin main")

    # 3. Bump version in pyproject.toml (single source of truth)
    print(f"\n[2/6] Bumping version: {current} -> {new_version}")
    if not args.dry_run:
        set_version(new_version)

    # 4. Add changelog header
    print(f"\n[3/6] Adding changelog header for {new_version}")
    if not args.dry_run:
        add_changelog_header(new_version)
    print("  -> Edit CHANGELOG.md now to add release notes, then re-run without --dry-run")
    print("     Or add notes before running this script.")

    # 5. Reinstall and run tests
    print("\n[4/6] Reinstalling and running tests...")
    if not args.dry_run:
        run("pip install -e \".[fastembed,dev]\" --quiet")
        result = run("python -m pytest tests/ -v", check=False)
        if result.returncode != 0:
            sys.exit("Tests failed — aborting release. Fix failures and re-run.")
        result = run("python -m ruff check src/ tests/", check=False)
        if result.returncode != 0:
            sys.exit("Lint check failed — aborting release. Fix lint errors and re-run.")

    # 6. Commit + tag
    print(f"\n[5/6] Committing v{new_version}...")
    if not args.dry_run:
        run("git add pyproject.toml CHANGELOG.md")
        run(f'git commit -m "v{new_version}"')
        run(f"git tag v{new_version}")

    # 7. Push
    if args.no_push:
        print("\n[6/6] Skipping push (--no-push). Run manually:")
        print(f"  git push origin main && git push origin v{new_version}")
    elif args.dry_run:
        print(f"\n[6/6] Would push main + tag v{new_version}")
    else:
        print(f"\n[6/6] Pushing main + tag v{new_version}...")
        run("git push origin main")
        run(f"git push origin v{new_version}")

    print(f"\nDone! v{new_version} {'would be' if args.dry_run else 'is'} released.")
    if not args.dry_run and not args.no_push:
        print(f"PyPI publish will trigger from the v{new_version} tag via GitHub Actions.")


if __name__ == "__main__":
    main()
