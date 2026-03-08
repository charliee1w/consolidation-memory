#!/usr/bin/env python3
"""Release automation for consolidation-memory.

Usage:
    python scripts/release.py --bump patch       # auto-bump patch, test, gate, commit, tag, push
    python scripts/release.py --bump minor       # auto-bump minor
    python scripts/release.py 0.14.0 --dry-run   # explicit version preview
    python scripts/release.py --bump patch --no-push
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess  # nosec B404
import sys
from datetime import date
from pathlib import Path
from urllib import error, request

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
CHANGELOG = ROOT / "CHANGELOG.md"
PACKAGE_NAME = "consolidation-memory"
RELEASE_SCOPE_USE_CASE = "Drift-aware debugging memory"
NOVELTY_RELEASE_RESULT = ROOT / "benchmarks" / "results" / "novelty_eval_release_full.json"
RELEASE_GATE_REPORT = ROOT / "benchmarks" / "results" / "release_gate_report.json"
SEMVER_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


def run(cmd: list[str], *, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(  # nosec B603
        cmd,
        cwd=str(ROOT),
        check=check,
        capture_output=capture,
        text=True,
    )


def parse_semver(version: str) -> tuple[int, int, int]:
    match = SEMVER_RE.match(version)
    if not match:
        raise ValueError(f"Invalid version format: {version} (expected X.Y.Z)")
    return tuple(int(part) for part in match.groups())


def bump_semver(version: str, bump: str) -> str:
    major, minor, patch = parse_semver(version)
    if bump == "patch":
        patch += 1
    elif bump == "minor":
        minor += 1
        patch = 0
    elif bump == "major":
        major += 1
        minor = 0
        patch = 0
    else:
        raise ValueError(f"Unsupported bump type: {bump}")
    return f"{major}.{minor}.{patch}"


def get_current_version() -> str:
    text = PYPROJECT.read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"(.+?)"', text, re.MULTILINE)
    if not match:
        raise RuntimeError("Could not find version in pyproject.toml")
    return match.group(1)


def _update_pyproject_version_text(text: str, new_version: str) -> str:
    return re.sub(
        r'^(version\s*=\s*)"(.+?)"',
        rf'\1"{new_version}"',
        text,
        count=1,
        flags=re.MULTILINE,
    )


def set_version(new_version: str) -> None:
    text = PYPROJECT.read_text(encoding="utf-8")
    updated = _update_pyproject_version_text(text, new_version)
    if updated == text:
        raise RuntimeError("Version substitution failed: no version line was updated in pyproject.toml")
    PYPROJECT.write_text(updated, encoding="utf-8")


def get_latest_tag() -> str | None:
    result = run(["git", "describe", "--tags", "--abbrev=0"], check=False, capture=True)
    if result.returncode != 0:
        return None
    tag = result.stdout.strip()
    return tag or None


def collect_release_notes(previous_tag: str | None) -> list[str]:
    cmd = ["git", "log", "--pretty=format:%s"]
    if previous_tag:
        cmd.append(f"{previous_tag}..HEAD")
    result = run(cmd, capture=True)
    notes: list[str] = []
    seen: set[str] = set()
    for line in result.stdout.splitlines():
        subject = line.strip()
        if not subject:
            continue
        if re.match(r"^v\d+\.\d+\.\d+\b", subject):
            continue
        if subject in seen:
            continue
        seen.add(subject)
        notes.append(subject)
    notes.reverse()
    return notes[:12]


def render_changelog_entry(new_version: str, notes: list[str]) -> str:
    today = date.today().isoformat()
    cleaned = [note.strip() for note in notes if note.strip()]
    if not cleaned:
        cleaned = ["Maintenance release."]
    bullets = "\n".join(f"- {note}" for note in cleaned)
    return f"## {new_version} - {today}\n\n### Highlights\n\n{bullets}\n"


def add_changelog_entry(new_version: str, notes: list[str]) -> bool:
    text = CHANGELOG.read_text(encoding="utf-8")
    marker = "# Changelog\n"
    if marker not in text:
        raise RuntimeError("Could not find '# Changelog' header in CHANGELOG.md")
    if re.search(rf"^##\s+{re.escape(new_version)}\b", text, flags=re.MULTILINE):
        print(f"  Changelog already has entry for {new_version}, skipping insertion")
        return False
    entry = render_changelog_entry(new_version, notes)
    updated = text.replace(marker, f"{marker}\n{entry}\n", 1)
    CHANGELOG.write_text(updated, encoding="utf-8")
    return True


def tag_exists(tag_name: str) -> bool:
    result = run(["git", "rev-parse", "-q", "--verify", f"refs/tags/{tag_name}"], check=False, capture=True)
    return result.returncode == 0


def version_exists_on_pypi(version: str) -> bool:
    url = f"https://pypi.org/pypi/{PACKAGE_NAME}/json"
    try:
        with request.urlopen(url, timeout=15) as response:  # nosec B310
            payload = json.load(response)
    except error.HTTPError as exc:
        if exc.code == 404:
            return False
        raise RuntimeError(f"Failed to query {url}: HTTP {exc.code}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Failed to query {url}: {exc.reason}") from exc
    releases = payload.get("releases")
    if not isinstance(releases, dict):
        raise RuntimeError("Unexpected PyPI payload: missing 'releases'")
    return version in releases


def rollback_release_files(pyproject_text: str, changelog_text: str) -> None:
    print("  Rolling back pyproject.toml and CHANGELOG.md...")
    PYPROJECT.write_text(pyproject_text, encoding="utf-8")
    CHANGELOG.write_text(changelog_text, encoding="utf-8")


def resolve_target_version(current: str, explicit_version: str | None, bump: str) -> str:
    if explicit_version:
        parse_semver(explicit_version)
        return explicit_version
    return bump_semver(current, bump)


def main() -> None:
    parser = argparse.ArgumentParser(description="Release consolidation-memory")
    parser.add_argument(
        "version",
        nargs="?",
        help="Explicit release version (e.g. 0.14.0). If omitted, --bump is applied.",
    )
    parser.add_argument(
        "--bump",
        choices=["patch", "minor", "major"],
        default="patch",
        help="Semver bump to apply when explicit version is omitted (default: patch).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen")
    parser.add_argument("--no-push", action="store_true", help="Commit + tag but don't push")
    parser.add_argument(
        "--skip-pypi-check",
        action="store_true",
        help="Skip remote PyPI version collision check.",
    )
    args = parser.parse_args()

    if not PYPROJECT.exists():
        sys.exit(f"pyproject.toml not found at {PYPROJECT}")
    if not CHANGELOG.exists():
        sys.exit(f"CHANGELOG.md not found at {CHANGELOG}")

    try:
        current = get_current_version()
        new_version = resolve_target_version(current, args.version, args.bump)
    except (RuntimeError, ValueError) as exc:
        sys.exit(str(exc))

    if new_version == current:
        sys.exit(
            f"Requested version {new_version} matches current version. "
            "Use a higher explicit version or --bump."
        )

    tag_name = f"v{new_version}"

    print(f"\nRelease: {current} -> {new_version}")
    print(f"{'(DRY RUN)' if args.dry_run else ''}\n")

    result = run(["git", "status", "--porcelain"], capture=True)
    if result.stdout.strip():
        sys.exit("Working tree is not clean. Commit or stash changes first.")

    print("\n[1/8] Pulling latest from origin...")
    if not args.dry_run:
        run(["git", "pull", "--ff-only", "origin", "main"])

    print(f"\n[2/8] Validating release target {tag_name}...")
    if tag_exists(tag_name):
        sys.exit(f"Tag {tag_name} already exists. Choose a different version.")
    if not args.skip_pypi_check:
        try:
            exists_on_pypi = version_exists_on_pypi(new_version)
        except RuntimeError as exc:
            if args.dry_run:
                print(f"  Warning: {exc}")
                exists_on_pypi = False
            else:
                sys.exit(str(exc))
        if exists_on_pypi:
            sys.exit(f"{PACKAGE_NAME} {new_version} already exists on PyPI.")

    previous_tag = get_latest_tag()
    release_notes = collect_release_notes(previous_tag)

    print(f"\n[3/8] Preparing files for {new_version}...")
    pyproject_before = PYPROJECT.read_text(encoding="utf-8")
    changelog_before = CHANGELOG.read_text(encoding="utf-8")
    if not args.dry_run:
        try:
            set_version(new_version)
            actual = get_current_version()
            if actual != new_version:
                raise RuntimeError(
                    f"Version substitution failed: expected {new_version}, got {actual} in pyproject.toml"
                )
            inserted = add_changelog_entry(new_version, release_notes)
            if inserted:
                source = previous_tag or "repository start"
                print(f"  Added changelog entry from commits since {source}.")
        except Exception as exc:  # pragma: no cover - defensive rollback path
            rollback_release_files(pyproject_before, changelog_before)
            sys.exit(f"Failed to prepare release files: {exc}")

    print("\n[4/8] Reinstalling and running tests...")
    if not args.dry_run:
        run([sys.executable, "-m", "pip", "install", "-e", ".[fastembed,dev]", "--quiet"])
        result = run([sys.executable, "-m", "pytest", "tests/", "-v"], check=False)
        if result.returncode != 0:
            rollback_release_files(pyproject_before, changelog_before)
            sys.exit("Tests failed - aborting release. Files reverted.")
        result = run([sys.executable, "-m", "ruff", "check", "src/", "tests/"], check=False)
        if result.returncode != 0:
            rollback_release_files(pyproject_before, changelog_before)
            sys.exit("Lint check failed - aborting release. Files reverted.")

    print("\n[5/8] Running release gate checks...")
    if not args.dry_run:
        NOVELTY_RELEASE_RESULT.parent.mkdir(parents=True, exist_ok=True)
        result = run(
            [
                sys.executable,
                "-m",
                "benchmarks.novelty_eval",
                "--mode",
                "full",
                "--output",
                str(NOVELTY_RELEASE_RESULT),
            ],
            check=False,
        )
        if result.returncode != 0:
            rollback_release_files(pyproject_before, changelog_before)
            sys.exit("Full novelty evaluation failed - aborting release. Files reverted.")

        result = run(
            [
                sys.executable,
                "scripts/verify_release_gates.py",
                "--novelty-result",
                str(NOVELTY_RELEASE_RESULT),
                "--scope-use-case",
                RELEASE_SCOPE_USE_CASE,
                "--output",
                str(RELEASE_GATE_REPORT),
            ],
            check=False,
        )
        if result.returncode != 0:
            rollback_release_files(pyproject_before, changelog_before)
            sys.exit(
                "Release gate verification failed - aborting release. "
                "Files reverted. See benchmarks/results/release_gate_report.json."
            )

    print(f"\n[6/8] Committing {tag_name}...")
    if not args.dry_run:
        run(["git", "add", "pyproject.toml", "CHANGELOG.md"])
        run(["git", "commit", "-m", tag_name])
        run(["git", "tag", tag_name])

    if args.no_push:
        print("\n[7/8] Skipping push (--no-push). Run manually:")
        print(f"  git push origin main {tag_name}")
    elif args.dry_run:
        print(f"\n[7/8] Would push main + tag {tag_name}")
    else:
        print(f"\n[7/8] Pushing main + tag {tag_name}...")
        run(["git", "push", "origin", "main", tag_name])

    print(f"\n[8/8] Done! {tag_name} {'would be' if args.dry_run else 'is'} released.")
    if not args.dry_run and not args.no_push:
        print(f"PyPI publish will trigger from the {tag_name} tag via GitHub Actions.")


if __name__ == "__main__":
    main()
