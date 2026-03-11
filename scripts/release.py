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
import os
import re
import subprocess  # nosec B404
import sys
import tempfile
from datetime import date
from pathlib import Path
from typing import NamedTuple
from urllib import error, request

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
CHANGELOG = ROOT / "CHANGELOG.md"
PACKAGE_NAME = "consolidation-memory"
RELEASE_SCOPE_USE_CASE = "Drift-aware debugging memory"
NOVELTY_RELEASE_RESULT = ROOT / "benchmarks" / "results" / "novelty_eval_release_full.json"
RELEASE_GATE_REPORT = ROOT / "benchmarks" / "results" / "release_gate_report.json"
EXPECTED_RELEASE_BRANCH = "main"
RELEASE_INSTALL_EXTRAS = ".[fastembed,rest,dev]"
SEMVER_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


class ReleaseCommand(NamedTuple):
    label: str
    cmd: list[str]
    failure_message: str
    env: dict[str, str] | None = None


def run(
    cmd: list[str],
    *,
    check: bool = True,
    capture: bool = False,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(cmd)}")
    merged_env = None if env is None else {**os.environ, **env}
    return subprocess.run(  # nosec B603
        cmd,
        cwd=str(ROOT),
        check=check,
        capture_output=capture,
        env=merged_env,
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


def get_current_branch() -> str:
    result = run(["git", "branch", "--show-current"], capture=True)
    branch = result.stdout.strip()
    if not branch:
        raise RuntimeError("Could not determine current git branch.")
    return branch


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


def remote_tag_exists(tag_name: str) -> bool:
    result = run(
        ["git", "ls-remote", "--tags", "--refs", "origin", f"refs/tags/{tag_name}"],
        check=False,
        capture=True,
    )
    return result.returncode == 0 and bool(result.stdout.strip())


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


def ensure_release_branch(expected_branch: str = EXPECTED_RELEASE_BRANCH) -> None:
    branch = get_current_branch()
    if branch != expected_branch:
        raise RuntimeError(
            f"Release automation must run from '{expected_branch}', found '{branch}'."
        )


def resolve_target_version(current: str, explicit_version: str | None, bump: str) -> str:
    if explicit_version:
        parse_semver(explicit_version)
        return explicit_version
    return bump_semver(current, bump)


def get_quality_gate_commands(python_executable: str, scratch_dir: Path) -> list[ReleaseCommand]:
    coverage_dir = scratch_dir / "coverage"
    coverage_dir.mkdir(parents=True, exist_ok=True)
    coverage_file = coverage_dir / ".coverage"
    coverage_xml = coverage_dir / "coverage.xml"
    coverage_env = {"COVERAGE_FILE": str(coverage_file)}

    return [
        ReleaseCommand(
            label="Install publish-grade release dependencies",
            cmd=[
                python_executable,
                "-m",
                "pip",
                "install",
                "-e",
                RELEASE_INSTALL_EXTRAS,
                "--quiet",
            ],
            failure_message="Dependency install failed - aborting release. Files reverted.",
        ),
        ReleaseCommand(
            label="Run test suite with coverage",
            cmd=[
                python_executable,
                "-m",
                "pytest",
                "tests/",
                "-v",
                "--cov=consolidation_memory",
                f"--cov-report=xml:{coverage_xml}",
            ],
            failure_message="Tests failed - aborting release. Files reverted.",
            env=coverage_env,
        ),
        ReleaseCommand(
            label="Run builder baseline smoke",
            cmd=[python_executable, "scripts/smoke_builder_base.py"],
            failure_message="Builder smoke failed - aborting release. Files reverted.",
        ),
        ReleaseCommand(
            label="Enforce ResourceWarning gate",
            cmd=[python_executable, "-m", "pytest", "tests/", "-q", "-W", "error::ResourceWarning"],
            failure_message="ResourceWarning gate failed - aborting release. Files reverted.",
        ),
        ReleaseCommand(
            label="Run lint",
            cmd=[python_executable, "-m", "ruff", "check", "src/", "tests/"],
            failure_message="Lint check failed - aborting release. Files reverted.",
        ),
        ReleaseCommand(
            label="Run type checks",
            cmd=[python_executable, "-m", "mypy", "src/consolidation_memory/"],
            failure_message="Type check failed - aborting release. Files reverted.",
        ),
        ReleaseCommand(
            label="Run security checks",
            cmd=[python_executable, "-m", "bandit", "-q", "-r", "src", "scripts"],
            failure_message="Security check failed - aborting release. Files reverted.",
        ),
    ]


def get_release_gate_commands(python_executable: str) -> list[ReleaseCommand]:
    NOVELTY_RELEASE_RESULT.parent.mkdir(parents=True, exist_ok=True)
    return [
        ReleaseCommand(
            label="Run full novelty evaluation",
            cmd=[
                python_executable,
                "-m",
                "benchmarks.novelty_eval",
                "--mode",
                "full",
                "--output",
                str(NOVELTY_RELEASE_RESULT),
            ],
            failure_message="Full novelty evaluation failed - aborting release. Files reverted.",
        ),
        ReleaseCommand(
            label="Enforce release gates",
            cmd=[
                python_executable,
                "scripts/verify_release_gates.py",
                "--novelty-result",
                str(NOVELTY_RELEASE_RESULT),
                "--scope-use-case",
                RELEASE_SCOPE_USE_CASE,
                "--output",
                str(RELEASE_GATE_REPORT),
            ],
            failure_message=(
                "Release gate verification failed - aborting release. "
                "Files reverted. See benchmarks/results/release_gate_report.json."
            ),
        ),
    ]


def run_release_commands(
    commands: list[ReleaseCommand],
    *,
    pyproject_text: str,
    changelog_text: str,
) -> None:
    current_label = "release step"
    try:
        for command in commands:
            current_label = command.label
            print(f"  {command.label}...")
            result = run(command.cmd, check=False, env=command.env)
            if result.returncode != 0:
                raise RuntimeError(command.failure_message)
    except Exception as exc:
        rollback_release_files(pyproject_text, changelog_text)
        if isinstance(exc, RuntimeError):
            sys.exit(str(exc))
        sys.exit(f"{current_label} failed - aborting release. Files reverted. ({exc})")


def verify_built_artifacts(artifact_dir: Path, version: str) -> list[Path]:
    artifacts = sorted(path for path in artifact_dir.iterdir() if path.is_file())
    wheels = [path for path in artifacts if path.suffix == ".whl"]
    sdists = [path for path in artifacts if path.name.endswith(".tar.gz")]
    expected_prefix = f"{PACKAGE_NAME.replace('-', '_')}-"

    if len(wheels) != 1 or len(sdists) != 1:
        raise RuntimeError(
            "Expected exactly one wheel and one sdist from build output. "
            f"Found wheels={len(wheels)}, sdists={len(sdists)} in {artifact_dir}."
        )

    expected_artifacts = wheels + sdists
    mismatched = [
        path.name
        for path in expected_artifacts
        if not path.name.startswith(expected_prefix) or version not in path.name
    ]
    if mismatched:
        raise RuntimeError(
            "Built artifact names do not match the expected package/version: "
            + ", ".join(mismatched)
        )

    return expected_artifacts


def build_and_validate_artifacts(
    python_executable: str,
    version: str,
    *,
    pyproject_text: str,
    changelog_text: str,
) -> None:
    try:
        with tempfile.TemporaryDirectory(prefix="cm-release-build-") as tmp:
            artifact_dir = Path(tmp) / "dist"
            print("  Build sdist and wheel...")
            build_result = run(
                [python_executable, "-m", "build", "--outdir", str(artifact_dir)],
                check=False,
            )
            if build_result.returncode != 0:
                raise RuntimeError("Build failed - aborting release. Files reverted.")

            artifacts = verify_built_artifacts(artifact_dir, version)

            print("  Validate built artifacts...")
            check_result = run(
                [
                    python_executable,
                    "-m",
                    "twine",
                    "check",
                    "--strict",
                    *[str(path) for path in artifacts],
                ],
                check=False,
            )
            if check_result.returncode != 0:
                raise RuntimeError("Artifact validation failed - aborting release. Files reverted.")
    except Exception as exc:
        rollback_release_files(pyproject_text, changelog_text)
        if isinstance(exc, RuntimeError):
            sys.exit(str(exc))
        sys.exit(f"Artifact validation failed - aborting release. Files reverted. ({exc})")


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

    print("\n[1/9] Validating git state...")
    result = run(["git", "status", "--porcelain"], capture=True)
    if result.stdout.strip():
        sys.exit("Working tree is not clean. Commit or stash changes first.")
    try:
        ensure_release_branch()
    except RuntimeError as exc:
        sys.exit(str(exc))

    print(f"\n[2/9] Pulling latest from origin/{EXPECTED_RELEASE_BRANCH}...")
    if not args.dry_run:
        run(["git", "pull", "--ff-only", "origin", EXPECTED_RELEASE_BRANCH])

    print(f"\n[3/9] Validating release target {tag_name}...")
    if tag_exists(tag_name):
        sys.exit(f"Tag {tag_name} already exists. Choose a different version.")
    if remote_tag_exists(tag_name):
        sys.exit(f"Tag {tag_name} already exists on origin. Choose a different version.")
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

    print(f"\n[4/9] Preparing files for {new_version}...")
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

    print("\n[5/9] Running publish-grade quality gates...")
    if not args.dry_run:
        with tempfile.TemporaryDirectory(prefix="cm-release-quality-") as tmp:
            run_release_commands(
                get_quality_gate_commands(sys.executable, Path(tmp)),
                pyproject_text=pyproject_before,
                changelog_text=changelog_before,
            )

    print("\n[6/9] Running release gate checks...")
    if not args.dry_run:
        run_release_commands(
            get_release_gate_commands(sys.executable),
            pyproject_text=pyproject_before,
            changelog_text=changelog_before,
        )

    print("\n[7/9] Building and validating release artifacts...")
    if not args.dry_run:
        build_and_validate_artifacts(
            sys.executable,
            new_version,
            pyproject_text=pyproject_before,
            changelog_text=changelog_before,
        )

    print(f"\n[8/9] Committing {tag_name}...")
    if not args.dry_run:
        run(["git", "add", "pyproject.toml", "CHANGELOG.md"])
        run(["git", "commit", "-m", tag_name])
        run(["git", "tag", tag_name])

    if args.no_push:
        print("\n[9/9] Skipping push (--no-push). Run manually:")
        print(f"  git push origin {EXPECTED_RELEASE_BRANCH} {tag_name}")
    elif args.dry_run:
        print(f"\n[9/9] Would push {EXPECTED_RELEASE_BRANCH} + tag {tag_name}")
    else:
        print(f"\n[9/9] Pushing {EXPECTED_RELEASE_BRANCH} + tag {tag_name}...")
        run(["git", "push", "origin", EXPECTED_RELEASE_BRANCH, tag_name])

    print(f"\nDone! {tag_name} {'would be' if args.dry_run else 'is'} released.")
    if not args.dry_run and not args.no_push:
        print(f"PyPI publish will trigger from the {tag_name} tag via GitHub Actions.")


if __name__ == "__main__":
    main()
