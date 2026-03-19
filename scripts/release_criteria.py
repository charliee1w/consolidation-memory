#!/usr/bin/env python3
"""Automated release criteria for main-branch pushes.

This script inspects commits since the latest tag and recommends one of:
- ``major``
- ``minor``
- ``patch``
- no release

Criteria (highest priority first):
1. Head commit contains ``[skip release]`` -> no release.
2. Head commit contains ``[release major|minor|patch]`` -> forced bump.
3. Conventional-commit signal over commits since latest tag:
   - breaking change (``!`` or ``BREAKING CHANGE``) -> major
   - ``feat`` -> minor
   - ``fix|perf|refactor|revert|security`` -> patch
4. Otherwise, no release.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess  # nosec B404
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

_BUMP_PRIORITY = {"patch": 1, "minor": 2, "major": 3}
_CONVENTIONAL_SUBJECT_RE = re.compile(
    r"^(?P<type>[a-z]+)(?:\([^)]+\))?(?P<bang>!)?:\s+",
    re.IGNORECASE,
)
_FORCE_RELEASE_RE = re.compile(r"\[(?:release|bump)\s+(major|minor|patch)\]", re.IGNORECASE)
_SKIP_RELEASE_RE = re.compile(r"\[(?:skip\s+release|release\s+skip)\]", re.IGNORECASE)


@dataclass(frozen=True)
class CommitMessage:
    subject: str
    body: str = ""

    @property
    def text(self) -> str:
        return f"{self.subject}\n{self.body}".strip()


def _run_git(args: list[str]) -> subprocess.CompletedProcess[str]:
    git_executable = shutil.which("git")
    if not git_executable:
        raise RuntimeError("git executable not found in PATH")
    return subprocess.run(  # nosec B603
        [str(Path(git_executable).resolve()), *args],
        cwd=str(ROOT),
        check=True,
        capture_output=True,
        text=True,
    )


def _get_latest_tag() -> str | None:
    git_executable = shutil.which("git")
    if not git_executable:
        raise RuntimeError("git executable not found in PATH")
    result = subprocess.run(  # nosec B603
        [str(Path(git_executable).resolve()), "describe", "--tags", "--abbrev=0"],
        cwd=str(ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    tag = result.stdout.strip()
    return tag or None


def _parse_git_log(output: str) -> list[CommitMessage]:
    commits: list[CommitMessage] = []
    for raw_entry in output.split("\x1f"):
        entry = raw_entry.strip()
        if not entry:
            continue
        lines = entry.splitlines()
        subject = lines[0].strip()
        body = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""
        if subject:
            commits.append(CommitMessage(subject=subject, body=body))
    return commits


def _collect_commits_since_tag(
    *,
    from_ref: str | None = None,
    to_ref: str = "HEAD",
) -> tuple[list[CommitMessage], str | None]:
    baseline = from_ref if from_ref is not None else _get_latest_tag()
    revspec = f"{baseline}..{to_ref}" if baseline else to_ref
    result = _run_git(["log", "--format=%s%n%b%x1f", revspec])
    return _parse_git_log(result.stdout), baseline


def _max_bump(left: str | None, right: str | None) -> str | None:
    if left is None:
        return right
    if right is None:
        return left
    return left if _BUMP_PRIORITY[left] >= _BUMP_PRIORITY[right] else right


def _forced_head_directive(head: CommitMessage | None) -> str | None:
    if head is None:
        return None
    text = head.text
    if _SKIP_RELEASE_RE.search(text):
        return "skip"
    forced = _FORCE_RELEASE_RE.search(text)
    if forced:
        return forced.group(1).lower()
    return None


def _classify_commit_bump(commit: CommitMessage) -> str | None:
    subject = commit.subject.strip()
    if not subject:
        return None
    match = _CONVENTIONAL_SUBJECT_RE.match(subject)
    if not match:
        return None

    commit_type = match.group("type").lower()
    bang = bool(match.group("bang"))
    body_upper = commit.body.upper()

    if bang or "BREAKING CHANGE" in body_upper:
        return "major"
    if commit_type == "feat":
        return "minor"
    if commit_type in {"fix", "perf", "refactor", "revert", "security"}:
        return "patch"
    return None


def decide_release(commits: list[CommitMessage]) -> dict[str, object]:
    if not commits:
        return {
            "should_release": False,
            "bump": None,
            "reason": "No commits found since latest tag.",
            "signals": {"major": 0, "minor": 0, "patch": 0},
        }

    forced = _forced_head_directive(commits[0])
    if forced == "skip":
        return {
            "should_release": False,
            "bump": None,
            "reason": "Head commit requested release skip via [skip release].",
            "signals": {"major": 0, "minor": 0, "patch": 0},
        }
    if forced in _BUMP_PRIORITY:
        return {
            "should_release": True,
            "bump": forced,
            "reason": f"Head commit forced bump via [release {forced}].",
            "signals": {"major": 0, "minor": 0, "patch": 0},
        }

    bump: str | None = None
    signals = {"major": 0, "minor": 0, "patch": 0}
    for commit in commits:
        signal = _classify_commit_bump(commit)
        if signal is None:
            continue
        signals[signal] += 1
        bump = _max_bump(bump, signal)

    if bump is None:
        return {
            "should_release": False,
            "bump": None,
            "reason": "No releasable conventional-commit signals found.",
            "signals": signals,
        }

    return {
        "should_release": True,
        "bump": bump,
        "reason": f"Derived {bump} bump from conventional commits since latest tag.",
        "signals": signals,
    }


def _write_github_output(path: str, payload: dict[str, object]) -> None:
    reason = str(payload.get("reason") or "").replace("\n", " ").strip()
    lines = [
        f"should_release={'true' if payload.get('should_release') else 'false'}",
        f"bump={payload.get('bump') or ''}",
        f"reason={reason}",
        f"from_ref={payload.get('from_ref') or ''}",
        f"commits_scanned={payload.get('commits_scanned')}",
    ]
    with open(path, "a", encoding="utf-8") as handle:
        for line in lines:
            handle.write(f"{line}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate automated release criteria.")
    parser.add_argument("--from-ref", help="Optional baseline git ref (default: latest tag).")
    parser.add_argument("--to-ref", default="HEAD", help="Target git ref to evaluate (default: HEAD).")
    parser.add_argument(
        "--github-output",
        help="Path to GITHUB_OUTPUT file to export should_release/bump/reason outputs.",
    )
    args = parser.parse_args()

    commits, baseline = _collect_commits_since_tag(from_ref=args.from_ref, to_ref=args.to_ref)
    result = decide_release(commits)
    payload: dict[str, object] = {
        **result,
        "from_ref": baseline,
        "to_ref": args.to_ref,
        "commits_scanned": len(commits),
    }
    print(json.dumps(payload, indent=2))

    output_path = args.github_output or os.environ.get("GITHUB_OUTPUT")
    if output_path:
        _write_github_output(output_path, payload)


if __name__ == "__main__":
    main()
