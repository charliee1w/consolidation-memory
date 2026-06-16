#!/usr/bin/env python3
"""Fetch and rebase the current branch onto its upstream remote.

Run at session start (or before substantive work) to keep the working copy
aligned with origin:

    python scripts/sync_working_copy.py
    python scripts/sync_working_copy.py --install-deps
    python scripts/sync_working_copy.py --stash   # stash dirty tree, sync, pop

Exits non-zero when the tree has uncommitted changes (unless --stash).
"""

from __future__ import annotations

import argparse
import subprocess  # nosec B404
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(
    cmd: list[str],
    *,
    check: bool = True,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(  # nosec B603
        cmd,
        cwd=str(ROOT),
        check=check,
        capture_output=capture,
        text=True,
    )


def git_output(*args: str) -> str:
    result = run(["git", *args], capture=True)
    return result.stdout.strip()


def is_dirty() -> bool:
    return bool(git_output("status", "--porcelain"))


def current_branch() -> str:
    branch = git_output("rev-parse", "--abbrev-ref", "HEAD")
    if branch == "HEAD":
        raise SystemExit("sync_working_copy: detached HEAD; checkout a branch first")
    return branch


def upstream_ref() -> str | None:
    result = run(
        ["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
        check=False,
        capture=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def behind_ahead(upstream: str) -> tuple[int, int]:
    result = run(["git", "rev-list", "--left-right", "--count", f"HEAD...{upstream}"], capture=True)
    ahead_s, behind_s = result.stdout.split()
    return int(behind_s), int(ahead_s)


def short_sha(ref: str = "HEAD") -> str:
    return git_output("rev-parse", "--short", ref)


def sync_working_copy(*, stash: bool = False, remote: str = "origin") -> int:
    if not (ROOT / ".git").exists():
        print("sync_working_copy: not a git repository", file=sys.stderr)
        return 1

    branch = current_branch()
    print(f"sync_working_copy: branch={branch}")

    dirty = is_dirty()
    if dirty and not stash:
        print(
            "sync_working_copy: working tree has uncommitted changes; "
            "commit/stash first or pass --stash",
            file=sys.stderr,
        )
        return 1

    stashed = False
    if dirty and stash:
        run(["git", "stash", "push", "-u", "-m", "sync_working_copy auto-stash"])
        stashed = True

    before = short_sha()
    run(["git", "fetch", remote])

    upstream = upstream_ref()
    if upstream is None:
        upstream = f"{remote}/{branch}"
        print(f"sync_working_copy: no upstream configured; using {upstream}")

    behind, ahead = behind_ahead(upstream)
    print(f"sync_working_copy: {behind} behind, {ahead} ahead of {upstream}")

    if behind:
        run(["git", "pull", "--rebase", remote, branch])
    else:
        print("sync_working_copy: already up to date")

    after = short_sha()
    if before != after:
        print(f"sync_working_copy: updated {before} -> {after}")
    else:
        print(f"sync_working_copy: at {after}")

    if stashed:
        pop = run(["git", "stash", "pop"], check=False)
        if pop.returncode != 0:
            print(
                "sync_working_copy: stash pop failed; resolve conflicts then "
                "`git stash pop` manually",
                file=sys.stderr,
            )
            return pop.returncode

    return 0


def install_deps() -> int:
    print("sync_working_copy: installing editable package with dev extras")
    run([sys.executable, "-m", "pip", "install", "-e", ".[all,dev]"])
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stash",
        action="store_true",
        help="stash dirty changes before sync, then pop afterward",
    )
    parser.add_argument(
        "--remote",
        default="origin",
        help="remote to fetch/pull from (default: origin)",
    )
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help='run pip install -e ".[all,dev]" after a successful sync',
    )
    args = parser.parse_args(argv)

    code = sync_working_copy(stash=args.stash, remote=args.remote)
    if code != 0:
        return code
    if args.install_deps:
        return install_deps()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())