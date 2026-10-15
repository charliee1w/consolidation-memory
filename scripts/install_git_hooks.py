#!/usr/bin/env python3
"""Install repository git hooks for local CI-style checks before push."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOOKS_DIR = ROOT / ".githooks"
PRE_PUSH = HOOKS_DIR / "pre-push"


def main() -> int:
    if not PRE_PUSH.is_file():
        print(f"install_git_hooks: missing {PRE_PUSH.relative_to(ROOT)}", file=sys.stderr)
        return 1
    PRE_PUSH.chmod(PRE_PUSH.stat().st_mode | 0o111)

    subprocess.run(
        ["git", "config", "core.hooksPath", ".githooks"],
        cwd=ROOT,
        check=True,
    )
    print("install_git_hooks: OK")
    print(f"  hooks path: {HOOKS_DIR.relative_to(ROOT)}")
    print("  pre-push runs: python scripts/pre_push_check.py")
    print("  bypass once: git push --no-verify")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())