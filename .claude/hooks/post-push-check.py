"""PostToolUse hook: after `git push`, remind to evaluate whether a release is due."""

import json
import os
import re
import subprocess
import sys


def git(*args: str) -> str:
    project_dir = os.environ.get("CLAUDE_PROJECT_DIR", ".")
    result = subprocess.run(
        ["git", "-C", project_dir, *args], capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def main() -> None:
    data = json.load(sys.stdin)
    command = (data.get("tool_input") or {}).get("command") or ""
    if not re.search(r"git\s+push", command):
        return
    last_tag = git("describe", "--tags", "--abbrev=0")
    if not last_tag:
        return
    count = git("rev-list", "--count", f"{last_tag}..HEAD")
    print(
        f"Pushed to remote. {count} commits since last release ({last_tag}). "
        "Evaluate whether a new release is warranted: new features = minor bump, "
        "bug fixes = patch bump. Skip for docs/test/refactor-only changes."
    )


if __name__ == "__main__":
    main()
