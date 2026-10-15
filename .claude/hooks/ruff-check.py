"""PostToolUse hook: run ruff on an edited Python file and report issues."""

import importlib.util
import json
import os
import subprocess
import sys


def main() -> None:
    data = json.load(sys.stdin)
    file_path = (data.get("tool_input") or {}).get("file_path") or ""
    if not file_path.endswith(".py") or not os.path.isfile(file_path):
        return
    # Skip silently where ruff is not installed for this interpreter.
    if importlib.util.find_spec("ruff") is None:
        return
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "check", file_path],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"ruff found issues in {file_path}:")
        print((result.stdout + result.stderr).strip())


if __name__ == "__main__":
    main()
