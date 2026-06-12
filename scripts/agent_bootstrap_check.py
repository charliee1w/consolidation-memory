#!/usr/bin/env python3
"""Quick sanity check that agent bootstrap docs and imports are present.

Run at session start (optional):
    python scripts/agent_bootstrap_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "AGENTS.md",
    "GOAL.md",
    "docs/AGENT_GOAL.md",
    "docs/VIBECODING.md",
    "docs/ARCHITECTURE.md",
    ".grok/rules/consolidation-memory.md",
    ".cursor/rules/consolidation-memory.mdc",
    "src/consolidation_memory/consolidation/fast_path.py",
]

REQUIRED_IMPORTS = [
    ("consolidation_memory.consolidation.fast_path", "try_fast_path_extraction"),
    ("consolidation_memory.consolidation.engine", "run_consolidation"),
    ("consolidation_memory.claim_graph", "claim_from_record"),
]


def main() -> int:
    errors: list[str] = []

    for rel in REQUIRED_FILES:
        path = ROOT / rel
        if not path.exists():
            errors.append(f"missing file: {rel}")

    sys.path.insert(0, str(ROOT / "src"))
    for module, attr in REQUIRED_IMPORTS:
        try:
            imported = __import__(module, fromlist=[attr])
            if not hasattr(imported, attr):
                errors.append(f"missing attribute: {module}.{attr}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"import failed: {module} ({exc})")

    goal = ROOT / "docs" / "AGENT_GOAL.md"
    if goal.exists():
        text = goal.read_text(encoding="utf-8")
        if "M1 — LLM-optional substrate" not in text:
            errors.append("docs/AGENT_GOAL.md missing M1 section")
        if "⬜" not in text:
            errors.append("docs/AGENT_GOAL.md has no open tasks (⬜)")

    if errors:
        print("agent bootstrap check: FAIL")
        for err in errors:
            print(f"  - {err}")
        return 1

    print("agent bootstrap check: OK")
    print(f"  root: {ROOT}")
    print("  next: read docs/AGENT_GOAL.md and pick first unchecked M1 task")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())