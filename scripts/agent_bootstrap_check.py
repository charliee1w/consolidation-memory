#!/usr/bin/env python3
"""Quick sanity check that contributor docs and core imports are present.

Run at session start (optional):
    python scripts/agent_bootstrap_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "CONTRIBUTING.md",
    "docs/ARCHITECTURE.md",
    "docs/FAST_PATH_EPISODES.md",
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

    if errors:
        print("agent bootstrap check: FAIL")
        for err in errors:
            print(f"  - {err}")
        return 1

    print("agent bootstrap check: OK")
    print("  next: python scripts/sync_working_copy.py (if tree clean)")
    print("  next: read CONTRIBUTING.md and docs/ARCHITECTURE.md before coding")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())