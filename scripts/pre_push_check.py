#!/usr/bin/env python3
"""Fast local checks that mirror the main CI test job gates.

Run before push (or via the installed pre-push git hook):
    python scripts/pre_push_check.py

Use --full for a complete pytest run after the quick gates pass.
"""

from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Optional extras not installed in the CI test matrix job (fastembed,dev only).
BLOCKED_OPTIONAL_IMPORTS = (
    "fastapi",
    "fastapi.testclient",
    "textual",
    "openai",
)


def _run(command: list[str], *, label: str) -> int:
    print(f"pre_push_check: {label}")
    print(f"  $ {' '.join(command)}")
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode != 0:
        print(f"pre_push_check: FAIL ({label})", file=sys.stderr)
    return completed.returncode


def check_optional_import_collection() -> int:
    """Fail if any test module imports optional deps at collection time."""
    print("pre_push_check: optional-import collection gate")
    for name in BLOCKED_OPTIONAL_IMPORTS:
        sys.modules[name] = None  # type: ignore[assignment]

    sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]
    import pytest

    exit_code = pytest.main(
        [
            str(ROOT / "tests"),
            "--collect-only",
            "-q",
            "--disable-warnings",
        ]
    )
    if exit_code != 0:
        print(
            "pre_push_check: FAIL (optional-import collection gate)\n"
            "  A test file likely imports an optional extra at module scope.\n"
            "  Defer the import or guard it like tests/test_rest.py.",
            file=sys.stderr,
        )
    return int(exit_code)


def check_ruff() -> int:
    try:
        import ruff  # noqa: F401
    except ImportError:
        print("pre_push_check: skip ruff (not installed)")
        return 0
    return _run([sys.executable, "-m", "ruff", "check", "src/", "tests/"], label="ruff")


def check_bandit() -> int:
    try:
        importlib.import_module("bandit")
    except ImportError:
        print("pre_push_check: skip bandit (not installed)")
        return 0
    return _run(
        [
            sys.executable,
            "-m",
            "bandit",
            "-q",
            "-ll",
            "-r",
            "src",
            "scripts",
            "-s",
            "B608,B110",
        ],
        label="bandit",
    )


def check_pytest_full() -> int:
    return _run([sys.executable, "-m", "pytest", "tests/", "-q"], label="pytest")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run the full pytest suite after quick gates (slower).",
    )
    parser.add_argument(
        "--skip-collection",
        action="store_true",
        help="Skip the optional-import collection gate.",
    )
    args = parser.parse_args(argv)

    failures = 0
    if not args.skip_collection and check_optional_import_collection() != 0:
        failures += 1
    if check_ruff() != 0:
        failures += 1
    if check_bandit() != 0:
        failures += 1
    if args.full and check_pytest_full() != 0:
        failures += 1

    if failures:
        print(f"pre_push_check: {failures} gate(s) failed", file=sys.stderr)
        return 1

    print("pre_push_check: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())