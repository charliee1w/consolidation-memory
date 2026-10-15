#!/usr/bin/env python3
"""Validate release gates from novelty benchmark evidence.

Usage:
    python scripts/verify_release_gates.py \
      --novelty-result benchmarks/results/novelty_eval_full.json \
      --scope-use-case "Drift-aware debugging memory"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEDGE_DOC = ROOT / "docs" / "NOVELTY_WEDGE.md"
SRC_PATH = ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from consolidation_memory.release_gates import evaluate_release_gates


def _scope_alignment(use_case: str, wedge_doc: Path) -> tuple[bool, str]:
    if not use_case.strip():
        return False, "scope use-case cannot be empty"
    if not wedge_doc.exists():
        return False, f"wedge doc not found: {wedge_doc}"
    content = wedge_doc.read_text(encoding="utf-8").lower()
    token = use_case.strip().lower()
    if token not in content:
        return False, f"use-case not found in wedge doc: '{use_case}'"
    return True, f"use-case '{use_case}' found in {wedge_doc}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify mandatory release gates")
    parser.add_argument(
        "--novelty-result",
        type=Path,
        required=True,
        help="Path to novelty_eval JSON output (must be mode=full for release)",
    )
    parser.add_argument(
        "--scope-use-case",
        type=str,
        required=True,
        help="Exact wedge use-case text that this release aligns with",
    )
    parser.add_argument(
        "--wedge-doc",
        type=Path,
        default=DEFAULT_WEDGE_DOC,
        help=f"Path to NOVELTY_WEDGE.md (default: {DEFAULT_WEDGE_DOC})",
    )
    parser.add_argument(
        "--max-age-days",
        type=int,
        default=7,
        help="Maximum acceptable benchmark evidence age in days (default: 7)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional release gate report path",
    )
    args = parser.parse_args()

    novelty_path = args.novelty_result
    if not novelty_path.exists():
        sys.exit(f"Novelty result file not found: {novelty_path}")

    novelty_results = json.loads(novelty_path.read_text(encoding="utf-8"))
    scope_pass, scope_note = _scope_alignment(args.scope_use_case, args.wedge_doc)

    report = evaluate_release_gates(
        novelty_results=novelty_results,
        max_age_days=args.max_age_days,
        required_mode="full",
        scope_alignment_pass=scope_pass,
        scope_alignment_note=scope_note,
    )
    report["source_novelty_result"] = str(novelty_path)

    output_path = args.output
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "overall_pass": report["overall_pass"],
                "gates": {name: gate["pass"] for name, gate in report["gates"].items()},
                "errors": report["errors"],
                "output_path": str(output_path) if output_path else None,
            },
            indent=2,
        )
    )

    if not report["overall_pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
