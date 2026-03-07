# External Review Playbook

Use this when asking outside contributors to evaluate whether the repo is a
solid base to build on.

## Reviewer Onboarding (Fresh Environment)

1. Clone repo and create a fresh virtual environment.
2. Install with `pip install -e ".[fastembed,dev]"`.
3. Run `python scripts/smoke_builder_base.py`.
4. Run `python -m pytest tests/ -q -W error::ResourceWarning`.
5. Read `README.md` and only the docs linked from it.

If any step is unclear or fails, open a review issue immediately.

## Canonical Extension Exercise

Ask each reviewer to implement one extension from docs only:

1. Create a plugin subclass of `PluginBase` with at least one hook.
2. Register it via config or entry point.
3. Demonstrate the hook firing in a minimal script or test.

The goal is to validate that a new builder can ship a real extension without
reverse-engineering internal implementation details.

## Issue Rubric

Report findings with:

1. Severity:
   - `P0`: Blocks basic use or data safety.
   - `P1`: Major workflow break or misleading docs.
   - `P2`: Friction that slows contributors materially.
   - `P3`: Minor papercut or clarity issue.
2. Reproduction steps (copy-paste commands).
3. Expected vs actual behavior.
4. Proposed fix direction.
5. Confidence (`high`, `medium`, `low`).

## Exit Criteria

A review round is complete when:

1. At least 3 external reviewers complete onboarding.
2. Each reviewer successfully completes the extension exercise or reports a
   blocker with reproducible evidence.
3. All `P0`/`P1` issues are fixed or explicitly deferred with owner/date.
