# Builder Baseline

This document defines the minimum functional base that must stay stable for
new contributors and external builders.

## Stable Foundation Contract

The following surfaces are considered foundation-level and must remain
functional on every PR:

1. Python API (`MemoryClient.store`, `recall`, `search`, `status`, `export`)
2. Tool dispatch API (`schemas.openai_tools`, `dispatch_tool_call`)
3. Storage lifecycle (fresh project init + store/recall/export on a clean data dir)
4. Test hygiene (`ResourceWarning` gate for unclosed resources)

## Required CI Gates

Foundation readiness requires all of the following to pass:

1. Unit/integration tests
2. Lint (`ruff`)
3. Type check (`mypy`)
4. Builder smoke test (`python scripts/smoke_builder_base.py`)
5. Resource hygiene gate (`pytest -W error::ResourceWarning`)

These gates are wired in `.github/workflows/test.yml`.

## Local Verification

Run this from a clean checkout before opening a PR:

```bash
pip install -e ".[fastembed,dev]"
python scripts/smoke_builder_base.py
python -m pytest tests/ -q -W error::ResourceWarning
ruff check src/ tests/
mypy src/consolidation_memory/
```

## Definition Of Ready-To-Build

A build is considered safe for external contributors when:

1. All required CI gates pass.
2. Quickstart works on a fresh machine without project-specific setup.
3. A minimal extension can be built using only public docs.
4. Open external-review findings are triaged with owner + severity.

## Compatibility Policy

For foundation surfaces:

1. Backward-compatible changes are preferred.
2. Breaking changes require changelog entry + migration note in README/docs.
3. Removing or renaming foundation APIs requires a deprecation window unless the
   project is in emergency/security mode.
