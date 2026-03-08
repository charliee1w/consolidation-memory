# Builder Baseline

This baseline defines the minimum quality bar before building on top of the repo.

## Required Local Checks

```bash
pip install -e ".[all,dev]"
python scripts/smoke_builder_base.py
pytest tests/ -q
pytest tests/ -q -W error::ResourceWarning
ruff check src/ tests/
mypy src/consolidation_memory/
```

## Required CI Signals

From GitHub Actions:

- `test.yml` matrix passes.
- `novelty_gates` job passes.
- No skipped critical checks for release-impacting changes.

## Builder-Ready Definition

A branch is builder-ready when:

1. All local checks above pass.
2. CI is green.
3. Changed behavior is documented.
4. No adapter surface is left semantically behind.

## Scope For This Baseline

This baseline validates repository health and developer ergonomics.
It does not by itself prove release readiness; release readiness requires `docs/RELEASE_GATES.md` compliance.
