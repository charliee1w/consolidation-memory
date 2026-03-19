# Release Gates

Release gating is fail-closed.

A release is allowed only when all required gates pass with complete and recent evidence.

## Source Of Truth

- Gate evaluator: `src/consolidation_memory/release_gates.py`
- Gate CLI: `scripts/verify_release_gates.py`
- Publish workflow: `.github/workflows/publish.yml`
- Automated release trigger: `.github/workflows/release-on-main.yml`
- Criteria evaluator: `scripts/release_criteria.py`

## Required Evidence

- Novelty evaluation JSON (`mode=full` for release).
- Scope alignment evidence (use-case string must exist in `docs/NOVELTY_WEDGE.md`).
- Gate report JSON produced by `scripts/verify_release_gates.py`.

## Mandatory Gates

The evaluator enforces these gates:

1. `scope_alignment_gate`
- Required scope use-case token must be present in wedge documentation.

2. `metric_threshold_gate`
- Novelty report must be `mode=full` and `overall_pass=true` with all section passes true.

3. `evidence_completeness_gate`
- Required top-level and per-section fields must exist in the novelty artifact.

4. `evidence_recency_gate`
- Evidence age must be less than or equal to `max_age_days` (default 7).

If any gate fails, overall release gate status is false.

## Local Verification

```bash
python scripts/release.py --bump patch --dry-run
python -m benchmarks.novelty_eval --mode full --output benchmarks/results/novelty_eval_release_full.json
python scripts/verify_release_gates.py \
  --novelty-result benchmarks/results/novelty_eval_release_full.json \
  --scope-use-case "Drift-aware debugging memory" \
  --output benchmarks/results/release_gate_report.json
```

`scripts/release.py` now fail-closes on the same publish-grade quality checks used by
tag publish: clean `main`, tests, builder smoke, `ResourceWarning` gate, lint,
type checks, security scan, full novelty gate enforcement, and artifact build +
`twine check`.

## CI Enforcement

- PR CI (`test.yml`) runs quick novelty checks.
- PR CI also validates wheel/sdist buildability and runs a dedicated optional-surface
  job with `rest`, `openai`, and `dashboard` extras installed.
- Main-branch automation (`release-on-main.yml`) evaluates release criteria and only
  creates a new release tag/version when eligible.
- Tag publish (`publish.yml`) requires the tagged commit to be on `origin/main`,
  runs release quality gates (tests/resource warnings/lint/mypy/security/smoke),
  then runs full novelty evaluation + gate enforcement before build/publish.
- Nightly (`novelty-full-nightly.yml`) refreshes full novelty + gate artifacts.

## Policy Notes

- Missing evidence is a failure, not a warning.
- Stale evidence is a failure.
- A passing local run does not override failing CI gate evidence.
