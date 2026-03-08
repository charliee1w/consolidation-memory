# Novelty Eval Guide

`benchmarks/novelty_eval.py` is the authoritative novelty harness.

## What It Validates

- Belief freshness after code drift.
- Contradiction resolution latency.
- Temporal claim retrieval quality.
- Provenance trace completeness.

The harness is deterministic and local-first by default.

## Run It

Quick mode (CI-like):

```bash
python -m benchmarks.novelty_eval --mode quick --output benchmarks/results/novelty_eval_ci_quick_local.json
```

Full mode (release-grade):

```bash
python -m benchmarks.novelty_eval --mode full --output benchmarks/results/novelty_eval_release_full.json
```

## Validate Release Gates

```bash
python scripts/verify_release_gates.py \
  --novelty-result benchmarks/results/novelty_eval_release_full.json \
  --scope-use-case "Drift-aware debugging memory" \
  --output benchmarks/results/release_gate_report.json
```

## Expected Output Shape

`novelty_eval.py` writes one JSON object with:

- `benchmark`
- `run_id`
- `mode`
- `generated_at`
- `sections`
- `overall_pass`

Each section includes `thresholds`, `measured`, and `pass`.

## Troubleshooting

1. Failing threshold in one section.
- Inspect section-level `measured` values first; do not adjust thresholds to hide regressions.

2. Gate validator fails on structure.
- Ensure novelty JSON includes required top-level and section fields.

3. Scope alignment fails.
- Ensure the exact use-case string is present in `docs/NOVELTY_WEDGE.md`.

4. Recency gate fails.
- Regenerate full novelty evidence.

## CI Mapping

- `test.yml` uses quick novelty checks.
- `publish.yml` and `novelty-full-nightly.yml` use full novelty checks.
