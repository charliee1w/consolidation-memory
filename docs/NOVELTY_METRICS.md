# Novelty Metrics

These metrics are enforced by `benchmarks/novelty_eval.py` and consumed by release gates.

## Evaluation Modes

- `quick`: smaller deterministic sample for CI feedback.
- `full`: release-grade sample for gating.

## Section 1: Belief Freshness After Code Drift

Definition:

- Fraction of impacted claims that are corrected/challenged after drift detection.

Measured fields:

- `freshness_after_drift`
- `stale_claim_leak_rate`
- `p95_challenge_lag_seconds`

Thresholds:

- `freshness_after_drift >= 0.97`
- `stale_claim_leak_rate <= 0.03`
- `p95_challenge_lag_seconds <= 120`

## Section 2: Contradiction Resolution Latency

Definition:

- Latency between contradictory evidence ingestion and resolution event.

Measured fields:

- `median_latency_seconds`
- `p95_latency_seconds`
- `unresolved_contradiction_scenarios`

Thresholds:

- `median_latency_seconds <= 90`
- `p95_latency_seconds <= 300`
- `unresolved_contradiction_scenarios == 0`

## Section 3: Temporal Belief Reconstruction

Definition:

- Claim retrieval precision@5 on temporal reconstruction queries.

Measured fields:

- `overall_macro_precision_at_5`
- `slice_macro_precision_at_5.temporal`

Thresholds:

- `overall_macro_precision_at_5 >= 0.80`
- `slice_macro_precision_at_5.temporal >= 0.70`

## Section 4: Provenance Trace Completeness

Definition:

- Share of returned claims with complete provenance links and lifecycle events.

Measured fields:

- `provenance_coverage`
- `missing_provenance_claims_per_1000`

Thresholds:

- `provenance_coverage >= 0.995`
- `missing_provenance_claims_per_1000 <= 5`

## Output Contract

`novelty_eval.py` outputs:

- Top-level fields required by release gates:
  - `benchmark`
  - `run_id`
  - `mode`
  - `generated_at`
  - `sections`
  - `overall_pass`
- Per-section fields required by release gates:
  - `aligned_metric_section`
  - `thresholds`
  - `measured`
  - `pass`

## Maintenance Rule

If thresholds or metric definitions change in code, update this file and `docs/RELEASE_GATES.md` in the same PR.
