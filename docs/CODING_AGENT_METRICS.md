# Coding-Agent Metrics

These metrics are enforced by `benchmarks/coding_agent_eval.py`.

CI runs `quick` mode on every push/PR to `main` (see `.github/workflows/test.yml`).
Run `full` mode locally or before releases for baseline tracking.

Unlike conversational-memory benchmarks (for example LoCoMo), this harness measures
trust workflows that matter for coding agents: debug-solution reuse, drift
suppression, scope isolation, contradiction visibility, and outcome-informed ranking.

## Evaluation Modes

- `quick`: smaller deterministic sample for local/CI feedback.
- `full`: larger release-grade sample for baseline tracking.

## Section 1: Debug Solution Ingest -> Consolidate -> Recall

Definition:

- Fraction of scenarios where a path-anchored debugging solution is recoverable
  after LLM-free fast-path consolidation.

Measured fields:

- `solution_recall_hit_rate`

Threshold:

- `solution_recall_hit_rate >= 0.90`

## Section 2: Stale Fix Suppression After Drift

Definition:

- After git drift challenges impacted claims, recall must not surface those stale
  claims as active reuse candidates.

Measured fields:

- `stale_fix_suppression_rate`
- `stale_fix_leak_rate`

Thresholds:

- `stale_fix_suppression_rate >= 0.95`
- `stale_fix_leak_rate <= 0.05`

## Section 3: Scope Isolation Under Recall

Definition:

- Private namespace memories must not leak into another namespace's recall.

Measured fields:

- `scope_isolation_rate`
- `scope_leak_rate`

Thresholds:

- `scope_isolation_rate == 1.0`
- `scope_leak_rate == 0.0`

## Section 4: Contradiction Visibility On Recall

Definition:

- Recall must surface contradiction warnings or per-claim uncertainty labels when
  conflicting fixes exist.

Measured fields:

- `contradiction_visibility_rate`

Threshold:

- `contradiction_visibility_rate >= 0.95`

## Section 5: Outcome-Informed Claim Ranking

Definition:

- Claims with recorded successful outcomes should outrank equivalent claims without
  outcome support.

Measured fields:

- `outcome_ranking_win_rate`
- `median_reliability_delta`

Thresholds:

- `outcome_ranking_win_rate >= 0.80`
- `median_reliability_delta >= 0.0`

## Run It

Quick mode:

```bash
python -m benchmarks.coding_agent_eval --mode quick
```

Full mode:

```bash
python -m benchmarks.coding_agent_eval --mode full --output benchmarks/results/coding_agent_eval_full.json
```

## Output Contract

`coding_agent_eval.py` outputs:

- `benchmark`
- `run_id`
- `mode`
- `generated_at`
- `sections`
- `overall_pass`

Each section includes `thresholds`, `measured`, and `pass`.