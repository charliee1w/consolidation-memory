# Real-World Metrics

`benchmarks/real_world_eval.py` evaluates the **active live project** using real
stored episodes, claims, and repository drift signals.

Unlike synthetic harnesses (`novelty_eval`, `coding_agent_eval`), this benchmark
uses your actual memory corpus and reports messy-world performance.

**CI note:** `ci` mode runs against an isolated fixture corpus in GitHub Actions.
`quick` and `full` remain manual on your live project. Unit tests cover harness
logic in `tests/test_real_world_eval.py`.

## Run It

```bash
python -m benchmarks.real_world_eval --mode ci
python -m benchmarks.real_world_eval --mode quick
python -m benchmarks.real_world_eval --mode full --output benchmarks/results/real_world_eval_full.json
```

Optional repo path for drift checks (defaults to repository root):

```bash
python -m benchmarks.real_world_eval --repo-path .
```

## Sections

1. **Live Solution Recall@5** — can real debugging episodes be found again?
2. **Live Claim Recall@5** — can active claims be retrieved by problem query?
3. **Challenged Claim Suppression** — do challenged claims stay out of active reuse?
4. **Live Provenance Coverage** — do recalled claims still carry complete provenance?
5. **Live Drift Response** — when repo paths change, are impacted claims challenged?
6. **Memory Health Snapshot** — backlog, trust profile, and health status.

## Threshold Philosophy

Thresholds are intentionally lower than synthetic harnesses because this
benchmark measures real, noisy memory — not controlled fixtures.

## Live corpus trend (project `universal`)

Published from `benchmarks/results/real_world_eval_full.json` (2026-06-16, `fastembed` embeddings; run `real_world_eval_full_16e26d13611a`).

| Section | Measured | Threshold | Pass |
| --- | --- | --- | --- |
| Live Solution Recall@5 | **84.2%** (101/120) | ≥ 35% | yes |
| Live Claim Recall@5 | **81.25%** (65/80) | ≥ 40% | yes |
| Challenged Claim Suppression | **100%** (60/60) | ≥ 90% | yes |
| Live Provenance Coverage | **100%** (300/300) | ≥ 85% | yes |
| Live Drift Response | **100%** (20/20 impacted) | ≥ 80% when paths change | yes |
| Memory Health Snapshot | healthy; backlog 5.1% (15 pending / 292 episodes) | backlog ≤ 20% | yes |

**Overall:** `overall_pass: true`

Prior baseline (2026-06-14): solution recall 81.7%, claim recall 77.5%, health degraded (45h stale consolidation).

Re-run after corpus or embedding backend changes:

```bash
consolidation-memory warmup
python -m benchmarks.real_world_eval --mode full --output benchmarks/results/real_world_eval_full.json --repo-path .
```