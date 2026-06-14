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