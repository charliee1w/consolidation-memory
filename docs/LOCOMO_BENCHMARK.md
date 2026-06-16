# LoCoMo Benchmark

[LoCoMo](https://github.com/snap-research/locomo) measures long conversational
memory with multi-hop and temporal questions. consolidation-memory ships a
harness at `benchmarks/locomo.py` for head-to-head comparison against a
full-context baseline and published Mem0 reference scores.

## Prerequisites

```bash
pip install -e ".[all,benchmark]"
export OPENAI_API_KEY=sk-...
```

## Quick validation (1 conversation, limited QA)

```bash
python -m benchmarks.locomo --mode episodes-only --dry-run
```

Or use the helper script:

```bash
./scripts/benchmark.sh --dry-run --mode episodes-only
```

## Full run (all modes)

```bash
./scripts/benchmark.sh --mode all
```

Modes:

| Mode | What it measures |
| --- | --- |
| `episodes-only` | Recall over stored episodes without consolidation |
| `full` | Episodes + consolidation + recall |
| `full-context` | Baseline: entire transcript in one prompt (no memory system) |

Results are written to `benchmarks/results/locomo_<mode>_<timestamp>.json`.

## Interpreting scores

The harness prints a markdown table with LLM-judge accuracy (%) per category:

- single-hop
- multi-hop
- temporal
- open-domain
- overall

Reference Mem0 scores are embedded in `benchmarks/locomo.py` (`MEM0_REF`) for
quick comparison. Coding-agent workloads should also consult
[REAL_WORLD_METRICS.md](REAL_WORLD_METRICS.md) — LoCoMo is conversational memory,
not solution recall on a live repo.

## Dataset

`scripts/benchmark.sh` downloads `benchmarks/data/locomo10.json` automatically.
To refresh manually:

```bash
curl -sL -o benchmarks/data/locomo10.json \
  https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json
```

## CI posture

LoCoMo is **manual** (requires `OPENAI_API_KEY` and paid judge calls). CI uses
`novelty_eval`, `coding_agent_eval`, and `real_world_eval --mode ci` instead.