# Novelty Execution Chunks

This file provides a repeatable chunking strategy for substantial novelty/trust work.

## Chunking Rules

1. Each chunk must end in runnable verification commands.
2. Each chunk must update affected docs in the same change.
3. Never defer schema/API contract updates to a later chunk.
4. Keep adapter-facing semantics centralized in canonical services.

## Suggested Chunk Order For New Work

1. Metrics and gate contract update.
2. Schema/data model update (with migration).
3. Core service-layer behavior change.
4. Adapter surface propagation (MCP/REST/OpenAI/Python).
5. Benchmark and regression tests.
6. Documentation and release-gate evidence refresh.

## Chunk Completion Template

For each chunk, capture:

- What changed.
- Which invariants were affected.
- Which commands were run.
- Which docs were updated.
- What remains for the next chunk.

## Verification Minimum Per Chunk

```bash
pytest tests/ -q
ruff check src/ tests/
mypy src/consolidation_memory/
```

If novelty or release-gate semantics changed:

```bash
python -m benchmarks.novelty_eval --mode quick --output benchmarks/results/novelty_eval_chunk_check.json
```

## Handoff Note Template

- Completed chunk:
- Files changed:
- Verification run:
- Risks/open questions:
- Next chunk:
