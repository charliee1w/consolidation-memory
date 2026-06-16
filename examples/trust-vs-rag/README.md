# Trust vs simple RAG — same bug twice

This example shows why consolidation-memory is not "just another vector DB."

## Scenario

1. You fix an auth bug and store the solution as an episode.
2. Consolidation emits a **claim** with provenance (episode → record → claim).
3. Weeks later, you refactor `auth.py` — the file anchor drifts.
4. `memory_detect_drift` **challenges** the old claim instead of silently serving it.
5. On the second occurrence, recall surfaces the fix **with trust signals**
   (provenance, precision, challenged status) — not an opaque snippet.

Simple RAG returns the nearest embedding. consolidation-memory maintains beliefs
with expiry, contradiction events, and drift auditability.

## Run the demo

From the repository root (LLM optional — uses fast-path / disabled consolidation):

```bash
pip install -e ".[fastembed,dev]"
python examples/trust-vs-rag/demo_flow.py
```

The script:

- Stores a solution episode with a file anchor
- Runs consolidation (fast-path when `LLM_BACKEND=disabled`)
- Simulates a git rename on the anchored path
- Runs drift detection and shows challenged claims
- Recalls the problem query and prints ranked claims with status

## Compare to RAG mentally

| Simple RAG | consolidation-memory |
| --- | --- |
| Embed text, return top-k | Episodes + records + **claims** |
| No provenance | Source episodes and anchors on every claim |
| Stale docs stay hot | Drift challenges + claim expiry on `forget()` |
| One-shot retrieval | `memory_status` scheduler + consolidation observability |

## Next steps

- Wire MCP with `memory_remember` / `memory_ask` for day-to-day use
- Run `memory_hygiene_scan` periodically on noisy corpora
- See [REAL_WORLD_METRICS.md](../../docs/REAL_WORLD_METRICS.md) for live recall numbers