# Roadmap

Outcome-based direction for `consolidation-memory`. Details may shift as the implementation evolves.

## Objective

Make local-first agent memory dependable for coding workflows: useful recall, explicit trust signals, and low operational overhead — **usable by anyone**, not only MCP power users.

## Shipped

- Hybrid retrieval over episodes, topics, records, and claims
- Temporal queries (`as_of`) on trust surfaces
- Claim graph with contradiction and lifecycle events
- Drift-aware claim challenges via file anchors and git deltas
- LLM-optional fast-path consolidation for structured episodes
- Claim precision ranking and outcome-driven consolidation scheduling
- Scope columns and persisted policy/ACL primitives
- Entity-centric recall — optional `entity` on `memory_recall` boosts path/subject-linked episodes, records, and claims via anchors
- Hypothesis competition — config `hypothesis_competition_enabled` keeps contradicted records during consolidation with lowered precision; optional `hypothesis_competition` on `memory_recall` surfaces competing claims
- Surface parity — MCP, REST, Python, and OpenAI dispatch through `MemoryClient`, with one strict allowed-argument set across all of them
- Policy ergonomics — `memory_policy_list` / `memory_policy_grant` on MCP, REST, and OpenAI dispatch; CLI `policy list|grant`
- Simple agent surface — `memory_remember` / `memory_ask` on MCP, REST (`POST /memory/remember`, `POST /memory/ask`) and OpenAI dispatch; `CONSOLIDATION_MEMORY_MCP_TOOL_PROFILE=simple` exposes 3 of the 29 full-profile tools
- Scope discovery — `memory_scope_list` (and `GET /memory/scopes`) aggregates existing scopes (no external registry needed) with per-table usage counts and pageable windows. Counts cover live rows only, and the result is intentionally **not** filtered by `read_visibility` — it is a topology audit, not a tenant boundary (see [ACL.md](ACL.md#trust-boundary)); returned envelopes are reusable as `scope` arguments
- MCP spec surface — host-negotiated protocol revisions, a typed `outputSchema` per tool, `isError` execution failures and one strict argument contract enforced on every surface — see [MCP_GUIDE.md](MCP_GUIDE.md)
- Corpus hygiene — `forget()` expires claims that lose all provenance; `memory_hygiene_scan` / `memory_hygiene_apply` clean noisy episodes and repair orphaned claims, exposed on every surface plus a Hygiene tab
- Graphical surfaces — browser (`consolidation-memory ui`, Ask · Remember · Browse · Health · Hygiene · Metrics) with `init --quick` and an in-browser setup wizard when config is missing; Textual TUI (`consolidation-memory dashboard`); native desktop (`consolidation-memory app`, PySide6, system tray) — see [UI.md](UI.md)
- Live recall evidence — trending `real_world_eval --mode full` on the live `universal` corpus; the published run and thresholds are in [REAL_WORLD_METRICS.md](REAL_WORLD_METRICS.md) and the CI fixture stays regression-only
- Benchmark and positioning narrative — [LOCOMO_BENCHMARK.md](LOCOMO_BENCHMARK.md) (full run needs `OPENAI_API_KEY`), [examples/trust-vs-rag/](../examples/trust-vs-rag/) with `demo_flow.py`, [PLUGIN_DEVELOPMENT.md](PLUGIN_DEVELOPMENT.md) and the [examples/](../examples/) adapters

## Shipped adoption gaps

| Priority | Gap | Done-when |
| --- | --- | --- |
| P0 | **No simple agent surface** — the full MCP profile overwhelms newcomers | ~~`memory_remember` / `memory_ask` + `CONSOLIDATION_MEMORY_MCP_TOOL_PROFILE=simple`~~ |
| P0 | **Live recall proof** — synthetic CI passes; messy corpora underperform | ~~Trending `real_world_eval --mode full` on the live corpus~~ ([REAL_WORLD_METRICS.md](REAL_WORLD_METRICS.md)) |
| P1 | **Setup friction** — Python path, embeddings, hooks, scope concepts | ~~One-command `init --quick` + `ui`; in-browser setup wizard when config missing~~ |
| P1 | **Ops opacity** — stale consolidation / embedding health unclear to casual users | ~~Actionable health in the UIs plus warnings and fix-it flows (consolidate, reindex, warmup)~~ |
| P1 | **Corpus hygiene** — forgetting episodes does not retract claims | ~~Claim expiry on `forget()`; `memory_hygiene_apply` for noisy corpora~~ |
| P2 | **Positioning vs simple RAG** — the wedge is hard to explain in 30s | ~~[examples/trust-vs-rag/](../examples/trust-vs-rag/) + `demo_flow.py`~~ |
| P2 | **Benchmark narrative** — LoCoMo / head-to-head not published | ~~[LOCOMO_BENCHMARK.md](LOCOMO_BENCHMARK.md) harness + dry-run docs~~ |
| P2 | **Ecosystem packaging** — adapter docs, plugin author guide, community templates | ~~[PLUGIN_DEVELOPMENT.md](PLUGIN_DEVELOPMENT.md) + [examples/](../examples/) adapters~~ |

## Next

- **Adapter maturity** — keep transport parity as new retrieval and trust
  features land, and hold every third-party agent adapter to the trust
  invariants in [CONTRIBUTING.md](../CONTRIBUTING.md#trust-invariants).
- **Evaluation depth** — trend live-corpus recall per release so release
  evidence grows with the feature surface, keeping drift response and
  provenance coverage at 100%.
- **Operational observability** — surface applied schema migrations and the
  consolidation/drift audit trail in `memory_status` rather than only in logs.

## Non-goals

- Generic “memory for everything” without measurable retrieval/trust evidence
- Feature sprawl that weakens provenance or temporal correctness
- Transport-only features that skip Python/MCP/REST/OpenAI parity
- Replacing the trust stack with opaque snippet search

## References

- [Architecture](ARCHITECTURE.md)
- [Fast-path episodes](FAST_PATH_EPISODES.md)
- [Real-world metrics](REAL_WORLD_METRICS.md)
- [Contributing](../CONTRIBUTING.md)
