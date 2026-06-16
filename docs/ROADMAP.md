# Roadmap

Outcome-based direction for `consolidation-memory`. Details may shift as the implementation evolves.

## Objective

Make local-first agent memory dependable for coding workflows: useful recall, explicit trust signals, and low operational overhead — **usable by anyone**, not only MCP power users.

## Shipped today

- Hybrid retrieval over episodes, topics, records, and claims
- Temporal queries (`as_of`) on trust surfaces
- Claim graph with contradiction and lifecycle events
- Drift-aware claim challenges via file anchors and git deltas
- LLM-optional fast-path consolidation for structured episodes
- Claim precision ranking and outcome-driven consolidation scheduling
- Scope columns and persisted policy/ACL primitives
- Entity-centric recall — optional `entity` on `memory_recall` boosts path/subject-linked episodes, records, and claims via anchors
- Hypothesis competition — config `hypothesis_competition_enabled` keeps contradicted records during consolidation with lowered precision; optional `hypothesis_competition` on `memory_recall` surfaces competing claims
- MCP, REST, Python, and OpenAI tool parity through `MemoryClient`
- ~~**Policy ergonomics**~~ — `memory_policy_list` / `memory_policy_grant` on MCP, REST, and OpenAI dispatch; CLI `policy list|grant`
- **Browser UI** — `consolidation-memory ui` serves `/ui/` (Ask · Remember · Browse · Health · Hygiene · Metrics); `init --quick` for zero-prompt setup; in-browser setup wizard when config is missing
- **Corpus hygiene** — `forget()` expires claims that lose all provenance; Hygiene tab + `corpus_hygiene` scan/apply for noisy episodes and orphaned claims
- **Native desktop UI** — `consolidation-memory app` (PySide6) with Ask · Remember · Browse and system tray icon
- ~~**MCP simple tools**~~ — `memory_remember` / `memory_ask` on MCP, REST (`POST /memory/remember`, `POST /memory/ask`), and OpenAI dispatch; browser UI uses the same aliases

## Adoption blockers (tracked)

Prioritized gaps between engineering maturity and broad adoption. Each item has a measurable done-when.

| Priority | Blocker | Done-when |
| --- | --- | --- |
| P0 | **No simple agent surface** — full MCP profile (30 tools) overwhelms newcomers | ~~`memory_remember` / `memory_ask` + `CONSOLIDATION_MEMORY_MCP_TOOL_PROFILE=simple`~~ (shipped) |
| P0 | **Live recall proof gap** — synthetic CI passes; messy corpora underperform | ~~Trending `real_world_eval --mode full` on live `universal` corpus~~ (2026-06-14, see [REAL_WORLD_METRICS.md](REAL_WORLD_METRICS.md)); CI fixture stays regression-only |
| P1 | **Setup friction** — Python path, embeddings, hooks, scope concepts | ~~One-command `init --quick` + `ui`; in-browser setup wizard when config missing~~ (shipped) |
| P1 | **Ops opacity** — stale consolidation / embedding health unclear to casual users | ~~Actionable health in UI + warnings; fix-it flows (consolidate, reindex, warmup)~~ (shipped) |
| P1 | **Corpus hygiene** — forget episodes does not retract claims | ~~Claim retraction/expiry on forget; cleanup wizard for noisy corpora~~ (shipped) |
| P2 | **Positioning vs simple RAG** — narrow wedge hard to explain in 30s | ~~[examples/trust-vs-rag/](../examples/trust-vs-rag/) + `demo_flow.py`~~ (shipped) |
| P2 | **Benchmark narrative** — LoCoMo / head-to-head not published | ~~[LOCOMO_BENCHMARK.md](LOCOMO_BENCHMARK.md) harness + dry-run docs~~ (shipped; full run needs `OPENAI_API_KEY`) |
| P2 | **Ecosystem packaging** — adapter docs, plugin author guide, community templates | ~~[PLUGIN_DEVELOPMENT.md](PLUGIN_DEVELOPMENT.md) + [examples/](examples/) adapters~~ (shipped) |

## Near term (adoption slice)

1. ~~**Agent simple profile**~~ — `CONSOLIDATION_MEMORY_MCP_TOOL_PROFILE=simple`; documented in README and CONTRIBUTING
2. ~~**UI setup wizard**~~ — detect missing config from `/ui/`, guide `init --quick`, show MCP snippets
3. ~~**Live metrics dashboard**~~ — Metrics tab charts bundled/live `real_world_eval` sections
4. **Adapter maturity** — keep transport parity as new retrieval and trust features land

## Mid term

- ~~Claim-level retraction when episodes are forgotten~~ (shipped via `forget()` + hygiene orphan repair)
- Broader agent-ecosystem adapters with trust invariants preserved
- Stronger operational tooling for migrations, consolidation observability, and drift audits
- Continued evaluation depth for release-quality evidence

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