# Roadmap

Outcome-based direction for `consolidation-memory`. Details may shift as the implementation evolves.

## Objective

Make local-first agent memory dependable for coding workflows: useful recall, explicit trust signals, and low operational overhead.

## Shipped today

- Hybrid retrieval over episodes, topics, records, and claims
- Temporal queries (`as_of`) on trust surfaces
- Claim graph with contradiction and lifecycle events
- Drift-aware claim challenges via file anchors and git deltas
- LLM-optional fast-path consolidation for structured episodes
- Claim precision ranking and outcome-driven consolidation scheduling
- Scope columns and persisted policy/ACL primitives
- MCP, REST, Python, and OpenAI tool parity through `MemoryClient`

## Near term

- **Hypothesis competition** — optional mode to keep competing claims visible with lowered precision instead of immediate expiry
- **Entity-centric recall** — thin structural layer from anchors and record subjects, without new heavyweight infrastructure
- **Policy ergonomics** — clearer admin surfaces for namespace and ACL management in self-hosted deployments
- **Adapter maturity** — keep transport parity as new retrieval and trust features land

## Mid term

- Broader agent-ecosystem adapters with trust invariants preserved
- Stronger operational tooling for migrations, consolidation observability, and drift audits
- Continued evaluation depth for release-quality evidence

## Non-goals

- Generic “memory for everything” without measurable retrieval/trust evidence
- Feature sprawl that weakens provenance or temporal correctness
- Transport-only features that skip Python/MCP/REST/OpenAI parity

## References

- [Architecture](ARCHITECTURE.md)
- [Fast-path episodes](FAST_PATH_EPISODES.md)
- [Contributing](../CONTRIBUTING.md)