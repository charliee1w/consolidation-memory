# Roadmap

This roadmap is outcome-based so it stays useful as implementation details evolve.

## Product Objective

Make `consolidation-memory` the most dependable local-first memory layer for coding agents: high recall usefulness, strong trust semantics, and low operational overhead.

**Commercial objective:** Keep the MIT core as the adoption engine; monetize **team governance, sync, and compliance operations** (see [MONETIZATION.md](MONETIZATION.md)).

## Current Platform Baseline

Already implemented:

- Hybrid retrieval over episodes/topics/records/claims.
- Temporal query support (`as_of`) across key trust surfaces.
- Claim graph + contradiction and lifecycle event tracking.
- Drift-aware claim challenge flow using stored anchors and git deltas.
- Release gate enforcement wired into CI and publish workflows.
- LLM-optional fast-path consolidation (M1), claim precision ranking (M2), outcome-driven utility scheduling (M3).
- Automated changelog + release-on-main workflows.

## Engineering Track (open core)

### Current: M4 — Hypothesis competition

- `CONTRADICTION_RESOLUTION_MODE=compete|expire_old` (default unchanged)
- Competing claims stay active with lowered precision + `contradicts` edges
- Novelty eval scenario for compete mode

**Done when:** novelty gates pass with compete mode enabled.

### Next: M5 — Thin structural layer

- Entity records derived from anchors + record subjects
- Entity-centric recall expansion
- Graph traversal over existing tables only

**Done when:** entity-centric recall beats raw vector-only on a fixed eval fixture.

### Ongoing platform work

1. **Scope hardening and policy semantics** — ACL enforcement shipped (schema v14); continue policy admin APIs.
2. **Adapter maturity** — MCP/REST/OpenAI/Python parity under new features.
3. **Evaluation depth** — expand novelty harness evidence for release decisions.
4. **Operational resilience** — migration safety, repair tooling, consolidation/drift observability.

## Commercial Track (paid surface)

Mapped to [MONETIZATION.md](MONETIZATION.md). Engineering prerequisites listed so open-core work does not block revenue accidentally.

| Milestone | Revenue intent | Engineering deliverables | Target signal |
| --- | --- | --- | --- |
| **C0 (now)** | Positioning + pipeline | Monetization doc, README/roadmap links, design-partner outreach | 3+ serious conversations |
| **C1** | First paid pilot | Policy admin REST + read-only trust dashboard; manual billing | 1 paying team namespace |
| **C2** | Team tier beta | Sync transport, hosted consolidation/drift workers, namespace billing hooks | Self-serve signup (waitlist → paid) |
| **C3** | Enterprise pilot | SSO, signed evidence exports, retention/legal-hold admin | Security review passed |

**Open-core rule:** C1–C3 add **control plane, transport, and ops** — not replacements for `MemoryClient` trust semantics.

## Mid-Term Priorities

1. **Universal integration adapters** — external agent ecosystems with trust invariants preserved.
2. **Canonical service boundary** — business semantics in query/service layers; transports stay thin.
3. **Deployment posture** — local-first default + documented self-hosted Team topology.

## Success Criteria

- Retrieval quality remains stable or improves across benchmark suites.
- Trust guarantees remain auditable (temporal correctness, contradiction traceability, provenance coverage).
- Release gates pass with recent evidence.
- API surface remains compatible across supported transports.
- Commercial milestones have explicit open-core boundaries before implementation starts.

## Non-Goals

- Broad “memory for everything” positioning without benchmark evidence.
- Feature sprawl that weakens trust semantics.
- Shipping major behavior changes without cross-surface contract checks.
- Paywalling drift, provenance, or local single-user usage.

## Planning References

- [Monetization Plan](MONETIZATION.md) — tiers, boundaries, C1–C3
- [Agent Goal](AGENT_GOAL.md) — living engineering backlog
- [Vibecoding Guide](VIBECODING.md) — rules for agent-assisted implementation
- [Release Gates](RELEASE_GATES.md)
- [Novelty Metrics](NOVELTY_METRICS.md)
- [Novelty Wedge](NOVELTY_WEDGE.md)
- [Universal strategy docs](strategy/)