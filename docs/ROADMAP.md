# Roadmap

This roadmap is outcome-based so it stays useful as implementation details evolve.

## Product Objective

Make `consolidation-memory` the most dependable local-first memory layer for coding agents: high recall usefulness, strong trust semantics, and low operational overhead.

## Current Platform Baseline

Already implemented:

- Hybrid retrieval over episodes/topics/records/claims.
- Temporal query support (`as_of`) across key trust surfaces.
- Claim graph + contradiction and lifecycle event tracking.
- Drift-aware claim challenge flow using stored anchors and git deltas.
- Release gate enforcement wired into CI and publish workflows.

## Near-Term Priorities

1. Scope hardening and policy semantics.
- Move from scope metadata + filtering to stricter policy enforcement primitives.

2. Adapter maturity.
- Strengthen cross-surface contracts so MCP/REST/OpenAI/Python remain behaviorally aligned under new features.

3. Evaluation depth.
- Expand benchmark coverage beyond current novelty harness and keep evidence fresh for release decisions.

4. Operational resilience.
- Continue improving migration safety, repair tooling, and observability around consolidation and drift workflows.

## Mid-Term Priorities

1. Universal integration adapters.
- Add first-class adapter paths for external agent ecosystems while preserving trust invariants.

2. Canonical service boundary.
- Keep business semantics centralized in query/service layers and reduce adapter duplication.

3. Deployment posture.
- Preserve local-first default while supporting stronger shared/self-hosted operation models.

## Success Criteria

- Retrieval quality remains stable or improves across benchmark suites.
- Trust guarantees remain auditable (temporal correctness, contradiction traceability, provenance coverage).
- Release gates pass with recent evidence.
- API surface remains compatible across supported transports.

## Non-Goals

- Broad “memory for everything” positioning without benchmark evidence.
- Feature sprawl that weakens trust semantics.
- Shipping major behavior changes without cross-surface contract checks.

## Planning References

- [Release Gates](RELEASE_GATES.md)
- [Novelty Metrics](NOVELTY_METRICS.md)
- [Novelty Wedge](NOVELTY_WEDGE.md)
- [Universal strategy docs](strategy/)
