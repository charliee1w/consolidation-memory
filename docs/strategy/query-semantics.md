# Canonical Query Semantics

## Purpose

Describe the query-time contract that every adapter surface must follow.

## Canonical Service Layer

`src/consolidation_memory/query_service.py` defines canonical query envelopes and execution paths:

- `RecallQuery`
- `EpisodeSearchQuery`
- `ClaimBrowseQuery`
- `ClaimSearchQuery`
- `DriftQuery`

Adapters should delegate to this layer instead of implementing independent semantics.

## Invariants

1. Temporal invariant
- `as_of` queries must reconstruct state valid at that timestamp.

2. Scope invariant
- Scope filters must be applied consistently across episodes, records, topics, and claims.

3. Provenance invariant
- Claim search/browse must preserve provenance-aware filtering behavior.

4. Drift invariant
- Drift detection must produce deterministic impacted/challenged claim output and audit events.

5. Error-contract invariant
- Adapter errors should map to surface-appropriate transport errors while preserving root cause clarity.

## Adapter Mapping

- Python API: `MemoryClient.query_*` and convenience wrappers.
- MCP: tool functions in `server.py`.
- REST: endpoints in `rest.py`.
- OpenAI-compatible: schemas + dispatcher in `schemas.py`.

## Regression Strategy

When query semantics change:

1. update canonical service tests,
2. update adapter contract tests,
3. update docs in the same change,
4. verify novelty/release gates if trust metrics are affected.
