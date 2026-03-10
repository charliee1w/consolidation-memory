# Universal Memory Architecture

## Purpose

Define current architecture boundaries and target structure for universal shared memory without losing existing trust guarantees.

## Current Architecture (Implemented)

### Control and service

- `MemoryClient` orchestrates operations.
- `CanonicalQueryService` centralizes query envelopes and trust-aware semantics.

### Storage

- SQLite for relational state (`memory.db`, schema v14).
- FAISS for semantic retrieval vectors.
- Markdown topic files for human-readable consolidated knowledge.

### Trust layer

- Temporal validity for records/claims.
- Claim graph, provenance links, contradiction and lifecycle events.
- Anchor extraction + drift challenge event pipeline.

### Adapter surfaces

- CLI, Python API, MCP, REST, OpenAI-compatible schemas/dispatch.

## Target Architecture (Incremental)

1. Scope and policy plane.
- Continue evolving from scope filtering + persisted ACL enforcement toward richer governance policy management APIs.

2. Canonical service layer.
- Keep all retrieval and trust semantics in service modules consumed by every adapter.

3. Adapter framework.
- Build explicit adapter contracts for additional ecosystems without semantic drift.

4. Observability and governance.
- Provide reliable introspection and audit workflows for shared deployments.

## Non-Negotiable Invariants

- Temporal query correctness.
- Provenance and contradiction auditability.
- Drift challenge event integrity.
- Cross-surface semantic consistency.

## Migration Posture

- Additive schema evolution.
- Backward-compatible defaults for existing single-project users.
- Explicit docs + tests for every trust-impacting change.

## Current vs Target Summary

| Area | Current | Target |
| --- | --- | --- |
| Scope | Metadata + persisted ACL semantics | Metadata + richer governance policy semantics |
| Query semantics | Mostly centralized | Fully centralized and adapter-agnostic |
| Adapters | Core surfaces only | Core + external ecosystem adapters |
| Governance | Operational basics | Rich policy + observability model |

## Implementation Note

This architecture intentionally preserves local-first behavior as a first-class deployment mode while enabling stronger shared-memory operation.
