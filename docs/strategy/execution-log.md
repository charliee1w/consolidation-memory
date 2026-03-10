# Universal Memory Execution Log

This log records meaningful strategy-level milestones and decisions.

## 2026-03-10: Policy primitives enforced in scoped write paths

- Added `scope.policy` support (`read_visibility`, `write_mode`) to canonical scope schema and coercion.
- Enforced scoped write deny mode in `MemoryClient` store paths (`write_mode=deny` returns `write_denied` without embedding/database writes).
- Added adapter parity updates:
  - MCP `memory_store` and `memory_store_batch` now accept optional `scope`.
  - OpenAI tool schema now includes `scope.policy`.
  - Cross-surface tests cover Python, MCP, REST, and OpenAI-dispatch write-deny behavior.
- Contract change:
  - `StoreResult.status` and `BatchStoreResult.status` now include `write_denied`.
- Remaining:
  - Persisted policy/ACL entities are still planned (policy is currently envelope-level semantics, not standalone DB objects).

## 2026-03-08: Documentation reset for durability

- Rewrote strategy docs to align with current schema v13 + canonical query service architecture.
- Removed stale timeline/test-count narrative from strategy artifacts.
- Standardized strategy docs around implementation status, migration posture, and trust invariants.

## 2026-03-07: Shared scope persistence landed

- Added persistent scope columns and indexes for episodes/topics/records.
- Added scope resolution and scope-aware filtering in client/query paths.
- Preserved backward-compatible defaults for legacy single-project flows.

## 2026-03-07: Universal planning artifacts created

- Added gap analysis, architecture target, object model, and execution plan docs.
- Identified governance/policy and adapter ecosystem as primary remaining gaps.

## Log Maintenance Rules

For each entry include:

- date,
- what shipped,
- what changed in contracts,
- what remains.

Do not use this file for speculative notes without implementation relevance.
