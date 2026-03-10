# Universal Memory Execution Log

This log records meaningful strategy-level milestones and decisions.

## 2026-03-10: Persisted ACL policy enforcement shipped (schema v14)

- Added first-class persisted policy/ACL entities:
  - `access_policies`
  - `policy_principals`
  - `policy_acl_entries`
- Implemented canonical policy resolution in `MemoryClient.resolve_scope()`:
  - `scope.policy` remains backward-compatible fallback.
  - persisted ACL is authoritative when matching rows exist.
  - conflict rules are explicit: write deny-overrides-allow; read uses most restrictive visibility.
- Enforced policy in canonical client/service paths:
  - writes: `store`, `store_batch`, scoped variants.
  - reads/queries: `recall`, `search`, claim browse/search.
  - topic/record retrieval paths now enforce persisted policy as applicable (`browse`, `read_topic`, `timeline`).
- Added cross-surface parity coverage (Python, MCP, REST, OpenAI dispatch) for persisted write-deny and read-visibility enforcement.
- Remaining:
  - no public policy administration APIs yet (policies currently managed via DB helpers/internal calls).

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
