# Schema Migration Plan For The Canonical Memory Model

Date: 2026-03-07
Status: Partially Implemented (A6 baseline)

## Purpose

Map the current SQLite schema to the canonical universal-memory model in additive, backward-compatible phases.

This plan starts from the current repository state:

- schema version: `13` in `src/consolidation_memory/database.py`
- current durable boundary: project directory plus persisted scope columns on episodes/records/topics
- current trust layer: claims, claim sources, claim events, contradictions, and episode anchors

## Assumptions

1. The current API surfaces must continue working during migration.
2. Initial migrations should add new tables and nullable foreign keys before any destructive cutover.
3. Existing project-isolated installs should auto-upgrade into one default namespace without user intervention.
4. Service-layer cutover happens after the schema scaffolding exists; this document does not require service refactors yet.

## Current Schema Inventory

### Current tables that already exist

- `episodes`
- `knowledge_topics`
- `knowledge_records`
- `consolidation_runs`
- `consolidation_metrics`
- `contradiction_log`
- `tag_cooccurrence`
- `claims`
- `claim_edges`
- `claim_sources`
- `claim_events`
- `episode_anchors`
- `schema_version`
- FTS tables managed separately

### Current gaps relative to the canonical model

- no `namespaces`
- no `principals`
- no `app_clients`
- no `agents`
- no first-class managed `sessions` table
- no first-class managed `projects` table
- no universal `memory_objects`
- no policy / ACL tables
- provenance and anchors are specialized, not generalized
- contradictions and lifecycle events are not generalized across object types

### A6 implementation update

The A6 step landed a production baseline before full table normalization:

- scope columns were added to `episodes`, `knowledge_records`, and `knowledge_topics`
- client/query/consolidation paths now enforce namespace/project boundaries with explicit sharing modes
- legacy data is backfilled to safe defaults during migration

This keeps behavior backward-compatible while deferring table-heavy identity management to later phases.

## Migration Strategy

The migration strategy is:

1. add missing scope and identity tables
2. backfill default rows from current local installs
3. add a universal `memory_objects` registry above existing payload tables
4. dual-write generalized provenance, anchors, contradictions, and lifecycle events
5. enforce scopes and ACLs in the service layer only after the additive schema exists

No current table should be dropped during the initial universal-memory rollout.

## Table Disposition

| Table or concept | Disposition | Notes |
| --- | --- | --- |
| `episodes` | Keep, evolve | Add scope FKs and universal memory ID |
| `knowledge_topics` | Keep, evolve | Continue as human-facing projection |
| `knowledge_records` | Keep, evolve | Add scope FKs and universal memory ID |
| `claims` | Keep, evolve | Add namespace and policy fields |
| `claim_edges` | Keep | Add namespace only if needed for query efficiency |
| `claim_sources` | Keep temporarily, later supersede | Backfill into generalized provenance links |
| `claim_events` | Keep temporarily, later supersede | Backfill into generalized lifecycle events |
| `episode_anchors` | Keep temporarily, later supersede | Backfill into generalized anchors |
| `contradiction_log` | Keep temporarily, later supersede | Backfill into generalized contradictions |
| `consolidation_runs` | Keep | Operational audit remains useful |
| `consolidation_metrics` | Keep | Operational metrics remain useful |
| `tag_cooccurrence` | Keep | Ranking support, optionally namespace-aware later |
| `schema_version` | Keep | Migration ledger |
| `Config.active_project` | Reinterpret | Client default only, not canonical boundary |

## Phased Migration

### Phase 1: Add identity and shared-scope tables

Target schema version: `12`

New tables:

- `namespaces`
- `principals`
- `app_clients`
- `agents`
- `sessions`
- `projects`

Add nullable columns:

- `episodes.namespace_id`
- `episodes.project_id`
- `episodes.principal_id`
- `episodes.app_id`
- `episodes.agent_id`
- `episodes.session_id`
- `knowledge_topics.namespace_id`
- `knowledge_topics.project_id`
- `knowledge_records.namespace_id`
- `knowledge_records.project_id`
- `knowledge_records.principal_id`
- `knowledge_records.app_id`
- `knowledge_records.agent_id`
- `knowledge_records.session_id`
- `claims.namespace_id`

Backfill rules:

- create namespace `default`
- create one synthetic principal such as `local_owner`
- create one synthetic app client such as `legacy_client`
- create one `projects` row per existing project storage directory
- map current `active_project` behavior to a default `projects` row
- when `episodes.source_session` is present, create or reuse a `sessions` row using that string as `external_key`

Compatibility rules:

- existing code can keep reading old tables without using the new columns
- if no scope is provided, resolve to `namespace=default` and the current project

Validation expectations:

- migration tests from schema v11 to v12
- legacy single-project store/recall regression tests
- project isolation tests still green

### Phase 2: Add the universal memory registry

Target schema version: `13`

New table:

- `memory_objects`

Suggested columns:

- `id`
- `namespace_id`
- `memory_kind`
- `semantic_kind`
- `project_id`
- `session_id`
- `principal_id`
- `app_id`
- `agent_id`
- `payload_ref_kind`
- `payload_ref_id`
- `status`
- `valid_from`
- `valid_until`
- `policy_id`
- `created_at`
- `updated_at`

Add nullable columns:

- `episodes.memory_object_id`
- `knowledge_records.memory_object_id`

Backfill rules:

- one `memory_objects` row per `episodes` row
- one `memory_objects` row per `knowledge_records` row
- `knowledge_topics` remain projections and do not need to become memory objects immediately

Compatibility rules:

- old table rows remain source-of-truth payloads during this phase
- new adapters can start referencing `memory_object_id` without changing payload formats

Validation expectations:

- backfill integrity tests: every episode and knowledge record gets one memory object
- temporal validity tests remain green after backfill
- no duplicate or orphaned `memory_object_id` values

### Phase 3: Generalize provenance and anchors

Target schema version: `14`

New tables:

- `provenance_links`
- `anchors`

Backfill sources:

- `claim_sources` -> `provenance_links`
- `episode_anchors` -> `anchors`

Compatibility rules:

- dual-write both old and new tables until the service layer changes
- existing claim recall behavior must continue to work from current tables

Design notes:

- generalized provenance must support memory-to-claim and memory-to-memory derivation
- generalized anchors must attach to `memory_objects`, not only `episodes`

Validation expectations:

- claim evidence parity tests
- source traceability tests
- drift invalidation tests

### Phase 4: Add policy and ACL tables

Target schema version: `15`

New tables:

- `access_policies`
- `acl_entries`

Add nullable columns:

- `namespaces.default_policy_id`
- `memory_objects.policy_id`
- `claims.policy_id`
- `projects.policy_id`

Backfill rules:

- create one default namespace-private policy that preserves current behavior
- bind all legacy objects to the default policy

Compatibility rules:

- this phase adds data only; enforcement is deferred to the service-layer refactor
- until enforcement lands, old behavior remains unchanged

Validation expectations:

- migration tests
- policy inheritance tests
- no regression in legacy same-project visibility

### Phase 5: Generalize contradictions and lifecycle events

Target schema version: `16`

New tables:

- `contradictions`
- `lifecycle_events`

Backfill sources:

- `contradiction_log` -> `contradictions`
- `claim_events` -> `lifecycle_events(subject_type=claim)`
- selected operational events may also be mirrored from consolidation workflows

Compatibility rules:

- keep existing contradiction and claim-event queries working
- dual-write until service-layer cutover

Validation expectations:

- contradiction visibility tests
- audit event ordering tests
- temporal recall plus claim timeline tests

### Phase 6: Service-layer cutover prerequisites

Target schema version: no new schema required if earlier phases are complete

Expected service-layer changes after the additive schema exists:

- resolve a canonical scope envelope on every read/write
- prefer `memory_objects` as the universal object ID
- use generalized provenance for new adapter work
- evaluate ACLs after scope resolution
- keep markdown topics as projections, not as the canonical contract

Validation expectations:

- cross-surface semantic parity tests
- trust regression suite
- explicit shared-scope and isolation tests

## Backfill Defaults

These defaults keep upgrades silent for current local users:

- `namespace.slug = "default"`
- `principal.display_name = "Local Owner"`
- `app_clients.name = "legacy_client"`
- `projects.slug = <existing active project name>`
- `sessions.external_key = episodes.source_session` when present

If any row lacks a meaningful legacy mapping, leave the new FK nullable during the first additive migration and let the later service layer fill it explicitly.

## Rollout Rules

1. Add tables before rewriting reads.
2. Backfill before dual-write.
3. Dual-write before cutover.
4. Keep old query paths until parity tests pass.
5. Do not drop legacy tables in the same release that introduces canonical writes.

## Deferred From This Migration Plan

These are intentionally not required for the first canonical-model rollout:

- group and organization membership tables
- managed multi-tenant authn/authz
- remote replication / sync bus
- storage-engine replacement beyond SQLite + FAISS
- Letta-style block payload tables as a first migration target

The model should allow those later, but this plan avoids overloading the first schema transition.

## Recommended Implementation Order

1. Implement Phase 1 and Phase 2 first.
2. Refactor the service layer only after those tables exist.
3. Generalize provenance before adding native adapters.
4. Add policy enforcement before enabling intentional cross-client shared writes.

That order preserves current trust behavior while creating the minimum durable model needed for universal adapters.
