# Shared Memory Scopes

Date: 2026-03-07  
Status: Implemented (A6 baseline)

## Purpose

Define the first production-usable shared-memory scoping model for multiple clients/apps while preserving safe defaults and legacy behavior.

## Canonical Scope Concepts

- `namespace`: top-level sharing boundary.
- `app/client`: calling surface identity (`name`, `type`, optional provider/external key).
- `agent`: logical agent identity (optional).
- `session`: short-lived interaction identity (optional).
- `project/repo`: project boundary (`project_slug`, optional repo metadata).

## Persistence Shape (Schema v13)

Scope columns are now persisted on:

- `episodes`
- `knowledge_records`
- `knowledge_topics`

Key columns:

- `namespace_slug`, `namespace_sharing_mode`
- `app_client_name`, `app_client_type`, `app_client_provider`, `app_client_external_key`
- `agent_name`, `agent_external_key`
- `session_external_key`, `session_kind`
- `project_slug`, `project_display_name`, `project_root_uri`, `project_repo_remote`, `project_default_branch`

Migration behavior:

- additive, rollback-safe migration (`schema_version` 13)
- existing rows backfilled to safe defaults
- `project_slug` backfilled from active project for legacy installs

## Defaults

When scope is omitted:

- `namespace_slug = "default"`
- `namespace_sharing_mode = "private"`
- `app_client_name = "legacy_client"`
- `app_client_type = "python_sdk"`
- `project_slug = active_project` (or `default`)

This preserves current single-project behavior.

## Read/Write Semantics

Writes:

- all `store`/`store_batch` paths now persist resolved scope metadata.
- dedup is scope-aware.

Reads:

- namespace + project boundaries are always enforced.
- `private` namespace mode also enforces app/client isolation.
- `shared`/`team`/`managed` modes intentionally allow cross-app reads inside the same namespace+project.
- agent/session fields are optional narrowing filters when provided.

## Conflict Behavior

- write conflicts do not overwrite: episodes remain append-only UUID rows.
- near-duplicate writes are deduplicated only within the effective read scope.
- consolidation now prevents cross-scope clustering by forcing cross-scope distances apart.
- topic creation/updates persist scope and use scope-prefixed filenames to avoid silent cross-scope topic collisions.

## Privacy Boundaries

- no implicit cross-namespace reads.
- no implicit cross-project reads.
- `private` mode prevents accidental cross-app memory sharing.
- sharing across clients/apps is explicit (`namespace_sharing_mode` set to non-private and same namespace/project selected).
- agent/session metadata is persisted as queryable scope metadata; it is not exposed as prompt/model-state sharing.

## Backward Compatibility

- legacy APIs remain valid (`store`, `recall`, `search`, REST endpoints).
- scope-aware wrappers are additive (`*_with_scope`) and schema-level scope arguments remain optional.
- old data remains readable through default-scope backfill.

## Follow-Up Hardening

- extend scope-aware filtering to broader trust/audit surfaces as needed (policy/ACL phase).
- formalize namespace/app identity tables if/when lifecycle management requirements exceed column-based persistence.
