# Shared Memory Scopes

## Purpose

Document the implemented scope model and its compatibility behavior.

## Scope Envelope

The canonical scope envelope includes:

- `namespace`
- `project`
- `app_client`
- `agent`
- `session`
- `policy`

`MemoryClient.resolve_scope()` fills defaults when fields are omitted.

## Persisted Shape (Schema v14)

Scope columns are persisted on:

- `episodes`
- `knowledge_topics`
- `knowledge_records`

Policy/ACL entities are persisted on:

- `access_policies` (scope-targeted policy containers)
- `policy_principals` (principal identity rows)
- `policy_acl_entries` (principal-to-policy bindings with read/write controls)

Column groups:

- namespace: `namespace_slug`, `namespace_sharing_mode`
- app client: `app_client_name`, `app_client_type`, `app_client_provider`, `app_client_external_key`
- agent/session: `agent_name`, `agent_external_key`, `session_external_key`, `session_kind`
- project metadata: `project_slug`, `project_display_name`, `project_root_uri`, `project_repo_remote`, `project_default_branch`

## Default Compatibility Behavior

If scope is omitted:

- namespace defaults to `default`
- app client defaults to `legacy_client`
- project defaults to active project

This keeps legacy single-project usage working.

## Read/Write Semantics

- Writes persist resolved scope columns.
- Reads apply scope filters built from resolved scope.
- Namespace sharing modes `shared`, `team`, and `managed` intentionally broaden app-client isolation behavior.
- Policy controls are resolved in this order:
  - `scope.policy` remains a compatibility fallback.
  - persisted ACL rows are authoritative when present for the resolved scope/principal.
- Conflict resolution:
  - write: deny-overrides-allow (`deny` wins)
  - read visibility: most restrictive visibility wins (`private` over `project` over `namespace`)

## Privacy Boundary Today

Current privacy boundary is enforced by canonical service/client logic using scope filters plus persisted ACL policy resolution.

## Verification

Inspect:

- `src/consolidation_memory/client.py` (`resolve_scope`, `_resolved_scope_to_db_row`, `_resolved_scope_to_query_filter`)
- `src/consolidation_memory/database.py` (scope columns + indexes)
