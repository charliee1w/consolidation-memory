# Canonical Memory Object Model

## Purpose

Define the conceptual objects used across current and future shared-memory features.

## Core Objects

1. Namespace
- Top-level sharing boundary.

2. Project
- Repository/workspace scope under a namespace.

3. App Client
- Calling integration identity (MCP, REST, CLI, SDK, etc.).

4. Agent
- Optional agent identity within an app client.

5. Session
- Optional conversation/thread/workflow identity.

6. Episode
- Raw stored memory event.

7. Knowledge Topic
- Consolidated markdown-level summary node.

8. Knowledge Record
- Typed structured unit (`fact`, `solution`, `preference`, `procedure`).

9. Claim
- Deterministic proposition derived from knowledge records.

10. Provenance Link
- Mapping from claim to source episode/topic/record.

11. Anchor
- Extracted path/tool/commit references from episodes.

12. Lifecycle Event
- Claim event timeline (`create`, `update`, `contradiction`, `code_drift_detected`, etc.).

## Implementation Status

| Object | Status |
| --- | --- |
| Namespace/Project/App/Agent/Session | Implemented as scope envelope + persisted columns |
| Episode/Topic/Record | Implemented |
| Claim/Provenance/Event/Anchor | Implemented |
| Policy/ACL object | Planned |

## Mapping To Current Schema

- Episodes -> `episodes`
- Topics -> `knowledge_topics`
- Records -> `knowledge_records`
- Claims -> `claims`
- Provenance -> `claim_sources`
- Claim relationships -> `claim_edges`
- Lifecycle -> `claim_events`
- Anchors -> `episode_anchors`

## Evolution Rule

Add new conceptual objects only when they create enforceable behavior or measurable trust improvements.
