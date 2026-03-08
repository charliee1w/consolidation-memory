# Novelty Wedge

## Product Wedge

`consolidation-memory` is focused on **Drift-aware debugging memory** for coding agents.

The system should retain high-value debugging knowledge, detect when code changes invalidate prior memory, and expose the trust trail needed to safely reuse past solutions.

## Primary User

A developer or coding assistant that iterates on codebases over time and needs memory that is:

- Local-first and inspectable.
- Temporal (able to reconstruct prior belief states).
- Provenance-aware (can show where a belief came from).
- Drift-aware (can challenge stale claims when files change).

## In-Scope Use Cases

1. Preserve debugging outcomes as durable solution records.
2. Recall prior fixes with confidence and source traceability.
3. Challenge affected claims when repository files change.
4. Query what was believed at a prior time (`as_of`).

## Out Of Scope

1. Generic consumer long-term memory assistants.
2. Opaque “black box” memory behavior without auditable sources.
3. Multi-tenant cloud control plane claims not yet implemented.

## Why This Wedge

- It maps directly to implemented trust semantics (claims, provenance, contradictions, drift).
- It is measurable by the current novelty harness.
- It avoids overpromising beyond shipped behavior.
