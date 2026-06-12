# consolidation-memory (Grok project rules)

Read [CONTRIBUTING.md](../../CONTRIBUTING.md) and [docs/ARCHITECTURE.md](../../docs/ARCHITECTURE.md) before code changes.

## Workflow

- Preserve trust invariants (temporal correctness, provenance, contradictions, drift auditability, scope isolation, surface parity).
- One focused slice per session; run targeted `pytest` + `ruff check src tests` before done.
- Update user-facing docs when behavior changes.

## Mantra

Deterministic belief maintenance first; LLMs only for unstructured residue; tests prove trust.