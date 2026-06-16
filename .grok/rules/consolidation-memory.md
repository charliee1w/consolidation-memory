# consolidation-memory (Grok project rules)

## Memory first

See [00-memory-first.md](00-memory-first.md). `memory_recall` is the mandatory first
tool on every user turn when `consolidation_memory` MCP is available.

Read [CONTRIBUTING.md](../../CONTRIBUTING.md) and [docs/ARCHITECTURE.md](../../docs/ARCHITECTURE.md) before code changes.

## Fresh working copy

After `memory_recall`, run `python scripts/sync_working_copy.py` as the first shell step
when the tree is clean. This fetches and rebases onto `origin/<branch>`. If the tree is
dirty, report it and ask whether to commit, stash (`--stash`), or continue on the
current base. Re-run sync before pushing or after long idle gaps.

## Workflow

- Preserve trust invariants (temporal correctness, provenance, contradictions, drift auditability, scope isolation, surface parity).
- One focused slice per session; run targeted `pytest` + `ruff check src tests` before done.
- Update user-facing docs when behavior changes.
- Read **Known architectural debt** in [Claude.md](../../Claude.md) before large refactors.

## Audit-aligned maintainer checks (2026-06-15)

- Full MCP profile = **30 tools**; simple profile = `memory_recall`, `memory_remember`, `memory_ask`.
- Hygiene: `memory_hygiene_scan` / `memory_hygiene_apply` (global-by-design); `forget()` expires orphan claims.
- Consolidated knowledge can lag code — use `memory_correct` or superseding episodes + consolidate.
- New tools: ship on MCP + REST + OpenAI dispatch + tests; document scope-aware vs global in CONTRIBUTING.

## Mantra

Deterministic belief maintenance first; LLMs only for unstructured residue; tests prove trust.