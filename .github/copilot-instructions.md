# Copilot instructions — consolidation-memory

Read before suggesting or applying changes:

1. `CONTRIBUTING.md` — trust invariants, scope-vs-global contract, local validation
2. `docs/ARCHITECTURE.md` — modules, persistence, contracts
3. `docs/MCP_GUIDE.md` — wire contract, scopes, errors

## Product stance

Claims are reusable beliefs; episodes are evidence. This is a trust layer, not generic RAG.

## Coding rules

- Prefer deterministic logic in `consolidation/fast_path.py` over LLM prompt changes.
- Keep Python / MCP / REST / OpenAI tool semantics aligned via `query_service.py`.
- Add tests for behavior changes; run `ruff check src tests/ scripts/` on touched files.
- Schema changes: add an additive migration in `src/consolidation_memory/db/migrations.py`,
  bump `CURRENT_SCHEMA_VERSION` in that module, and cover it in `tests/test_core.py` with a
  test that calls `ensure_schema()`. `database.py` only re-exports from `db/`.

## Current focus

Open work is tracked in the **Debt ledger** section of `CLAUDE.md`, ranked P0 → P2.
