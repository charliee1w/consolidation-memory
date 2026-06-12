# Copilot instructions — consolidation-memory

Read before suggesting or applying changes:

1. `docs/AGENT_GOAL.md` — current milestone and next task
2. `docs/VIBECODING.md` — trust invariants, verification, milestone order
3. `AGENTS.md` — session workflow

## Product stance

Claims are reusable beliefs; episodes are evidence. This is a trust layer, not generic RAG.

## Coding rules

- Prefer deterministic logic in `consolidation/fast_path.py` over LLM prompt changes.
- Keep Python / MCP / REST / OpenAI tool semantics aligned via `query_service.py`.
- Add tests for behavior changes; run `ruff check src tests` on touched files.
- Use additive DB migrations in `database.py` when schema changes.

## Current focus

M1 — LLM-optional substrate. See unchecked items in `docs/AGENT_GOAL.md`.