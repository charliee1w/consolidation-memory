# consolidation-memory (Grok project rules)

Read at session start:

1. `docs/AGENT_GOAL.md` — first unchecked ⬜ task (M1 unless user redirects)
2. `docs/VIBECODING.md` — trust rules override vague instructions

## Workflow

- `memory_recall` before coding (MCP `consolidation_memory` if configured)
- One slice per session; tests + `ruff check src tests` before done
- Update `docs/AGENT_GOAL.md` when a task completes
- `memory_recall` before final response

## Default user intent

User says **"go"** → continue agent goal, pick first open M1 task.

## Mantra

Vibe deterministic belief maintenance; LLMs only for unstructured residue; tests prove trust.