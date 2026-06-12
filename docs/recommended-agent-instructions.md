# Recommended Agent Instructions

Copy blocks below into agent hosts (Cursor rules, Claude `CLAUDE.md`, Copilot, custom system prompts).

**This repo now ships most of this automatically:**

- Cursor: `.cursor/rules/consolidation-memory.mdc` (`alwaysApply: true`)
- Copilot: `.github/copilot-instructions.md`
- All agents: `AGENTS.md`, `GOAL.md`, `docs/AGENT_GOAL.md`, `docs/VIBECODING.md`

---

## Minimal block (any agent host)

```text
Working on consolidation-memory.
1) Read docs/AGENT_GOAL.md — first unchecked task.
2) Read docs/VIBECODING.md — follow trust rules.
3) memory_recall before coding; memory_store progress; memory_recall before done.
4) One slice per session; pytest + ruff before claiming done.
5) Update docs/AGENT_GOAL.md when a task completes.
```

---

## Default session prompt (user)

```text
Continue toward the agent goal — pick the first unchecked M1 task.
```

---

## Memory workflow (MCP)

- **Start:** `memory_recall` with user goal + `include_knowledge=true`
- **Progress:** `memory_store` self-contained notes (problem + fix for solutions)
- **After edits:** `memory_detect_drift` once if anchors may have changed
- **End:** `memory_recall` to align with current claims

Valid `memory_store` content types: `exchange`, `fact`, `preference`, `solution`.  
Pass `tags` as a list, not a comma-separated string.

---

## Startup checklist

```text
[ ] docs/AGENT_GOAL.md — task selected
[ ] docs/VIBECODING.md — rules acknowledged
[ ] python scripts/agent_bootstrap_check.py (optional)
[ ] memory_recall
[ ] implement one slice + tests
[ ] ruff + targeted pytest
[ ] docs/AGENT_GOAL.md updated
[ ] memory_recall before response
```

---

## Full rule reference

See [VIBECODING.md](VIBECODING.md) — do not duplicate; link instead.