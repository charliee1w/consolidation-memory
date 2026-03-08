# Recommended Agent Instructions

Use this block in agent instruction files (for example `AGENTS.md`, `.github/copilot-instructions.md`, or `.cursor/rules/*.md`).

## Memory Workflow

- At the start of each new task, call `memory_recall` using the user goal.
- After meaningful progress, call `memory_store` with concise, self-contained notes.
- Store both problem and fix for solution memories.
- Tag memories with useful scope (feature, file path, issue id) when available.
- After substantial file edits, call `memory_detect_drift`.
- Before final response, call `memory_recall` again to verify current claim state.

## Storage Guidance

- Do store durable facts, solutions, and preferences.
- Do not store trivial chatter.
- Prefer precise and auditable wording over broad summaries.

## Optional Session Reminder

```text
Startup checklist:
1) memory_recall
2) implement with scope-aware trust semantics
3) memory_store milestones
4) memory_detect_drift after major edits
5) memory_recall before final response
```
