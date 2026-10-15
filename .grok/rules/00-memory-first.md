# Memory first (non-negotiable)

`consolidation_memory` MCP is configured for this project. **Your first tool call on
every user turn must be `memory_recall`.** No exceptions before recall completes.

## Startup sequence

1. `memory_recall` — short query from the user's goal; prefer `n_results=3`.
   Use `include_knowledge=true` when you need consolidated topics/claims. While
   record embeddings are still warming the server returns episodes only plus a
   `warnings` entry, and retries once itself within a ~3s budget
   (`CONSOLIDATION_MEMORY_DEFERRED_KNOWLEDGE_RETRY_SECONDS`); if the warning
   survives, call `memory_recall` again.
2. Then — shell, read, grep, edit, subagents, etc.

If the first recall is slow or times out, retry once with a shorter query. If the
host blocks the turn on a memory gate, the gate releases as soon as one recall
attempt returns, so a retry is never a deadlock.

## Ongoing workflow

- `memory_store` after meaningful progress. Only `content` is required, but pass
  `content_type` and `tags` — untagged, uncategorized rows retrieve worse.
- `memory_consolidate` when recall is noisy or contradictory.
- `memory_detect_drift` after substantial code edits.
- `memory_hygiene_scan` when the corpus feels noisy. `memory_hygiene_apply`
  mutates, is not gated by `write_mode`, and reports `status` `dry_run` or
  `applied` — so treat consolidated knowledge as potentially stale versus
  current code.
- Final `memory_recall` before closing the turn.

## If tools are unavailable

Say so once, continue without inventing memory content.