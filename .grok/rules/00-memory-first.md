# Memory first (non-negotiable)

`consolidation_memory` MCP is configured for this project. **Your first tool call on
every user turn must be `memory_recall`.** No exceptions before recall completes.

## Startup sequence

1. `memory_recall` — short query from the user's goal; prefer `n_results=3`.
   Use `include_knowledge=true` when you need consolidated topics/claims; the MCP
   server returns episodes-only while record embeddings warm, then full knowledge
   on the next call (~sub-second when warm).
2. Then — shell, read, grep, edit, subagents, etc.

If the first recall is slow or times out, retry once with a shorter query. The
memory gate auto-unblocks after a recall attempt so the turn is not frozen.

## Ongoing workflow

- `memory_store` after meaningful progress (`content_type` + `tags` required).
- `memory_consolidate` when recall is noisy or contradictory.
- `memory_detect_drift` after substantial code edits.
- Final `memory_recall` before closing the turn.

## If tools are unavailable

Say so once, continue without inventing memory content.