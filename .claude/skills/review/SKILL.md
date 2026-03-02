---
name: review
description: Deep code review of recent changes or a specific area
argument-hint: "[area|file|commit-range] or empty for uncommitted changes"
allowed-tools: Bash, Read, Grep, Glob
---

# Code Review for consolidation-memory

Review `$ARGUMENTS` (or uncommitted changes if empty).

## Scope

If `$ARGUMENTS` is empty, review all uncommitted changes:
```
git diff
git diff --cached
```

If `$ARGUMENTS` is a file or glob pattern, review those files.

If `$ARGUMENTS` looks like a commit range (e.g., `v0.9.0..HEAD`), review that range:
```
git diff <range>
```

## Review Checklist

For each changed file, evaluate:

### Correctness
- Logic errors, off-by-one, race conditions
- Edge cases: empty inputs, None values, large inputs
- Error handling: correct exceptions, no swallowed errors
- Thread safety: shared state protected by locks?

### Architecture
- Does this fit the existing patterns? (Config singleton, atomic writes, structured records)
- Are new dependencies justified?
- Is the consolidation engine's threading model respected?

### Security
- Prompt injection vectors in LLM-facing strings
- Path traversal in file operations
- SQL injection (should use parameterized queries)
- Input validation and size limits

### Performance
- FAISS operations: batch vs single-item?
- SQLite: indexes used? N+1 queries?
- Embedding calls: batched?
- Memory: unbounded collections?

### Testing
- Are new code paths covered?
- Do existing tests still make sense?
- Are mocks patching the right module paths? (common issue after refactors)

### Code Quality
- Type hints present and correct
- ruff-clean (line length 100, Python 3.10+ target)
- No dead code or commented-out blocks
- Docstrings where behavior is non-obvious

## Output Format

Organize findings by severity:

**CRITICAL** — bugs, data loss risks, security issues
**HIGH** — incorrect behavior, race conditions, missing error handling
**MEDIUM** — performance issues, missing tests, architectural concerns
**LOW** — style, naming, minor improvements
**GOOD** — patterns worth noting as positive examples

Include file paths and line numbers. Quote the problematic code.
