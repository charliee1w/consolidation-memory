# Fast-path consolidation episode shapes

Fast-path consolidation extracts structured knowledge **without calling the LLM**. Use these episode shapes when you want deterministic consolidation (for example with `llm.backend = "disabled"`).

## How it works

1. Unconsolidated episodes are clustered (embedding similarity + scope isolation).
2. For each cluster, `try_fast_path_extraction()` runs **before** LLM extraction.
3. **Every episode in the cluster** must parse successfully. If any episode is
   ambiguous, the cluster falls back to the LLM (or fails when the LLM backend
   is disabled).
4. Successful fast-path merges use deterministic merge logic only — no LLM merge prompts.

Inspect results via `memory_status` / `MemoryClient.status()`:

- `fast_path_hits` / `llm_fallbacks` on the last consolidation run
- `consolidation_quality.fast_path_rate` across recent runs

## Configuration

| Setting | Default | Purpose |
| --- | --- | --- |
| `CONSOLIDATION_FAST_PATH_ENABLED` | `true` | Master switch for deterministic extraction |
| `CONSOLIDATION_MIN_CLUSTER_SIZE` | `2` | Set to `1` to consolidate singleton structured episodes in tests or small corpora |
| `llm.backend` | `lmstudio` | Set to `disabled` for LLM-free consolidation of eligible episodes only |

Every scalar `Config` field is also settable as
`CONSOLIDATION_MEMORY_<FIELD_NAME>`, so the first two are `llm.backend =
"disabled"`, `consolidation.min_cluster_size = 1` in TOML or
`CONSOLIDATION_MEMORY_LLM_BACKEND=disabled`,
`CONSOLIDATION_MEMORY_CONSOLIDATION_MIN_CLUSTER_SIZE=1` in the environment.

TOML example:

```toml
[llm]
backend = "disabled"

[consolidation]
min_cluster_size = 1
```

## Parser priority (per episode)

1. **Structured JSON** — content starts with `{` and validates as a knowledge record (any `content_type`).
2. **`preference`** content type — preference text patterns (below).
3. **`procedure`** content type — procedure text patterns (below).
4. **`solution` or `fact`** content type — path-anchored solution parser (requires a file path in content).

## Structured JSON (all record types)

Content must start with `{` and parse as an object. Required and optional keys:

| `type` | Required | Optional |
| --- | --- | --- |
| `fact` | `subject`, `info` | — |
| `solution` | `problem`, `fix` | `context` |
| `preference` | `key`, `value` | `context` |
| `procedure` | `trigger`, `steps` | `context` |
| `strategy` | `problem_pattern`, `strategy` | `preconditions`, `expected_signals`, `failure_modes`, `context` |

Non-string values are coerced to text; a list becomes its items joined with
`" | "`. Missing required fields or an unknown `type` does **not** fast-path —
the cluster falls back to the LLM.

### Fact

```json
{"type": "fact", "subject": "Auth service", "info": "Uses JWT bearer tokens"}
```

```python
mem.store(
    '{"type": "fact", "subject": "Auth service", "info": "Uses JWT bearer tokens"}',
    content_type="fact",
)
```

### Solution

```json
{
  "type": "solution",
  "problem": "JWT tests fail without secret",
  "fix": "Set AUTH_JWT_SECRET in .env",
  "context": "tests/test_auth.py"
}
```

### Preference

```json
{"type": "preference", "key": "reviews", "value": "Short PR summaries with file paths"}
```

### Procedure

`steps` may be a string or a JSON array (arrays are joined with ` | `).

```json
{
  "type": "procedure",
  "trigger": "before merging a PR",
  "steps": ["run ruff", "run targeted pytest", "update changelog"],
  "context": "release workflow"
}
```

### Strategy

`strategy` is a consolidation-only record type: store the JSON with any ingest
`content_type` (see [CONTRIBUTING.md](../CONTRIBUTING.md#episode-content_type-vs-record-type)).

```json
{
  "type": "strategy",
  "problem_pattern": "flaky CI on Windows path separators",
  "strategy": "Normalize paths before diffing artifacts",
  "preconditions": "CI logs show mixed slash styles"
}
```

## Preference text (`content_type="preference"`)

**User prefers …**

```text
User prefers short PR summaries with concrete file paths.
```

The key is the `for …` clause when present; otherwise it comes from the episode
tags (the single tag, or the first three joined with `, `), falling back to
`general` when there are no tags.

**With explicit key**

```text
User prefers concise commit messages for workflow.
```

**Key/value form**

```text
preference: reviews = short PR summaries with concrete file paths
```

## Procedure text (`content_type="procedure"`)

**Trigger / Steps lines**

```text
Trigger: CI fails on pull request
Steps: inspect logs, reproduce locally, add regression test
```

Episode tags (up to three) are copied into `context` when present.

**Single-line form**

```text
procedure: trigger=before cutting a release, steps=run release gates, bump version, publish changelog
```

## Recommended solution store shape for agents

When storing `content_type="solution"`, use a problem-first shape so recall queries
(problem symptoms) match the embedded vector. Full narrative is still stored; only
the FAISS embedding is problem-skewed.

**Preferred text template**

```text
Problem: <symptom or error>
Fix: <specific steps, commands, or config changes>
Context: path:src/consolidation_memory/client.py
```

**Structured JSON alternative**

```json
{"type": "solution", "problem": "MCP recall times out", "fix": "Enable embedding disk cache", "context": "src/consolidation_memory/embedding_disk_cache.py"}
```

Episodes without `Problem:`, JSON `problem`+`fix`, or a path anchor still store
successfully but return `shape_warnings` advising better structure.

Re-embed existing episodes after upgrading: `consolidation-memory reindex`.

## Path-anchored solution (`content_type="solution"` or `"fact"`)

The solution parser requires at least one **file path** in the episode content
(detected by the anchor extractor). Without a path, the episode is not
fast-path eligible via this parser (use structured JSON instead). It recognizes
`problem:` / `issue:` / `error:` and `fix:` / `solution:` / `resolution:` line
labels, in any case.

**Problem + Fix lines**

```text
Problem: tests fail in tests/test_auth.py when JWT secret is missing
Fix: set AUTH_JWT_SECRET in .env and run pytest tests/test_auth.py
```

**Inline fix**

```text
Tests fail in tests/test_auth.py when JWT secret is missing. Fix: set AUTH_JWT_SECRET in .env.
```

**Implicit problem/fix**

```text
Tests fail in tests/test_auth.py when JWT secret is missing.
Set AUTH_JWT_SECRET in .env and run pytest tests/test_auth.py
```

Resolution order: an explicit `problem:` line, else the inline-fix prefix, else
the first sentence. The fix is an explicit `fix:` line, else the inline
sentence after `Fix:`, else the first line containing a detected path, else the
whole content. Detected paths (up to five) are stored in `context`.

## What does not fast-path

- Freeform `exchange` episodes with no structure
- Facts/solutions **without** JSON structure and **without** path anchors
- Partial JSON (`{"type": "fact", "subject": "only"}` missing `info`)
- Mixed clusters where one episode parses and another does not

## LLM-off workflow

1. Set `llm.backend = "disabled"`.
2. Store episodes using the shapes above (set `content_type` appropriately).
3. Run `consolidate` / `memory_consolidate` / `client.consolidate()`.
4. Confirm `fast_path_hits > 0`, `api_calls == 0`, and new rows in `knowledge_records`.

Integration coverage: `tests/test_integration.py::TestLlmDisabledStructuredConsolidation`.

## Related docs

- [ARCHITECTURE.md](ARCHITECTURE.md) — consolidation and scheduler overview
- [CONTRIBUTING.md](../CONTRIBUTING.md) — trust invariants and verification for consolidation changes
- [MODEL_SUPPORT.md](MODEL_SUPPORT.md) — embedding and LLM backend matrix