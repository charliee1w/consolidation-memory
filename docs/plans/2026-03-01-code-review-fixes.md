# Code Review Fixes — consolidation-memory

Fix all issues from the comprehensive code review below. Work through them systematically, running tests after each logical group. The repo is at `D:/consolidation-memory`. Run `python -m pytest tests/ -x -q` to verify. All 275+ tests must pass after each group, plus any new tests you add.

## CRITICAL — Fix these first

### 1. Non-atomic two-file save in vector_store.py (lines 127-128)

`_save()` does two sequential `os.replace()` calls for the FAISS index and JSON id-map. A crash between them leaves inconsistent state, and the mismatch detector rebuilds an empty index — losing all vectors.

**Fix**: Write a generation counter (integer) to a small `.gen` file atomically after both files are written. On `_load_or_create`, check that all three files share the same generation. If they don't, fall back to the previous generation's files (keep one backup pair). Alternatively, write both files first to temp names, then rename id-map first (it's the source of truth — having extra IDs is safer than too few), then rename the index.

### 2. Half-life formula in context_assembler.py (line 63)

```python
# Current (wrong — gives 0.368 at half-life):
return math.exp(-age_days / half_life_days)

# Correct:
return math.exp(-age_days * math.log(2) / half_life_days)
```

One-line fix. Update any tests that assert on specific decay values.

### 3. Tag filter after SQL LIMIT in database.py (lines 891-905)

`search_episodes` fetches `LIMIT N` rows then filters by tags in Python, returning fewer results than requested.

**Fix**: Over-fetch by multiplying the limit (e.g., `fetch_limit = limit * 5`), apply tag filtering in Python, then slice to `limit`. This matches the pattern already used in `context_assembler.recall()` with `fetch_k`. Add a test that stores 20 episodes (10 with tag "a", 10 with tag "b"), requests `limit=10, tags=["a"]`, and verifies exactly 10 results are returned.

## HIGH — Bugs

### 4. Surprise boost is cumulative across runs (scoring.py lines 44-47)

Every consolidation run re-boosts high-access episodes. Score inflates to `SURPRISE_MAX` over cycles.

**Fix**: Make the boost idempotent. Calculate the target score for the current access level and only boost if current score is below that target. Or: track a `last_boost_access_count` on each episode and only apply the boost for the delta since last adjustment.

Simpler approach: change the boost to be absolute rather than additive — compute the target surprise from access count, and set `new_score = max(current_score, target)` instead of `new_score += boost`.

### 5. Five Config fields not loaded from TOML (config.py `_build_config()`)

These fields exist on the dataclass but have no TOML mapping:
- `FAISS_SIZE_WARNING_THRESHOLD` — add under `[faiss]` as `size_warning_threshold`
- `FAISS_COMPACTION_THRESHOLD` — add under `[faiss]` as `compaction_threshold`
- `KNOWLEDGE_MAX_VERSIONS` — add under `[consolidation]` as `knowledge_max_versions`
- `MAX_BACKUPS` — add under `[storage]` or `[general]` as `max_backups`
- `CONSOLIDATION_PRIORITY_WEIGHTS` — add under `[consolidation]` as `priority_weights` (dict)

Add the `.get()` calls in `_build_config()` alongside the existing ones. Add a test that passes TOML data with these fields and asserts they are loaded.

### 6. Truncated cluster episodes silently abandoned (engine.py lines 364-378)

When a cluster is truncated to `CONSOLIDATION_MAX_CLUSTER_SIZE`, dropped episodes never get `consolidation_attempts` incremented and are re-clustered every run.

**Fix**: After truncation, call `increment_consolidation_attempts()` on the dropped episode IDs. Add a log warning listing the dropped episode count.

### 7. No guard against LLM dropping records during merge (engine.py lines 296-301)

All existing records are soft-deleted and replaced with the LLM's output. If the LLM hallucinates, records are lost.

**Fix**: Add a guard after the merge: if `len(merged_records) < len(existing_db_records) * 0.5` and `len(existing_db_records) >= 4`, reject the merge and log a warning. Keep existing records intact. Add a test that simulates an LLM returning drastically fewer records and verifies the merge is rejected.

### 8. store_batch dedup misses within-batch duplicates (client.py lines 261-283)

Each item checks against the existing FAISS index, but earlier items from the same batch aren't indexed yet.

**Fix**: After the main FAISS dedup check, also compare each item's embedding against all previously accepted items in the current batch using cosine similarity. Use a simple numpy dot product against a growing list of accepted embeddings. Add a test that calls `store_batch([A, A])` with identical content and verifies only one is stored.

### 9. openai_backend generate() can return None (openai_backend.py line 106)

`message.content` is `str | None`. Returning `None` violates `LLMBackend` protocol.

**Fix**: Check for `None` and return `""` or raise `ValueError("LLM returned empty response")`. The latter is better since callers already handle exceptions.

### 10. cmd_import crashes on None tags (cli.py line 511)

Passes `None` to `json.loads()`.

**Fix**: Add `if ep.get("tags") is None: ep["tags"] = []` before the json.loads line. Add a test.

### 11. memory_compact and memory_consolidate missing from OpenAI schemas (schemas.py)

**Fix**: Add function schemas for both tools to `openai_tools` list. Add them to `dispatch_tool_call`. Add tests.

### 12. override_config doesn't recompute paths (config.py lines 489-499)

**Fix**: Call `cfg._recompute_paths()` at the end of `__enter__` and `__exit__`. Add a test that overrides `_base_data_dir` via `override_config` and asserts `DATA_DIR` etc. are updated.

### 13. Silent fallthrough when CONSOLIDATION_MEMORY_CONFIG points to nonexistent file (config.py lines 61-65)

**Fix**: If the env var is set but the file doesn't exist, raise `FileNotFoundError(f"Config file specified by CONSOLIDATION_MEMORY_CONFIG does not exist: {p}")`. Add a test.

## MEDIUM — Security & Robustness

### 14. Prompt injection: </episode> not sanitized (prompting.py lines 76-81)

**Fix**: Add `</episode>` and `<episode>` to the sanitize regex. Also add `<\|im_start\|>`, `<\|im_end\|>`, `\[INST\]`, `<<SYS>>`, `Human:`, `User:`, and `Assistant:` patterns. Sanitize `<` and `>` inside episode content by replacing with `＜` and `＞` (fullwidth) or HTML entities, since the surrounding XML tags are the structural boundary. Add tests for each pattern.

### 15. API keys visible in default __repr__ (config.py)

**Fix**: Add a `__repr__` method to Config that redacts `EMBEDDING_API_KEY` and `LLM_API_KEY` fields (show `"***"` instead of the value). Use `dataclasses.fields(self)` to iterate and selectively redact.

### 16. async tools calling blocking I/O (server.py lines 56-268)

**Fix**: Wrap each `_client.method()` call in `await asyncio.to_thread(...)`. This is mechanical — every tool function body that calls `_client.X()` becomes `result = await asyncio.to_thread(_client.X, args...)`. Do this for all tool functions.

### 17. Timed-out LLM futures never cancelled (prompting.py lines 122-130)

**Fix**: After catching `TimeoutError`, call `future.cancel()`. This won't interrupt in-flight requests but prevents queued ones from running.

```python
except (TimeoutError, concurrent.futures.TimeoutError):
    future.cancel()
    logger.warning(...)
```

### 18. Ollama missing nomic query/document prefixes (ollama.py lines 59-60)

**Fix**: Add the same prefix logic as LMStudioEmbeddingBackend. Check if the model name contains "nomic" and prepend `"search_query: "` or `"search_document: "` as appropriate. Add `encode_query` that uses the query prefix. Add a test.

## MEDIUM — Performance

### 19. Python-loop reconstruct() in vector_store.py (lines 177-179, 371-372)

**Fix**: Replace the Python loop with bulk extraction:
```python
vectors = faiss.rev_swig_ptr(self._index.get_xb(), n * dim).reshape(n, dim).copy()
```
This works for IndexFlatIP. For IVF indexes in compact(), use the reconstruct loop as fallback (IVF doesn't expose get_xb). Add a comment explaining the difference.

### 20. record_cache include_expired bypasses cache entirely (record_cache.py lines 47-58)

**Fix**: Maintain two cache slots — one for `include_expired=True`, one for `False`. The `True` slot is the superset. When the `False` slot is requested, filter expired records from the `True` cache if it exists and is fresh.

### 21. cmd_import embeds one-by-one (cli.py lines 521-525)

**Fix**: Collect all episode contents into a list, call `encode_documents(all_texts)` once, then pair each embedding with its episode for insertion. Batch size of 50 like `cmd_reindex`.

## LOW — Code Quality

### 22. Dead code removal

- Remove `_embed_single` from `ollama.py` (lines 30-39)
- Verify `_llm_with_validation` and `_validate_llm_output` in `prompting.py` (lines 531-559) have no callers, then remove
- Remove dead `set_active_project("default")` cleanup calls in `test_project_isolation.py`

### 23. Type improvements in types.py

- Change `HealthStatus.status` and similar `status: str` fields to `Literal["healthy", "degraded", "error"]` (and appropriate literals for each type)
- Change `StatusResult` fields from `X | dict[str, Any]` to `Optional[X]` with `default=None`
- Export `CompactResult`, `ContentType`, `RecordType` from `__init__.py`

### 24. Consistency fixes

- `lmstudio.py`: Switch embedding backend from `urllib` to `httpx` for consistency with the LLM backend
- Remove hardcoded `"stop": ["<|im_end|>"]` from `lmstudio.py` LLM backend or make it configurable
- Add `n_results` upper-bound clamping in `server.py` (`n_results = min(n_results, 50)`)
- Add content length validation on `memory_store` (reject content over 50KB)
- Move `_task_indicators` set in `context_assembler.py` to module-level constant

### 25. Validation improvements in config.py

Add these validations to `_validate_config()`:
- `CONSOLIDATION_INTERVAL_HOURS > 0`
- `CONSOLIDATION_MAX_DURATION > 0`
- `LLM_CALL_TIMEOUT > 0`
- `CONSOLIDATION_MIN_CLUSTER_SIZE >= 2`
- `CONSOLIDATION_MAX_CLUSTER_SIZE >= CONSOLIDATION_MIN_CLUSTER_SIZE`
- `SURPRISE_MIN < SURPRISE_MAX`
- `CIRCUIT_BREAKER_THRESHOLD >= 1`

## TEST SUITE IMPROVEMENTS

### 26. Add circuit breaker tests (HIGH PRIORITY)

Create `tests/test_circuit_breaker.py` with tests for:
- CLOSED -> OPEN after N failures
- OPEN -> HALF_OPEN after cooldown
- HALF_OPEN -> CLOSED on success
- HALF_OPEN -> OPEN on failure
- `check()` raising `ConnectionError` when OPEN
- `reset()` behavior
- Thread safety under concurrent failures

### 27. Add context_assembler unit tests

Create `tests/test_context_assembler.py` with tests for:
- `_recency_decay` calculation (verify half-life math after fix #2)
- Tag overlap boosting
- Procedure record 15% boost
- Knowledge topic ranking
- Edge cases: zero-norm vectors, empty knowledge base

### 28. Fix concurrency tests

In `test_concurrency.py`, after `t.join(timeout=N)`, add:
```python
assert not t.is_alive(), f"Thread {t.name} still alive (possible deadlock)"
```

### 29. CI improvements

In `.github/workflows/test.yml`:
- Add Python 3.12 to the matrix
- Add `pip install pytest-cov` and run with `--cov=consolidation_memory --cov-report=xml`
- Add pip caching via `actions/cache`

### 30. Reset caches in conftest

In the autouse `tmp_data_dir` fixture in `conftest.py`, add resets for `topic_cache` and `record_cache` module-level state alongside the existing `reset_backends()` call.

---

## Execution order

1. Critical fixes (#1-3) — run tests
2. High bugs (#4-13) — run tests
3. Security/robustness (#14-18) — run tests
4. Performance (#19-21) — run tests
5. Code quality (#22-25) — run tests
6. Test suite (#26-30) — run full suite

After all fixes, run `ruff check src/ tests/` and `python -m mypy src/` to verify no lint/type regressions. All existing + new tests must pass.
