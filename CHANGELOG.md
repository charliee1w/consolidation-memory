# Changelog

## 0.4.0 — 2026-02-28

### Bug Fixes

- **Fix `executescript()` breaking transaction atomicity** — `ensure_schema()` now uses individual `execute()` calls instead of `executescript()`, which implicitly commits before running
- **Fix upsert race condition in knowledge topics** — `upsert_knowledge_topic()` catches `IntegrityError` from concurrent inserts and falls back to update
- **Fix LIKE injection in keyword search** — `search_episodes()` now escapes `%`, `_`, and `\` in user input with proper `ESCAPE` clause
- **Fix store rollback leaving dedup-visible orphans** — `store()` now uses `hard_delete_episode()` instead of soft-delete when FAISS add fails, preventing dedup from finding rolled-back episodes
- **Fix `store_batch()` issuing per-item FAISS adds** — batch now collects embeddings and calls `add_batch()` once, with proper rollback on failure
- **Fix ThreadPoolExecutor timeout in `_call_llm()` and `_consolidation_loop()`** — no longer uses context manager (whose `__exit__` calls `shutdown(wait=True)`, blocking until thread finishes even after timeout)
- **Fix knowledge filename collision** — `_process_cluster()` now appends counter suffix when target filename already exists
- **Fix `_parse_tags()` crash on malformed JSON** — catches `ValueError` alongside `JSONDecodeError`
- **Fix recency decay returning >1.0 for future-dated episodes** — clamps `age_days` to non-negative
- **Fix variable shadowing of `tags` parameter in recall** — renamed local to `ep_parsed_tags`

### Security

- **Path traversal guard in `correct()`** — validates that resolved filepath stays within `KNOWLEDGE_DIR`
- **Path traversal guard in `export()`** — skips knowledge files whose resolved path escapes `KNOWLEDGE_DIR`
- **Path traversal guard in CLI `import`** — skips imported knowledge files with directory traversal in filename

### Robustness

- **Null-guard all MCP server tool functions** — returns error JSON instead of `AttributeError` if client not initialized
- **Null-guard all REST API endpoints** — returns 503 instead of crashing if client not initialized
- **Added `hard_delete_episode()` to database layer** — permanent delete for rollback scenarios

### Internal

- Removed redundant `import json` inside `search()` method

## 0.3.0 — 2026-02-28

### Bug Fixes

- **Fix stuck consolidation on backend failure** — episodes that hit `max_attempts` during an LLM outage now get their attempt counter reset after 24h, so they consolidate once the backend recovers
- **Validate LLM merge output before writing** — `_merge_into_existing()` now rejects empty or frontmatter-less LLM output instead of silently corrupting knowledge files
- **Fix dedup check missing duplicates** — dedup now checks top-3 FAISS results instead of top-1, preventing tombstone-filtered vectors from masking real duplicates
- **Move knowledge versioning after validation** — `_version_knowledge_file()` now runs only after merge output passes validation, eliminating noise in version history from failed merges

### Performance

- **Paginate surprise score adjustment** — `_adjust_surprise_scores()` now processes episodes in batches of 1000 and computes median access via SQL, reducing peak memory usage at 10K+ episodes
- **Cache backend health probe** — `_probe_backend()` caches results for 30s, eliminating redundant HTTP requests on repeated `status()` calls
- **Cap FAISS search over-fetch** — absolute limit of `max(k*3, 200)` on `fetch_k` prevents pathological full-index scans when filters request large candidate sets

### Internal

- Schema migration v4: index on `consolidation_attempts` column
- New DB functions: `reset_stale_consolidation_attempts()`, `get_median_access_count()`, `get_active_episodes_paginated()`

## 0.1.0 — 2026-02-24

Initial public release.

### Features

- **Episode storage** with SQLite persistence and FAISS vector indexing
- **Semantic recall** with cosine similarity, weighted by surprise score, recency, and access frequency
- **Automatic consolidation** — background thread clusters related episodes via agglomerative clustering, then synthesizes structured knowledge documents using a local LLM
- **4 embedding backends**: FastEmbed (zero-config), LM Studio, OpenAI, Ollama
- **3 LLM backends** + disabled mode: LM Studio, OpenAI, Ollama
- **MCP server** for Claude Desktop / Claude Code / Cursor integration
- **REST API** (FastAPI) for language-agnostic HTTP access
- **Python client** (`MemoryClient`) with context manager support
- **OpenAI function calling schemas** with dispatch for any OpenAI-compatible LLM
- **CLI**: `init`, `serve`, `status`, `consolidate`, `export`, `import`, `reindex`
- **TOML configuration** with platform-specific path defaults via `platformdirs`
- **Store-time deduplication** via FAISS cosine similarity threshold
- **Knowledge versioning** — backups before overwrites, configurable retention
- **Adaptive surprise scoring** — access-boosted, decay for inactive episodes
- **Atomic writes** for FAISS persistence (tempfile + os.replace)
- **LLM output validation** with structured checks and retry
- **Export/import** for backup and migration between installations
- 88 tests across 4 test files
