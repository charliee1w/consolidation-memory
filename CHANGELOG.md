# Changelog

## 0.12.0 — 2026-03-02

Diff-aware merge validation for consolidation trust.

### Features

- **Diff-aware merge validation** — after LLM merge, each pre-merge record is compared against merged output via cosine similarity. Records with no semantic match (max similarity below threshold) are flagged as "silently dropped" and logged to `contradiction_log` with `resolution='silent_drop'` for audit. Prevents silent merge drift, the #1 trust issue with consolidation
- **2 new config fields** — `merge_drop_detection_enabled` (default true), `merge_drop_similarity_threshold` (default 0.5) under `[consolidation]`

### Internal

- 407 tests (4 new for merge validation)

## 0.11.0 — 2026-03-02

Hybrid search and recall quality improvements.

### Features

- **Hybrid BM25 + semantic search** — FTS5 virtual table mirrors episode content; recall now runs both FAISS (cosine similarity) and FTS5 (BM25 keyword) searches, merges candidates, and computes a weighted hybrid score. Fixes recall failures on exact terms, acronyms, and proper nouns (e.g., "CORS bug in AuthService" now ranks correctly)
- **Schema migration v10** — `episodes_fts` FTS5 table with automatic backfill from existing episodes; gracefully degrades if FTS5 is unavailable
- **4 new config fields** — `hybrid_search_enabled`, `hybrid_semantic_weight` (0.7), `hybrid_keyword_weight` (0.3), `hybrid_fts_candidates` (50) under `[retrieval]`
- **Tag co-occurrence graph** — episodes with co-occurring tags get a 10% recall boost, clustering results around intent motifs
- **Contradiction audit log** — schema v8 `contradiction_log` table tracks when knowledge records contradict each other during consolidation
- **Plugin system** — entry-point based plugin wiring with `PLUGINS_ENABLED` config

### Bug Fixes

- **mypy type error** in co-occurrence boost resolved
- **Phase 1 quality fixes** — plugin wiring, API parity, miscellaneous bug fixes

### Documentation

- Public roadmap added to README
- REST endpoint list updated

### Internal

- 403 tests (24 new for hybrid search)

## 0.10.0 — 2026-02-28

Knowledge introspection, forgetting transparency, and adoption tooling.

### Features

- **LoCoMo benchmark harness** — `benchmarks/` package for comparable evaluation against Mem0/Zep/OpenAI Memory using the LoCoMo-10 dataset (token F1, BLEU-1, LLM judge scoring across 5 question categories)
- **Knowledge browser** — `memory_browse` and `memory_read_topic` MCP tools + `browse` CLI command for inspecting consolidated knowledge topics with record counts, confidence scores, and full markdown content
- **Temporal timeline** — `memory_timeline` MCP tool showing how understanding of a topic evolved over time, with supersession detection via embedding similarity
- **Decay transparency** — `memory_decay_report` MCP tool previews what consolidation would prune without deleting anything
- **Episode protection** — `memory_protect` MCP tool marks episodes or entire tags as immune to pruning; schema v7 adds `protected` column
- **Recent activity** — `memory_status` now includes `recent_activity` field with last 5 consolidation run summaries
- **Configurable decay policies** — `[decay_policies]` in config.toml for tag-based retention overrides (e.g., keep architecture decisions for a year, forget debugging sessions after a week)
- **`setup-claude` CLI command** — appends recommended memory instructions to `~/.claude/CLAUDE.md` with confirmation prompt so Claude Code proactively uses memory tools

### Documentation

- Cross-client memory section in README explaining shared memory across Claude Code, Cursor, Windsurf, and any MCP client
- Example MCP configs in `docs/examples/` for Claude Code, Cursor, VS Code + Continue, and generic clients
- Recommended CLAUDE.md snippet in `docs/recommended-claude-md.md`
- Updated README tools list and CLI table

## 0.9.0 — 2026-02-28

Comprehensive code review: 30 fixes across correctness, security, performance, and code quality.

### Critical Fixes

- **Non-atomic two-file save** — reversed rename order in vector_store so id-map (source of truth) is written first; added graceful recovery on mismatch
- **Half-life formula** — `_recency_decay` now uses correct `exp(-age * ln2 / half_life)` instead of `exp(-age / half_life)`
- **Tag filter after SQL LIMIT** — over-fetch 5x when tags specified so post-filter doesn't silently truncate results

### Bug Fixes

- **Surprise boost cumulative** — switched from additive to absolute-max approach so repeated access doesn't inflate indefinitely
- **5 config fields not loaded from TOML** — `FAISS_SIZE_WARNING_THRESHOLD`, `FAISS_COMPACTION_THRESHOLD`, `CONSOLIDATION_PRIORITY_WEIGHTS`, `KNOWLEDGE_MAX_VERSIONS`, `MAX_BACKUPS`
- **Truncated cluster episodes silently abandoned** — dropped episodes now get consolidation_attempts incremented
- **No guard against LLM dropping records during merge** — reject merge if merged records < 50% of existing (when >= 4 exist)
- **store_batch intra-batch duplicates** — compare new embeddings against already-accepted batch entries via dot product
- **openai_backend generate() None** — raise ValueError instead of returning None
- **cmd_import crashes on None tags** — handle `None` tags gracefully
- **memory_compact/consolidate missing from OpenAI schemas** — added schemas and dispatch handlers
- **override_config doesn't recompute paths** — call `_recompute_paths()` on enter and exit
- **Silent fallthrough for missing config file** — raise FileNotFoundError when env var points to nonexistent file

### Security & Robustness

- **Prompt injection sanitization** — extended patterns with `<episode>`, `<|im_start|>`, `[INST]`, `<<SYS>>`, `Human:`, `Assistant:`, plus fullwidth char replacement
- **API keys visible in __repr__** — custom `__repr__` redacts `EMBEDDING_API_KEY` and `LLM_API_KEY`
- **Async tools calling blocking I/O** — all MCP tool functions now use `asyncio.to_thread()`
- **Timed-out LLM futures never cancelled** — added `future.cancel()` after TimeoutError
- **Ollama nomic query/document prefixes** — nomic models get correct `search_document:`/`search_query:` prefixes

### Performance

- **FAISS bulk vector extraction** — replaced Python-loop `reconstruct()` with `faiss.rev_swig_ptr()` for IndexFlatIP
- **record_cache dual slots** — two cache slots (all/unexpired) so `include_expired=True` doesn't bypass cache
- **cmd_import batch embed** — embed in batches of 50 instead of one-by-one

### Code Quality

- Remove dead code: `_embed_single` (ollama), `_validate_llm_output`/`_llm_with_validation` (prompting)
- Literal types for all status fields, Optional for StatusResult fields
- Export `CompactResult`, `ContentType`, `RecordType` from package
- LM Studio embedding backend switched from urllib to httpx
- Remove hardcoded `<|im_end|>` stop token from LM Studio LLM
- Add `n_results` clamping (max 50) and content length validation (max 50KB) in MCP server
- Move `_task_indicators` to module-level frozenset
- 7 new config validations (interval, duration, timeout, cluster sizes, surprise range, circuit breaker)

### Tests

- New `test_circuit_breaker.py` — 14 tests covering all state transitions and thread safety
- New `test_context_assembler.py` — 11 tests for recency decay, priority scoring, task indicators
- Thread alive assertions in all concurrency tests
- Reset topic_cache and record_cache in conftest autouse fixture
- 320 tests (up from 299)

### CI

- Add Python 3.12 to test matrix
- Add pytest-cov with XML coverage report
- Add pip caching via actions/cache

## 0.8.3 — 2026-02-28

### Bug Fixes

- **valid_from marking** — contradiction metadata was silently skipped due to checking set truthiness instead of contradiction count
- **ThreadPoolExecutor leak** — LLM timeout retries created a new executor each attempt, leaking zombie threads; now uses a shared module-level pool
- **correct() quality check bypass** — `correct()` overwrote knowledge files even when the LLM output failed frontmatter validation; now returns an error
- **_slugify empty for non-ASCII** — pure CJK/emoji topic names produced an empty slug, causing file errors; falls back to hash-based slug
- **compact() IVF downgrade** — compaction rebuilt the FAISS index as flat even when it was previously IVF; now re-upgrades after rebuild

### Refactoring

- Extract shared `normalize_l2` to `backends/base.py` (was duplicated in 4 backends)
- LM Studio backend uses shared `retry_with_backoff` instead of hand-rolled retry
- Extract `_check_specifics_preservation` helper (was duplicated in 2 validators)
- Narrow exception catching in `encode_documents`/`encode_query` to transient errors only
- Remove dead `_build_distillation_prompt` function
- Clean up `consolidation/__init__.py` to only export `run_consolidation`

### Docs

- Add `test` and `dashboard` CLI commands to README
- Fix config path in ARCHITECTURE.md (`consolidation-memory` → `consolidation_memory`)
- Add default value comments to TOML config example
- Add Changelog URL to pyproject.toml

### Internal

- 299 tests (up from 292)

## 0.8.2 — 2026-02-28

### Features

- **Environment variable config overrides** — every scalar Config field can now be set via `CONSOLIDATION_MEMORY_<FIELD_NAME>` env vars, enabling Docker and CI configuration without a TOML file; priority: defaults < TOML < env vars < `reset_config()`

### Internal

- 292 tests (up from 281)

## 0.8.1 — 2026-02-28

### Features

- **`consolidation-memory test` CLI command** — end-to-end smoke test verifying store, embed, recall, forget, and LLM connectivity after install; prints colored pass/fail summary and always cleans up test data

### Internal

- 281 tests (up from 275)

## 0.8.0 — 2026-02-28

### Features

- **Procedure record type** — fourth knowledge record type capturing learned workflow patterns (trigger, steps, context); 1.15x relevance boost for task-oriented recall queries
- **Temporal fact tracking** — knowledge records now carry `valid_from` / `valid_until` fields; recall can filter expired records via `include_expired` parameter
- **Contradiction detection** — during consolidation, new records are compared against existing ones (semantic similarity >= 0.7); optional LLM verification marks contradicting records as expired and replaces them
- **FAISS IVF auto-migration** — when the index exceeds 10,000 vectors, it is automatically rebuilt as `IndexIVFFlat` with `nlist=sqrt(n)` and `nprobe=nlist/4` for faster approximate search; configurable via `faiss.ivf_upgrade_threshold`

### Refactoring

- **Config dataclass singleton** — replaced ~60 module-level constants with a `@dataclass Config` accessed via `get_config()`; test fixtures use `reset_config()` instead of 40+ `mock.patch` calls
- **Consolidation package split** — split 1,400-line `consolidation.py` monolith into `consolidation/` package: `clustering.py`, `prompting.py`, `scoring.py`, `engine.py`

### Tooling

- **mypy type checking** — added `py.typed` marker and fixed all type errors; mypy overrides for `tomli`, `fastapi`, `uvicorn`

### Documentation

- **Architecture overview** — `docs/ARCHITECTURE.md` with Mermaid diagrams covering data flow, threading model, storage layout, consolidation internals, retrieval pipeline, and security

### Internal

- 275 tests (up from 237)

## 0.7.0 — 2026-02-28

### Features

- **TUI dashboard** — `consolidation-memory dashboard` launches an interactive terminal UI (powered by Textual) with 4 tabs:
  - **Episodes browser** — sortable table showing content preview, type, tags, surprise score, creation time, and consolidation status
  - **Knowledge topics** — topic list with record detail panel; select a topic to view its extracted facts, solutions, and preferences
  - **Consolidation history** — table of all consolidation runs with timestamps, episodes processed, clusters formed, topics created/updated, and status
  - **Memory stats** — live-refreshing display of episode counts by type, FAISS index size, tombstone count, DB size, and last consolidation time
- **Keybindings**: `q` quit, `r` refresh, `1-4` switch tabs
- **Lightweight data layer** — dashboard queries SQLite directly without initializing FAISS or embedding backends, keeping startup instant

### Dependencies

- New optional extra: `pip install consolidation-memory[dashboard]` (adds `textual>=1.0.0`)
- `all` extra now includes `dashboard`

### Internal

- New `DashboardData` class in `dashboard_data.py` — read-only SQLite queries with content truncation, tag parsing, FAISS sidecar file reading
- 23 new tests (237 total) covering all data-fetching methods

## 0.6.0 — 2026-02-28

### Features

- **Multi-project namespace support** — isolate SQLite DB, FAISS index, and knowledge files per project via `--project` CLI flag or `CONSOLIDATION_MEMORY_PROJECT` environment variable
- **Auto-migration** — existing flat-layout data directories are automatically migrated to `projects/default/` on first run
- **Project-aware CLI** — `consolidation-memory --project work status`, `consolidation-memory --project personal serve`
- **Project logging** — MCP server and REST API log the active project at startup; REST `/health` endpoint includes project name

### Internal

- Consumer modules (`database`, `vector_store`, `consolidation`, `context_assembler`) refactored to access config path constants dynamically, enabling runtime project switching
- `validate_project_name()` enforces lowercase alphanumeric, hyphens, underscores (1-64 chars)
- `set_active_project()` / `get_active_project()` API for programmatic project switching
- `maybe_migrate_to_projects()` handles flat-to-project directory migration with rollback on failure
- 40 new tests (214 total), including project isolation integration tests
- Simplified `conftest.py` — removed 8 redundant consumer-module path patches

## 0.5.0 — 2026-02-28

### Features

- **Schema-guided knowledge extraction** — consolidation now outputs structured JSON records instead of free-form markdown, making individual facts, solutions, and preferences independently searchable
- **New `knowledge_records` table** (schema v5) — each knowledge record is stored as a typed row with its own embedding text, linked to a parent `knowledge_topics` entry
- **Three record types**: `fact` (subject + info), `solution` (problem + fix + context), `preference` (key + value + context)
- **Record-level recall** — `memory_recall` now returns a `records` field with individually ranked knowledge records alongside episodes and knowledge documents
- **Record embedding cache** — new `record_cache` module (same thread-safe pattern as `topic_cache`) caches record embeddings for fast numpy matmul search during recall
- **Markdown rendering from records** — optional (enabled by default via `render_markdown` config), generates human-readable .md files from structured records
- **Record merge on consolidation** — when merging into existing topics, old records are soft-deleted and replaced with LLM-merged records
- **JSON extraction validation** — validates LLM output as valid JSON with required fields, record type validation, and specifics preservation checks
- **Export/import includes records** — export format bumped to v1.1, includes `knowledge_records` array

### Configuration

- `records_semantic_weight` (default 0.9) — semantic similarity weight for record search
- `records_keyword_weight` (default 0.1) — keyword match weight for record search
- `records_relevance_threshold` (default 0.3) — minimum relevance score for record results
- `records_max_results` (default 15) — maximum records returned per recall
- `render_markdown` (default true) — whether to render .md files from records

### Internal

- Schema migration v5: `knowledge_records` table with indexes on `topic_id`, `record_type`, `deleted`
- `get_stats()` now includes `total_records` and `records_by_type` breakdown
- CLI `status` shows record counts by type
- 24 new tests (174 total, 7 skipped)

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
