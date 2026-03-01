# Changelog

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
