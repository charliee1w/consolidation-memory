# Changelog

## 0.14.0 - 2026-03-19

### Highlights

- Add outcome tracking model
- Add strategy memory support
- Fix timeout root cause
- Fix timeout root causes
- Fix mypy outcome_type errors
- Fix mypy overload errors
- Add reliability scoring for claims
- Add reliability scoring
- feat(claim-search): add trust-aware ranking for claim retrieval
- fix(mcp): disable auto consolidation by default for server runtime
- fix(claim-search): filter low-signal semantic-only temporal noise
- fix(types): satisfy mypy optional claim vector assignment


## 0.13.7 - 2026-03-13

### Highlights

- Fix memory client hardening issues
- Harden multi-agent pipeline
- Harden multi-agent codebase
- Remove unused scope_row assignment
- Fix fastembed cache handling
- Harden multi-agent system
- Fix type errors in tool dispatch
- Harden multi-agent consolidation
- Fix memory server scope validation
- Add provenance episode insert
- Fix checkout auth failure
- Enable automated memory workflow


## 0.13.6 - 2026-03-10

### Highlights

- Start consolidation MCP server
- Add auto MCP server start
- Determine MCP timeout root cause
- Resolve context assembler mypy error
- Clean up MCP processes
- Assess codebase and plan fixes
- Fix consolidation workflow issues
- Fix memory drift detection
- Adjust consolidation force limits
- Implement persisted policy ACLs
- Implement policy ACL entities
- Fix B608 warnings in database


## 0.13.5 - 2026-03-08

### Highlights

- Fix MCP startup reliability with lazy MemoryClient init
- Redo documentation for longevity
- Update section title from 'What Ships Today' to 'What It Is'
- Address MemoryClient risks
- Review MemoryClient docs
- Update memory architecture docs
- Fix CP1252 unicode output
- Fix CP1252 Unicode output crash
- Fix mypy errors in vector_store
- Summarize recent test output
- Handle consolidation timeout
- Adjust MemoryClient init timeout


## 0.13.2 - 2026-03-08

Public-readiness stabilization release with a full-system snapshot for external users.

### Highlights

- Completed release-readiness validation on current tree:
  - `pytest -q`: 681 passed
  - `ruff check .`: passed
  - `mypy src/consolidation_memory/`: passed (non-failing annotation notes only)
  - `python scripts/smoke_builder_base.py`: passed
  - `python scripts/verify_release_gates.py --novelty-result ... --scope-use-case "Drift-aware debugging memory"`: passed all gates
- Fixed Windows CP1252 console compatibility in `consolidation-memory test` by falling back to ASCII status markers when Unicode symbols are not encodable.
- Added regression coverage for CP1252 stdout behavior in CLI tests.
- Resolved Windows typing friction in vector-store file locking by using typed `getattr` bindings for `msvcrt` lock members.
- Eliminated local ACL warning noise in test/lint workflows by pinning pytest cache/temp paths to repo-local directories and ignoring locked novelty runtime folders.
- Fixed dashboard episode ordering nondeterminism when `created_at` values tie by adding deterministic SQL tie-break ordering.
- Added regression coverage for tied `created_at` ordering in `tests/test_dashboard_data.py`.
- Fixed dashboard consolidation-run ordering nondeterminism when `started_at` values tie by adding deterministic SQL tie-break ordering.
- Added regression coverage for tied `started_at` ordering in `tests/test_dashboard_data.py`.
- Expanded CI test matrix to include Python 3.11 so advertised classifier support is exercised.
- Added REST dependency installation on the Ubuntu Python 3.13 CI lane so FastAPI REST tests run in CI rather than being skipped everywhere.

### Full System Snapshot (OG -> Current)

- **Core memory lifecycle**
  - Episode store/recall/forget with SQLite + FAISS and adaptive relevance scoring (semantic + recency + access/surprise signals).
  - Store-time deduplication and durable export/import pipelines.
  - Multi-project partitioning plus scoped shared-memory foundations (schema v13 scope columns and scope-aware operations).
- **Trust and temporal reasoning**
  - Structured knowledge records (fact/solution/preference/procedure) with markdown topic rendering.
  - Temporal validity windows (`valid_from`/`valid_until`) and `as_of` recall for prior-belief reconstruction.
  - Contradiction detection/audit, provenance links, claim graph, and drift-aware claim challenge flows.
- **Retrieval quality and scale**
  - Hybrid semantic + BM25 recall pipeline with tag co-occurrence boost and recall dedup controls.
  - FAISS tombstones + compaction and auto-upgrade path to IVF for larger corpora.
  - Canonical query service and shared query envelopes across transports.
- **Interfaces and integration**
  - Python client, MCP tools, REST API, CLI workflows, and OpenAI-function schemas with parity-focused validation.
  - Optional dashboard and benchmark harnesses for operational visibility and comparative quality measurement.
- **Release and operations maturity**
  - Fail-closed release gates with novelty evidence checks, recency requirements, and scope-alignment enforcement.
  - Nightly novelty workflows, builder smoke baseline, and expanded regression coverage across core, API, and trust surfaces.

## 0.13.1 - 2026-03-08

Canonical first public release for external builders.

### Highlights

- Finalized and published the first public baseline for `0.13.x`.
- Added robust release automation with automatic semver bump options (`--bump patch|minor|major`), changelog scaffolding, and rollback safety.
- Added release preflight checks for existing tag collisions and existing PyPI versions.
- Hardened publish workflow validation by enforcing tag/version parity against `pyproject.toml`.
- Added a structured PyPI release existence check in CI to skip duplicate publish attempts safely.
- Added tests for release automation helpers (`tests/test_release_script.py`).
- Included structured GitHub discussion category templates and README community links for external contributor onboarding.

## 0.13.0 - 2026-03-07

Pre-1.0 foundation release focused on trust-preserving queries, scoped shared-memory foundations, and fail-closed release readiness for external builders.

### Highlights

- Introduced canonical scope envelope domain model (`NamespaceScope`, `AppClientScope`, `AgentScope`, `SessionScope`, `ProjectRepoScope`) and coercion helpers for typed scope-aware operations.
- Added schema v13 scope columns plus indexes across episodes and knowledge tables, enabling namespace/project/app/agent/session partitioning without separate deployments.
- Added canonical `CanonicalQueryService` and query envelopes to unify recall/search/claim browse/search semantics across Python client, CLI, MCP server, and REST surfaces.
- Added trust-preserving claim scope filtering based on claim provenance source rows to prevent cross-scope claim leakage.
- Standardized on the vendor-neutral instruction bootstrap command `setup-memory --path ...`.
- Added universal project guidance and config examples to run one shared memory project across clients.
- Added fail-closed release gate enforcement with novelty evidence validation in release automation and publish workflows.
- Added a builder baseline smoke test and `ResourceWarning` gating to lock a minimum stable foundation for contributors.

### Query Core And Scope Model

- Added typed scope envelopes and mapping coercion (`coerce_scope_envelope`) with literal-constrained sharing modes, app types, and session kinds.
- Threaded scope metadata through store/recall/search/claim operations across client, server, REST, and schema layers.
- Added shared query semantics helpers for payload parsing and scope matching.
- Added claim-source scope lookup and query-time filtering for trust-preserving claim retrieval.
- Added and expanded regression coverage for project isolation and scope behavior.

### Release And Quality Gates

- Added `src/consolidation_memory/release_gates.py` to enforce scope alignment, metric thresholds, evidence completeness, and evidence recency.
- Added `scripts/verify_release_gates.py` to validate novelty artifacts and emit machine-readable gate reports.
- Updated release automation and publish workflow to run a full novelty evaluation and block publish on failed, missing, or stale evidence.
- Added nightly full-novelty workflow artifact publishing for auditability.
- Added and updated release-gate and novelty-eval documentation.

### Builder Experience

- Added deterministic `scripts/smoke_builder_base.py` end-to-end smoke coverage for Python API, tool dispatch, status, recall/search, and export on a fresh project.
- Added builder baseline docs, external review playbook, minimal plugin example, and a structured issue template for outside reviewer feedback.
- Updated README development guidance with builder smoke and `ResourceWarning` commands.
- Updated MCP config examples to recommend explicit shared `--project` usage for cross-client memory continuity.

### Hardening And Fixes

- Resolved outstanding mypy errors in scope literal coercion and database row typing.
- Hardened database connection lifecycle for teardown paths and added explicit close helpers used in tests and smoke flows.
- Expanded test coverage for contradictions, server schemas, temporal records, query service behavior, project isolation, and vector-store consistency.
- Added targeted fixes and regression tests across backend, consolidation, and API surfaces.

### Docs And Strategy

- Added architecture and execution docs covering universal memory design, gap analysis, schema migration, shared scopes, and query semantics.
- Added an execution log and reusable prompt set for architecture and review cycles.

### Compatibility

- Pre-1.0 SemVer: this is a minor feature release (`0.13.0`).
- Existing external APIs remain available; scope/query enhancements are additive.
- Instruction setup now uses `setup-memory --path <instruction-file>` only.

## 0.12.4 - 2026-03-06

Novelty release hardening and claim-graph drift support.

### Highlights

- Added code drift detection (`detect-drift`) across Python client, CLI, and REST with deterministic impacted-claim outputs and `code_drift_detected` audit events.
- Added adaptive utility-based consolidation scheduling with deterministic weighted scoring and interval fallback trigger.
- Added novelty benchmark harness (`benchmarks/novelty_eval.py`) with quick/full modes and pass/fail metric gating output.
- Added CI novelty gate job that runs targeted claim-graph, drift, and adaptive scheduler suites plus novelty quick eval, failing build on metric gate miss.
- Extended export/import portability to round-trip claim graph entities: `claims`, `claim_edges`, `claim_sources`, `claim_events`, and `episode_anchors`.
- Extended integration and regression coverage for drift invalidation, utility scheduling, claim retrieval, and end-to-end claim drift recall flows.

## 0.12.3 — 2026-03-03

Comprehensive security and robustness hardening across 17 files.

### Security

- **Prompt injection mitigation** — all user-supplied content fed into LLM consolidation/contradiction prompts is now sanitized via `_sanitize_for_prompt()`
- **Release script shell injection** — `subprocess.run(shell=True)` replaced with list-form invocation
- **Path traversal guard** — `_merge_into_existing` in consolidation engine now rejects filenames with `..` or `/`
- **REST filename validation** — topic read/correct endpoints reject unsafe filenames

### Bug Fixes

- **Database re-entrant connections** — `get_connection()` now uses nest counting so nested calls don't double-commit/rollback
- **Atomic FTS operations** — `insert_episode`, `soft_delete_episode`, `hard_delete_episode` now perform FTS updates inside the same connection context
- **Circuit breaker TOCTOU** — `check()` now acquires lock internally for atomic state transitions
- **Config bool coercion** — `bool("false")` trap fixed with `_coerce_bool()` helper for all TOML boolean fields
- **Context assembler datetime comparison** — ISO string comparison replaced with proper `parse_datetime()` calls
- **Engine version file collision** — timestamp format now includes microseconds
- **Dashboard None guards** — surprise_score, consolidation run fields, and `_topics` init all guarded

### Hardening

- **Config thread safety** — `get_config()` uses double-checked locking
- **API exception isolation** — all 16 MCP tool handlers and REST endpoints wrapped in try/except with logging
- **API validation parity** — content length (50KB), batch size (100), n_results bounds enforced across MCP/REST/OpenAI surfaces
- **REST async correctness** — all endpoints now use `asyncio.to_thread()` for blocking client calls
- **Client batch validation** — episodes without `"content"` key are skipped; surprise coerced to float
- **Background consolidation safety** — loop checks stop event and pool before submitting
- **FAISS index safety** — local VectorStore wrapped in try/finally with `_save()`
- **Plugin dispatch safety** — `fire()` iterates a copy of the plugin list
- **Release script rollback** — version bump reverted on test/lint failure
- **Retry validation** — `retry_with_backoff` validates `max_retries >= 1`
- **Backend cleanup** — `reset_backends()` closes existing backends

### Internal

- 526 tests (2 new for schemas.py validation)

## 0.12.2 — 2026-03-02

Temporal belief queries, shared utils, and code quality improvements.

### Features

- **Temporal belief queries** (Phase 3.4) — `as_of` parameter on `recall()` across all API surfaces (MCP, REST, OpenAI schemas, Python client). When set, recall returns only records and episodes that were valid at that point in time, enabling questions like "what did I know about X last month?"
- **`EVOLVING_TOPIC_LOOKBACK_DAYS` config field** — the 30-day lookback window for "evolving topic" signals is now configurable via `[retrieval]` in TOML instead of hardcoded

### Refactoring

- **Shared `utils` module** — extracted `parse_json_list()` and `parse_datetime()` into `src/consolidation_memory/utils.py`, replacing duplicated patterns across 15 call sites in 8 modules
- **Content-type validation deduplication** — extracted `_normalize_content_type()` in `client.py`, shared between `store()` and `store_batch()`
- **Code-fence stripping deduplication** — consolidated duplicate `_strip_code_fences` implementations into a single reused helper
- **`RunStatus` Literal type** — added `RunStatus` type and `RUN_STATUS_RUNNING` / `RUN_STATUS_COMPLETED` / `RUN_STATUS_FAILED` constants in `types.py`, replacing raw status strings across `database.py`, `engine.py`, and `client.py`
- **Plugin hook name validation** — `fire()` in `plugins.py` now validates against a `HOOK_NAMES` frozenset, raising `ValueError` on typos instead of silently doing nothing
- **Magic number replaced** — hardcoded contradictions-per-run estimate in `consolidation_log()` replaced with `_CONTRADICTIONS_PER_RUN_ESTIMATE` named constant
- **Redundant `len()` guards simplified** — removed 3 unnecessary `len(prunable) if prunable else 0` patterns in `engine.py`

### Bug Fixes

- **mypy fixes** — `ThreadPoolExecutor | None` typing corrected across modules
- **CI fix** — REST API tests now skip gracefully when `fastapi` is not installed

### Internal

- 524 tests (30 new: 23 for temporal belief queries in `test_temporal_belief_queries.py`, 7 for shared utils in `test_utils.py`)

## 0.12.1 — 2026-03-02


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
