# consolidation-memory contributor context

Compact maintainer guide aligned with the codebase.

## Repository shape

```text
src/consolidation_memory/
  client.py          orchestration + tool-facing operations
  database.py        SQLite schema/migrations
  query_service.py   canonical query envelopes
  context_assembler.py  hybrid recall
  claim_graph.py     deterministic claim canonicalization
  drift.py           git-based drift challenge flow
  consolidation/       engine, fast_path, prompting
  schemas.py         OpenAI tools + MCP dispatch
```

## Important facts

- Package version: `pyproject.toml`
- Schema version: `database.py` (`CURRENT_SCHEMA_VERSION`)
- Tool schemas: `schemas.py` (`openai_tools`, `dispatch_tool_call`)

## Local verification

```bash
pip install -e ".[all,dev]"
python scripts/smoke_builder_base.py
pytest tests/ -q
ruff check src tests/
mypy src/consolidation_memory/
```

## Guardrails

1. Keep semantics aligned across Python, MCP, REST, and OpenAI surfaces.
2. Preserve trust invariants (see [CONTRIBUTING.md](CONTRIBUTING.md)).
3. Update user-facing docs when behavior changes.
4. Avoid hard-coded test counts or stale timeline statements in docs.

## Known architectural debt (audit 2026-06-13)

Prioritized blind spots — check this before large refactors; update when fixed.

**P0 (trust / scope)** — largely addressed 2026-06-13
- ~~Scope on audit APIs (explicit `scope` arg)~~: `contradictions`, `decay_report`, `consolidation_log`, `status` filter when `scope` is passed. Remaining: unscoped calls still global; `trust_profile` in `status()` always global; default-resolved scope on `None` not yet applied to audit paths.
- ~~`content_type` validation~~: shared `validate_episode_content_type()` in `types.py`.

**P1 (enforcement / ops)** — largely addressed 2026-06-13
- ~~`coding_agent_eval` CI gate~~: `quick` mode in `novelty_gates` job. `real_world_eval` remains manual (live corpus).
- ~~`embedding_disk_cache` cross-process lock~~: `.embedding_cache_write.lock` via `process_write_lock.py`.
- ~~`SECURITY.md` + MCP trust boundary~~: updated for `0.16.x` with stdio trust model documented.

**P2 (structure / tests)**
- `ContentType` (4 ingest types) vs `RecordType` (+ `procedure`, `strategy` post-consolidation only).
- Global-by-design tools (`consolidate`, `compact`, `detect_drift`) vs scope leaks — document contract in CONTRIBUTING.
- Migration regression thin for schema v17–v20; concurrency tests thread-only, not multi-process.
- `database.py` god-module (~5.4k lines); split by domain when touching migrations heavily.

**Keep (do not rewrite)**
- Episodes → records → claims → topics stack; `tool_dispatch` seam; FAISS write lease; fast-path before LLM; `query_service` envelopes.

## Core docs

- [README.md](README.md)
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [docs/FAST_PATH_EPISODES.md](docs/FAST_PATH_EPISODES.md)
- [CONTRIBUTING.md](CONTRIBUTING.md)