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
- ~~Scope on audit APIs (explicit `scope` arg)~~: `contradictions`, `decay_report`, `consolidation_log`, `status` filter when `scope` is passed. Remaining: unscoped calls still global (intentional); default-resolved scope on `None` not yet applied to audit paths.
- ~~`content_type` validation~~: shared `validate_episode_content_type()` in `types.py`.
- ~~`trust_profile` in scoped `status()`~~: `get_claim_trust_stats`, `count_active_challenged_claims`, `get_recently_contradicted_topic_ids` accept `scope`.

**P1 (enforcement / ops)** — largely addressed 2026-06-13
- ~~`coding_agent_eval` CI gate~~: `quick` mode in `novelty_gates` job. `real_world_eval` remains manual (live corpus).
- ~~`embedding_disk_cache` cross-process lock~~: `.embedding_cache_write.lock` via `process_write_lock.py`.
- ~~`SECURITY.md` + MCP trust boundary~~: updated for `0.16.x` with stdio trust model documented.

**P2 (structure / tests)**
- ~~`ContentType` vs `RecordType`~~: `procedure` ingest type added; `strategy` remains JSON-only at store time (see CONTRIBUTING + FAST_PATH_EPISODES).
- ~~Global-by-design tools~~: documented in CONTRIBUTING (scope-aware vs global contract).
- ~~Migration regression v17–v20~~: additive tests in `tests/test_core.py`.
- ~~Multi-process lock test~~: `tests/test_process_write_lock.py` for `ProcessWriteLease`.
- `database.py` god-module (~5.4k lines); split by domain when touching migrations heavily.

**Keep (do not rewrite)**
- Episodes → records → claims → topics stack; `tool_dispatch` seam; FAISS write lease; fast-path before LLM; `query_service` envelopes.

## Core docs

- [README.md](README.md)
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [docs/FAST_PATH_EPISODES.md](docs/FAST_PATH_EPISODES.md)
- [CONTRIBUTING.md](CONTRIBUTING.md)