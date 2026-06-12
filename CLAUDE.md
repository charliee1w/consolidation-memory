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

## Core docs

- [README.md](README.md)
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [docs/FAST_PATH_EPISODES.md](docs/FAST_PATH_EPISODES.md)
- [CONTRIBUTING.md](CONTRIBUTING.md)