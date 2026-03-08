# consolidation-memory contributor context

This file is a compact, code-aligned maintainer guide.

## Repository shape

```text
src/consolidation_memory/
  __init__.py
  cli.py
  client.py
  server.py
  rest.py
  schemas.py
  database.py
  vector_store.py
  context_assembler.py
  query_service.py
  query_semantics.py
  claim_graph.py
  anchors.py
  drift.py
  config.py
  release_gates.py
  plugins.py
  record_cache.py
  topic_cache.py
  types.py
```

## Important facts

- Package version source: `pyproject.toml`.
- DB schema version source: `database.py` (`CURRENT_SCHEMA_VERSION`).
- Tool schemas source: `schemas.py` (`openai_tools` + `dispatch_tool_call`).
- Release gates source: `release_gates.py` + `scripts/verify_release_gates.py`.

## Local verification

```bash
pip install -e ".[all,dev]"
python scripts/smoke_builder_base.py
pytest tests/ -q
pytest tests/ -q -W error::ResourceWarning
ruff check src/ tests/
mypy src/consolidation_memory/
```

## CI and release references

- `.github/workflows/test.yml`
- `.github/workflows/publish.yml`
- `.github/workflows/novelty-full-nightly.yml`

## Guardrails for changes

1. Keep semantics aligned across Python/MCP/REST/OpenAI surfaces.
2. Preserve trust invariants:
- temporal correctness
- provenance traceability
- contradiction visibility
- drift challenge auditability
- scope isolation behavior
3. Update docs in the same change when behavior changes.
4. Avoid hard-coded test counts or stale timeline statements in docs.

## Core docs

- `README.md`
- `docs/ARCHITECTURE.md`
- `docs/ROADMAP.md`
- `docs/RELEASE_GATES.md`
- `docs/NOVELTY_METRICS.md`
