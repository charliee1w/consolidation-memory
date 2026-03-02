# consolidation-memory

Local-first persistent memory for AI agents — store, recall, and consolidate knowledge across sessions using FAISS, SQLite, and any LLM.

## Project Structure

```
src/consolidation_memory/
├── server.py              MCP server (primary interface)
├── client.py              MemoryClient API + MCP tool dispatch
├── database.py            SQLite (schema v9, WAL mode)
├── vector_store.py        FAISS index (flat → IVF auto-migration at 10K vectors)
├── config.py              Singleton Config dataclass, TOML + env var loading
├── context_assembler.py   Retrieval pipeline (FAISS ANN + priority scoring)
├── consolidation/         Knowledge extraction engine
│   ├── clustering.py      scipy UPGMA hierarchical clustering (threshold 0.78)
│   ├── prompting.py       LLM prompt builders, JSON parsing, validation
│   ├── scoring.py         Surprise score adjustment
│   └── engine.py          Orchestration, contradiction detection, merging
├── backends/              Embedding backends (fastembed, lmstudio, ollama, openai)
├── types.py               Episode, KnowledgeTopic, KnowledgeRecord, RecordType
├── schemas.py             OpenAI function calling schemas
├── record_cache.py        Thread-safe embedding cache for knowledge records
├── topic_cache.py         Thread-safe embedding cache for topics
├── circuit_breaker.py     LLM failure handling (3 failures → 60s cooldown)
├── cli.py                 CLI commands
├── rest.py                FastAPI REST API
├── dashboard.py           TUI dashboard (Textual)
└── plugins.py             Entry-point plugin system
```

## Development

### Setup

```
pip install -e ".[all,dev]"
```

### Testing

```
python -m pytest tests/ -v
```

Run the full suite before any commit. All 379+ tests must pass.

### Linting

```
python -m ruff check src/ tests/
python -m ruff format --check src/ tests/
```

### Type Checking

```
python -m mypy src/consolidation_memory/
```

## Code Style

- Python 3.10+ target
- Line length: 100 characters
- Type hints on all function signatures
- ruff for linting and formatting
- No bare `except:` — catch specific exceptions
- Google-style docstrings where needed
- Tests use pytest fixtures; `conftest.py` sets up isolated temp dirs

## Architecture Decisions

- **Config singleton**: `get_config()` returns a shared dataclass instance. Tests use `reset_config()`. All ~50 fields loadable from TOML + env var overrides (`CONSOLIDATION_MEMORY_<FIELD>`).
- **FAISS persistence**: Atomic writes via tempfile + os.replace. ID-map written first (source of truth). Auto-migrates to IndexIVFFlat at 10K vectors.
- **Knowledge records**: 4 structured types — `fact`, `solution`, `preference`, `procedure`. Stored as JSON in SQLite, rendered to markdown. Each record has its own embedding for independent retrieval.
- **Consolidation engine**: Background daemon thread with `consolidation_lock`. Scipy UPGMA clustering at threshold 0.78. LLM extracts structured records from episode clusters.
- **Contradiction detection**: Semantic similarity >= 0.7 between new and existing records triggers LLM verification. Contradicted records get `valid_until` set (temporal expiry).
- **Thread safety**: SQLite WAL mode, FAISS threading lock, per-thread DB connections via `threading.local`.
- **Security**: Prompt injection sanitization (XML wrapping, pattern stripping), path traversal guards, input validation, circuit breaker for LLM failures.

## Release Process

Use `/release` skill or `python scripts/release.py <version>`.

### Versioning (pre-1.0 semver)

- New features → minor bump
- Bug fixes only → patch bump
- Breaking changes → major bump

### When to Release

Release when a meaningful milestone lands — a feature, a noteworthy bug fix, or before starting a big refactor. Do NOT release for: internal refactors, test-only changes, docs-only, or minor tooling (batch these into the next real release).

### Process

1. Ensure working tree is clean
2. Bump version in `pyproject.toml` (single source of truth)
3. Update `CHANGELOG.md` — match existing format (Features, Bug Fixes, Refactoring, etc.)
4. Commit as `vX.Y.Z` or `vX.Y.Z: short description`
5. Tag `vX.Y.Z`
6. Push main + tag
7. Create GitHub release: `gh release create vX.Y.Z --notes-from-tag`

## Git Conventions

- Branch: `main` (solo dev, no feature branches)
- Commit messages: descriptive, present tense
- Release commits: `vX.Y.Z` or `vX.Y.Z: short description`

## Roadmap

See `.claude/ROADMAP.md` for the implementation roadmap.
**Current focus: Phase 2 (Recall Quality Leap).**

When asked to "continue the roadmap", read that file and implement the next
incomplete item. Commit after each feature. Run the full test suite before committing.

## Key Files to Know

- `pyproject.toml` — version (single source of truth), all deps and tool config
- `CHANGELOG.md` — full release history from v0.1.0
- `docs/ARCHITECTURE.md` — detailed architecture with Mermaid diagrams
- `.claude/ROADMAP.md` — project roadmap (gitignored, internal only)
- `tests/conftest.py` — test isolation setup (temp dirs, config reset, cache clearing)
- `scripts/release.py` — automated release script
