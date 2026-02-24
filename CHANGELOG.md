# Changelog

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
