# consolidation-memory

[![CI](https://github.com/charliee1w/consolidation-memory/actions/workflows/test.yml/badge.svg)](https://github.com/charliee1w/consolidation-memory/actions/workflows/test.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://pypi.org/project/consolidation-memory/)

**Your AI forgets everything between sessions. This fixes that.**

A local-first memory system that stores, retrieves, and *consolidates* knowledge across conversations. Episodes go in, structured knowledge comes out — automatically, via a background LLM that clusters and synthesizes what it's learned.

No cloud dependency. No subscriptions. Your data stays on your machine.

```
You: "My build is failing with a linker error"
AI:  (recalls your project uses CMake + MSVC on Windows)
     (recalls you hit the same error last month — it was a missing vcpkg dependency)
     "Last time this happened it was a missing vcpkg package. Want me to
      check if your vcpkg.json changed since we fixed it?"
```

## How It Works

```
 ┌─────────┐     ┌───────────┐     ┌────────────┐
 │  Store   │────▶│  Embed    │────▶│ FAISS Index │
 │ episodes │     │ (any LLM) │     │ + SQLite DB │
 └─────────┘     └───────────┘     └──────┬─────┘
                                          │
                 ┌───────────┐     ┌──────▼─────┐
                 │ Knowledge │◀────│   Recall    │
                 │   Docs    │     │ (semantic)  │
                 └─────┬─────┘     └────────────┘
                       │
                ┌──────▼──────┐
                │ Consolidate │  ← background thread
                │ (cluster +  │    clusters episodes
                │  LLM synth) │    into knowledge docs
                └─────────────┘
```

1. **Store** — Save episodes (facts, solutions, preferences) with embeddings into SQLite + FAISS
2. **Recall** — Semantic search with priority scoring (surprise, recency, access frequency)
3. **Consolidate** — Background LLM clusters related episodes and synthesizes structured markdown knowledge documents

### How Consolidation Works

The consolidation engine runs on a background daemon thread (default: every 6 hours). It fetches all unconsolidated episodes, embeds them, and groups them using agglomerative hierarchical clustering with a configurable distance threshold. Each cluster represents a coherent topic.

For each cluster, the engine checks existing knowledge topics for semantic overlap. If a matching topic exists (above the topic-match threshold), the cluster's episodes are merged into the existing document. Otherwise, a new knowledge document is synthesized from scratch.

The LLM receives the cluster's episodes (with prompt injection patterns sanitized) and produces a structured markdown document with YAML frontmatter (title, summary, tags, confidence score). The engine validates the output, versions the previous document, writes the new one, and updates the SQLite metadata. Episodes that have been consolidated and aged past the prune threshold are soft-deleted to keep the FAISS index lean.

All backends retry transient failures with exponential backoff. If 3 consecutive clusters fail (indicating the LLM backend is down), consolidation aborts early rather than burning through timeouts.

## Quick Start

```bash
pip install consolidation-memory[fastembed]
consolidation-memory init
```

That's it. FastEmbed runs locally, no external services needed.

### MCP Server (Claude Desktop / Claude Code / Cursor)

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "consolidation_memory": {
      "command": "consolidation-memory"
    }
  }
}
```

Nine tools become available:

| Tool | What it does |
|------|-------------|
| `memory_store` | Save an episode (fact, solution, preference, exchange) |
| `memory_store_batch` | Store multiple episodes in one call (single embed + FAISS batch) |
| `memory_recall` | Semantic search over episodes + knowledge, with optional filters |
| `memory_search` | Keyword/metadata search — works without embedding backend |
| `memory_status` | System stats + health diagnostics + consolidation metrics |
| `memory_forget` | Soft-delete an episode |
| `memory_export` | Export everything to JSON |
| `memory_correct` | Fix outdated knowledge documents |

`memory_recall` supports optional filters: `content_types`, `tags`, `after`, `before` — all applied post-vector-search so you can narrow results to specific episode types or date ranges.

`memory_search` does plain text `LIKE` matching in SQLite. No embedding backend needed. Supports the same filters (`content_types`, `tags`, `after`, `before`) plus a `limit` parameter.

### Python API

```python
from consolidation_memory import MemoryClient

with MemoryClient() as mem:
    mem.store("User prefers dark mode", content_type="preference", tags=["ui"])

    result = mem.recall("user interface preferences")
    for ep in result.episodes:
        print(ep["content"], ep["similarity"])

    stats = mem.status()
    print(stats.health)  # {"status": "healthy", "issues": [], "backend_reachable": true}
```

### OpenAI Function Calling

Works with any OpenAI-compatible API (LM Studio, Ollama, OpenAI, Azure):

```python
from consolidation_memory import MemoryClient
from consolidation_memory.schemas import openai_tools, dispatch_tool_call

mem = MemoryClient()
# Pass openai_tools to your chat completion, dispatch results with dispatch_tool_call()
```

### REST API

```bash
pip install consolidation-memory[rest]
consolidation-memory serve --rest --port 8080
```

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Version + status |
| `POST` | `/memory/store` | Store episode |
| `POST` | `/memory/store/batch` | Store multiple episodes |
| `POST` | `/memory/recall` | Semantic search (with optional filters) |
| `POST` | `/memory/search` | Keyword/metadata search (no embedding needed) |
| `GET` | `/memory/status` | System statistics + consolidation metrics |
| `DELETE` | `/memory/episodes/{id}` | Forget episode |
| `POST` | `/memory/consolidate` | Trigger consolidation |
| `POST` | `/memory/correct` | Correct knowledge doc |
| `POST` | `/memory/export` | Export to JSON |

## Embedding Backends

| Backend | Install | Model | Dimensions | Runs locally? |
|---------|---------|-------|-----------|:---:|
| **FastEmbed** (default) | `pip install consolidation-memory[fastembed]` | bge-small-en-v1.5 | 384 | Yes |
| LM Studio | Built-in | nomic-embed-text-v1.5 | 768 | Yes |
| Ollama | Built-in | nomic-embed-text | 768 | Yes |
| OpenAI | `pip install consolidation-memory[openai]` | text-embedding-3-small | 1536 | No |

## LLM Backends (for consolidation)

The consolidation step needs a chat-capable LLM to synthesize clusters into knowledge documents. Set `backend = "disabled"` to skip consolidation and use store/recall only.

| Backend | Requirements |
|---------|-------------|
| **LM Studio** (default) | LM Studio running with any chat model |
| Ollama | Ollama running with any chat model |
| OpenAI | API key |
| Disabled | None — no consolidation, pure vector search |

## Configuration

```bash
consolidation-memory init  # Interactive setup
```

Or edit the config directly:

| Platform | Path |
|----------|------|
| Linux/macOS | `~/.config/consolidation_memory/config.toml` |
| Windows | `%APPDATA%\consolidation_memory\config.toml` |
| Override | `CONSOLIDATION_MEMORY_CONFIG` env var |

```toml
[embedding]
backend = "fastembed"

[llm]
backend = "lmstudio"
api_base = "http://localhost:1234/v1"
model = "qwen2.5-7b-instruct"

[consolidation]
auto_run = true
interval_hours = 6
cluster_threshold = 0.72
prune_enabled = true
prune_after_days = 60
```

## CLI

```bash
consolidation-memory serve              # MCP server (default)
consolidation-memory serve --rest       # REST API
consolidation-memory init               # Interactive setup
consolidation-memory status             # Show stats
consolidation-memory consolidate        # Manual consolidation
consolidation-memory export             # Export to JSON
consolidation-memory import PATH        # Import from JSON
consolidation-memory reindex            # Re-embed everything (after switching backends)
```

## Data Storage

All data stays local:

| Platform | Path |
|----------|------|
| Linux | `~/.local/share/consolidation_memory/` |
| macOS | `~/Library/Application Support/consolidation_memory/` |
| Windows | `%LOCALAPPDATA%\consolidation_memory\` |

Override with `data_dir` under `[paths]` in config.

## Migrating

Already have a data directory? Point your config at it:

```toml
[paths]
data_dir = "/path/to/your/existing/data"
```

Switching embedding backends (different dimensions)?

```bash
consolidation-memory reindex
```

## Development

```bash
git clone https://github.com/charliee1w/consolidation-memory
cd consolidation-memory
pip install -e ".[fastembed,dev]"
python -m pytest tests/ -v      # 88 tests, no external services needed
python -m ruff check src/ tests/
```

## License

MIT
