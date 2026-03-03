# consolidation-memory

[![PyPI](https://img.shields.io/pypi/v/consolidation-memory)](https://pypi.org/project/consolidation-memory/)
[![CI](https://img.shields.io/github/actions/workflow/status/charliee1w/consolidation-memory/test.yml?label=tests)](https://github.com/charliee1w/consolidation-memory/actions)
[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://pypi.org/project/consolidation-memory/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Local-first persistent memory for AI agents. SQLite + FAISS, runs on a laptop, no cloud.

Agents store episodes (conversations, facts, solutions). A background thread periodically clusters related episodes and uses a local LLM to synthesize them into structured knowledge records. Old episodes get pruned. Knowledge compounds over time instead of degrading.

## Install

```bash
pip install consolidation-memory[fastembed]
consolidation-memory init
consolidation-memory setup-claude  # Add memory instructions to CLAUDE.md
```

FastEmbed runs locally. No API keys needed. The `setup-claude` command adds instructions to your `~/.claude/CLAUDE.md` so Claude Code proactively uses memory tools.

## MCP Server

```json
{
  "mcpServers": {
    "consolidation_memory": {
      "command": "consolidation-memory"
    }
  }
}
```

Tools: `memory_store`, `memory_store_batch`, `memory_recall`, `memory_search`, `memory_status`, `memory_forget`, `memory_export`, `memory_correct`, `memory_compact`, `memory_consolidate`, `memory_browse`, `memory_read_topic`, `memory_timeline`, `memory_decay_report`, `memory_protect`

## Python API

```python
from consolidation_memory import MemoryClient

with MemoryClient() as mem:
    mem.store("User prefers dark mode", content_type="preference", tags=["ui"])

    result = mem.recall("user interface preferences")
    for ep in result.episodes:
        print(ep["content"], ep["similarity"])
```

## OpenAI Function Calling

Works with any OpenAI-compatible API (LM Studio, Ollama, OpenAI, Azure):

```python
from consolidation_memory import MemoryClient
from consolidation_memory.schemas import openai_tools, dispatch_tool_call

mem = MemoryClient()
# Pass openai_tools to your chat completion, dispatch results with dispatch_tool_call()
```

## REST API

```bash
pip install consolidation-memory[rest]
consolidation-memory serve --rest --port 8080
```

`POST /memory/store` | `POST /memory/store/batch` | `POST /memory/recall` | `POST /memory/search` | `GET /memory/status` | `DELETE /memory/episodes/{id}` | `POST /memory/consolidate` | `POST /memory/correct` | `POST /memory/export` | `POST /memory/compact` | `GET /memory/browse` | `GET /memory/topics/{filename}` | `POST /memory/timeline` | `POST /memory/contradictions` | `POST /memory/protect` | `GET /memory/decay-report` | `GET /health`

## How Consolidation Works

```
store episodes → SQLite + FAISS
                      ↓
        background thread (every 6h)
                      ↓
     hierarchical clustering by similarity
                      ↓
        LLM synthesizes knowledge records
        (facts, solutions, preferences, procedures)
                      ↓
     records feed back into recall, old episodes pruned
```

Episodes are grouped by semantic similarity using agglomerative clustering. Each cluster is matched against existing knowledge topics. The LLM either creates a new topic or merges into an existing one. Output is validated, versioned, and written as structured records with their own embeddings for independent search.

Three consecutive LLM failures trip a circuit breaker. Pruned episodes still count toward consolidation history.

## Backends

### Embedding

| Backend | Install | Model | Local |
|---------|---------|-------|:-----:|
| **FastEmbed** (default) | `pip install consolidation-memory[fastembed]` | bge-small-en-v1.5 | Y |
| LM Studio | Built-in | nomic-embed-text-v1.5 | Y |
| Ollama | Built-in | nomic-embed-text | Y |
| OpenAI | `pip install consolidation-memory[openai]` | text-embedding-3-small | N |

### LLM (for consolidation)

| Backend | Requirements |
|---------|-------------|
| **LM Studio** (default) | LM Studio running with any chat model |
| Ollama | Ollama running with any chat model |
| OpenAI | API key |
| Disabled | None — no consolidation, pure vector search |

## Configuration

```bash
consolidation-memory init
```

<details>
<summary>Manual config</summary>

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
cluster_threshold = 0.72  # default: 0.78
prune_enabled = true
prune_after_days = 60  # default: 30
```

</details>

<details>
<summary>Environment variable overrides</summary>

Every setting can be overridden with `CONSOLIDATION_MEMORY_<FIELD_NAME>`:

```bash
CONSOLIDATION_MEMORY_EMBEDDING_BACKEND=lmstudio
CONSOLIDATION_MEMORY_EMBEDDING_DIMENSION=768
CONSOLIDATION_MEMORY_LLM_BACKEND=openai
CONSOLIDATION_MEMORY_LLM_API_KEY=sk-...
CONSOLIDATION_MEMORY_CONSOLIDATION_INTERVAL_HOURS=12
CONSOLIDATION_MEMORY_CONSOLIDATION_AUTO_RUN=false
```

Priority: defaults < TOML < env vars < `reset_config()` (tests).

</details>

## CLI

| Command | Description |
|---------|-------------|
| `consolidation-memory serve` | Start MCP server (default) |
| `consolidation-memory serve --rest` | Start REST API |
| `consolidation-memory --project work serve` | MCP server for a specific project |
| `consolidation-memory init` | Interactive setup |
| `consolidation-memory status` | Show stats |
| `consolidation-memory consolidate` | Manual consolidation |
| `consolidation-memory export` | Export to JSON |
| `consolidation-memory import PATH` | Import from JSON |
| `consolidation-memory reindex` | Re-embed everything (after switching backends) |
| `consolidation-memory browse` | Browse knowledge topics |
| `consolidation-memory setup-claude` | Add memory instructions to CLAUDE.md |
| `consolidation-memory test` | Post-install verification |
| `consolidation-memory dashboard` | TUI dashboard |

## Multi-Project

Isolate memories per project:

```bash
consolidation-memory --project work status
CONSOLIDATION_MEMORY_PROJECT=work consolidation-memory serve
```

MCP config for multiple projects:

```json
{
  "mcpServers": {
    "memory-work": {
      "command": "consolidation-memory",
      "env": { "CONSOLIDATION_MEMORY_PROJECT": "work" }
    },
    "memory-personal": {
      "command": "consolidation-memory",
      "env": { "CONSOLIDATION_MEMORY_PROJECT": "personal" }
    }
  }
}
```

Each project gets its own database, vector index, and knowledge files.

## Cross-Client Memory

One consolidation-memory instance serves every MCP client on your machine. Claude Code, Cursor, Windsurf, VS Code + Continue — all share the same SQLite database and FAISS index. A fact stored from Cursor is recalled in Claude Code. No cloud sync needed.

This is the local-first alternative to cloud-based memory passports. Your data never leaves your machine.

<details>
<summary>Example configs for each client</summary>

**Claude Code** (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "consolidation_memory": {
      "command": "consolidation-memory"
    }
  }
}
```

**Cursor** (`.cursor/mcp.json`):

```json
{
  "mcpServers": {
    "consolidation_memory": {
      "command": "consolidation-memory"
    }
  }
}
```

**VS Code + Continue** (`.continue/config.json`):

```json
{
  "mcpServers": [
    {
      "name": "consolidation_memory",
      "command": "consolidation-memory"
    }
  ]
}
```

**Generic MCP client** (any client supporting stdio transport):

```json
{
  "command": "consolidation-memory",
  "transport": "stdio"
}
```

All configs above point at the default data directory. To share memories across clients with a specific project:

```json
{
  "command": "consolidation-memory",
  "env": { "CONSOLIDATION_MEMORY_PROJECT": "my-project" }
}
```

Every client using the same project name reads and writes to the same database.

</details>

## Data Storage

All data stays local.

| Platform | Path |
|----------|------|
| Linux | `~/.local/share/consolidation_memory/projects/<name>/` |
| macOS | `~/Library/Application Support/consolidation_memory/projects/<name>/` |
| Windows | `%LOCALAPPDATA%\consolidation_memory\projects\<name>\` |

Switching embedding backends? `consolidation-memory reindex`

## Roadmap

- [x] Hybrid search (BM25 + semantic fusion)
- [x] Diff-aware merge validation for consolidation
- [x] Recall result deduplication
- [x] Confidence-aware recall ranking
- [x] Source traceability in recall results
- [x] Consolidation changelog
- [x] Uncertainty signaling for low-confidence records
- [x] Temporal belief queries (`as_of` parameter)
- [ ] Adaptive memory (query-driven consolidation, access-weighted ranking)
- [ ] Cross-project recall
- [ ] First-party plugins (project context, git history)

## Development

```bash
git clone https://github.com/charliee1w/consolidation-memory
cd consolidation-memory
pip install -e ".[all,dev]"
pytest tests/ -v
ruff check src/ tests/
```

## License

MIT
