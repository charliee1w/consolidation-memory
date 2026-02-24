# consolidation-memory

Local-first persistent memory for AI agents. Store facts, solutions, and preferences as episodes — then a local LLM automatically clusters and distills them into structured knowledge documents. Your AI remembers what it learned, not just what you said.

Works with **any LLM** (LM Studio, Ollama, OpenAI) and **any interface** — MCP (Claude Desktop/Code/Cursor), Python API, REST API, or OpenAI function calling.

## Quick Start

```bash
pip install consolidation-memory[fastembed]
consolidation-memory init
```

### MCP (Claude Desktop / Claude Code / Cursor)

Add to your Claude Desktop config (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "consolidation_memory": {
      "command": "consolidation-memory"
    }
  }
}
```

### Python API

```python
from consolidation_memory import MemoryClient

with MemoryClient() as mem:
    # Store
    result = mem.store("User prefers dark mode", content_type="preference", tags=["ui"])
    print(result.id)  # UUID of stored episode

    # Recall
    result = mem.recall("user interface preferences")
    for ep in result.episodes:
        print(ep["content"], ep["similarity"])
    for doc in result.knowledge:
        print(doc["title"])

    # Status
    stats = mem.status()
    print(stats.episodic_buffer["total"])
    print(stats.knowledge_base["total_topics"])

    # Forget
    mem.forget(episode_id="some-uuid")

    # Export
    export = mem.export()
    print(export.path)  # backup JSON file

    # Correct knowledge
    mem.correct("vr_setup.md", "SteamVR version is now 2.7, not 2.5")

    # Manual consolidation
    mem.consolidate()
```

### OpenAI Function Calling (any OpenAI-compatible LLM)

```python
import json
from openai import OpenAI
from consolidation_memory import MemoryClient
from consolidation_memory.schemas import openai_tools, dispatch_tool_call

client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
mem = MemoryClient()

messages = [{"role": "user", "content": "What do you remember about my VR setup?"}]

response = client.chat.completions.create(
    model="qwen2.5-7b-instruct",
    messages=messages,
    tools=openai_tools,
)

for call in response.choices[0].message.tool_calls or []:
    result = dispatch_tool_call(mem, call.function.name, json.loads(call.function.arguments))
    messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})

mem.close()
```

Works with LM Studio, Ollama, OpenAI, Azure, any OpenAI-compatible API.

### REST API

```bash
pip install consolidation-memory[rest]
consolidation-memory serve --rest --port 8080
```

```bash
# Store
curl -X POST http://localhost:8080/memory/store \
  -H "Content-Type: application/json" \
  -d '{"content": "User runs SteamVR on Windows 11", "content_type": "fact"}'

# Recall
curl -X POST http://localhost:8080/memory/recall \
  -H "Content-Type: application/json" \
  -d '{"query": "VR setup"}'

# Status
curl http://localhost:8080/memory/status

# Health
curl http://localhost:8080/health
```

All endpoints:

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Version + status |
| POST | `/memory/store` | Store episode |
| POST | `/memory/recall` | Semantic search |
| GET | `/memory/status` | System statistics |
| DELETE | `/memory/episodes/{id}` | Forget episode |
| POST | `/memory/consolidate` | Run consolidation |
| POST | `/memory/correct` | Correct knowledge doc |
| POST | `/memory/export` | Export to JSON |

## How It Works

```
Store → Embed → FAISS Index
                    ↓
            Recall (semantic search + priority scoring)
                    ↓
        Consolidation (cluster → LLM synthesis → knowledge docs)
```

1. **Store**: Episodes (facts, solutions, preferences) are embedded and stored in SQLite + FAISS
2. **Recall**: Queries are embedded and matched against episodes using cosine similarity, weighted by surprise score, recency, and access frequency
3. **Consolidate**: Background thread clusters related episodes via agglomerative clustering, then synthesizes them into structured markdown knowledge documents using a local LLM

## MCP Tools

| Tool | Description |
|------|-------------|
| `memory_store` | Store a memory episode |
| `memory_recall` | Semantic search over episodes + knowledge |
| `memory_status` | System statistics |
| `memory_forget` | Remove an episode |
| `memory_export` | Export to JSON snapshot |
| `memory_correct` | Correct a knowledge document |

## CLI Commands

```bash
consolidation-memory serve              # Start MCP server (default)
consolidation-memory serve --rest       # Start REST API on 127.0.0.1:8080
consolidation-memory serve --rest --port 9000 --host 0.0.0.0
consolidation-memory init               # Interactive setup
consolidation-memory status             # Show statistics
consolidation-memory consolidate        # Run consolidation manually
consolidation-memory export             # Export to JSON
consolidation-memory import PATH        # Import from JSON export
consolidation-memory reindex            # Re-embed with current backend
```

## Embedding Backends

| Backend | Config Value | Model | Dimensions | Requirements |
|---------|-------------|-------|-----------|--------------|
| **FastEmbed** (default) | `fastembed` | bge-small-en-v1.5 | 384 | `pip install consolidation-memory[fastembed]` |
| LM Studio | `lmstudio` | nomic-embed-text-v1.5 | 768 | LM Studio running |
| OpenAI | `openai` | text-embedding-3-small | 1536 | API key |
| Ollama | `ollama` | nomic-embed-text | 768 | Ollama running |

## LLM Backends (Consolidation)

Consolidation requires an LLM to synthesize episode clusters into knowledge documents. Set `backend = "disabled"` under `[llm]` to use store/recall without consolidation.

| Backend | Config Value | Requirements |
|---------|-------------|--------------|
| LM Studio (default) | `lmstudio` | LM Studio running with chat model |
| OpenAI | `openai` | API key |
| Ollama | `ollama` | Ollama running with chat model |
| Disabled | `disabled` | None (no consolidation) |

## Configuration

Config file location:
- **Linux/macOS**: `~/.config/consolidation_memory/config.toml`
- **Windows**: `%APPDATA%\consolidation_memory\config.toml`
- **Override**: `CONSOLIDATION_MEMORY_CONFIG` env var

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

[dedup]
enabled = true
similarity_threshold = 0.95
```

Run `consolidation-memory init` to generate a config interactively.

## Data Directory

- **Linux**: `~/.local/share/consolidation_memory/`
- **macOS**: `~/Library/Application Support/consolidation_memory/`
- **Windows**: `%LOCALAPPDATA%\consolidation_memory\`

Override with `data_dir` under `[paths]` in config.

## Migrating from Existing Installation

If you have an existing data directory, point your config at it:

```toml
[paths]
data_dir = "C:\\Users\\you\\Documents\\consolidation_memory\\data"
```

If switching embedding backends (different dimensions), run:

```bash
consolidation-memory reindex
```

## Installation Extras

```bash
pip install consolidation-memory                     # Core (MCP + Python API)
pip install consolidation-memory[fastembed]           # + FastEmbed (recommended)
pip install consolidation-memory[openai]              # + OpenAI SDK
pip install consolidation-memory[rest]                # + REST API (FastAPI + Uvicorn)
pip install consolidation-memory[all]                 # Everything
```

## Development

```bash
git clone https://github.com/charliee1w/consolidation-memory
cd consolidation-memory
pip install -e ".[fastembed,dev]"
python -m pytest tests/ -v      # 88 tests
```

## License

MIT
