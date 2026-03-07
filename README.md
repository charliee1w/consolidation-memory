# consolidation-memory

[![PyPI](https://img.shields.io/pypi/v/consolidation-memory)](https://pypi.org/project/consolidation-memory/)
[![CI](https://img.shields.io/github/actions/workflow/status/charliee1w/consolidation-memory/test.yml?label=tests)](https://github.com/charliee1w/consolidation-memory/actions)
[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://pypi.org/project/consolidation-memory/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Local-first persistent memory for coding agents.

`consolidation-memory` stores raw episodes, consolidates them into structured knowledge, and now tracks claim-level provenance and contradiction history. It runs on SQLite + FAISS and can be used through MCP, Python, REST, or OpenAI-style function calling.

## What It Does

- Stores episodes (`exchange`, `fact`, `solution`, `preference`) with vector embeddings.
- Recalls by semantic + keyword ranking with metadata boosts.
- Consolidates episodes into knowledge topics and structured records:
  - `fact`, `solution`, `preference`, `procedure`
- Tracks temporal validity (`valid_from`, `valid_until`) and supports `as_of` recall queries.
- Logs contradictions and supports contradiction-aware merge behavior.
- Maintains a claim graph:
  - claims
  - claim edges (for example, `contradicts`)
  - claim sources
  - claim events
- Extracts and persists episode anchors (paths, commits, tool references).
- Detects code drift from git changes and challenges impacted claims with audit events.
- Uses an adaptive consolidation scheduler (utility score + interval fallback).
- Returns claim results in `recall()` alongside episodes, topics, and records.

## Quick Start

```bash
pip install consolidation-memory[fastembed]
consolidation-memory init
consolidation-memory test
consolidation-memory serve
```

Notes:
- `fastembed` is local and does not require API keys.
- Consolidation requires an LLM backend (`lmstudio`, `ollama`, `openai`) unless explicitly disabled.

## MCP Server

Add to your MCP client config:

```json
{
  "mcpServers": {
    "consolidation_memory": {
      "command": "consolidation-memory",
      "args": ["--project", "universal", "serve"]
    }
  }
}
```

Use one shared project name (for example, `universal`) across all clients
to keep a single knowledge set.

Available tools:
- `memory_store`
- `memory_store_batch`
- `memory_recall`
- `memory_search`
- `memory_claim_browse`
- `memory_claim_search`
- `memory_detect_drift`
- `memory_status`
- `memory_forget`
- `memory_export`
- `memory_correct`
- `memory_compact`
- `memory_consolidate`
- `memory_protect`
- `memory_timeline`
- `memory_contradictions`
- `memory_browse`
- `memory_read_topic`
- `memory_decay_report`
- `memory_consolidation_log`

## Python API

```python
from consolidation_memory import MemoryClient

with MemoryClient(auto_consolidate=False) as mem:
    mem.store(
        "User prefers dark mode in terminal tools.",
        content_type="preference",
        tags=["ui", "terminal"],
    )

    result = mem.recall(
        "terminal preferences",
        n_results=5,
        include_knowledge=True,
        as_of="2026-03-01T00:00:00+00:00",
    )

    print("episodes:", len(result.episodes))
    print("knowledge topics:", len(result.knowledge))
    print("records:", len(result.records))
    print("claims:", len(result.claims))
    print("warnings:", result.warnings)
```

`RecallResult` is backward compatible and includes:
- `episodes`
- `knowledge`
- `records`
- `claims`
- `warnings`

## Consolidation Model

```text
episodes -> SQLite + FAISS
           -> recall (semantic + keyword)

background consolidation:
episodes -> cluster -> extract/merge records -> knowledge topics
         -> contradiction detection -> temporal expiration + audit log
         -> claim emission (claims/sources/events/edges)
```

## Claims And Anchors

### Claim graph

Consolidation emits deterministic claims for merged records and writes:
- `claims`: normalized claim payload and lifecycle state
- `claim_sources`: links to episodes/topics/records
- `claim_events`: `create`, `update`, `expire`, `contradiction`, etc.
- `claim_edges`: relationship graph (for example, `contradicts`)

Claim retrieval is exposed through:
- Python: `MemoryClient.browse_claims(...)` and `MemoryClient.search_claims(...)`
- MCP/OpenAI tools: `memory_claim_browse` and `memory_claim_search`
- REST: `POST /memory/claims/browse` and `POST /memory/claims/search`
- Temporal claim-state queries: pass `as_of` to claim browse/search interfaces

### Anchor persistence

Stored episode content is parsed for anchors and written to `episode_anchors`:
- file paths (POSIX + Windows)
- commit hashes
- tool references (`pytest`, `uvicorn`, `docker`, `git`, etc.)

Anchors are used for drift workflows and claim challenge operations.

### Drift detection interfaces

- CLI: `consolidation-memory detect-drift [--base-ref origin/main] [--repo-path <path>]`
- Python: `MemoryClient.detect_drift(base_ref=..., repo_path=...)`
- REST: `POST /memory/detect-drift`

Drift detection maps changed files to anchored claims, challenges impacted active
claims, and records `claim_events` with event type `code_drift_detected`.

## REST API

Install extras and run:

```bash
pip install consolidation-memory[rest]
consolidation-memory serve --rest --host 127.0.0.1 --port 8080
```

Endpoints:
- `GET /health`
- `POST /memory/store`
- `POST /memory/store/batch`
- `POST /memory/recall`
- `POST /memory/search`
- `POST /memory/claims/browse`
- `POST /memory/claims/search`
- `POST /memory/detect-drift`
- `GET /memory/status`
- `DELETE /memory/episodes/{episode_id}`
- `POST /memory/consolidate`
- `POST /memory/correct`
- `POST /memory/export`
- `POST /memory/compact`
- `GET /memory/browse`
- `GET /memory/topics/{filename}`
- `POST /memory/timeline`
- `POST /memory/contradictions`
- `POST /memory/protect`
- `POST /memory/consolidation-log`
- `GET /memory/decay-report`

## OpenAI Function Calling

Use the provided tool schemas and dispatch helper:

```python
from consolidation_memory import MemoryClient
from consolidation_memory.schemas import openai_tools, dispatch_tool_call

client = MemoryClient(auto_consolidate=False)

# Pass openai_tools to your model
# Then route tool calls through dispatch_tool_call(client, name, arguments)
```

## Backends

### Embedding

| Backend | Local | Default model | Typical dimension |
| --- | --- | --- | --- |
| `fastembed` (default) | yes | `BAAI/bge-small-en-v1.5` | 384 |
| `lmstudio` | yes | `text-embedding-nomic-embed-text-v1.5` | 768 |
| `ollama` | yes | `nomic-embed-text` | 768 |
| `openai` | no | `text-embedding-3-small` | 1536 |

### LLM (for consolidation/extraction)

| Backend | Notes |
| --- | --- |
| `lmstudio` (default) | local chat model |
| `ollama` | local chat model |
| `openai` | API-backed |
| `disabled` | store/recall only, no LLM consolidation |

## Configuration

Generate config interactively:

```bash
consolidation-memory init
```

Default config file locations:
- Linux: `~/.config/consolidation_memory/config.toml`
- macOS: `~/Library/Application Support/consolidation_memory/config.toml`
- Windows: `%APPDATA%\\consolidation_memory\\config.toml`
- Override path: `CONSOLIDATION_MEMORY_CONFIG`

Every scalar config field can be overridden with:
- `CONSOLIDATION_MEMORY_<FIELD_NAME>`

Examples:

```bash
CONSOLIDATION_MEMORY_EMBEDDING_BACKEND=fastembed
CONSOLIDATION_MEMORY_LLM_BACKEND=lmstudio
CONSOLIDATION_MEMORY_CONSOLIDATION_INTERVAL_HOURS=6
CONSOLIDATION_MEMORY_PROJECT=work
```

## CLI Commands

| Command | Purpose |
| --- | --- |
| `consolidation-memory serve` | start MCP server |
| `consolidation-memory serve --rest` | start REST server |
| `consolidation-memory init` | interactive setup |
| `consolidation-memory test` | installation/self-check |
| `consolidation-memory status` | show memory stats |
| `consolidation-memory consolidate` | run consolidation now |
| `consolidation-memory detect-drift` | challenge claims impacted by changed files |
| `consolidation-memory export` | export JSON snapshot |
| `consolidation-memory import PATH` | import JSON snapshot |
| `consolidation-memory reindex` | rebuild embeddings/index |
| `consolidation-memory browse` | inspect knowledge topics |
| `consolidation-memory setup-memory --path AGENTS.md` | write memory integration block to any instruction file |
| `consolidation-memory setup-claude` | legacy alias for `setup-memory --path ~/.claude/CLAUDE.md` |
| `consolidation-memory dashboard` | launch Textual dashboard |

## Agent Instruction Setup

Use the vendor-neutral setup helper to add proactive recall/store guidance to your agent instructions:

```bash
consolidation-memory setup-memory --path AGENTS.md
```

For Claude Desktop specifically, use:

```bash
consolidation-memory setup-memory --path ~/.claude/CLAUDE.md
```

## Multi-project Isolation

Each project has isolated storage:

```bash
consolidation-memory --project work status
CONSOLIDATION_MEMORY_PROJECT=work consolidation-memory serve
```

This keeps separate:
- SQLite DB
- FAISS index
- knowledge topics
- consolidation logs

## Data Layout

Base directory is `platformdirs.user_data_dir("consolidation_memory")`.

Per project:

```text
projects/<project>/
  memory.db
  faiss_index.bin
  faiss_id_map.json
  faiss_tombstones.json
  knowledge/
  consolidation_logs/
  backups/
```

Export/import snapshots include:

- episodes + knowledge topics/records
- claims
- claim edges
- claim sources
- claim events
- episode anchors

## Development

```bash
git clone https://github.com/charliee1w/consolidation-memory
cd consolidation-memory
pip install -e ".[all,dev]"
pytest tests/ -q
ruff check src/ tests/
```

## License

MIT
