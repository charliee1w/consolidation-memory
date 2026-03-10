# consolidation-memory

[![PyPI](https://img.shields.io/pypi/v/consolidation-memory)](https://pypi.org/project/consolidation-memory/)
[![CI](https://img.shields.io/github/actions/workflow/status/charliee1w/consolidation-memory/test.yml?label=tests)](https://github.com/charliee1w/consolidation-memory/actions)
[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://pypi.org/project/consolidation-memory/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Local-first persistent memory for coding agents.



`consolidation-memory` stores episodic events, consolidates them into structured knowledge, and exposes a trust-aware retrieval stack (temporal recall, contradiction tracking, claim provenance, and drift challenge workflows).

## What It Is

- Episode storage with semantic dedup and FAISS indexing.
- Hybrid recall across episodes, knowledge topics, structured records, and claims.
- Claim graph with provenance (`claim_sources`) and lifecycle events (`claim_events`).
- Temporal queries (`as_of`) for both knowledge and claims.
- Drift detection that maps changed files to anchored claims and marks impacted claims challenged.
- Multi-scope persistence (namespace/project/app/agent/session columns) with compatibility defaults.
- Four access surfaces:
  - MCP server (`consolidation-memory serve`)
  - Python API (`MemoryClient`)
  - REST API (`consolidation-memory serve --rest`)
  - OpenAI-style tool schemas (`consolidation_memory.schemas.openai_tools`)

## Install

```bash
pip install consolidation-memory[fastembed]
```

Common extras:

- `consolidation-memory[rest]` for FastAPI endpoints
- `consolidation-memory[dashboard]` for the Textual dashboard
- `consolidation-memory[all,dev]` for full local development

## Quick Start

```bash
consolidation-memory init
consolidation-memory test
consolidation-memory serve
```

`consolidation-memory` with no subcommand defaults to `serve`.

## CLI Commands

```text
serve            Start MCP server (default command)
serve --rest     Start REST API
init             Interactive setup
test             End-to-end health check
status           Runtime/system stats
consolidate      Trigger consolidation run
detect-drift     Challenge claims impacted by changed files
export           Export full snapshot JSON
import PATH      Import snapshot JSON
reindex          Rebuild vectors with current embedding backend
browse           Browse knowledge topics
setup-memory     Add reusable memory instructions to an agent file
dashboard        Launch Textual dashboard
```

## MCP Setup

```json
{
  "mcpServers": {
    "consolidation_memory": {
      "command": "/absolute/path/to/python",
      "args": ["-m", "consolidation_memory", "--project", "default", "serve"],
      "env": {
        "PYTHONUNBUFFERED": "1",
        "CONSOLIDATION_MEMORY_IDLE_TIMEOUT_SECONDS": "0"
      }
    }
  }
}
```

Prefer an exact Python interpreter over the `consolidation-memory` console script. It avoids PATH/env drift and is more reliable on Windows when MCP hosts restart the server.
For long-lived MCP hosts, keep `CONSOLIDATION_MEMORY_IDLE_TIMEOUT_SECONDS=0` unless you explicitly want the server to auto-exit when idle.

MCP tools exposed by `server.py`:

- `memory_store`
- `memory_recall`
- `memory_store_batch`
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
- `memory_consolidation_log`
- `memory_decay_report`
- `memory_protect`
- `memory_timeline`
- `memory_contradictions`
- `memory_browse`
- `memory_read_topic`

## Python Example

```python
from consolidation_memory import MemoryClient

with MemoryClient(auto_consolidate=False) as mem:
    mem.store(
        "User prefers short PR summaries with concrete file paths.",
        content_type="preference",
        tags=["workflow", "reviews"],
    )

    result = mem.recall(
        "how should I format PR summaries?",
        n_results=5,
        include_knowledge=True,
    )

    print(len(result.episodes), len(result.knowledge), len(result.records), len(result.claims))
```

## REST API

Run:

```bash
pip install consolidation-memory[rest]
consolidation-memory serve --rest --host 127.0.0.1 --port 8080
```

For non-loopback binds (for example `--host 0.0.0.0`), set auth first:

```bash
export CONSOLIDATION_MEMORY_REST_AUTH_TOKEN="change-me"
consolidation-memory serve --rest --host 0.0.0.0 --port 8080
```

When auth is enabled, send `Authorization: Bearer <token>` on all endpoints except `/health`.

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

## OpenAI-Compatible Tools

Use:

- `consolidation_memory.schemas.openai_tools`
- `consolidation_memory.schemas.dispatch_tool_call`

This keeps tool definitions and dispatch behavior aligned with the same semantics used by MCP and REST.

## Scope Model (Compatibility + Shared Use)

By default, existing single-project usage still works.

When a scope envelope is provided, records are persisted with explicit scope dimensions:

- `namespace_*`
- `project_*`
- `app_client_*`
- `agent_*`
- `session_*`

This allows selective sharing without mixing unrelated contexts.

Optional `scope.policy` controls:

- `read_visibility`: `private` (default), `project`, `namespace`
- `write_mode`: `allow` (default), `deny`

Persisted ACL entities are also supported (`access_policies`, `policy_principals`, `policy_acl_entries`).
When persisted ACL rows match the resolved scope/principal, they are authoritative over `scope.policy`.
Conflict rules: write `deny` overrides `allow`; read visibility resolves to the most restrictive level.

## Storage Layout

Data is under `platformdirs.user_data_dir("consolidation_memory")/projects/<project>/`.

```text
memory.db
faiss_index.bin
faiss_id_map.json
faiss_tombstones.json
.faiss_reload
knowledge/
knowledge/versions/
consolidation_logs/
backups/
```

## Configuration

Config file discovery:

1. `CONSOLIDATION_MEMORY_CONFIG`
2. Platform default config path
3. Built-in defaults

Every scalar field can be overridden with `CONSOLIDATION_MEMORY_<FIELD_NAME>`.

Examples:

```bash
CONSOLIDATION_MEMORY_PROJECT=work
CONSOLIDATION_MEMORY_EMBEDDING_BACKEND=fastembed
CONSOLIDATION_MEMORY_LLM_BACKEND=ollama
CONSOLIDATION_MEMORY_CONSOLIDATION_INTERVAL_HOURS=6
```

## Documentation Map

- [Architecture](docs/ARCHITECTURE.md)
- [Roadmap](docs/ROADMAP.md)
- [Release Gates](docs/RELEASE_GATES.md)
- [Novelty Metrics](docs/NOVELTY_METRICS.md)
- [Novelty Eval Guide](docs/NOVELTY_EVAL_GUIDE.md)
- [Builder Baseline](docs/BUILDER_BASELINE.md)
- [External Review Playbook](docs/EXTERNAL_REVIEW_PLAYBOOK.md)
- [Recommended Agent Instructions](docs/recommended-agent-instructions.md)
- [Universal-memory strategy docs](docs/strategy/)

## Development

```bash
git clone https://github.com/charliee1w/consolidation-memory
cd consolidation-memory
pip install -e ".[all,dev]"
python scripts/smoke_builder_base.py
pytest tests/ -q
ruff check src/ tests/
mypy src/consolidation_memory/
```

## Community

- Contributors: [CONTRIBUTORS.md](CONTRIBUTORS.md)
- Issues: [GitHub Issues](https://github.com/charliee1w/consolidation-memory/issues)
- Discussions: [GitHub Discussions](https://github.com/charliee1w/consolidation-memory/discussions)

## License, Etc.
Project policies:
- [Security](SECURITY.md)
- [Contributing](CONTRIBUTING.md)
- [Code of Conduct](CODE_OF_CONDUCT.md)
MIT
