# Multi-Project Namespace Support

## Problem

All episodes, knowledge, and indexes live in a single flat DATA_DIR. Users with multiple projects (work, side-project, personal) get cross-contamination — work memories appear in personal recalls and vice versa.

## Solution

Isolate SQLite DB, FAISS index, and knowledge files into per-project subdirectories under `DATA_DIR/projects/<name>/`. Default to `"default"` when unspecified.

## Directory Layout

```
DATA_DIR/
├── projects/
│   ├── default/
│   │   ├── memory.db
│   │   ├── faiss_index.bin
│   │   ├── faiss_id_map.json
│   │   ├── faiss_tombstones.json
│   │   ├── knowledge/
│   │   │   └── versions/
│   │   ├── backups/
│   │   └── consolidation_logs/
│   ├── work/
│   │   ├── memory.db
│   │   └── ...
│   └── side-project/
│       ├── memory.db
│       └── ...
├── config.toml          # shared
└── logs/                # shared
```

## Project Resolution Priority

1. `--project` CLI flag (highest)
2. `CONSOLIDATION_MEMORY_PROJECT` env var
3. `"default"` (fallback)

Project is set **once at process startup**, not per-request. Both MCP and REST are one-project-per-process. To serve multiple projects, run multiple instances.

## Project Name Validation

- Lowercase alphanumeric, hyphens, underscores
- 1-64 characters
- Regex: `^[a-z0-9][a-z0-9_-]{0,63}$`

## Changes by Module

### config.py

- Add `_active_project: str = "default"` module-level state
- Add `set_active_project(name: str)` — validates name, sets state, recalculates all path constants
- Add `get_active_project() -> str`
- Add `validate_project_name(name: str) -> str` — raises ValueError on invalid
- Add `get_project_data_dir(project: str | None = None) -> Path`
- Path constants (DB_PATH, FAISS_*_PATH, KNOWLEDGE_DIR, BACKUP_DIR, CONSOLIDATION_LOG_DIR) recalculated on `set_active_project()` call
- Read `CONSOLIDATION_MEMORY_PROJECT` env var during initial config load

### cli.py

- Add global `--project` argument to top-level parser
- All subcommands inherit it
- Call `set_active_project(args.project)` before dispatching to subcommand
- Falls back to env var via config.py defaults

### server.py (MCP)

- Read project from env var at startup (already handled by config.py init)
- No tool-level project parameter
- Multiple Claude Desktop entries use different env vars:
  ```json
  {
    "work-memory": {
      "command": "consolidation-memory",
      "args": ["serve"],
      "env": {"CONSOLIDATION_MEMORY_PROJECT": "work"}
    },
    "personal-memory": {
      "command": "consolidation-memory",
      "args": ["serve"],
      "env": {"CONSOLIDATION_MEMORY_PROJECT": "personal"}
    }
  }
  ```

### rest.py

- Project set at app startup via env var or CLI flag
- No per-request project switching (avoids concurrency issues with shared MemoryClient)
- Multiple instances on different ports for multiple projects

### client.py

- No changes needed — already reads paths from config module

## Migration

On startup, if `DATA_DIR/memory.db` exists (old flat layout) and `DATA_DIR/projects/` does not exist:
1. Create `DATA_DIR/projects/default/`
2. Move memory.db, faiss_*.{bin,json}, knowledge/, backups/, consolidation_logs/ into projects/default/
3. Log migration message to stderr

If both flat and projects/ exist, skip migration (manual resolution needed).

## Tests

New `tests/test_project_isolation.py`:
- Store in project A, recall in project B → empty results
- Store in project A, recall in project A → finds results
- Default project when unspecified
- Invalid project names rejected (special chars, empty, too long)
- `set_active_project()` recalculates all paths correctly
- Migration from flat layout to projects/default/

Update `tests/conftest.py`:
- Patch `_active_project` to `"default"`
- Update path patches to use projects/default/ subdirectory

## README Updates

- New "Multi-Project Support" section with examples
- CLI usage: `consolidation-memory --project work status`
- Env var: `CONSOLIDATION_MEMORY_PROJECT=work consolidation-memory serve`
- Claude Desktop config with multiple project servers
- REST: multiple instances on different ports

## Version

This is a minor feature addition. Bump to v0.6.0.
