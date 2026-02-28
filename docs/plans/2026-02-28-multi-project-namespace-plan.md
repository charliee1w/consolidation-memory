# Multi-Project Namespace Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Isolate SQLite DB, FAISS index, and knowledge files into per-project subdirectories so memories from different projects don't cross-contaminate.

**Architecture:** `DATA_DIR/projects/<name>/` holds all per-project data. Project resolved once at process startup from `--project` CLI flag > `CONSOLIDATION_MEMORY_PROJECT` env var > `"default"`. Consumer modules (`database.py`, `vector_store.py`, `consolidation.py`, `context_assembler.py`) that import path constants at module level are refactored to access `config.ATTR` dynamically so `set_active_project()` propagates correctly.

**Tech Stack:** Python 3.10+, pytest, argparse, TOML config, FAISS, SQLite

---

### Task 1: config.py — Add project validation and path recalculation

**Files:**
- Modify: `src/consolidation_memory/config.py`
- Test: `tests/test_project_isolation.py`

**Step 1: Write the failing tests for project name validation**

Create `tests/test_project_isolation.py`:

```python
"""Tests for multi-project namespace isolation."""

import re

import pytest


class TestProjectNameValidation:
    """Test validate_project_name()."""

    def test_valid_simple_name(self):
        from consolidation_memory.config import validate_project_name
        assert validate_project_name("work") == "work"

    def test_valid_with_hyphens(self):
        from consolidation_memory.config import validate_project_name
        assert validate_project_name("my-project") == "my-project"

    def test_valid_with_underscores(self):
        from consolidation_memory.config import validate_project_name
        assert validate_project_name("my_project") == "my_project"

    def test_valid_with_numbers(self):
        from consolidation_memory.config import validate_project_name
        assert validate_project_name("project2") == "project2"

    def test_invalid_empty(self):
        from consolidation_memory.config import validate_project_name
        with pytest.raises(ValueError, match="empty"):
            validate_project_name("")

    def test_invalid_special_chars(self):
        from consolidation_memory.config import validate_project_name
        with pytest.raises(ValueError, match="Invalid project name"):
            validate_project_name("my project!")

    def test_invalid_uppercase(self):
        from consolidation_memory.config import validate_project_name
        with pytest.raises(ValueError, match="Invalid project name"):
            validate_project_name("MyProject")

    def test_invalid_too_long(self):
        from consolidation_memory.config import validate_project_name
        with pytest.raises(ValueError, match="Invalid project name"):
            validate_project_name("a" * 65)

    def test_invalid_starts_with_hyphen(self):
        from consolidation_memory.config import validate_project_name
        with pytest.raises(ValueError, match="Invalid project name"):
            validate_project_name("-project")

    def test_invalid_path_traversal(self):
        from consolidation_memory.config import validate_project_name
        with pytest.raises(ValueError, match="Invalid project name"):
            validate_project_name("../evil")
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_project_isolation.py::TestProjectNameValidation -v`
Expected: FAIL — `validate_project_name` does not exist yet

**Step 3: Implement project validation and set_active_project in config.py**

Add to the end of `config.py` (before `_validate_config()` call), replacing the existing path block:

```python
import re as _re

_PROJECT_NAME_RE = _re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


def validate_project_name(name: str) -> str:
    """Validate and return a project name. Raises ValueError on invalid."""
    if not name:
        raise ValueError("Project name cannot be empty")
    if not _PROJECT_NAME_RE.match(name):
        raise ValueError(
            f"Invalid project name {name!r}. "
            "Must be 1-64 chars, lowercase alphanumeric/hyphens/underscores, "
            "starting with alphanumeric."
        )
    return name


def get_active_project() -> str:
    """Return the active project name."""
    return _active_project


def set_active_project(name: str | None = None) -> str:
    """Set active project and recalculate all path constants.

    Args:
        name: Project name, or None to resolve from env var / default.

    Returns:
        The resolved project name.
    """
    global _active_project
    global DATA_DIR, DB_PATH, FAISS_INDEX_PATH, FAISS_ID_MAP_PATH
    global FAISS_TOMBSTONE_PATH, FAISS_RELOAD_SIGNAL
    global KNOWLEDGE_DIR, KNOWLEDGE_VERSIONS_DIR
    global CONSOLIDATION_LOG_DIR, BACKUP_DIR

    if name is None:
        name = os.environ.get("CONSOLIDATION_MEMORY_PROJECT", "default")

    name = validate_project_name(name)
    _active_project = name

    DATA_DIR = _base_data_dir / "projects" / name
    DB_PATH = DATA_DIR / "memory.db"
    FAISS_INDEX_PATH = DATA_DIR / "faiss_index.bin"
    FAISS_ID_MAP_PATH = DATA_DIR / "faiss_id_map.json"
    FAISS_TOMBSTONE_PATH = DATA_DIR / "faiss_tombstones.json"
    FAISS_RELOAD_SIGNAL = DATA_DIR / ".faiss_reload"
    KNOWLEDGE_DIR = DATA_DIR / "knowledge"
    KNOWLEDGE_VERSIONS_DIR = KNOWLEDGE_DIR / "versions"
    CONSOLIDATION_LOG_DIR = DATA_DIR / "consolidation_logs"
    BACKUP_DIR = DATA_DIR / "backups"

    return name
```

Also refactor the existing path block at the top to save `_base_data_dir`:

```python
_data_dir_str = _paths.get("data_dir", "")
_base_data_dir = Path(_data_dir_str).expanduser().resolve() if _data_dir_str else _default_data

# Initialize project from env var
_active_project = os.environ.get("CONSOLIDATION_MEMORY_PROJECT", "default")
try:
    validate_project_name(_active_project)
except ValueError:
    _active_project = "default"

DATA_DIR = _base_data_dir / "projects" / _active_project
DB_PATH = DATA_DIR / "memory.db"
FAISS_INDEX_PATH = DATA_DIR / "faiss_index.bin"
FAISS_ID_MAP_PATH = DATA_DIR / "faiss_id_map.json"
FAISS_TOMBSTONE_PATH = DATA_DIR / "faiss_tombstones.json"
FAISS_RELOAD_SIGNAL = DATA_DIR / ".faiss_reload"
KNOWLEDGE_DIR = DATA_DIR / "knowledge"
CONSOLIDATION_LOG_DIR = DATA_DIR / "consolidation_logs"
LOG_DIR = _base_data_dir / "logs"         # shared across projects
BACKUP_DIR = DATA_DIR / "backups"
```

Note: `LOG_DIR` stays at `_base_data_dir / "logs"` (shared). `KNOWLEDGE_VERSIONS_DIR` derived from `KNOWLEDGE_DIR`.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_project_isolation.py::TestProjectNameValidation -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/consolidation_memory/config.py tests/test_project_isolation.py
git commit -m "feat: add project name validation and set_active_project to config"
```

---

### Task 2: Refactor consumer modules to use dynamic config path access

**Files:**
- Modify: `src/consolidation_memory/database.py`
- Modify: `src/consolidation_memory/vector_store.py`
- Modify: `src/consolidation_memory/consolidation.py`
- Modify: `src/consolidation_memory/context_assembler.py`

**Context:** These modules import path constants at module level (`from config import DB_PATH`). This creates a local copy — when `set_active_project()` updates `config.DB_PATH`, the local copy is stale. Fix: access `config.DB_PATH` dynamically.

**Step 1: Refactor database.py**

Change line 17 from:
```python
from consolidation_memory.config import DB_PATH
```
to:
```python
from consolidation_memory import config as _config
```

Then replace all `DB_PATH` references with `_config.DB_PATH`. The main usage is in the connection function — grep for `DB_PATH` in the file and replace each occurrence.

**Step 2: Refactor vector_store.py**

Change lines 25-33 from:
```python
from consolidation_memory.config import (
    FAISS_INDEX_PATH,
    FAISS_ID_MAP_PATH,
    FAISS_TOMBSTONE_PATH,
    FAISS_RELOAD_SIGNAL,
    EMBEDDING_DIMENSION,
    FAISS_SEARCH_FETCH_K_PADDING,
    FAISS_SIZE_WARNING_THRESHOLD,
)
```
to:
```python
from consolidation_memory import config as _config
from consolidation_memory.config import (
    EMBEDDING_DIMENSION,
    FAISS_SEARCH_FETCH_K_PADDING,
    FAISS_SIZE_WARNING_THRESHOLD,
)
```

Replace all `FAISS_INDEX_PATH` → `_config.FAISS_INDEX_PATH`, `FAISS_ID_MAP_PATH` → `_config.FAISS_ID_MAP_PATH`, `FAISS_TOMBSTONE_PATH` → `_config.FAISS_TOMBSTONE_PATH`, `FAISS_RELOAD_SIGNAL` → `_config.FAISS_RELOAD_SIGNAL` throughout the file.

Non-path constants (`EMBEDDING_DIMENSION`, `FAISS_SEARCH_FETCH_K_PADDING`, `FAISS_SIZE_WARNING_THRESHOLD`) stay as local imports since they don't change per project.

**Step 3: Refactor consolidation.py**

Change lines 24-50: split path imports to use dynamic config access. Keep all non-path constants as local imports.

From:
```python
from consolidation_memory.config import (
    ...
    CONSOLIDATION_LOG_DIR,
    ...
    KNOWLEDGE_DIR,
    KNOWLEDGE_VERSIONS_DIR,
    ...
)
```

To:
```python
from consolidation_memory import config as _config
from consolidation_memory.config import (
    FAISS_COMPACTION_THRESHOLD,
    LLM_VALIDATION_RETRY,
    CONSOLIDATION_CLUSTER_THRESHOLD,
    CONSOLIDATION_MAX_CLUSTER_SIZE,
    # ... all non-path constants stay ...
)
```

Replace: `KNOWLEDGE_DIR` → `_config.KNOWLEDGE_DIR`, `KNOWLEDGE_VERSIONS_DIR` → `_config.KNOWLEDGE_VERSIONS_DIR`, `CONSOLIDATION_LOG_DIR` → `_config.CONSOLIDATION_LOG_DIR`.

**Step 4: Refactor context_assembler.py**

Change line 17: `KNOWLEDGE_DIR` → dynamic access.

From:
```python
from consolidation_memory.config import (
    CONSOLIDATION_PRIORITY_WEIGHTS,
    KNOWLEDGE_DIR,
    ...
)
```

To:
```python
from consolidation_memory import config as _config
from consolidation_memory.config import (
    CONSOLIDATION_PRIORITY_WEIGHTS,
    KNOWLEDGE_KEYWORD_WEIGHT,
    # ... all non-path constants stay ...
)
```

Replace `KNOWLEDGE_DIR` → `_config.KNOWLEDGE_DIR`.

**Step 5: Update conftest.py — remove stale consumer-module patches**

Remove patches that targeted consumer module path copies since they now read from config dynamically:

Remove these patches:
```python
patch("consolidation_memory.database.DB_PATH", db_path),
patch("consolidation_memory.vector_store.FAISS_INDEX_PATH", faiss_idx),
patch("consolidation_memory.vector_store.FAISS_ID_MAP_PATH", faiss_map),
patch("consolidation_memory.vector_store.FAISS_TOMBSTONE_PATH", faiss_tombstone),
patch("consolidation_memory.vector_store.FAISS_RELOAD_SIGNAL", faiss_reload),
patch("consolidation_memory.consolidation.KNOWLEDGE_DIR", knowledge),
patch("consolidation_memory.consolidation.KNOWLEDGE_VERSIONS_DIR", knowledge_versions),
patch("consolidation_memory.consolidation.CONSOLIDATION_LOG_DIR", consol_log),
```

The remaining patches for non-path constants (`EMBEDDING_DIMENSION`, `CONSOLIDATION_PRUNE_ENABLED`, tuning weights, etc.) stay because they're still local imports in their consumer modules.

**Step 6: Run full test suite**

Run: `pytest tests/ -v`
Expected: All 174 tests pass (7 skipped). This verifies the refactor didn't break anything.

**Step 7: Commit**

```bash
git add src/consolidation_memory/database.py src/consolidation_memory/vector_store.py \
    src/consolidation_memory/consolidation.py src/consolidation_memory/context_assembler.py \
    tests/conftest.py
git commit -m "refactor: consumer modules use dynamic config path access"
```

---

### Task 3: Add set_active_project path recalculation tests

**Files:**
- Modify: `tests/test_project_isolation.py`

**Step 1: Write tests for set_active_project**

Add to `tests/test_project_isolation.py`:

```python
class TestSetActiveProject:
    """Test set_active_project() recalculates paths."""

    def test_default_project(self, tmp_data_dir):
        from consolidation_memory import config
        config.set_active_project("default")
        assert config.get_active_project() == "default"
        assert "projects" in str(config.DATA_DIR)
        assert str(config.DATA_DIR).endswith("default")

    def test_custom_project(self, tmp_data_dir):
        from consolidation_memory import config
        config.set_active_project("work")
        assert config.get_active_project() == "work"
        assert str(config.DATA_DIR).endswith("work")
        assert str(config.DB_PATH).endswith("work/memory.db") or \
               str(config.DB_PATH).endswith("work\\memory.db")

    def test_all_paths_recalculated(self, tmp_data_dir):
        from consolidation_memory import config
        config.set_active_project("test-proj")
        proj_str = "test-proj"
        assert proj_str in str(config.DATA_DIR)
        assert proj_str in str(config.DB_PATH)
        assert proj_str in str(config.FAISS_INDEX_PATH)
        assert proj_str in str(config.FAISS_ID_MAP_PATH)
        assert proj_str in str(config.FAISS_TOMBSTONE_PATH)
        assert proj_str in str(config.KNOWLEDGE_DIR)
        assert proj_str in str(config.BACKUP_DIR)
        assert proj_str in str(config.CONSOLIDATION_LOG_DIR)

    def test_log_dir_shared(self, tmp_data_dir):
        """LOG_DIR should NOT contain project name — it's shared."""
        from consolidation_memory import config
        config.set_active_project("work")
        assert "work" not in str(config.LOG_DIR)

    def test_none_resolves_from_env_default(self, tmp_data_dir, monkeypatch):
        from consolidation_memory import config
        monkeypatch.delenv("CONSOLIDATION_MEMORY_PROJECT", raising=False)
        config.set_active_project(None)
        assert config.get_active_project() == "default"

    def test_none_resolves_from_env_var(self, tmp_data_dir, monkeypatch):
        from consolidation_memory import config
        monkeypatch.setenv("CONSOLIDATION_MEMORY_PROJECT", "fromenv")
        config.set_active_project(None)
        assert config.get_active_project() == "fromenv"

    def test_invalid_name_raises(self, tmp_data_dir):
        from consolidation_memory import config
        with pytest.raises(ValueError):
            config.set_active_project("INVALID!")
```

**Step 2: Run tests**

Run: `pytest tests/test_project_isolation.py::TestSetActiveProject -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/test_project_isolation.py
git commit -m "test: add set_active_project path recalculation tests"
```

---

### Task 4: Add project isolation integration tests

**Files:**
- Modify: `tests/test_project_isolation.py`

**Step 1: Write isolation tests**

```python
class TestProjectIsolation:
    """Store in project A should not appear in project B recalls."""

    def test_store_in_a_not_visible_in_b(self, tmp_data_dir):
        from consolidation_memory import config
        from consolidation_memory.client import MemoryClient

        # Store in project "alpha"
        config.set_active_project("alpha")
        with MemoryClient(auto_consolidate=False) as client_a:
            client_a.store("secret alpha fact", content_type="fact", tags=["alpha"])

        # Recall in project "beta" — should find nothing
        config.set_active_project("beta")
        with MemoryClient(auto_consolidate=False) as client_b:
            result = client_b.recall("secret alpha fact", n_results=10)
            assert len(result.episodes) == 0

    def test_store_in_a_visible_in_a(self, tmp_data_dir):
        from consolidation_memory import config
        from consolidation_memory.client import MemoryClient

        config.set_active_project("alpha")
        with MemoryClient(auto_consolidate=False) as client:
            client.store("visible alpha fact", content_type="fact", tags=["alpha"])

        # Re-open same project — should find it
        config.set_active_project("alpha")
        with MemoryClient(auto_consolidate=False) as client:
            result = client.recall("visible alpha fact", n_results=10)
            assert len(result.episodes) > 0
            assert "visible alpha fact" in result.episodes[0]["content"]

    def test_default_project_works(self, tmp_data_dir):
        from consolidation_memory import config
        from consolidation_memory.client import MemoryClient

        config.set_active_project("default")
        with MemoryClient(auto_consolidate=False) as client:
            store_result = client.store("default project fact", content_type="fact")
            assert store_result.status == "stored"
```

**Step 2: Run tests**

Run: `pytest tests/test_project_isolation.py::TestProjectIsolation -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/test_project_isolation.py
git commit -m "test: add project isolation integration tests"
```

---

### Task 5: cli.py — Add --project flag

**Files:**
- Modify: `src/consolidation_memory/cli.py`

**Step 1: Add --project to the top-level parser**

In `main()`, after `parser.add_argument("--version", ...)` add:

```python
parser.add_argument(
    "--project", "-p",
    default=None,
    help="Project namespace (default: CONSOLIDATION_MEMORY_PROJECT env var or 'default')",
)
```

Before the command dispatch block, add project activation:

```python
args = parser.parse_args()

# Activate project namespace before any command
if args.project or os.environ.get("CONSOLIDATION_MEMORY_PROJECT"):
    from consolidation_memory.config import set_active_project
    set_active_project(args.project)
```

**Step 2: Update cmd_status to show active project**

Add after the version line:
```python
from consolidation_memory.config import get_active_project
print(f"Project:     {get_active_project()}")
```

**Step 3: Run existing tests + manual verification**

Run: `pytest tests/ -v`
Expected: All pass

**Step 4: Commit**

```bash
git add src/consolidation_memory/cli.py
git commit -m "feat: add --project flag to CLI"
```

---

### Task 6: server.py — Log active project at startup

**Files:**
- Modify: `src/consolidation_memory/server.py`

**Step 1: Add project logging to lifespan**

In the `lifespan()` function, after the version log line add:

```python
from consolidation_memory.config import get_active_project
logger.info("Active project: %s", get_active_project())
```

**Step 2: Commit**

```bash
git add src/consolidation_memory/server.py
git commit -m "feat: log active project at MCP server startup"
```

---

### Task 7: rest.py — Log active project at startup

**Files:**
- Modify: `src/consolidation_memory/rest.py`

**Step 1: Add project info to lifespan and health endpoint**

In `create_app()` lifespan, log the active project. In `/health`, include project:

```python
@app.get("/health")
async def health():
    from consolidation_memory.config import get_active_project
    return {"status": "ok", "version": __version__, "project": get_active_project()}
```

**Step 2: Commit**

```bash
git add src/consolidation_memory/rest.py
git commit -m "feat: include active project in REST health endpoint"
```

---

### Task 8: Migration from flat layout

**Files:**
- Modify: `src/consolidation_memory/config.py`
- Modify: `tests/test_project_isolation.py`

**Step 1: Write migration test**

```python
class TestMigration:
    """Test auto-migration from flat DATA_DIR to projects/default/."""

    def test_migrate_flat_to_projects(self, tmp_path):
        from consolidation_memory.config import maybe_migrate_to_projects
        base = tmp_path / "data"
        base.mkdir()
        # Create flat layout
        (base / "memory.db").write_text("fake db")
        (base / "faiss_index.bin").write_bytes(b"fake index")
        (base / "faiss_id_map.json").write_text("[]")
        (base / "faiss_tombstones.json").write_text("[]")
        (base / "knowledge").mkdir()
        (base / "knowledge" / "topic.md").write_text("knowledge")
        (base / "backups").mkdir()
        (base / "consolidation_logs").mkdir()

        migrated = maybe_migrate_to_projects(base)
        assert migrated is True

        default_dir = base / "projects" / "default"
        assert (default_dir / "memory.db").exists()
        assert (default_dir / "faiss_index.bin").exists()
        assert (default_dir / "knowledge" / "topic.md").exists()
        assert not (base / "memory.db").exists()  # moved, not copied

    def test_no_migration_if_already_projects(self, tmp_path):
        from consolidation_memory.config import maybe_migrate_to_projects
        base = tmp_path / "data"
        base.mkdir()
        (base / "projects").mkdir()
        (base / "memory.db").write_text("fake db")

        migrated = maybe_migrate_to_projects(base)
        assert migrated is False  # skip: projects/ already exists

    def test_no_migration_if_no_flat_data(self, tmp_path):
        from consolidation_memory.config import maybe_migrate_to_projects
        base = tmp_path / "data"
        base.mkdir()

        migrated = maybe_migrate_to_projects(base)
        assert migrated is False  # nothing to migrate
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_project_isolation.py::TestMigration -v`
Expected: FAIL — `maybe_migrate_to_projects` does not exist

**Step 3: Implement migration in config.py**

```python
def maybe_migrate_to_projects(base_dir: Path) -> bool:
    """Migrate flat DATA_DIR layout to projects/default/ structure.

    Returns True if migration was performed.
    """
    flat_db = base_dir / "memory.db"
    projects_dir = base_dir / "projects"

    if not flat_db.exists():
        return False  # nothing to migrate

    if projects_dir.exists():
        return False  # already migrated or manual setup

    import shutil
    import sys

    default_dir = projects_dir / "default"
    default_dir.mkdir(parents=True)

    _FILES_TO_MOVE = [
        "memory.db", "faiss_index.bin", "faiss_id_map.json",
        "faiss_tombstones.json", ".faiss_reload",
    ]
    _DIRS_TO_MOVE = ["knowledge", "backups", "consolidation_logs"]

    for fname in _FILES_TO_MOVE:
        src = base_dir / fname
        if src.exists():
            shutil.move(str(src), str(default_dir / fname))

    for dname in _DIRS_TO_MOVE:
        src = base_dir / dname
        if src.exists():
            shutil.move(str(src), str(default_dir / dname))

    print(
        f"[consolidation-memory] Migrated data to {default_dir}",
        file=sys.stderr,
    )
    return True
```

Call `maybe_migrate_to_projects(_base_data_dir)` at the end of `config.py` init, after the path constants are set but before `_validate_config()`.

**Step 4: Run tests**

Run: `pytest tests/test_project_isolation.py::TestMigration -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/consolidation_memory/config.py tests/test_project_isolation.py
git commit -m "feat: add auto-migration from flat layout to projects/default"
```

---

### Task 9: Update conftest.py for project-aware paths

**Files:**
- Modify: `tests/conftest.py`

**Step 1: Update tmp_data_dir fixture**

The fixture currently creates `data/` and patches `DATA_DIR` to it. Now it should:
1. Create `data/projects/default/` structure
2. Patch `_base_data_dir` as well as `_active_project`
3. Keep patching `DATA_DIR` to `data/projects/default/`

Update the fixture so `data_dir = tmp_path / "data" / "projects" / "default"` and also patch `_base_data_dir` to `tmp_path / "data"` and `_active_project` to `"default"`.

**Step 2: Run full test suite**

Run: `pytest tests/ -v`
Expected: All pass

**Step 3: Commit**

```bash
git add tests/conftest.py
git commit -m "fix: update conftest for project-aware path layout"
```

---

### Task 10: Update README with multi-project docs

**Files:**
- Modify: `README.md`

**Step 1: Add Multi-Project Support section**

Add after the "CLI" section and before "Data Storage":

````markdown
## Multi-Project Support

Isolate memories per project — work memories stay in work, personal stays in personal.

```bash
# CLI flag
consolidation-memory --project work status
consolidation-memory --project personal serve --rest --port 8081

# Environment variable
CONSOLIDATION_MEMORY_PROJECT=work consolidation-memory serve
```

### MCP (Claude Desktop) — Multiple Projects

Add separate server entries per project:

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

### Data Layout

```
~/.local/share/consolidation_memory/
├── projects/
│   ├── default/        # used when no project specified
│   │   ├── memory.db
│   │   ├── faiss_index.bin
│   │   └── knowledge/
│   ├── work/
│   └── personal/
├── config.toml         # shared across projects
└── logs/               # shared
```

Existing users are auto-migrated to `projects/default/` on first run.
````

**Step 2: Update CLI table to include --project**

**Step 3: Update Data Storage section path table to mention projects/**

**Step 4: Commit**

```bash
git add README.md
git commit -m "docs: add multi-project namespace documentation"
```

---

### Task 11: Version bump

**Files:**
- Modify: `pyproject.toml`
- Modify: `CHANGELOG.md`

**Step 1: Bump version to 0.6.0 in pyproject.toml**

Change `version = "0.5.0"` to `version = "0.6.0"`.

**Step 2: Add CHANGELOG entry**

**Step 3: Run full test suite one final time**

Run: `pytest tests/ -v`
Expected: All pass

Run: `ruff check src/ tests/`
Expected: Clean

**Step 4: Commit and tag**

```bash
git add pyproject.toml CHANGELOG.md
git commit -m "v0.6.0: multi-project namespace support"
git tag v0.6.0
```

---

### Task 12: Push to GitHub

```bash
git push origin main --tags
```
