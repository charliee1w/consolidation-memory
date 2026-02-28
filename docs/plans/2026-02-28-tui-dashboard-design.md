# TUI Dashboard Design

**Date:** 2026-02-28
**Status:** Approved

## Overview

Add a Textual-based TUI dashboard to consolidation-memory, accessible via `consolidation-memory dashboard`. Provides read-only visibility into episodes, knowledge topics, consolidation history, and system stats.

## Architecture

Two new modules + CLI/config changes:

- `src/consolidation_memory/dashboard_data.py` -- Read-only SQLite data layer
- `src/consolidation_memory/dashboard.py` -- Textual TUI app (4 tabbed screens)
- `cli.py` modification -- New `dashboard` subcommand
- `pyproject.toml` modification -- `[dashboard]` optional dependency group

### Decision: Direct SQLite Access

The dashboard uses `database.py` functions and direct SQLite queries rather than `MemoryClient`. This avoids initializing FAISS/embedding backends, keeping startup instant and dependencies minimal. The dashboard is read-only so the full client is unnecessary.

## Data Layer (`dashboard_data.py`)

```python
class DashboardData:
    def __init__(self, db_path: Path | None = None)

    def get_episodes(self, sort_by="created_at", desc=True,
                     content_type=None, limit=500) -> list[dict]
    # Fields: id, content (truncated ~80 chars), content_type,
    #         tags, surprise_score, created_at, consolidated

    def get_knowledge_topics(self) -> list[dict]
    # Fields: filename, title, summary, fact_count, confidence,
    #         created_at, updated_at, source_episode_count

    def get_records_for_topic(self, topic_id: str) -> list[dict]
    # Fields: record_type, content, confidence, created_at

    def get_consolidation_runs(self, limit=100) -> list[dict]
    # Fields: id, started_at, completed_at, episodes_processed,
    #         clusters_formed, topics_created, topics_updated,
    #         episodes_pruned, status, error_message

    def get_stats(self) -> dict
    # Fields: episode counts by type, totals, knowledge topic count,
    #         record count, last consolidation time, db_size_mb

    def get_faiss_stats(self) -> dict
    # Fields: index_size, tombstone_count, tombstone_ratio
    # Reads FAISS metadata files directly, no FAISS import
```

Each method opens a short-lived read-only SQLite connection.

## TUI App (`dashboard.py`)

Textual app with `TabbedContent`, 4 tabs.

### Tab 1: Episodes Browser

- `DataTable` with columns: Content (preview), Type, Tags, Surprise, Created, Status
- Sortable by clicking column headers
- Color-coded consolidated status (pending/consolidated/pruned)
- Footer with total episode count

### Tab 2: Knowledge Topics

- `DataTable` listing topics: Filename, Title, Facts, Confidence, Updated
- Selecting a row shows records for that topic in a detail panel below

### Tab 3: Consolidation History

- `DataTable` of runs: Started, Completed, Episodes, Clusters, Created, Updated, Pruned, Status
- Color-coded status (running=yellow, completed=green, error=red)

### Tab 4: Memory Stats

- Static widgets in a grid layout showing:
  - Episode counts by type (exchange, fact, solution, preference)
  - FAISS index size and tombstone count
  - Last consolidation time
  - DB size on disk
- Auto-refreshes every 5 seconds via timer

### Global Keybindings

- `q` -- quit
- `r` -- refresh current tab
- `1-4` -- switch tabs

## CLI Integration

New subcommand in `cli.py`:

```python
def cmd_dashboard(args):
    try:
        from consolidation_memory.dashboard import DashboardApp
    except ImportError:
        print("Dashboard requires textual. Install: pip install consolidation-memory[dashboard]")
        sys.exit(1)
    app = DashboardApp()
    app.run()
```

## Dependencies

```toml
[project.optional-dependencies]
dashboard = ["textual>=1.0.0"]
```

## Testing

`tests/test_dashboard_data.py` -- Tests for every `DashboardData` method:

- `test_get_episodes_default` -- insert episodes, verify fields and sort order
- `test_get_episodes_filtered_by_type` -- content_type filtering
- `test_get_episodes_sort_options` -- sorting by different columns
- `test_get_knowledge_topics` -- insert topics, verify fields
- `test_get_records_for_topic` -- insert records, verify retrieval
- `test_get_consolidation_runs` -- insert runs, verify ordering
- `test_get_stats` -- stat aggregation correctness
- `test_get_faiss_stats_no_index` -- graceful when no FAISS files
- `test_empty_database` -- all methods return empty on fresh DB

No Textual/UI tests -- data layer only.

## Files Changed

| File | Change |
|------|--------|
| `src/consolidation_memory/dashboard_data.py` | New -- data fetching layer |
| `src/consolidation_memory/dashboard.py` | New -- Textual TUI app |
| `src/consolidation_memory/cli.py` | Add `dashboard` subcommand |
| `pyproject.toml` | Add `[dashboard]` optional deps |
| `tests/test_dashboard_data.py` | New -- data layer tests |
