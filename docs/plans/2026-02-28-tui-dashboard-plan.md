# TUI Dashboard Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a Textual TUI dashboard to consolidation-memory with episodes browser, knowledge topics, consolidation history, and memory stats tabs.

**Architecture:** Two new modules (`dashboard_data.py` for read-only SQLite queries, `dashboard.py` for Textual app) plus CLI integration. Direct SQLite access avoids FAISS/embedding startup.

**Tech Stack:** Textual >= 1.0.0, SQLite (via existing database.py), existing config.py for path resolution.

---

### Task 1: Add `textual` to optional dependencies

**Files:**
- Modify: `pyproject.toml:35-40`

**Step 1: Add dashboard optional dependency**

In `pyproject.toml`, add the `dashboard` extra after the existing `rest` line (line 38) and update the `all` extra:

```toml
[project.optional-dependencies]
fastembed = ["fastembed>=0.4.0"]
openai = ["openai>=1.0.0"]
rest = ["fastapi>=0.115.0", "uvicorn[standard]>=0.34.0"]
dashboard = ["textual>=1.0.0"]
all = ["consolidation-memory[fastembed,openai,rest,dashboard]"]
dev = ["pytest>=8.0.0", "ruff>=0.7.0", "httpx>=0.28.0"]
```

**Step 2: Install the dashboard extra**

Run: `pip install -e "D:\consolidation-memory[dashboard]"`
Expected: textual installed successfully

**Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "feat: add textual as optional dashboard dependency"
```

---

### Task 2: Create the data layer with tests (TDD)

**Files:**
- Create: `src/consolidation_memory/dashboard_data.py`
- Create: `tests/test_dashboard_data.py`

**Step 1: Write all data layer tests**

Create `tests/test_dashboard_data.py`. Tests use the existing `tmp_data_dir` autouse fixture from `conftest.py` which patches all config paths to temp directories and calls `ensure_schema()` implicitly via database functions.

```python
"""Tests for dashboard data-fetching layer."""

import json
from datetime import datetime, timezone

import pytest

from consolidation_memory.database import (
    ensure_schema,
    insert_episode,
    upsert_knowledge_topic,
    insert_knowledge_records,
    start_consolidation_run,
    complete_consolidation_run,
)
from consolidation_memory.dashboard_data import DashboardData


@pytest.fixture
def data():
    """Create DashboardData instance with schema initialized."""
    ensure_schema()
    return DashboardData()


class TestGetEpisodes:
    def test_default_returns_episodes_sorted_by_created_at_desc(self, data):
        ensure_schema()
        id1 = insert_episode("first episode", content_type="fact")
        id2 = insert_episode("second episode", content_type="exchange")
        id3 = insert_episode("third episode", content_type="solution")

        episodes = data.get_episodes()
        assert len(episodes) == 3
        # Most recent first
        assert episodes[0]["id"] == id3
        assert episodes[1]["id"] == id2
        assert episodes[2]["id"] == id1

    def test_returns_expected_fields(self, data):
        insert_episode(
            "test content here",
            content_type="fact",
            tags=["python", "tips"],
            surprise_score=0.7,
        )

        episodes = data.get_episodes()
        assert len(episodes) == 1
        ep = episodes[0]
        assert "id" in ep
        assert "content_preview" in ep
        assert "content_type" in ep
        assert ep["content_type"] == "fact"
        assert "tags" in ep
        assert ep["tags"] == ["python", "tips"]
        assert "surprise_score" in ep
        assert ep["surprise_score"] == pytest.approx(0.7)
        assert "created_at" in ep
        assert "consolidated" in ep

    def test_content_preview_truncated(self, data):
        long_content = "x" * 200
        insert_episode(long_content)

        episodes = data.get_episodes()
        assert len(episodes[0]["content_preview"]) <= 83  # 80 + "..."

    def test_filter_by_content_type(self, data):
        insert_episode("fact episode", content_type="fact")
        insert_episode("exchange episode", content_type="exchange")
        insert_episode("another fact", content_type="fact")

        facts = data.get_episodes(content_type="fact")
        assert len(facts) == 2
        assert all(e["content_type"] == "fact" for e in facts)

    def test_sort_by_surprise_score(self, data):
        insert_episode("low surprise", surprise_score=0.1)
        insert_episode("high surprise", surprise_score=0.9)
        insert_episode("mid surprise", surprise_score=0.5)

        episodes = data.get_episodes(sort_by="surprise_score", desc=True)
        scores = [e["surprise_score"] for e in episodes]
        assert scores == sorted(scores, reverse=True)

    def test_sort_by_content_type(self, data):
        insert_episode("z episode", content_type="solution")
        insert_episode("a episode", content_type="exchange")
        insert_episode("m episode", content_type="fact")

        episodes = data.get_episodes(sort_by="content_type", desc=False)
        types = [e["content_type"] for e in episodes]
        assert types == sorted(types)

    def test_limit(self, data):
        for i in range(10):
            insert_episode(f"episode {i}")

        episodes = data.get_episodes(limit=3)
        assert len(episodes) == 3

    def test_excludes_deleted(self, data):
        from consolidation_memory.database import soft_delete_episode

        id1 = insert_episode("keep me")
        id2 = insert_episode("delete me")
        soft_delete_episode(id2)

        episodes = data.get_episodes()
        assert len(episodes) == 1
        assert episodes[0]["id"] == id1


class TestGetKnowledgeTopics:
    def test_returns_topics_with_fields(self, data):
        topic_id = upsert_knowledge_topic(
            filename="python_tips.md",
            title="Python Tips",
            summary="Tips for Python development",
            source_episodes=["ep1", "ep2", "ep3"],
            fact_count=5,
            confidence=0.85,
        )

        topics = data.get_knowledge_topics()
        assert len(topics) == 1
        t = topics[0]
        assert t["id"] == topic_id
        assert t["filename"] == "python_tips.md"
        assert t["title"] == "Python Tips"
        assert t["summary"] == "Tips for Python development"
        assert t["fact_count"] == 5
        assert t["confidence"] == pytest.approx(0.85)
        assert t["source_episode_count"] == 3
        assert "created_at" in t
        assert "updated_at" in t

    def test_ordered_by_updated_at_desc(self, data):
        upsert_knowledge_topic("a.md", "A", "summary A", ["ep1"])
        upsert_knowledge_topic("b.md", "B", "summary B", ["ep2"])
        # Update A to make it most recent
        upsert_knowledge_topic("a.md", "A Updated", "new summary", ["ep3"])

        topics = data.get_knowledge_topics()
        assert topics[0]["filename"] == "a.md"
        assert topics[1]["filename"] == "b.md"


class TestGetRecordsForTopic:
    def test_returns_records(self, data):
        topic_id = upsert_knowledge_topic("test.md", "Test", "summary", ["ep1"])
        insert_knowledge_records(topic_id, [
            {"record_type": "fact", "content": '{"key": "value"}', "embedding_text": "test fact", "confidence": 0.9},
            {"record_type": "solution", "content": '{"fix": "it"}', "embedding_text": "test solution"},
        ], source_episodes=["ep1"])

        records = data.get_records_for_topic(topic_id)
        assert len(records) == 2
        assert records[0]["record_type"] in ("fact", "solution")
        assert "content" in records[0]
        assert "confidence" in records[0]
        assert "created_at" in records[0]

    def test_empty_for_nonexistent_topic(self, data):
        records = data.get_records_for_topic("nonexistent-id")
        assert records == []


class TestGetConsolidationRuns:
    def test_returns_runs_newest_first(self, data):
        run1 = start_consolidation_run()
        complete_consolidation_run(
            run1, episodes_processed=10, clusters_formed=3,
            topics_created=2, topics_updated=1, episodes_pruned=5,
        )
        run2 = start_consolidation_run()
        complete_consolidation_run(
            run2, episodes_processed=20, clusters_formed=5,
            topics_created=1, topics_updated=3,
        )

        runs = data.get_consolidation_runs()
        assert len(runs) == 2
        assert runs[0]["id"] == run2  # newest first

        r = runs[0]
        assert r["episodes_processed"] == 20
        assert r["clusters_formed"] == 5
        assert r["topics_created"] == 1
        assert r["topics_updated"] == 3
        assert r["status"] == "completed"
        assert "started_at" in r
        assert "completed_at" in r

    def test_limit(self, data):
        for _ in range(5):
            rid = start_consolidation_run()
            complete_consolidation_run(rid)

        runs = data.get_consolidation_runs(limit=2)
        assert len(runs) == 2

    def test_includes_error_runs(self, data):
        rid = start_consolidation_run()
        complete_consolidation_run(rid, status="error", error_message="LLM timeout")

        runs = data.get_consolidation_runs()
        assert len(runs) == 1
        assert runs[0]["status"] == "error"
        assert runs[0]["error_message"] == "LLM timeout"


class TestGetStats:
    def test_episode_counts_by_type(self, data):
        insert_episode("e1", content_type="exchange")
        insert_episode("e2", content_type="exchange")
        insert_episode("e3", content_type="fact")
        insert_episode("e4", content_type="solution")
        insert_episode("e5", content_type="preference")

        stats = data.get_stats()
        assert stats["episodes_by_type"]["exchange"] == 2
        assert stats["episodes_by_type"]["fact"] == 1
        assert stats["episodes_by_type"]["solution"] == 1
        assert stats["episodes_by_type"]["preference"] == 1
        assert stats["total_episodes"] == 5

    def test_knowledge_and_record_counts(self, data):
        tid = upsert_knowledge_topic("t.md", "T", "s", ["ep1"], fact_count=3)
        insert_knowledge_records(tid, [
            {"record_type": "fact", "content": "{}", "embedding_text": "f"},
            {"record_type": "solution", "content": "{}", "embedding_text": "s"},
        ])

        stats = data.get_stats()
        assert stats["knowledge_topic_count"] == 1
        assert stats["record_count"] == 2

    def test_db_size_present(self, data):
        stats = data.get_stats()
        assert "db_size_mb" in stats
        assert isinstance(stats["db_size_mb"], float)

    def test_last_consolidation(self, data):
        rid = start_consolidation_run()
        complete_consolidation_run(rid)

        stats = data.get_stats()
        assert stats["last_consolidation"] is not None

    def test_empty_db(self, data):
        stats = data.get_stats()
        assert stats["total_episodes"] == 0
        assert stats["knowledge_topic_count"] == 0
        assert stats["record_count"] == 0
        assert stats["last_consolidation"] is None


class TestGetFaissStats:
    def test_no_index_files(self, data):
        stats = data.get_faiss_stats()
        assert stats["index_size"] == 0
        assert stats["tombstone_count"] == 0
        assert stats["tombstone_ratio"] == 0.0

    def test_reads_from_metadata_files(self, data, tmp_data_dir):
        from consolidation_memory import config

        # Write mock FAISS metadata files
        id_map = ["id1", "id2", "id3", "id4", "id5"]
        config.FAISS_ID_MAP_PATH.write_text(json.dumps(id_map))

        tombstones = ["id2", "id4"]
        config.FAISS_TOMBSTONE_PATH.write_text(json.dumps(tombstones))

        stats = data.get_faiss_stats()
        assert stats["index_size"] == 3  # 5 total - 2 tombstones
        assert stats["tombstone_count"] == 2
        assert stats["tombstone_ratio"] == pytest.approx(2 / 5)


class TestEmptyDatabase:
    def test_all_methods_return_empty(self, data):
        assert data.get_episodes() == []
        assert data.get_knowledge_topics() == []
        assert data.get_records_for_topic("any-id") == []
        assert data.get_consolidation_runs() == []

        stats = data.get_stats()
        assert stats["total_episodes"] == 0

        faiss = data.get_faiss_stats()
        assert faiss["index_size"] == 0
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_dashboard_data.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'consolidation_memory.dashboard_data'`

**Step 3: Write the dashboard_data module**

Create `src/consolidation_memory/dashboard_data.py`:

```python
"""Read-only data layer for the TUI dashboard.

Queries SQLite directly via database.py to avoid initializing
FAISS/embedding backends. All methods return plain dicts/lists.
"""

import json
from pathlib import Path

from consolidation_memory import config as _config
from consolidation_memory.database import ensure_schema, get_connection


class DashboardData:
    """Lightweight read-only data access for the dashboard."""

    def __init__(self) -> None:
        ensure_schema()

    def get_episodes(
        self,
        sort_by: str = "created_at",
        desc: bool = True,
        content_type: str | None = None,
        limit: int = 500,
    ) -> list[dict]:
        """Fetch episodes for the browser table.

        Returns dicts with: id, content_preview, content_type, tags,
        surprise_score, created_at, consolidated.
        """
        allowed_sorts = {
            "created_at", "content_type", "surprise_score", "consolidated",
        }
        if sort_by not in allowed_sorts:
            sort_by = "created_at"

        direction = "DESC" if desc else "ASC"
        conditions = ["deleted = 0"]
        params: list = []

        if content_type:
            conditions.append("content_type = ?")
            params.append(content_type)

        where = " AND ".join(conditions)
        sql = (
            f"SELECT id, content, content_type, tags, surprise_score, "
            f"created_at, consolidated "
            f"FROM episodes WHERE {where} "
            f"ORDER BY {sort_by} {direction} LIMIT ?"
        )
        params.append(limit)

        with get_connection() as conn:
            rows = conn.execute(sql, params).fetchall()

        results = []
        for row in rows:
            r = dict(row)
            content = r.pop("content")
            r["content_preview"] = (
                content[:80] + "..." if len(content) > 80 else content
            )
            tags_raw = r["tags"]
            r["tags"] = json.loads(tags_raw) if isinstance(tags_raw, str) else tags_raw
            results.append(r)
        return results

    def get_knowledge_topics(self) -> list[dict]:
        """Fetch all knowledge topics with source episode counts."""
        with get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM knowledge_topics ORDER BY updated_at DESC"
            ).fetchall()

        results = []
        for row in rows:
            r = dict(row)
            src = r.get("source_episodes", "[]")
            episodes = json.loads(src) if isinstance(src, str) else src
            r["source_episode_count"] = len(episodes)
            results.append(r)
        return results

    def get_records_for_topic(self, topic_id: str) -> list[dict]:
        """Fetch active records for a specific knowledge topic."""
        with get_connection() as conn:
            rows = conn.execute(
                "SELECT record_type, content, confidence, created_at "
                "FROM knowledge_records WHERE topic_id = ? AND deleted = 0",
                (topic_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_consolidation_runs(self, limit: int = 100) -> list[dict]:
        """Fetch consolidation run history, newest first."""
        with get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM consolidation_runs ORDER BY started_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_stats(self) -> dict:
        """Aggregate memory statistics for the stats tab."""
        with get_connection() as conn:
            # Episode counts by type
            type_rows = conn.execute(
                "SELECT content_type, COUNT(*) as cnt "
                "FROM episodes WHERE deleted = 0 "
                "GROUP BY content_type"
            ).fetchall()
            episodes_by_type = {row["content_type"]: row["cnt"] for row in type_rows}

            total_episodes = sum(episodes_by_type.values())

            # Knowledge topics
            kt_row = conn.execute(
                "SELECT COUNT(*) as cnt FROM knowledge_topics"
            ).fetchone()
            knowledge_topic_count = kt_row["cnt"] if kt_row else 0

            # Knowledge records
            rec_row = conn.execute(
                "SELECT COUNT(*) as cnt FROM knowledge_records WHERE deleted = 0"
            ).fetchone()
            record_count = rec_row["cnt"] if rec_row else 0

            # Last consolidation
            last_run = conn.execute(
                "SELECT * FROM consolidation_runs "
                "ORDER BY started_at DESC LIMIT 1"
            ).fetchone()

        # DB size
        db_size_mb = 0.0
        if _config.DB_PATH.exists():
            db_size_mb = round(
                _config.DB_PATH.stat().st_size / (1024 * 1024), 2
            )

        return {
            "episodes_by_type": episodes_by_type,
            "total_episodes": total_episodes,
            "knowledge_topic_count": knowledge_topic_count,
            "record_count": record_count,
            "db_size_mb": db_size_mb,
            "last_consolidation": dict(last_run) if last_run else None,
        }

    def get_faiss_stats(self) -> dict:
        """Read FAISS metadata without importing faiss.

        Parses the JSON sidecar files to get index size and tombstone info.
        """
        id_map_path = _config.FAISS_ID_MAP_PATH
        tombstone_path = _config.FAISS_TOMBSTONE_PATH

        total = 0
        tombstones = 0

        if id_map_path.exists():
            try:
                ids = json.loads(id_map_path.read_text(encoding="utf-8"))
                total = len(ids)
            except (json.JSONDecodeError, OSError):
                pass

        if tombstone_path.exists():
            try:
                tombs = json.loads(tombstone_path.read_text(encoding="utf-8"))
                tombstones = len(tombs)
            except (json.JSONDecodeError, OSError):
                pass

        active = max(0, total - tombstones)
        ratio = tombstones / total if total > 0 else 0.0

        return {
            "index_size": active,
            "tombstone_count": tombstones,
            "tombstone_ratio": ratio,
        }
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_dashboard_data.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/consolidation_memory/dashboard_data.py tests/test_dashboard_data.py
git commit -m "feat: add dashboard data layer with tests"
```

---

### Task 3: Create the Textual TUI app

**Files:**
- Create: `src/consolidation_memory/dashboard.py`

**Step 1: Write the Textual dashboard app**

Create `src/consolidation_memory/dashboard.py`:

```python
"""TUI dashboard for consolidation-memory.

Requires: pip install consolidation-memory[dashboard]
"""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import (
    DataTable,
    Footer,
    Header,
    Label,
    Static,
    TabbedContent,
    TabPane,
)
from textual.timer import Timer

from consolidation_memory.dashboard_data import DashboardData


# ── Helpers ──────────────────────────────────────────────────────────────────

_CONSOLIDATED_LABELS = {0: "Pending", 1: "Consolidated", 2: "Pruned"}

_STATUS_STYLE = {
    "running": "bold yellow",
    "completed": "bold green",
    "error": "bold red",
}


def _fmt_ts(ts: str | None) -> str:
    """Format an ISO timestamp for display (drop sub-second precision)."""
    if not ts:
        return "-"
    return ts.replace("T", " ")[:19]


# ── Tab: Episodes ────────────────────────────────────────────────────────────

class EpisodesTab(Container):
    """Sortable episodes browser."""

    def compose(self) -> ComposeResult:
        yield DataTable(id="episodes-table")
        yield Label("", id="episodes-footer")

    def on_mount(self) -> None:
        table = self.query_one("#episodes-table", DataTable)
        table.cursor_type = "row"
        table.add_columns(
            "Content", "Type", "Tags", "Surprise", "Created", "Status",
        )
        self.load_data()

    def load_data(self, sort_by: str = "created_at", desc: bool = True) -> None:
        data = DashboardData()
        episodes = data.get_episodes(sort_by=sort_by, desc=desc)

        table = self.query_one("#episodes-table", DataTable)
        table.clear()

        for ep in episodes:
            tags = ", ".join(ep["tags"]) if ep["tags"] else ""
            status = _CONSOLIDATED_LABELS.get(ep["consolidated"], "?")
            table.add_row(
                ep["content_preview"],
                ep["content_type"],
                tags,
                f"{ep['surprise_score']:.2f}",
                _fmt_ts(ep["created_at"]),
                status,
            )

        footer = self.query_one("#episodes-footer", Label)
        footer.update(f" {len(episodes)} episodes")

    def on_data_table_header_selected(self, event: DataTable.HeaderSelected) -> None:
        col_map = {
            "Content": "created_at",  # Can't sort by preview, fall back
            "Type": "content_type",
            "Tags": "created_at",
            "Surprise": "surprise_score",
            "Created": "created_at",
            "Status": "consolidated",
        }
        sort_by = col_map.get(str(event.label), "created_at")
        self.load_data(sort_by=sort_by)


# ── Tab: Knowledge Topics ────────────────────────────────────────────────────

class KnowledgeTab(Container):
    """Knowledge topics list with record detail panel."""

    def compose(self) -> ComposeResult:
        with Vertical():
            yield DataTable(id="topics-table")
            yield Static("Select a topic to view records", id="records-panel")

    def on_mount(self) -> None:
        table = self.query_one("#topics-table", DataTable)
        table.cursor_type = "row"
        table.add_columns(
            "Filename", "Title", "Facts", "Confidence", "Episodes", "Updated",
        )
        self.load_data()

    def load_data(self) -> None:
        data = DashboardData()
        self._topics = data.get_knowledge_topics()

        table = self.query_one("#topics-table", DataTable)
        table.clear()

        for t in self._topics:
            table.add_row(
                t["filename"],
                t["title"],
                str(t["fact_count"]),
                f"{t['confidence']:.2f}",
                str(t["source_episode_count"]),
                _fmt_ts(t["updated_at"]),
            )

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        idx = event.cursor_row
        if 0 <= idx < len(self._topics):
            topic = self._topics[idx]
            data = DashboardData()
            records = data.get_records_for_topic(topic["id"])

            panel = self.query_one("#records-panel", Static)
            if not records:
                panel.update(f"No records for {topic['title']}")
                return

            lines = [f"Records for: {topic['title']}\n"]
            for r in records:
                lines.append(
                    f"  [{r['record_type']}] (confidence: {r['confidence']:.2f}) "
                    f"{r['content'][:100]}"
                )
            panel.update("\n".join(lines))


# ── Tab: Consolidation History ───────────────────────────────────────────────

class ConsolidationTab(Container):
    """Consolidation run history."""

    def compose(self) -> ComposeResult:
        yield DataTable(id="runs-table")

    def on_mount(self) -> None:
        table = self.query_one("#runs-table", DataTable)
        table.cursor_type = "row"
        table.add_columns(
            "Started", "Completed", "Episodes", "Clusters",
            "Created", "Updated", "Pruned", "Status",
        )
        self.load_data()

    def load_data(self) -> None:
        data = DashboardData()
        runs = data.get_consolidation_runs()

        table = self.query_one("#runs-table", DataTable)
        table.clear()

        for r in runs:
            table.add_row(
                _fmt_ts(r["started_at"]),
                _fmt_ts(r.get("completed_at")),
                str(r["episodes_processed"]),
                str(r["clusters_formed"]),
                str(r["topics_created"]),
                str(r["topics_updated"]),
                str(r["episodes_pruned"]),
                r["status"],
            )


# ── Tab: Memory Stats ────────────────────────────────────────────────────────

class StatsTab(Container):
    """Live memory statistics display."""

    _refresh_timer: Timer | None = None

    def compose(self) -> ComposeResult:
        yield Static("Loading...", id="stats-content")

    def on_mount(self) -> None:
        self.load_data()
        self._refresh_timer = self.set_interval(5.0, self.load_data)

    def load_data(self) -> None:
        data = DashboardData()
        stats = data.get_stats()
        faiss = data.get_faiss_stats()

        lines = []
        lines.append("=== Episode Counts ===")
        by_type = stats["episodes_by_type"]
        for ct in ("exchange", "fact", "solution", "preference"):
            lines.append(f"  {ct:12s}: {by_type.get(ct, 0)}")
        lines.append(f"  {'total':12s}: {stats['total_episodes']}")

        lines.append("")
        lines.append("=== Knowledge Base ===")
        lines.append(f"  Topics:  {stats['knowledge_topic_count']}")
        lines.append(f"  Records: {stats['record_count']}")

        lines.append("")
        lines.append("=== FAISS Index ===")
        lines.append(f"  Active vectors:  {faiss['index_size']}")
        lines.append(f"  Tombstones:      {faiss['tombstone_count']}")
        lines.append(f"  Tombstone ratio: {faiss['tombstone_ratio']:.1%}")

        lines.append("")
        lines.append("=== Storage ===")
        lines.append(f"  Database size: {stats['db_size_mb']} MB")

        last = stats["last_consolidation"]
        lines.append("")
        lines.append("=== Last Consolidation ===")
        if last:
            lines.append(f"  Started:  {_fmt_ts(last['started_at'])}")
            lines.append(f"  Status:   {last['status']}")
            lines.append(f"  Episodes: {last['episodes_processed']}")
        else:
            lines.append("  No consolidation runs yet")

        widget = self.query_one("#stats-content", Static)
        widget.update("\n".join(lines))


# ── Main App ─────────────────────────────────────────────────────────────────

class DashboardApp(App):
    """Consolidation Memory TUI Dashboard."""

    TITLE = "consolidation-memory dashboard"

    CSS = """
    #episodes-table, #topics-table, #runs-table {
        height: 1fr;
    }
    #episodes-footer {
        height: 1;
        dock: bottom;
        background: $surface;
        padding: 0 1;
    }
    #records-panel {
        height: 10;
        border-top: solid $primary;
        padding: 1;
        overflow-y: auto;
    }
    #stats-content {
        padding: 1 2;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
        Binding("1", "tab_1", "Episodes", show=False),
        Binding("2", "tab_2", "Knowledge", show=False),
        Binding("3", "tab_3", "Consolidation", show=False),
        Binding("4", "tab_4", "Stats", show=False),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent(
            "Episodes", "Knowledge", "Consolidation", "Stats",
        ):
            with TabPane("Episodes", id="tab-episodes"):
                yield EpisodesTab()
            with TabPane("Knowledge", id="tab-knowledge"):
                yield KnowledgeTab()
            with TabPane("Consolidation", id="tab-consolidation"):
                yield ConsolidationTab()
            with TabPane("Stats", id="tab-stats"):
                yield StatsTab()
        yield Footer()

    def action_refresh(self) -> None:
        """Reload data in the active tab."""
        tabbed = self.query_one(TabbedContent)
        active_id = tabbed.active
        if active_id == "tab-episodes":
            self.query_one(EpisodesTab).load_data()
        elif active_id == "tab-knowledge":
            self.query_one(KnowledgeTab).load_data()
        elif active_id == "tab-consolidation":
            self.query_one(ConsolidationTab).load_data()
        elif active_id == "tab-stats":
            self.query_one(StatsTab).load_data()

    def action_tab_1(self) -> None:
        self.query_one(TabbedContent).active = "tab-episodes"

    def action_tab_2(self) -> None:
        self.query_one(TabbedContent).active = "tab-knowledge"

    def action_tab_3(self) -> None:
        self.query_one(TabbedContent).active = "tab-consolidation"

    def action_tab_4(self) -> None:
        self.query_one(TabbedContent).active = "tab-stats"
```

**Step 2: Verify import works**

Run: `python -c "from consolidation_memory.dashboard import DashboardApp; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add src/consolidation_memory/dashboard.py
git commit -m "feat: add Textual TUI dashboard app"
```

---

### Task 4: Add `dashboard` CLI subcommand

**Files:**
- Modify: `src/consolidation_memory/cli.py:1-11` (docstring)
- Modify: `src/consolidation_memory/cli.py:543-591` (main function)

**Step 1: Add cmd_dashboard function**

Add before the `main()` function (after `cmd_reindex`, around line 541):

```python
def cmd_dashboard():
    """Launch the TUI dashboard."""
    try:
        from consolidation_memory.dashboard import DashboardApp
    except ImportError:
        print("Dashboard requires textual. Install with: pip install consolidation-memory[dashboard]")
        sys.exit(1)
    app = DashboardApp()
    app.run()
```

**Step 2: Register the subcommand in main()**

In the `main()` function, add the subparser after the `reindex` parser (line 566):

```python
    sub.add_parser("dashboard", help="Launch TUI dashboard")
```

Add the elif branch in the command dispatch (after the reindex elif, line 587):

```python
    elif args.command == "dashboard":
        cmd_dashboard()
```

**Step 3: Update module docstring**

Update line 1-11 docstring to include the dashboard command:

```python
"""CLI entry point for consolidation-memory.

Usage:
    consolidation-memory serve       # Start MCP server (default)
    consolidation-memory init        # Interactive first-run setup
    consolidation-memory consolidate # Run consolidation manually
    consolidation-memory status      # Show system stats
    consolidation-memory export      # Export to JSON
    consolidation-memory import PATH # Import from JSON export
    consolidation-memory reindex     # Re-embed all episodes with current backend
    consolidation-memory dashboard   # Launch TUI dashboard
"""
```

**Step 4: Verify the CLI help shows the command**

Run: `consolidation-memory --help`
Expected: `dashboard` appears in the subcommands list

**Step 5: Commit**

```bash
git add src/consolidation_memory/cli.py
git commit -m "feat: add dashboard CLI subcommand"
```

---

### Task 5: Final verification

**Step 1: Run the full test suite**

Run: `pytest tests/ -v --tb=short`
Expected: ALL PASS (including new test_dashboard_data.py)

**Step 2: Run ruff linting**

Run: `ruff check src/consolidation_memory/dashboard_data.py src/consolidation_memory/dashboard.py`
Expected: No errors

**Step 3: Manual smoke test**

Run: `consolidation-memory dashboard`
Expected: TUI opens with 4 tabs, shows data if any exists, q to quit

**Step 4: Commit any fixes if needed, then final commit message**

If all clean, no action needed. If ruff catches issues, fix and commit:

```bash
git add -A
git commit -m "fix: address linting issues in dashboard modules"
```
