"""TUI dashboard for consolidation-memory.

Requires: pip install consolidation-memory[dashboard]
"""

from __future__ import annotations

from typing import Any

try:
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Container, Vertical
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
    _TEXTUAL_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised in tests without dashboard extra
    _TEXTUAL_AVAILABLE = False

from consolidation_memory.dashboard_data import DashboardData

DashboardApp: Any


# -- Helpers -------------------------------------------------------------------

_CONSOLIDATED_LABELS = {0: "Pending", 1: "Consolidated", 2: "Pruned"}


def _fmt_ts(ts: str | None) -> str:
    """Format an ISO timestamp for display (drop sub-second precision)."""
    if not ts:
        return "-"
    return ts.replace("T", " ")[:19]


if not _TEXTUAL_AVAILABLE:
    class _DashboardFallbackApp:  # pragma: no cover - simple runtime guard
        """Fallback app that preserves a clear error when textual is unavailable."""

        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs
            raise ImportError(
                "Dashboard requires textual. Install with: "
                "pip install consolidation-memory[dashboard]"
            )

        def run(self) -> None:
            raise ImportError(
                "Dashboard requires textual. Install with: "
                "pip install consolidation-memory[dashboard]"
            )
    DashboardApp = _DashboardFallbackApp
else:
    # -- Tab: Episodes -------------------------------------------------------------

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
                    f"{ep.get('surprise_score') or 0:.2f}",
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

    # -- Tab: Knowledge Topics -----------------------------------------------------

    class KnowledgeTab(Container):
        """Knowledge topics list with record detail panel."""

        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self._topics: list[dict] = []

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

    # -- Tab: Consolidation History ------------------------------------------------

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
                    _fmt_ts(r.get("started_at")),
                    _fmt_ts(r.get("completed_at")),
                    str(r.get("episodes_processed", 0)),
                    str(r.get("clusters_formed", 0)),
                    str(r.get("topics_created", 0)),
                    str(r.get("topics_updated", 0)),
                    str(r.get("episodes_pruned", 0)),
                    r.get("status", "unknown"),
                )

    # -- Tab: Memory Stats ---------------------------------------------------------

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

    # -- Main App ------------------------------------------------------------------

    class _DashboardTextualApp(App):
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

    DashboardApp = _DashboardTextualApp
