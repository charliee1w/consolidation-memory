"""Backend for the native desktop app (no Qt dependency)."""

from __future__ import annotations

from consolidation_memory import __version__
from consolidation_memory.config import get_active_project
from consolidation_memory.dashboard_data import DashboardData
from consolidation_memory.tool_dispatch import execute_tool_call


def build_health_snapshot(last_run: dict | None) -> tuple[str, str]:
    """Return (health, health_note) from the latest consolidation run."""
    health = "ok"
    health_note = "Memory is ready."
    if isinstance(last_run, dict):
        status = str(last_run.get("status") or "")
        if status == "error":
            health = "warning"
            health_note = "Last consolidation run reported an error."
    return health, health_note


class DesktopBackend:
    """Synchronous operations backing the native desktop UI."""

    def __init__(self, data: DashboardData | None = None) -> None:
        self._data = data or DashboardData()

    def overview(self) -> dict[str, object]:
        stats = self._data.get_stats()
        faiss = self._data.get_faiss_stats()
        health, health_note = build_health_snapshot(stats.get("last_consolidation"))
        return {
            "version": __version__,
            "project": get_active_project(),
            "health": health,
            "health_note": health_note,
            "stats": stats,
            "faiss": faiss,
        }

    def ask(self, query: str, *, n_results: int = 8) -> dict[str, object]:
        return execute_tool_call(
            "memory_ask",
            {"query": query, "n_results": n_results},
        )

    def remember(
        self,
        content: str,
        *,
        kind: str = "note",
        tags: list[str] | None = None,
    ) -> dict[str, object]:
        remember_args: dict[str, object] = {
            "content": content,
            "kind": kind,
            "tags": tags or ["desktop"],
        }
        return execute_tool_call("memory_remember", remember_args)

    def consolidate(self) -> dict[str, object]:
        return execute_tool_call("memory_consolidate", {})

    def forget(self, episode_id: str) -> dict[str, object]:
        return execute_tool_call("memory_forget", {"episode_id": episode_id})

    def recent_episodes(self, *, limit: int = 40) -> list[dict]:
        bounded = max(1, min(limit, 200))
        return self._data.get_episodes(limit=bounded)