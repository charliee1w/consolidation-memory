"""Machine-facing operational REST routes for consolidation-memory.

These endpoints mirror the browser UI fix-it flows under stable ``/ops/*`` paths
for adapters and automation. Memory tool semantics remain on ``/memory/*``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from fastapi import FastAPI, HTTPException

from consolidation_memory.dashboard_data import DashboardData
from consolidation_memory.maintenance import reindex_all_episodes, warmup_recall_caches
from consolidation_memory.ui_ops import build_ops_overview, load_metrics_for_ui

ExecuteFn = Callable[..., Awaitable[dict[str, object]]]


def register_ops_routes(app: FastAPI, *, execute: ExecuteFn) -> None:
    """Attach operational health and maintenance routes to a FastAPI app."""
    data = DashboardData()

    @app.get("/ops/overview")
    async def ops_overview() -> dict[str, object]:
        """Health snapshot, warnings, fix actions, and maintenance daemon state."""
        return build_ops_overview(data)

    @app.get("/ops/metrics")
    async def ops_metrics() -> dict[str, object]:
        """Best available real_world_eval summary for release-quality trending."""
        return load_metrics_for_ui()

    @app.get("/ops/daemon/status")
    async def ops_daemon_status() -> dict[str, object]:
        """Maintenance daemon install and run state."""
        from consolidation_memory.daemon_service import daemon_status

        return daemon_status()

    @app.post("/ops/daemon/install")
    async def ops_daemon_install() -> dict[str, object]:
        """Register the login-time maintenance daemon for the active project."""
        from consolidation_memory.daemon_service import install_daemon

        result = install_daemon()
        if result.get("status") == "error":
            raise HTTPException(status_code=500, detail=str(result.get("message")))
        return result

    @app.post("/ops/consolidate")
    async def ops_consolidate() -> dict[str, object]:
        """Run consolidation across unconsolidated episodes."""
        result = await execute("memory_consolidate", {})
        if isinstance(result, dict) and result.get("status") == "already_running":
            raise HTTPException(status_code=409, detail="Consolidation already running")
        return result

    @app.post("/ops/warmup")
    async def ops_warmup() -> dict[str, object]:
        """Warm record embedding caches for full knowledge recall."""
        return warmup_recall_caches()

    @app.post("/ops/reindex")
    async def ops_reindex() -> dict[str, object]:
        """Rebuild the episode vector index from stored episodes."""
        result = reindex_all_episodes()
        if result.get("status") == "error":
            raise HTTPException(status_code=500, detail=str(result.get("message")))
        return result