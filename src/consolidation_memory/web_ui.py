"""Browser UI and simplified REST surface for consolidation-memory.

The web UI exposes a small vocabulary — remember, ask, browse — on top of the
full memory stack. Advanced tools (claims, drift, policy) stay on the REST API.
"""

from __future__ import annotations

import importlib.resources
from collections.abc import Awaitable, Callable
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel, Field

from consolidation_memory import __version__
from consolidation_memory.dashboard_data import DashboardData
from consolidation_memory.simple_api import build_ask_recall_arguments

ExecuteFn = Callable[..., Awaitable[dict[str, object]]]

_MAX_CONTENT_LENGTH = 50_000
_MAX_QUERY_LENGTH = 10_000


def load_index_html() -> str:
    """Load the packaged browser UI document."""
    path = importlib.resources.files("consolidation_memory.web") / "index.html"
    return path.read_text(encoding="utf-8")


class RememberRequest(BaseModel):
    content: str = Field(min_length=1, max_length=_MAX_CONTENT_LENGTH)
    kind: Literal["note", "fact", "fix", "preference"] = "note"
    tags: list[str] | None = Field(default=None, max_length=20)


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=_MAX_QUERY_LENGTH)
    n_results: int = Field(default=8, ge=1, le=20)


def register_web_ui_routes(app: FastAPI, *, execute: ExecuteFn) -> None:
    """Attach browser UI pages and simplified JSON helpers to a FastAPI app."""
    data = DashboardData()

    @app.get("/", include_in_schema=False)
    async def root_redirect() -> RedirectResponse:
        return RedirectResponse(url="/ui/", status_code=302)

    @app.get("/ui", include_in_schema=False)
    async def ui_redirect() -> RedirectResponse:
        return RedirectResponse(url="/ui/", status_code=302)

    @app.get("/ui/", response_class=HTMLResponse, include_in_schema=False)
    async def ui_page() -> HTMLResponse:
        return HTMLResponse(content=load_index_html())

    @app.get("/ui/api/overview")
    async def ui_overview() -> dict[str, object]:
        from consolidation_memory.config import get_active_project

        stats = data.get_stats()
        faiss = data.get_faiss_stats()
        last_run = stats.get("last_consolidation")
        health = "ok"
        health_note = "Memory is ready."
        if isinstance(last_run, dict):
            status = str(last_run.get("status") or "")
            if status == "error":
                health = "warning"
                health_note = "Last consolidation run reported an error."
        return {
            "version": __version__,
            "project": get_active_project(),
            "health": health,
            "health_note": health_note,
            "stats": stats,
            "faiss": faiss,
        }

    @app.get("/ui/api/recent")
    async def ui_recent(limit: int = 40) -> dict[str, object]:
        bounded = max(1, min(limit, 200))
        episodes = data.get_episodes(limit=bounded)
        return {"episodes": episodes}

    @app.post("/ui/api/remember")
    async def ui_remember(req: RememberRequest) -> dict[str, object]:
        remember_args = req.model_dump()
        if not remember_args.get("tags"):
            remember_args["tags"] = ["ui"]
        return await execute("memory_remember", remember_args)

    @app.post("/ui/api/ask")
    async def ui_ask(req: AskRequest) -> dict[str, object]:
        return await execute(
            "memory_ask",
            build_ask_recall_arguments(req.model_dump()),
        )

    @app.post("/ui/api/consolidate")
    async def ui_consolidate() -> dict[str, object]:
        result = await execute("memory_consolidate", {})
        if isinstance(result, dict) and result.get("status") == "already_running":
            raise HTTPException(status_code=409, detail="Consolidation already running")
        return result

    @app.delete("/ui/api/episodes/{episode_id}")
    async def ui_forget_episode(episode_id: str) -> dict[str, object]:
        result = await execute("memory_forget", {"episode_id": episode_id})
        if result.get("status") == "not_found":
            raise HTTPException(status_code=404, detail="Episode not found")
        return result