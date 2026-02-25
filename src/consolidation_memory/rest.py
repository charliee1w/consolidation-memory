"""REST API for consolidation-memory.

Requires: ``pip install consolidation-memory[rest]``

Launch via CLI::

    consolidation-memory serve --rest
    consolidation-memory serve --rest --port 9000 --host 0.0.0.0

Or programmatically::

    from consolidation_memory.rest import create_app
    import uvicorn
    app = create_app()
    uvicorn.run(app, host="127.0.0.1", port=8080)
"""

from __future__ import annotations

import dataclasses

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
except ImportError:
    raise ImportError(
        "REST API requires FastAPI. Install with: pip install consolidation-memory[rest]"
    )

from contextlib import asynccontextmanager

from consolidation_memory import __version__
from consolidation_memory.client import MemoryClient


# ── Pydantic request models ─────────────────────────────────────────────────

class StoreRequest(BaseModel):
    content: str
    content_type: str = "exchange"
    tags: list[str] | None = None
    surprise: float = Field(default=0.5, ge=0.0, le=1.0)


class RecallRequest(BaseModel):
    query: str
    n_results: int = Field(default=10, ge=1, le=50)
    include_knowledge: bool = True
    content_types: list[str] | None = None
    tags: list[str] | None = None
    after: str | None = None
    before: str | None = None


class BatchStoreRequest(BaseModel):
    episodes: list[dict]


class SearchRequest(BaseModel):
    query: str | None = None
    content_types: list[str] | None = None
    tags: list[str] | None = None
    after: str | None = None
    before: str | None = None
    limit: int = Field(default=20, ge=1, le=50)


class CorrectRequest(BaseModel):
    topic_filename: str
    correction: str


# ── App factory ──────────────────────────────────────────────────────────────

_client: MemoryClient | None = None


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global _client
        _client = MemoryClient()
        yield
        _client.close()
        _client = None

    app = FastAPI(
        title="Consolidation Memory",
        description="Persistent semantic memory for AI conversations",
        version=__version__,
        lifespan=lifespan,
    )

    # ── Endpoints ────────────────────────────────────────────────────────

    @app.get("/health")
    async def health():
        """Health check."""
        return {"status": "ok", "version": __version__}

    @app.post("/memory/store")
    async def store(req: StoreRequest):
        """Store a memory episode."""
        result = _client.store(
            content=req.content,
            content_type=req.content_type,
            tags=req.tags,
            surprise=req.surprise,
        )
        return dataclasses.asdict(result)

    @app.post("/memory/store/batch")
    async def store_batch(req: BatchStoreRequest):
        """Store multiple memory episodes in a single operation."""
        result = _client.store_batch(episodes=req.episodes)
        return dataclasses.asdict(result)

    @app.post("/memory/recall")
    async def recall(req: RecallRequest):
        """Retrieve relevant memories by semantic similarity."""
        result = _client.recall(
            query=req.query,
            n_results=req.n_results,
            include_knowledge=req.include_knowledge,
            content_types=req.content_types,
            tags=req.tags,
            after=req.after,
            before=req.before,
        )
        return dataclasses.asdict(result)

    @app.post("/memory/search")
    async def search(req: SearchRequest):
        """Keyword/metadata search over episodes (no embedding needed)."""
        result = _client.search(
            query=req.query,
            content_types=req.content_types,
            tags=req.tags,
            after=req.after,
            before=req.before,
            limit=req.limit,
        )
        return dataclasses.asdict(result)

    @app.get("/memory/status")
    async def status():
        """Get memory system statistics."""
        result = _client.status()
        return dataclasses.asdict(result)

    @app.delete("/memory/episodes/{episode_id}")
    async def forget(episode_id: str):
        """Soft-delete an episode."""
        result = _client.forget(episode_id)
        if result.status == "not_found":
            raise HTTPException(status_code=404, detail="Episode not found")
        return dataclasses.asdict(result)

    @app.post("/memory/consolidate")
    async def consolidate():
        """Run consolidation manually."""
        return _client.consolidate()

    @app.post("/memory/correct")
    async def correct(req: CorrectRequest):
        """Correct a knowledge document."""
        result = _client.correct(
            topic_filename=req.topic_filename,
            correction=req.correction,
        )
        if result.status == "not_found":
            raise HTTPException(status_code=404, detail="Knowledge topic not found")
        return dataclasses.asdict(result)

    @app.post("/memory/export")
    async def export():
        """Export all episodes and knowledge to a JSON snapshot."""
        result = _client.export()
        return dataclasses.asdict(result)

    return app
