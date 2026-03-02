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


class ProtectRequest(BaseModel):
    episode_id: str | None = None
    tag: str | None = None


class TimelineRequest(BaseModel):
    topic: str


class ContradictionsRequest(BaseModel):
    topic: str | None = None


# ── App factory ──────────────────────────────────────────────────────────────

_client: MemoryClient | None = None


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global _client
        _client = MemoryClient()
        from consolidation_memory.config import get_active_project
        import logging
        logging.getLogger("consolidation_memory").info("REST API active project: %s", get_active_project())
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
        from consolidation_memory.config import get_active_project
        return {"status": "ok", "version": __version__, "project": get_active_project()}

    def _require_client() -> MemoryClient:
        if _client is None:
            raise HTTPException(status_code=503, detail="Memory system not initialized")
        return _client

    @app.post("/memory/store")
    async def store(req: StoreRequest):
        """Store a memory episode."""
        client = _require_client()
        result = client.store(
            content=req.content,
            content_type=req.content_type,
            tags=req.tags,
            surprise=req.surprise,
        )
        return dataclasses.asdict(result)

    @app.post("/memory/store/batch")
    async def store_batch(req: BatchStoreRequest):
        """Store multiple memory episodes in a single operation."""
        result = _require_client().store_batch(episodes=req.episodes)
        return dataclasses.asdict(result)

    @app.post("/memory/recall")
    async def recall(req: RecallRequest):
        """Retrieve relevant memories by semantic similarity."""
        result = _require_client().recall(
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
        result = _require_client().search(
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
        result = _require_client().status()
        return dataclasses.asdict(result)

    @app.delete("/memory/episodes/{episode_id}")
    async def forget(episode_id: str):
        """Soft-delete an episode."""
        result = _require_client().forget(episode_id)
        if result.status == "not_found":
            raise HTTPException(status_code=404, detail="Episode not found")
        return dataclasses.asdict(result)

    @app.post("/memory/consolidate")
    async def consolidate():
        """Run consolidation manually."""
        return _require_client().consolidate()

    @app.post("/memory/correct")
    async def correct(req: CorrectRequest):
        """Correct a knowledge document."""
        result = _require_client().correct(
            topic_filename=req.topic_filename,
            correction=req.correction,
        )
        if result.status == "not_found":
            raise HTTPException(status_code=404, detail="Knowledge topic not found")
        return dataclasses.asdict(result)

    @app.post("/memory/export")
    async def export():
        """Export all episodes and knowledge to a JSON snapshot."""
        result = _require_client().export()
        return dataclasses.asdict(result)

    @app.post("/memory/compact")
    async def compact():
        """Compact the FAISS index by removing tombstoned vectors."""
        result = _require_client().compact()
        return dataclasses.asdict(result)

    @app.get("/memory/browse")
    async def browse():
        """Browse all knowledge topics with summaries and metadata."""
        result = _require_client().browse()
        return dataclasses.asdict(result)

    @app.get("/memory/topics/{filename}")
    async def read_topic(filename: str):
        """Read the full markdown content of a knowledge topic."""
        result = _require_client().read_topic(filename)
        if result.status == "not_found":
            raise HTTPException(status_code=404, detail="Knowledge topic not found")
        if result.status == "error":
            raise HTTPException(status_code=400, detail=result.message)
        return dataclasses.asdict(result)

    @app.post("/memory/timeline")
    async def timeline(req: TimelineRequest):
        """Show how understanding of a topic has changed over time."""
        result = _require_client().timeline(topic=req.topic)
        return dataclasses.asdict(result)

    @app.post("/memory/contradictions")
    async def contradictions(req: ContradictionsRequest):
        """List detected contradictions from the audit log."""
        result = _require_client().contradictions(topic=req.topic)
        return dataclasses.asdict(result)

    @app.post("/memory/protect")
    async def protect(req: ProtectRequest):
        """Mark episodes as immune to pruning."""
        result = _require_client().protect(
            episode_id=req.episode_id,
            tag=req.tag,
        )
        if result.status == "not_found":
            raise HTTPException(status_code=404, detail=result.message)
        if result.status == "error":
            raise HTTPException(status_code=400, detail=result.message)
        return dataclasses.asdict(result)

    @app.get("/memory/decay-report")
    async def decay_report():
        """Show what would be forgotten if pruning ran right now."""
        result = _require_client().decay_report()
        return dataclasses.asdict(result)

    return app
