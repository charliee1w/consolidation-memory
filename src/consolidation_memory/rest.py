"""REST API for consolidation-memory.

Requires: ``pip install consolidation-memory[rest]``

Launch via CLI::

    consolidation-memory serve --rest
    CONSOLIDATION_MEMORY_REST_AUTH_TOKEN=changeme consolidation-memory serve --rest --port 9000 --host 0.0.0.0

Or programmatically::

    from consolidation_memory.rest import create_app
    import uvicorn
    app = create_app()
    uvicorn.run(app, host="127.0.0.1", port=8080)
"""

from __future__ import annotations

import asyncio
import dataclasses
import ipaddress
import os
import re
import secrets
from typing import Literal, cast

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
except ImportError:
    raise ImportError(
        "REST API requires FastAPI. Install with: pip install consolidation-memory[rest]"
    )

from contextlib import asynccontextmanager

from consolidation_memory import __version__
from consolidation_memory.client import MemoryClient
from consolidation_memory.types import DriftOutput

# Valid content types accepted by the memory system.
_ContentTypeLiteral = Literal["exchange", "fact", "solution", "preference"]

# Maximum number of episodes in a single batch store request.
_MAX_BATCH_SIZE = 100
_MEMORY_DETECT_DRIFT_TIMEOUT_SECONDS = float(
    os.environ.get("CONSOLIDATION_MEMORY_DRIFT_TIMEOUT_SECONDS", "180")
)
_REST_AUTH_TOKEN_ENV = "CONSOLIDATION_MEMORY_REST_AUTH_TOKEN"  # nosec B105
_REST_ALLOW_PUBLIC_BIND_ENV = "CONSOLIDATION_MEMORY_REST_ALLOW_PUBLIC_BIND"
_AUTH_EXEMPT_PATHS = frozenset({"/health"})


def _drift_timeout_seconds() -> float:
    configured = _MEMORY_DETECT_DRIFT_TIMEOUT_SECONDS
    return configured if configured > 0 else 180.0


def _run_detect_drift(
    *,
    base_ref: str | None = None,
    repo_path: str | None = None,
) -> DriftOutput:
    # Drift detection uses git + claim DB state only, and does not require
    # embedding/vector initialization.
    from consolidation_memory.drift import detect_code_drift

    return detect_code_drift(
        base_ref=base_ref,
        repo_path=repo_path,
    )


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def get_rest_auth_token() -> str | None:
    """Return configured REST bearer token, if any."""
    token = os.environ.get(_REST_AUTH_TOKEN_ENV, "").strip()
    return token or None


def _is_loopback_host(host: str) -> bool:
    token = host.strip().strip("[]")
    if token.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(token).is_loopback
    except ValueError:
        return False


def validate_rest_bind(host: str) -> None:
    """Fail closed for non-loopback binds without explicit auth/override."""
    if _is_loopback_host(host):
        return
    if get_rest_auth_token():
        return
    if _truthy_env(_REST_ALLOW_PUBLIC_BIND_ENV):
        return
    raise RuntimeError(
        "Refusing to bind REST API to non-loopback host without auth. "
        f"Set {_REST_AUTH_TOKEN_ENV} to require Bearer auth, or set "
        f"{_REST_ALLOW_PUBLIC_BIND_ENV}=true to override (not recommended)."
    )

# Reject filenames that attempt path traversal or contain path separators.
_UNSAFE_FILENAME_RE = re.compile(r"[/\\]|\.\.")


# ── Pydantic request models ─────────────────────────────────────────────────


class StoreRequest(BaseModel):
    content: str = Field(max_length=50_000)
    content_type: _ContentTypeLiteral = "exchange"
    tags: list[str] | None = None
    surprise: float = Field(default=0.5, ge=0.0, le=1.0)
    scope: dict[str, object] | None = None


class RecallRequest(BaseModel):
    query: str
    n_results: int = Field(default=10, ge=1, le=50)
    include_knowledge: bool = True
    content_types: list[_ContentTypeLiteral] | None = None
    tags: list[str] | None = None
    after: str | None = None
    before: str | None = None
    include_expired: bool = False
    as_of: str | None = None
    scope: dict[str, object] | None = None


class EpisodeInput(BaseModel):
    content: str = Field(max_length=50_000)
    content_type: _ContentTypeLiteral = "exchange"
    tags: list[str] | None = None
    surprise: float = Field(default=0.5, ge=0.0, le=1.0)


class BatchStoreRequest(BaseModel):
    episodes: list[EpisodeInput] = Field(max_length=_MAX_BATCH_SIZE)
    scope: dict[str, object] | None = None


class SearchRequest(BaseModel):
    query: str | None = None
    content_types: list[_ContentTypeLiteral] | None = None
    tags: list[str] | None = None
    after: str | None = None
    before: str | None = None
    limit: int = Field(default=20, ge=1, le=50)
    scope: dict[str, object] | None = None


class ClaimBrowseRequest(BaseModel):
    claim_type: str | None = None
    as_of: str | None = None
    limit: int = Field(default=50, ge=1, le=200)
    scope: dict[str, object] | None = None


class ClaimSearchRequest(BaseModel):
    query: str
    claim_type: str | None = None
    as_of: str | None = None
    limit: int = Field(default=50, ge=1, le=200)
    scope: dict[str, object] | None = None


class DetectDriftRequest(BaseModel):
    base_ref: str | None = None
    repo_path: str | None = None


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


class ConsolidationLogRequest(BaseModel):
    last_n: int = Field(default=5, ge=1, le=20)


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

        logging.getLogger("consolidation_memory").info(
            "REST API active project: %s", get_active_project()
        )
        yield
        _client.close()
        _client = None

    app = FastAPI(
        title="Consolidation Memory",
        description="Persistent semantic memory for AI conversations",
        version=__version__,
        lifespan=lifespan,
    )
    _install_auth_middleware(app)
    _register_memory_routes(app)
    return app

def _install_auth_middleware(app: FastAPI) -> None:
    @app.middleware("http")
    async def _rest_auth_middleware(request: Request, call_next):  # pragma: no cover - exercised in REST tests
        token = get_rest_auth_token()
        if token is None or request.method == "OPTIONS" or request.url.path in _AUTH_EXEMPT_PATHS:
            return await call_next(request)

        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing or invalid Authorization header"},
                headers={"WWW-Authenticate": "Bearer"},
            )
        provided = auth_header[len("Bearer ") :].strip()
        if not provided or not secrets.compare_digest(provided, token):
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid bearer token"},
                headers={"WWW-Authenticate": "Bearer"},
            )
        return await call_next(request)

def _register_memory_routes(app: FastAPI) -> None:
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
        if req.scope is not None:
            result = await asyncio.to_thread(
                client.store_with_scope,
                content=req.content,
                content_type=req.content_type,
                tags=req.tags,
                surprise=req.surprise,
                scope=req.scope,
            )
        else:
            result = await asyncio.to_thread(
                client.store,
                content=req.content,
                content_type=req.content_type,
                tags=req.tags,
                surprise=req.surprise,
            )
        return dataclasses.asdict(result)

    @app.post("/memory/store/batch")
    async def store_batch(req: BatchStoreRequest):
        """Store multiple memory episodes in a single operation."""
        client = _require_client()
        episodes = [ep.model_dump() for ep in req.episodes]
        if req.scope is not None:
            result = await asyncio.to_thread(
                client.store_batch_with_scope,
                episodes=episodes,
                scope=req.scope,
            )
        else:
            result = await asyncio.to_thread(client.store_batch, episodes=episodes)
        return dataclasses.asdict(result)

    @app.post("/memory/recall")
    async def recall(req: RecallRequest):
        """Retrieve relevant memories by semantic similarity."""
        client = _require_client()
        # cast() widens list[Literal[...]] to list[str] for mypy invariance.
        ct = cast(list[str] | None, req.content_types)
        result = await asyncio.to_thread(
            client.query_recall,
            query=req.query,
            n_results=req.n_results,
            include_knowledge=req.include_knowledge,
            content_types=ct,
            tags=req.tags,
            after=req.after,
            before=req.before,
            include_expired=req.include_expired,
            as_of=req.as_of,
            scope=req.scope,
        )
        return dataclasses.asdict(result)

    @app.post("/memory/search")
    async def search(req: SearchRequest):
        """Keyword/metadata search over episodes (no embedding needed)."""
        client = _require_client()
        ct = cast(list[str] | None, req.content_types)
        result = await asyncio.to_thread(
            client.query_search,
            query=req.query,
            content_types=ct,
            tags=req.tags,
            after=req.after,
            before=req.before,
            limit=req.limit,
            scope=req.scope,
        )
        return dataclasses.asdict(result)

    @app.post("/memory/claims/browse")
    async def browse_claims(req: ClaimBrowseRequest):
        """Browse claims with optional type and temporal filtering."""
        client = _require_client()
        result = await asyncio.to_thread(
            client.query_browse_claims,
            claim_type=req.claim_type,
            as_of=req.as_of,
            limit=req.limit,
            scope=req.scope,
        )
        return dataclasses.asdict(result)

    @app.post("/memory/claims/search")
    async def search_claims(req: ClaimSearchRequest):
        """Search claims by text with optional type and temporal filtering."""
        client = _require_client()
        result = await asyncio.to_thread(
            client.query_search_claims,
            query=req.query,
            claim_type=req.claim_type,
            as_of=req.as_of,
            limit=req.limit,
            scope=req.scope,
        )
        return dataclasses.asdict(result)

    @app.post("/memory/detect-drift")
    async def detect_drift(req: DetectDriftRequest):
        """Detect code drift and challenge anchored claims."""
        try:
            timeout_seconds = _drift_timeout_seconds()
            return await asyncio.wait_for(
                asyncio.to_thread(
                    _run_detect_drift,
                    base_ref=req.base_ref,
                    repo_path=req.repo_path,
                ),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError as e:
            raise HTTPException(
                status_code=408,
                detail=(
                    f"memory_detect_drift timed out after {timeout_seconds:g}s. "
                    "Try scoping repo_path to a smaller repository or set "
                    "CONSOLIDATION_MEMORY_DRIFT_TIMEOUT_SECONDS to a higher value."
                ),
            ) from e
        except RuntimeError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    @app.get("/memory/status")
    async def status():
        """Get memory system statistics."""
        result = await asyncio.to_thread(_require_client().status)
        return dataclasses.asdict(result)

    @app.delete("/memory/episodes/{episode_id}")
    async def forget(episode_id: str):
        """Soft-delete an episode."""
        result = await asyncio.to_thread(_require_client().forget, episode_id)
        if result.status == "not_found":
            raise HTTPException(status_code=404, detail="Episode not found")
        return dataclasses.asdict(result)

    @app.post("/memory/consolidate")
    async def consolidate():
        """Run consolidation manually."""
        result = await asyncio.to_thread(_require_client().consolidate)
        if isinstance(result, dict) and result.get("status") == "already_running":
            raise HTTPException(status_code=409, detail="Consolidation already running")
        return result

    @app.post("/memory/correct")
    async def correct(req: CorrectRequest):
        """Correct a knowledge document."""
        client = _require_client()
        result = await asyncio.to_thread(
            client.correct,
            topic_filename=req.topic_filename,
            correction=req.correction,
        )
        if result.status == "not_found":
            raise HTTPException(status_code=404, detail="Knowledge topic not found")
        return dataclasses.asdict(result)

    @app.post("/memory/export")
    async def export():
        """Export all episodes and knowledge to a JSON snapshot."""
        result = await asyncio.to_thread(_require_client().export)
        return dataclasses.asdict(result)

    @app.post("/memory/compact")
    async def compact():
        """Compact the FAISS index by removing tombstoned vectors."""
        result = await asyncio.to_thread(_require_client().compact)
        return dataclasses.asdict(result)

    @app.get("/memory/browse")
    async def browse():
        """Browse all knowledge topics with summaries and metadata."""
        result = await asyncio.to_thread(_require_client().browse)
        return dataclasses.asdict(result)

    @app.get("/memory/topics/{filename}")
    async def read_topic(filename: str):
        """Read the full markdown content of a knowledge topic."""
        if _UNSAFE_FILENAME_RE.search(filename):
            raise HTTPException(
                status_code=400,
                detail="Invalid filename: must not contain '/', '\\', or '..'",
            )
        result = await asyncio.to_thread(_require_client().read_topic, filename)
        if result.status == "not_found":
            raise HTTPException(status_code=404, detail="Knowledge topic not found")
        if result.status == "error":
            raise HTTPException(status_code=400, detail=result.message)
        return dataclasses.asdict(result)

    @app.post("/memory/timeline")
    async def timeline(req: TimelineRequest):
        """Show how understanding of a topic has changed over time."""
        result = await asyncio.to_thread(_require_client().timeline, topic=req.topic)
        return dataclasses.asdict(result)

    @app.post("/memory/contradictions")
    async def contradictions(req: ContradictionsRequest):
        """List detected contradictions from the audit log."""
        result = await asyncio.to_thread(_require_client().contradictions, topic=req.topic)
        return dataclasses.asdict(result)

    @app.post("/memory/protect")
    async def protect(req: ProtectRequest):
        """Mark episodes as immune to pruning."""
        client = _require_client()
        result = await asyncio.to_thread(
            client.protect,
            episode_id=req.episode_id,
            tag=req.tag,
        )
        if result.status == "not_found":
            raise HTTPException(status_code=404, detail=result.message)
        if result.status == "error":
            raise HTTPException(status_code=400, detail=result.message)
        return dataclasses.asdict(result)

    @app.post("/memory/consolidation-log")
    async def consolidation_log(req: ConsolidationLogRequest):
        """Show recent consolidation activity as a human-readable changelog."""
        result = await asyncio.to_thread(_require_client().consolidation_log, last_n=req.last_n)
        return dataclasses.asdict(result)

    @app.get("/memory/decay-report")
    async def decay_report():
        """Show what would be forgotten if pruning ran right now."""
        result = await asyncio.to_thread(_require_client().decay_report)
        return dataclasses.asdict(result)
