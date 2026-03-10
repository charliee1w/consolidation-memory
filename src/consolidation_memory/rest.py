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
from consolidation_memory.runtime import MemoryRuntime
from consolidation_memory.tool_dispatch import execute_tool_call
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


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    runtime = MemoryRuntime()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        runtime.startup()
        from consolidation_memory.config import get_active_project
        import logging

        logging.getLogger("consolidation_memory").info(
            "REST API active project: %s", get_active_project()
        )
        yield
        runtime.shutdown()

    app = FastAPI(
        title="Consolidation Memory",
        description="Persistent semantic memory for AI conversations",
        version=__version__,
        lifespan=lifespan,
    )
    _install_auth_middleware(app)
    _register_memory_routes(app, runtime)
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

def _register_memory_routes(app: FastAPI, runtime: MemoryRuntime) -> None:
    # ── Endpoints ────────────────────────────────────────────────────────
    @app.get("/health")
    async def health():
        """Health check."""
        from consolidation_memory.config import get_active_project

        return {"status": "ok", "version": __version__, "project": get_active_project()}

    async def _execute(
        name: str,
        arguments: dict[str, object],
        *,
        timeout: float | None = None,
    ) -> dict[str, object]:
        client = None
        if name != "memory_detect_drift":
            client = await runtime.get_client_with_timeout()
        return await runtime.run_blocking(
            execute_tool_call,
            name,
            arguments,
            client=client,
            timeout=timeout,
        )

    @app.post("/memory/store")
    async def store(req: StoreRequest):
        """Store a memory episode."""
        return await _execute("memory_store", req.model_dump())

    @app.post("/memory/store/batch")
    async def store_batch(req: BatchStoreRequest):
        """Store multiple memory episodes in a single operation."""
        return await _execute(
            "memory_store_batch",
            {"episodes": [ep.model_dump() for ep in req.episodes], "scope": req.scope},
        )

    @app.post("/memory/recall")
    async def recall(req: RecallRequest):
        """Retrieve relevant memories by semantic similarity."""
        payload = req.model_dump()
        payload["content_types"] = cast(list[str] | None, req.content_types)
        return await _execute("memory_recall", payload)

    @app.post("/memory/search")
    async def search(req: SearchRequest):
        """Keyword/metadata search over episodes (no embedding needed)."""
        payload = req.model_dump()
        payload["content_types"] = cast(list[str] | None, req.content_types)
        return await _execute("memory_search", payload)

    @app.post("/memory/claims/browse")
    async def browse_claims(req: ClaimBrowseRequest):
        """Browse claims with optional type and temporal filtering."""
        return await _execute("memory_claim_browse", req.model_dump())

    @app.post("/memory/claims/search")
    async def search_claims(req: ClaimSearchRequest):
        """Search claims by text with optional type and temporal filtering."""
        return await _execute("memory_claim_search", req.model_dump())

    @app.post("/memory/detect-drift")
    async def detect_drift(req: DetectDriftRequest):
        """Detect code drift and challenge anchored claims."""
        try:
            timeout_seconds = _drift_timeout_seconds()
            return await runtime.run_blocking(
                _run_detect_drift,
                base_ref=req.base_ref,
                repo_path=req.repo_path,
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
        return await _execute("memory_status", {})

    @app.delete("/memory/episodes/{episode_id}")
    async def forget(episode_id: str):
        """Soft-delete an episode."""
        result = await _execute("memory_forget", {"episode_id": episode_id})
        if result.get("status") == "not_found":
            raise HTTPException(status_code=404, detail="Episode not found")
        return result

    @app.post("/memory/consolidate")
    async def consolidate():
        """Run consolidation manually."""
        result = await _execute("memory_consolidate", {})
        if isinstance(result, dict) and result.get("status") == "already_running":
            raise HTTPException(status_code=409, detail="Consolidation already running")
        return result

    @app.post("/memory/correct")
    async def correct(req: CorrectRequest):
        """Correct a knowledge document."""
        result = await _execute("memory_correct", req.model_dump())
        if result.get("status") == "not_found":
            raise HTTPException(status_code=404, detail="Knowledge topic not found")
        return result

    @app.post("/memory/export")
    async def export():
        """Export all episodes and knowledge to a JSON snapshot."""
        return await _execute("memory_export", {})

    @app.post("/memory/compact")
    async def compact():
        """Compact the FAISS index by removing tombstoned vectors."""
        return await _execute("memory_compact", {})

    @app.get("/memory/browse")
    async def browse():
        """Browse all knowledge topics with summaries and metadata."""
        return await _execute("memory_browse", {})

    @app.get("/memory/topics/{filename}")
    async def read_topic(filename: str):
        """Read the full markdown content of a knowledge topic."""
        if _UNSAFE_FILENAME_RE.search(filename):
            raise HTTPException(
                status_code=400,
                detail="Invalid filename: must not contain '/', '\\', or '..'",
            )
        result = await _execute("memory_read_topic", {"filename": filename})
        if result.get("status") == "not_found":
            raise HTTPException(status_code=404, detail="Knowledge topic not found")
        if result.get("status") == "error":
            raise HTTPException(status_code=400, detail=str(result.get("message", "Unknown error")))
        return result

    @app.post("/memory/timeline")
    async def timeline(req: TimelineRequest):
        """Show how understanding of a topic has changed over time."""
        return await _execute("memory_timeline", req.model_dump())

    @app.post("/memory/contradictions")
    async def contradictions(req: ContradictionsRequest):
        """List detected contradictions from the audit log."""
        return await _execute("memory_contradictions", req.model_dump())

    @app.post("/memory/protect")
    async def protect(req: ProtectRequest):
        """Mark episodes as immune to pruning."""
        result = await _execute("memory_protect", req.model_dump())
        if result.get("status") == "not_found":
            raise HTTPException(status_code=404, detail=str(result.get("message", "Not found")))
        if result.get("status") == "error":
            raise HTTPException(status_code=400, detail=str(result.get("message", "Unknown error")))
        return result

    @app.post("/memory/consolidation-log")
    async def consolidation_log(req: ConsolidationLogRequest):
        """Show recent consolidation activity as a human-readable changelog."""
        return await _execute("memory_consolidation_log", req.model_dump())

    @app.get("/memory/decay-report")
    async def decay_report():
        """Show what would be forgotten if pruning ran right now."""
        return await _execute("memory_decay_report", {})
