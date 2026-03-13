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
import math
import os
import re
import secrets
from typing import Literal, TypeAlias, cast

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
_ScopeInput: TypeAlias = dict[str, object] | str

# Maximum number of episodes in a single batch store request.
_MAX_BATCH_SIZE = 100
_MAX_QUERY_LENGTH = 10_000
_MAX_TOPIC_LENGTH = 500
_MAX_FILENAME_LENGTH = 255
_MAX_PATH_LENGTH = 4096
_REST_AUTH_TOKEN_ENV = "CONSOLIDATION_MEMORY_REST_AUTH_TOKEN"  # nosec B105
_REST_ALLOW_PUBLIC_BIND_ENV = "CONSOLIDATION_MEMORY_REST_ALLOW_PUBLIC_BIND"
_AUTH_EXEMPT_PATHS = frozenset({"/health"})
_SYNTHETIC_LOOPBACK_HOSTS = frozenset({"testserver"})


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    token = raw.strip()
    if not token:
        return default
    try:
        value = float(token)
    except ValueError:
        return default
    return value if math.isfinite(value) else default


_MEMORY_DETECT_DRIFT_TIMEOUT_SECONDS = _env_float(
    "CONSOLIDATION_MEMORY_DRIFT_TIMEOUT_SECONDS",
    180.0,
)
_MEMORY_RECALL_TIMEOUT_SECONDS = _env_float(
    "CONSOLIDATION_MEMORY_RECALL_TIMEOUT_SECONDS",
    45.0,
)
_MEMORY_RECALL_FALLBACK_TIMEOUT_SECONDS = _env_float(
    "CONSOLIDATION_MEMORY_RECALL_FALLBACK_TIMEOUT_SECONDS",
    10.0,
)
_CLIENT_INIT_TIMEOUT_SECONDS = _env_float(
    "CONSOLIDATION_MEMORY_CLIENT_INIT_TIMEOUT_SECONDS",
    45.0,
)


def _drift_timeout_seconds() -> float:
    configured = _MEMORY_DETECT_DRIFT_TIMEOUT_SECONDS
    return configured if configured > 0 else 180.0


def _recall_timeout_seconds() -> float:
    configured = _MEMORY_RECALL_TIMEOUT_SECONDS
    return configured if configured > 0 else 90.0


def _recall_fallback_timeout_seconds() -> float:
    configured = _MEMORY_RECALL_FALLBACK_TIMEOUT_SECONDS
    return configured if configured > 0 else 20.0


def _client_init_timeout_seconds() -> float:
    configured = _CLIENT_INIT_TIMEOUT_SECONDS
    return configured if configured > 0 else 90.0


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


def _request_bind_host(request: Request) -> str | None:
    server = request.scope.get("server")
    if not isinstance(server, (tuple, list)) or not server:
        return None
    host = server[0]
    return host if isinstance(host, str) and host else None


def _request_bind_is_safe(host: str | None) -> bool:
    if host is None:
        return False
    if host in _SYNTHETIC_LOOPBACK_HOSTS:
        return True
    return _is_loopback_host(host)


def _public_bind_detail() -> str:
    return (
        "Refusing unauthenticated REST requests on a non-loopback bind. "
        f"Set {_REST_AUTH_TOKEN_ENV} to require Bearer auth, or set "
        f"{_REST_ALLOW_PUBLIC_BIND_ENV}=true to override (not recommended)."
    )

# Reject filenames that attempt path traversal or contain path separators.
_UNSAFE_FILENAME_RE = re.compile(r"[/\\]|\.\.")


def _extract_bearer_token(auth_header: str) -> str | None:
    parts = auth_header.strip().split()
    if len(parts) != 2:
        return None
    scheme, token = parts
    if scheme.lower() != "bearer":
        return None
    return token or None


# ── Pydantic request models ─────────────────────────────────────────────────


class StoreRequest(BaseModel):
    content: str = Field(max_length=50_000)
    content_type: _ContentTypeLiteral = "exchange"
    tags: list[str] | None = None
    surprise: float = Field(default=0.5, ge=0.0, le=1.0)
    scope: _ScopeInput | None = None


class RecallRequest(BaseModel):
    query: str = Field(max_length=_MAX_QUERY_LENGTH)
    n_results: int = Field(default=10, ge=1, le=50)
    include_knowledge: bool = True
    content_types: list[_ContentTypeLiteral] | None = None
    tags: list[str] | None = None
    after: str | None = Field(default=None, max_length=64)
    before: str | None = Field(default=None, max_length=64)
    include_expired: bool = False
    as_of: str | None = Field(default=None, max_length=64)
    scope: _ScopeInput | None = None


class EpisodeInput(BaseModel):
    content: str = Field(max_length=50_000)
    content_type: _ContentTypeLiteral = "exchange"
    tags: list[str] | None = None
    surprise: float = Field(default=0.5, ge=0.0, le=1.0)


class BatchStoreRequest(BaseModel):
    episodes: list[EpisodeInput] = Field(max_length=_MAX_BATCH_SIZE)
    scope: _ScopeInput | None = None


class SearchRequest(BaseModel):
    query: str | None = Field(default=None, max_length=_MAX_QUERY_LENGTH)
    content_types: list[_ContentTypeLiteral] | None = None
    tags: list[str] | None = None
    after: str | None = Field(default=None, max_length=64)
    before: str | None = Field(default=None, max_length=64)
    limit: int = Field(default=20, ge=1, le=50)
    scope: _ScopeInput | None = None


class ClaimBrowseRequest(BaseModel):
    claim_type: str | None = Field(default=None, max_length=64)
    as_of: str | None = Field(default=None, max_length=64)
    limit: int = Field(default=50, ge=1, le=200)
    scope: _ScopeInput | None = None


class ClaimSearchRequest(BaseModel):
    query: str = Field(max_length=_MAX_QUERY_LENGTH)
    claim_type: str | None = Field(default=None, max_length=64)
    as_of: str | None = Field(default=None, max_length=64)
    limit: int = Field(default=50, ge=1, le=200)
    scope: _ScopeInput | None = None


class OutcomeAnchorInput(BaseModel):
    anchor_type: str = Field(max_length=64)
    anchor_value: str = Field(max_length=_MAX_PATH_LENGTH)


class OutcomeRecordRequest(BaseModel):
    action_summary: str = Field(max_length=_MAX_QUERY_LENGTH)
    outcome_type: Literal["success", "failure", "partial_success", "reverted", "superseded"]
    source_claim_ids: list[str] | None = None
    source_record_ids: list[str] | None = None
    source_episode_ids: list[str] | None = None
    code_anchors: list[OutcomeAnchorInput] | None = None
    issue_ids: list[str] | None = None
    pr_ids: list[str] | None = None
    action_key: str | None = Field(default=None, max_length=_MAX_FILENAME_LENGTH)
    summary: str | None = Field(default=None, max_length=50_000)
    details: dict[str, object] | str | None = None
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    provenance: dict[str, object] | str | None = None
    observed_at: str | None = Field(default=None, max_length=64)
    scope: _ScopeInput | None = None


class OutcomeBrowseRequest(BaseModel):
    outcome_type: Literal["success", "failure", "partial_success", "reverted", "superseded"] | None = None
    action_key: str | None = Field(default=None, max_length=_MAX_FILENAME_LENGTH)
    source_claim_id: str | None = Field(default=None, max_length=_MAX_FILENAME_LENGTH)
    source_record_id: str | None = Field(default=None, max_length=_MAX_FILENAME_LENGTH)
    source_episode_id: str | None = Field(default=None, max_length=_MAX_FILENAME_LENGTH)
    as_of: str | None = Field(default=None, max_length=64)
    limit: int = Field(default=50, ge=1, le=200)
    scope: _ScopeInput | None = None


class DetectDriftRequest(BaseModel):
    base_ref: str | None = Field(default=None, max_length=_MAX_FILENAME_LENGTH)
    repo_path: str | None = Field(default=None, max_length=_MAX_PATH_LENGTH)


class CorrectRequest(BaseModel):
    topic_filename: str = Field(max_length=_MAX_FILENAME_LENGTH)
    correction: str = Field(max_length=50_000)
    scope: _ScopeInput | None = None


class ExportRequest(BaseModel):
    scope: _ScopeInput | None = None


class ForgetRequest(BaseModel):
    episode_id: str = Field(max_length=_MAX_FILENAME_LENGTH)
    scope: _ScopeInput | None = None


class ProtectRequest(BaseModel):
    episode_id: str | None = Field(default=None, max_length=_MAX_FILENAME_LENGTH)
    tag: str | None = Field(default=None, max_length=100)
    scope: _ScopeInput | None = None


class TimelineRequest(BaseModel):
    topic: str = Field(max_length=_MAX_TOPIC_LENGTH)
    scope: _ScopeInput | None = None


class BrowseRequest(BaseModel):
    scope: _ScopeInput | None = None


class ReadTopicRequest(BaseModel):
    filename: str = Field(max_length=_MAX_FILENAME_LENGTH)
    scope: _ScopeInput | None = None


class ContradictionsRequest(BaseModel):
    topic: str | None = Field(default=None, max_length=_MAX_FILENAME_LENGTH)


class ConsolidationLogRequest(BaseModel):
    last_n: int = Field(default=5, ge=1, le=20)


def create_app(*, bind_host: str | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    if bind_host is not None:
        validate_rest_bind(bind_host)
    runtime = MemoryRuntime()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        del app
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
        description="Trust-calibrated working memory for coding agents",
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
        allow_public_bind = _truthy_env(_REST_ALLOW_PUBLIC_BIND_ENV)
        if (
            token is None
            and not allow_public_bind
            and request.method != "OPTIONS"
            and request.url.path not in _AUTH_EXEMPT_PATHS
            and not _request_bind_is_safe(_request_bind_host(request))
        ):
            return JSONResponse(
                status_code=503,
                content={"detail": _public_bind_detail()},
            )

        if token is None or request.method == "OPTIONS" or request.url.path in _AUTH_EXEMPT_PATHS:
            return await call_next(request)

        auth_header = request.headers.get("Authorization", "")
        provided = _extract_bearer_token(auth_header)
        if provided is None:
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing or invalid Authorization header"},
                headers={"WWW-Authenticate": "Bearer"},
            )
        if not secrets.compare_digest(provided, token):
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
            timeout_seconds = _client_init_timeout_seconds()
            try:
                client = await runtime.get_client_with_timeout(
                    timeout=timeout_seconds
                )
            except TimeoutError as exc:
                raise HTTPException(
                    status_code=408,
                    detail=(
                        f"MemoryClient initialization timed out after "
                        f"{timeout_seconds:g}s. Retry in a few seconds, or "
                        "increase CONSOLIDATION_MEMORY_CLIENT_INIT_TIMEOUT_SECONDS."
                    ),
                ) from exc
        try:
            return await runtime.run_blocking(
                execute_tool_call,
                name,
                arguments,
                client=client,
                timeout=timeout,
            )
        except TimeoutError as exc:
            timeout_detail = (
                f"{name} timed out after {timeout:g}s."
                if timeout is not None
                else f"{name} timed out."
            )
            raise HTTPException(status_code=408, detail=timeout_detail) from exc
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

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
        timeout_seconds = _recall_timeout_seconds()
        fallback_timeout = _recall_fallback_timeout_seconds()
        try:
            return await _execute("memory_recall", payload, timeout=timeout_seconds)
        except HTTPException as exc:
            if exc.status_code != 408:
                raise

        search_payload: dict[str, object] = {
            "query": req.query,
            "content_types": cast(list[str] | None, req.content_types),
            "tags": req.tags,
            "after": req.after,
            "before": req.before,
            "limit": req.n_results,
            "scope": req.scope,
        }
        try:
            keyword_result = await _execute(
                "memory_search",
                search_payload,
                timeout=fallback_timeout,
            )
        except HTTPException as exc:
            if exc.status_code == 408:
                raise HTTPException(
                    status_code=408,
                    detail=(
                        f"memory_recall timed out after {timeout_seconds:g}s and keyword "
                        f"fallback timed out after {fallback_timeout:g}s. Try a shorter query, "
                        "reduce n_results, or set "
                        "CONSOLIDATION_MEMORY_RECALL_TIMEOUT_SECONDS higher."
                    ),
                ) from exc
            raise HTTPException(
                status_code=500,
                detail=(
                    f"memory_recall timed out after {timeout_seconds:g}s and keyword fallback "
                    f"failed: {exc.detail}"
                ),
            ) from exc

        warnings = [
            f"Recall timed out after {timeout_seconds:g}s; returned episodes-only fallback."
        ]
        if req.include_knowledge:
            warnings.append("Knowledge retrieval skipped in fallback mode.")

        episodes_value = keyword_result.get("episodes")
        fallback_episodes = episodes_value if isinstance(episodes_value, list) else []
        total_matches_value = keyword_result.get("total_matches")
        if isinstance(total_matches_value, int):
            total_episodes = total_matches_value
        elif isinstance(total_matches_value, str):
            try:
                total_episodes = int(total_matches_value)
            except ValueError:
                total_episodes = len(fallback_episodes)
        else:
            total_episodes = len(fallback_episodes)

        return {
            "episodes": fallback_episodes,
            "knowledge": [],
            "records": [],
            "claims": [],
            "total_episodes": total_episodes,
            "total_knowledge_topics": 0,
            "message": "Semantic recall timed out; returned keyword episodes-only fallback.",
            "warnings": warnings,
        }

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

    @app.post("/memory/outcomes/record")
    async def record_outcome(req: OutcomeRecordRequest):
        """Record an action outcome observation."""
        payload = req.model_dump()
        if req.code_anchors is not None:
            payload["code_anchors"] = [anchor.model_dump() for anchor in req.code_anchors]
        return await _execute("memory_outcome_record", payload)

    @app.post("/memory/outcomes/browse")
    async def browse_outcomes(req: OutcomeBrowseRequest):
        """Browse recorded action outcomes."""
        return await _execute("memory_outcome_browse", req.model_dump())

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

    @app.post("/memory/forget")
    async def forget_with_scope(req: ForgetRequest):
        """Soft-delete an episode with optional explicit scope."""
        result = await _execute("memory_forget", req.model_dump())
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
        if result.get("status") == "error":
            raise HTTPException(status_code=400, detail=str(result.get("message", "Unknown error")))
        return result

    @app.post("/memory/export")
    async def export(req: ExportRequest | None = None):
        """Export all episodes and knowledge to a JSON snapshot."""
        payload = {} if req is None else req.model_dump()
        return await _execute("memory_export", payload)

    @app.post("/memory/compact")
    async def compact():
        """Compact the FAISS index by removing tombstoned vectors."""
        return await _execute("memory_compact", {})

    @app.get("/memory/browse")
    async def browse():
        """Browse all knowledge topics with summaries and metadata."""
        return await _execute("memory_browse", {})

    @app.post("/memory/browse")
    async def browse_with_scope(req: BrowseRequest | None = None):
        """Browse knowledge topics with optional explicit scope."""
        payload = {} if req is None else req.model_dump()
        return await _execute("memory_browse", payload)

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

    @app.post("/memory/topics/read")
    async def read_topic_with_scope(req: ReadTopicRequest):
        """Read a knowledge topic with optional explicit scope."""
        if _UNSAFE_FILENAME_RE.search(req.filename):
            raise HTTPException(
                status_code=400,
                detail="Invalid filename: must not contain '/', '\\', or '..'",
            )
        result = await _execute("memory_read_topic", req.model_dump())
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
