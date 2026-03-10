"""Consolidation Memory MCP Server.

Thin wrapper over MemoryClient. Exposes memory tools to any MCP-capable
client via stdio transport. All business logic lives in client.py.
"""

import asyncio
import concurrent.futures
import dataclasses
import functools
import json
import logging
import os
import sys
import threading
import time
import traceback
from contextlib import asynccontextmanager
from typing import Awaitable, Callable, TypeVar

from mcp.server.fastmcp import FastMCP

# Configure logging to stderr (stdout is the MCP JSON-RPC channel)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("consolidation_memory")

_T = TypeVar("_T")

# ── Global client initialized lazily on first tool call ───────────────────
_client = None
_client_lock = threading.Lock()
_client_initializing = False
_client_init_owner_thread_id: int | None = None
_client_init_error: Exception | None = None
_client_init_cond = threading.Condition(_client_lock)


_MAX_BATCH_SIZE = 100
_MEMORY_DETECT_DRIFT_TIMEOUT_SECONDS = float(
    os.environ.get("CONSOLIDATION_MEMORY_DRIFT_TIMEOUT_SECONDS", "90")
)
_MEMORY_RECALL_TIMEOUT_SECONDS = float(
    os.environ.get("CONSOLIDATION_MEMORY_RECALL_TIMEOUT_SECONDS", "45")
)
_MEMORY_RECALL_FALLBACK_TIMEOUT_SECONDS = float(
    os.environ.get("CONSOLIDATION_MEMORY_RECALL_FALLBACK_TIMEOUT_SECONDS", "10")
)
_CLIENT_INIT_TIMEOUT_SECONDS = float(
    # Cold-start imports (faiss/embedding deps) can exceed 20s on Windows;
    # keep a conservative default to avoid first-call MCP timeouts.
    os.environ.get("CONSOLIDATION_MEMORY_CLIENT_INIT_TIMEOUT_SECONDS", "45")
)
_MCP_BLOCKING_WORKERS = max(
    1,
    int(os.environ.get("CONSOLIDATION_MEMORY_MCP_BLOCKING_WORKERS", "16")),
)
_WARMUP_ON_START = os.environ.get(
    "CONSOLIDATION_MEMORY_WARMUP_ON_START",
    "1",
).strip().lower() not in {"0", "false", "no", "off"}
_IDLE_TIMEOUT_SECONDS = float(
    os.environ.get("CONSOLIDATION_MEMORY_IDLE_TIMEOUT_SECONDS", "0")
)
_IDLE_CHECK_INTERVAL_SECONDS = float(
    os.environ.get("CONSOLIDATION_MEMORY_IDLE_CHECK_INTERVAL_SECONDS", "15")
)
_DUMP_STACKS_ON_CLIENT_INIT_TIMEOUT = os.environ.get(
    "CONSOLIDATION_MEMORY_DUMP_STACKS_ON_CLIENT_INIT_TIMEOUT",
    "0",
).strip().lower() in {"1", "true", "yes", "on"}
_PRELOAD_NUMERIC_BACKENDS_ON_START = os.environ.get(
    "CONSOLIDATION_MEMORY_PRELOAD_NUMERIC_BACKENDS_ON_START",
    "1",
).strip().lower() not in {"0", "false", "no", "off"}

_warmup_task: asyncio.Task | None = None
_idle_task: asyncio.Task | None = None
_active_tool_calls = 0
_last_activity_monotonic = time.monotonic()
_blocking_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=_MCP_BLOCKING_WORKERS,
    thread_name_prefix="consolidation_memory_mcp",
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


def _warmup_on_start() -> bool:
    return _WARMUP_ON_START


def _preload_numeric_backends() -> None:
    """Preload numpy/faiss on the main thread to avoid worker-thread import stalls."""
    if not _PRELOAD_NUMERIC_BACKENDS_ON_START:
        return
    started = time.monotonic()
    try:
        import numpy  # noqa: F401
        import faiss  # noqa: F401
    except Exception as exc:
        logger.warning("Numeric backend preload failed: %s", exc)
        return
    logger.info("Preloaded numpy/faiss in %.3fs", time.monotonic() - started)


def _idle_timeout_seconds() -> float:
    configured = _IDLE_TIMEOUT_SECONDS
    return configured if configured > 0 else 0.0


def _idle_check_interval_seconds() -> float:
    configured = _IDLE_CHECK_INTERVAL_SECONDS
    return configured if configured > 0 else 15.0


def _touch_activity() -> None:
    global _last_activity_monotonic
    _last_activity_monotonic = time.monotonic()


def _begin_tool_call() -> None:
    global _active_tool_calls
    _active_tool_calls += 1
    _touch_activity()


def _end_tool_call() -> None:
    global _active_tool_calls
    _active_tool_calls = max(0, _active_tool_calls - 1)
    _touch_activity()


async def _run_blocking(
    func: Callable[..., _T],
    /,
    *args: object,
    timeout: float | None = None,
    **kwargs: object,
) -> _T:
    """Run blocking work on a dedicated executor to avoid default-pool starvation."""
    _touch_activity()
    loop = asyncio.get_running_loop()
    work = functools.partial(func, *args, **kwargs)
    future = loop.run_in_executor(_blocking_executor, work)
    if timeout is None:
        return await future
    return await asyncio.wait_for(future, timeout=timeout)


async def _idle_shutdown_monitor() -> None:
    """Exit long-idle MCP server processes to prevent stale-process buildup."""
    timeout_seconds = _idle_timeout_seconds()
    if timeout_seconds <= 0:
        return

    check_seconds = _idle_check_interval_seconds()
    while True:
        await asyncio.sleep(check_seconds)
        if _active_tool_calls > 0:
            continue
        idle_for = time.monotonic() - _last_activity_monotonic
        if idle_for < timeout_seconds:
            continue
        logger.info(
            "MCP server idle for %.1fs (threshold %.1fs); shutting down process",
            idle_for,
            timeout_seconds,
        )
        os._exit(0)


def _get_client():
    """Return the global client, creating it on first access."""
    global _client
    global _client_initializing
    global _client_init_owner_thread_id
    global _client_init_error

    client = _client
    if client is not None:
        return client

    should_initialize = False
    current_thread_id = threading.get_ident()

    with _client_init_cond:
        if _client is not None:
            return _client

        # Re-entrant protection: if initialization path invokes _get_client()
        # again on the same thread, fail fast instead of deadlocking.
        if _client_initializing and _client_init_owner_thread_id == current_thread_id:
            raise RuntimeError(
                "Re-entrant MemoryClient initialization detected. "
                "Avoid calling memory tools/hooks during client startup."
            )

        if not _client_initializing:
            _client_initializing = True
            _client_init_owner_thread_id = current_thread_id
            _client_init_error = None
            should_initialize = True
        else:
            while _client_initializing and _client is None:
                _client_init_cond.wait(timeout=0.5)
            if _client is not None:
                return _client
            if _client_init_error is not None:
                raise RuntimeError(
                    f"MemoryClient initialization failed: {_client_init_error}"
                ) from _client_init_error

    if not should_initialize:
        if _client is None:
            raise RuntimeError("MemoryClient initialization did not complete")
        return _client

    from consolidation_memory.client import MemoryClient

    logger.info("Initializing MemoryClient...")
    try:
        initialized_client = MemoryClient()
    except Exception as exc:
        with _client_init_cond:
            _client_initializing = False
            _client_init_owner_thread_id = None
            _client_init_error = exc
            _client_init_cond.notify_all()
        raise

    with _client_init_cond:
        _client = initialized_client
        _client_initializing = False
        _client_init_owner_thread_id = None
        _client_init_error = None
        _client_init_cond.notify_all()

    return initialized_client


def _format_thread_stacks() -> str:
    """Return a debug string containing stack traces for all live threads."""
    frames = sys._current_frames()
    chunks: list[str] = []
    for thread in threading.enumerate():
        thread_ident = thread.ident
        frame = frames.get(thread_ident) if thread_ident is not None else None
        chunks.append(
            f"\n--- thread name={thread.name!r} ident={thread_ident} daemon={thread.daemon} ---"
        )
        if frame is None:
            chunks.append("<no frame available>")
            continue
        chunks.extend(traceback.format_stack(frame))
    return "\n".join(chunks)


async def _get_client_with_timeout():
    timeout_seconds = _client_init_timeout_seconds()
    try:
        started = time.monotonic()
        client = await _run_blocking(
            _get_client,
            timeout=timeout_seconds,
        )
        elapsed = time.monotonic() - started
        if elapsed > timeout_seconds * 0.8:
            logger.warning(
                "MemoryClient initialization was slow (%.2fs, budget %.2fs)",
                elapsed,
                timeout_seconds,
            )
        return client
    except asyncio.TimeoutError as exc:
        if _DUMP_STACKS_ON_CLIENT_INIT_TIMEOUT:
            logger.error(
                "Client init timeout thread dump:%s",
                _format_thread_stacks(),
            )
        raise TimeoutError(
            f"MemoryClient initialization timed out after {timeout_seconds:g}s. "
            "Retry in a few seconds, or increase "
            "CONSOLIDATION_MEMORY_CLIENT_INIT_TIMEOUT_SECONDS."
        ) from exc


async def _warm_client_background():
    try:
        client = await _get_client_with_timeout()
        await _run_blocking(_warm_recall_caches, client)
    except Exception as exc:
        logger.warning("Background client warmup failed: %s", exc)


def _warm_recall_caches(client=None) -> None:
    """Prime recall caches so first user recall avoids bulk embedding work."""
    from consolidation_memory import claim_cache, record_cache, topic_cache
    from consolidation_memory.config import get_config

    cfg = get_config()
    topic_cache.get_topic_vecs()
    record_cache.get_record_vecs(include_expired=False)
    if client is not None:
        try:
            from consolidation_memory.client import _resolved_scope_to_query_filter

            default_scope_filter = _resolved_scope_to_query_filter(client.resolve_scope())
            record_cache.get_record_vecs(include_expired=False, scope=default_scope_filter)
        except Exception as e:
            logger.debug("Scoped warmup skipped: %s", e)
    warmed_claims = claim_cache.warm_active_claim_vecs(
        limit=max(cfg.RECORDS_MAX_RESULTS * 10, cfg.RECALL_MAX_N * 5)
    )
    logger.info("Warmup complete (claims cached=%d)", warmed_claims)


@asynccontextmanager
async def lifespan(server: FastMCP):
    """Log startup metadata and close any initialized client on shutdown."""
    global _client, _warmup_task, _idle_task
    global _client_initializing, _client_init_owner_thread_id, _client_init_error

    from consolidation_memory import __version__

    logger.info("Starting consolidation_memory MCP server v%s...", __version__)
    from consolidation_memory.config import get_active_project
    logger.info("Active project: %s", get_active_project())
    _preload_numeric_backends()
    if _warmup_on_start():
        _warmup_task = asyncio.create_task(_warm_client_background())
    if _idle_timeout_seconds() > 0:
        _idle_task = asyncio.create_task(_idle_shutdown_monitor())

    yield

    if _idle_task is not None:
        _idle_task.cancel()
        try:
            await _idle_task
        except asyncio.CancelledError:
            pass
        _idle_task = None

    if _warmup_task is not None:
        _warmup_task.cancel()
        try:
            await _warmup_task
        except asyncio.CancelledError:
            pass
        _warmup_task = None

    if _client is not None:
        _client.close()
        _client = None
    _client_initializing = False
    _client_init_owner_thread_id = None
    _client_init_error = None
    logger.info("Shutting down consolidation_memory MCP server.")


mcp = FastMCP(
    "consolidation_memory",
    lifespan=lifespan,
)


def _tracked_tool() -> Callable[[Callable[..., Awaitable[str]]], Callable[..., Awaitable[str]]]:
    """Wrap MCP tools with lightweight activity accounting for idle shutdown."""

    def _decorator(
        func: Callable[..., Awaitable[str]],
    ) -> Callable[..., Awaitable[str]]:
        @mcp.tool()
        @functools.wraps(func)
        async def _wrapped(*args: object, **kwargs: object) -> str:
            _begin_tool_call()
            try:
                return await func(*args, **kwargs)
            finally:
                _end_tool_call()

        return _wrapped

    return _decorator


# ── Tools ────────────────────────────────────────────────────────────────────

@_tracked_tool()
async def memory_store(
    content: str,
    content_type: str = "exchange",
    tags: list[str] | None = None,
    surprise: float = 0.5,
    scope: dict[str, object] | None = None,
) -> str:
    """Store a memory episode in the episodic buffer.

    IMPORTANT: Always store memories when you learn something new about the user,
    solve a problem, discover a preference, or encounter something surprising.
    Write content as a self-contained note that future-you can understand without context.
    Include both the problem AND solution for solution-type memories.
    Do NOT store trivial exchanges like greetings.

    Args:
        content: The text content to store. Include relevant context.
        content_type: One of 'exchange' (conversation), 'fact' (learned info),
                      'solution' (problem+fix), 'preference' (user preference).
        tags: Optional topic tags for organization (e.g., ['vr', 'steamvr']).
        surprise: How novel this is, 0.0 (routine) to 1.0 (very surprising).
        scope: Optional canonical scope envelope for namespace/project/client isolation.
    """
    try:
        client = await _get_client_with_timeout()
        if len(content) > 50_000:
            return json.dumps({"error": "Content exceeds maximum length of 50KB"})
        if scope is not None:
            result = await _run_blocking(
                client.store_with_scope,
                content,
                content_type,
                tags,
                surprise,
                scope,
            )
        else:
            result = await _run_blocking(client.store, content, content_type, tags, surprise)
        return json.dumps(dataclasses.asdict(result), default=str)
    except Exception as e:
        logger.exception("memory_store failed")
        return json.dumps({"error": str(e)})


@_tracked_tool()
async def memory_recall(
    query: str,
    n_results: int = 10,
    include_knowledge: bool = True,
    content_types: list[str] | None = None,
    tags: list[str] | None = None,
    after: str | None = None,
    before: str | None = None,
    include_expired: bool = False,
    as_of: str | None = None,
    scope: dict[str, object] | None = None,
) -> str:
    """Retrieve relevant memories by semantic similarity.

    CRITICAL: You MUST call this at the START of EVERY new conversation, using a
    query that matches the user's opening message topic. Also call when the user
    references past interactions or when context about their setup/preferences
    would improve your response. This is your persistent memory — use it.

    Args:
        query: Natural language description of what to recall.
        n_results: Maximum number of episode results (1-50). Default 10.
        include_knowledge: Whether to include consolidated knowledge. Default True.
        content_types: Filter to specific types (e.g. ['solution', 'fact']).
        tags: Filter to episodes with at least one matching tag.
        after: Only episodes created after this ISO date (e.g. '2025-01-01').
        before: Only episodes created before this ISO date.
        include_expired: Include temporally expired knowledge records. Default False.
        as_of: ISO datetime for temporal belief queries. Returns knowledge state
            at that point in time, including records since superseded.
        scope: Optional canonical scope envelope for namespace/project/client isolation.
    """
    try:
        client = await _get_client_with_timeout()
        n_results = max(1, min(n_results, 50))
        recall_timeout = _recall_timeout_seconds()

        def _run_recall(include_knowledge_flag: bool):
            return client.query_recall(
                query,
                n_results,
                include_knowledge_flag,
                content_types=content_types,
                tags=tags,
                after=after,
                before=before,
                include_expired=include_expired,
                as_of=as_of,
                scope=scope,
            )

        def _run_keyword_fallback():
            return client.query_search(
                query=query,
                content_types=content_types,
                tags=tags,
                after=after,
                before=before,
                limit=n_results,
                scope=scope,
            )

        try:
            result = await _run_blocking(
                _run_recall,
                include_knowledge,
                timeout=recall_timeout,
            )
        except asyncio.TimeoutError:
            # Graceful degradation for heavy recall requests:
            # use keyword-only episode search instead of another semantic recall,
            # so we still return useful results when embedding operations stall.
            fallback_timeout = _recall_fallback_timeout_seconds()
            logger.warning(
                "memory_recall timed out after %.2fs; retrying keyword episodes-only fallback",
                recall_timeout,
            )
            try:
                keyword_result = await _run_blocking(
                    _run_keyword_fallback,
                    timeout=fallback_timeout,
                )
            except asyncio.TimeoutError:
                message = (
                    f"memory_recall timed out after {recall_timeout:g}s and keyword fallback "
                    f"timed out after {fallback_timeout:g}s. "
                    "Try a shorter query, reduce n_results, or set "
                    "CONSOLIDATION_MEMORY_RECALL_TIMEOUT_SECONDS higher."
                )
                logger.error(message)
                return json.dumps({"error": message})
            except Exception as fallback_error:
                message = (
                    f"memory_recall timed out after {recall_timeout:g}s and keyword fallback "
                    f"failed: {fallback_error}"
                )
                logger.error(message)
                return json.dumps({"error": message})

            warnings: list[str] = [
                (
                    f"Recall timed out after {recall_timeout:g}s; "
                    "returned episodes-only fallback."
                )
            ]
            if include_knowledge:
                warnings.append("Knowledge retrieval skipped in fallback mode.")

            payload = {
                "episodes": list(keyword_result.episodes),
                "knowledge": [],
                "records": [],
                "claims": [],
                "total_episodes": int(keyword_result.total_matches),
                "total_knowledge_topics": 0,
                "message": (
                    "Semantic recall timed out; returned keyword episodes-only fallback."
                ),
                "warnings": warnings,
            }
            return json.dumps(payload, default=str)

        return json.dumps(dataclasses.asdict(result), default=str)
    except Exception as e:
        logger.exception("memory_recall failed")
        return json.dumps({"error": str(e)})


@_tracked_tool()
async def memory_store_batch(
    episodes: list[dict],
    scope: dict[str, object] | None = None,
) -> str:
    """Store multiple memory episodes in a single operation.

    More efficient than calling memory_store repeatedly. Single embedding call
    and batch FAISS insertion.

    Args:
        episodes: List of episode objects, each with:
            - content (str, required): The text content to store.
            - content_type (str): One of 'exchange', 'fact', 'solution', 'preference'.
            - tags (list[str]): Optional topic tags.
            - surprise (float): Novelty score 0.0-1.0.
        scope: Optional canonical scope envelope for namespace/project/client isolation.
    """
    try:
        client = await _get_client_with_timeout()
        if len(episodes) > _MAX_BATCH_SIZE:
            return json.dumps({"error": f"Batch size {len(episodes)} exceeds maximum of {_MAX_BATCH_SIZE}"})
        if scope is not None:
            result = await _run_blocking(client.store_batch_with_scope, episodes, scope)
        else:
            result = await _run_blocking(client.store_batch, episodes)
        return json.dumps(dataclasses.asdict(result), default=str)
    except Exception as e:
        logger.exception("memory_store_batch failed")
        return json.dumps({"error": str(e)})


@_tracked_tool()
async def memory_search(
    query: str | None = None,
    content_types: list[str] | None = None,
    tags: list[str] | None = None,
    after: str | None = None,
    before: str | None = None,
    limit: int = 20,
    scope: dict[str, object] | None = None,
) -> str:
    """Keyword/metadata search over episodes. Works without embedding backend.

    Unlike memory_recall (semantic similarity), this does plain text matching
    in SQLite. Use when the embedding backend is down, or for exact substring
    searches. At least one filter parameter should be provided.

    Args:
        query: Text substring to search for in episode content (case-insensitive).
        content_types: Filter to specific types (e.g. ['solution', 'fact']).
        tags: Filter to episodes with at least one matching tag.
        after: Only episodes created after this ISO date (e.g. '2025-01-01').
        before: Only episodes created before this ISO date.
        limit: Maximum results (default 20, max 50).
        scope: Optional canonical scope envelope for namespace/project/client isolation.
    """
    try:
        client = await _get_client_with_timeout()
        result = await _run_blocking(
            lambda: client.query_search(
                query=query,
                content_types=content_types,
                tags=tags,
                after=after,
                before=before,
                limit=min(limit, 50),
                scope=scope,
            )
        )
        return json.dumps(dataclasses.asdict(result), default=str)
    except Exception as e:
        logger.exception("memory_search failed")
        return json.dumps({"error": str(e)})


@_tracked_tool()
async def memory_claim_browse(
    claim_type: str | None = None,
    as_of: str | None = None,
    limit: int = 50,
    scope: dict[str, object] | None = None,
) -> str:
    """Browse claims from the claim graph.

    Args:
        claim_type: Optional claim type filter (e.g. 'fact', 'solution').
        as_of: Optional ISO datetime for temporal claim queries.
        limit: Maximum results (default 50, max 200).
        scope: Optional canonical scope envelope for namespace/project/client isolation.
    """
    try:
        client = await _get_client_with_timeout()
        result = await _run_blocking(
            client.query_browse_claims,
            claim_type=claim_type,
            as_of=as_of,
            limit=min(limit, 200),
            scope=scope,
        )
        return json.dumps(dataclasses.asdict(result), default=str)
    except Exception as e:
        logger.exception("memory_claim_browse failed")
        return json.dumps({"error": str(e)})


@_tracked_tool()
async def memory_claim_search(
    query: str,
    claim_type: str | None = None,
    as_of: str | None = None,
    limit: int = 50,
    scope: dict[str, object] | None = None,
) -> str:
    """Search claims by text with optional temporal snapshot filtering.

    Args:
        query: Search text to match claim canonical text and payload.
        claim_type: Optional claim type filter (e.g. 'fact', 'solution').
        as_of: Optional ISO datetime for temporal claim queries.
        limit: Maximum matches (default 50, max 200).
        scope: Optional canonical scope envelope for namespace/project/client isolation.
    """
    try:
        client = await _get_client_with_timeout()
        result = await _run_blocking(
            client.query_search_claims,
            query=query,
            claim_type=claim_type,
            as_of=as_of,
            limit=min(limit, 200),
            scope=scope,
        )
        return json.dumps(dataclasses.asdict(result), default=str)
    except Exception as e:
        logger.exception("memory_claim_search failed")
        return json.dumps({"error": str(e)})


@_tracked_tool()
async def memory_detect_drift(
    base_ref: str | None = None,
    repo_path: str | None = None,
) -> str:
    """Detect code drift and challenge claims impacted by changed file anchors.

    Args:
        base_ref: Optional git base ref for comparison (e.g. 'origin/main').
        repo_path: Optional repository path (defaults to current working directory).
    """
    try:
        client = await _get_client_with_timeout()
        timeout_seconds = _drift_timeout_seconds()
        result = await _run_blocking(
            client.query_detect_drift,
            base_ref=base_ref,
            repo_path=repo_path,
            timeout=timeout_seconds,
        )
        return json.dumps(result, default=str)
    except asyncio.TimeoutError:
        message = (
            f"memory_detect_drift timed out after {timeout_seconds:g}s. "
            "Try scoping repo_path to a smaller repository or set "
            "CONSOLIDATION_MEMORY_DRIFT_TIMEOUT_SECONDS to a higher value."
        )
        logger.error(message)
        return json.dumps({"error": message})
    except Exception as e:
        logger.exception("memory_detect_drift failed")
        return json.dumps({"error": str(e)})


@_tracked_tool()
async def memory_status() -> str:
    """Show memory system statistics.

    Call this to check the health and state of the memory system.
    """
    try:
        client = await _get_client_with_timeout()
        result = await _run_blocking(client.status)
        return json.dumps(dataclasses.asdict(result), default=str)
    except Exception as e:
        logger.exception("memory_status failed")
        return json.dumps({"error": str(e)})


@_tracked_tool()
async def memory_forget(episode_id: str) -> str:
    """Mark an episode for removal from the memory system.

    Call this to forget specific memories that are incorrect,
    outdated, or that the user wants removed.

    Args:
        episode_id: The UUID of the episode to forget.
    """
    try:
        client = await _get_client_with_timeout()
        result = await _run_blocking(client.forget, episode_id)
        return json.dumps(dataclasses.asdict(result), default=str)
    except Exception as e:
        logger.exception("memory_forget failed")
        return json.dumps({"error": str(e)})


@_tracked_tool()
async def memory_export() -> str:
    """Export all episodes and knowledge to a JSON snapshot.

    Creates a timestamped JSON file in the backups directory containing
    all episodes (non-deleted) and knowledge topics with their content.
    Returns the file path.
    """
    try:
        client = await _get_client_with_timeout()
        result = await _run_blocking(client.export)
        return json.dumps(dataclasses.asdict(result), default=str)
    except Exception as e:
        logger.exception("memory_export failed")
        return json.dumps({"error": str(e)})


@_tracked_tool()
async def memory_correct(topic_filename: str, correction: str) -> str:
    """Correct a knowledge document with new information.

    Use this when you discover that a knowledge document contains outdated
    or incorrect information and needs to be updated.

    Args:
        topic_filename: The filename of the knowledge topic (e.g., 'vr_setup.md').
        correction: Description of what needs to be corrected and the correct information.
    """
    try:
        client = await _get_client_with_timeout()
        result = await _run_blocking(client.correct, topic_filename, correction)
        return json.dumps(dataclasses.asdict(result), default=str)
    except Exception as e:
        logger.exception("memory_correct failed")
        return json.dumps({"error": str(e)})


@_tracked_tool()
async def memory_compact() -> str:
    """Compact the FAISS index by removing tombstoned vectors.

    Call when memory_status shows high tombstone count or ratio.
    Tombstones accumulate from forget and prune operations.
    Compaction rebuilds the index without dead vectors.
    """
    try:
        client = await _get_client_with_timeout()
        result = await _run_blocking(client.compact)
        return json.dumps(dataclasses.asdict(result), default=str)
    except Exception as e:
        logger.exception("memory_compact failed")
        return json.dumps({"error": str(e)})


@_tracked_tool()
async def memory_consolidate() -> str:
    """Manually trigger a consolidation run.

    Clusters unconsolidated episodes by semantic similarity, synthesizes
    knowledge documents via LLM, prunes old episodes, and compacts FAISS.
    Returns a run report. Will refuse if a consolidation is already in progress.

    NOTE: This can take several minutes depending on episode count and LLM speed.
    """
    try:
        client = await _get_client_with_timeout()
        result = await _run_blocking(client.consolidate)
        if isinstance(result, dict) and result.get("status") == "already_running":
            payload = dict(result)
            payload.setdefault("message", "A consolidation run is already in progress")
            return json.dumps(payload, default=str)
        return json.dumps(result, default=str)
    except Exception as e:
        logger.exception("memory_consolidate failed")
        return json.dumps({"error": str(e)})


@_tracked_tool()
async def memory_consolidation_log(last_n: int = 5) -> str:
    """Show recent consolidation activity as a human-readable changelog.

    Returns summaries of recent consolidation runs: topics created/updated,
    contradictions detected, episodes pruned. Use this to understand what
    the memory system has been doing and how knowledge has changed.

    Args:
        last_n: Number of recent runs to show (1-20, default 5).
    """
    try:
        client = await _get_client_with_timeout()
        last_n = max(1, min(last_n, 20))
        result = await _run_blocking(client.consolidation_log, last_n)
        return json.dumps(dataclasses.asdict(result), default=str)
    except Exception as e:
        logger.exception("memory_consolidation_log failed")
        return json.dumps({"error": str(e)})


@_tracked_tool()
async def memory_decay_report() -> str:
    """Show what would be forgotten if pruning ran right now.

    Reports prunable episodes (consolidated and older than threshold),
    low-confidence records, and protected episode counts.
    Does NOT actually delete anything — just reports.
    """
    try:
        client = await _get_client_with_timeout()
        result = await _run_blocking(client.decay_report)
        return json.dumps(dataclasses.asdict(result), default=str)
    except Exception as e:
        logger.exception("memory_decay_report failed")
        return json.dumps({"error": str(e)})


@_tracked_tool()
async def memory_protect(
    episode_id: str | None = None,
    tag: str | None = None,
) -> str:
    """Mark episodes as immune to pruning.

    Protect specific episodes or all episodes with a given tag from
    being pruned during consolidation. Use this for important memories
    that should never be forgotten.

    Args:
        episode_id: Protect a specific episode by its UUID.
        tag: Protect all episodes with this tag.
    """
    try:
        client = await _get_client_with_timeout()
        result = await _run_blocking(client.protect, episode_id, tag)
        return json.dumps(dataclasses.asdict(result), default=str)
    except Exception as e:
        logger.exception("memory_protect failed")
        return json.dumps({"error": str(e)})


@_tracked_tool()
async def memory_timeline(topic: str) -> str:
    """Show how understanding of a topic has changed over time.

    Returns all knowledge records matching the topic sorted chronologically,
    including expired/superseded records. Shows what was believed, when it
    changed, and what replaced it. Useful for questions like "how has my
    understanding of X evolved?"

    Args:
        topic: Natural language topic to query (e.g., 'frontend framework preference').
    """
    try:
        client = await _get_client_with_timeout()
        result = await _run_blocking(client.timeline, topic)
        return json.dumps(dataclasses.asdict(result), default=str)
    except Exception as e:
        logger.exception("memory_timeline failed")
        return json.dumps({"error": str(e)})


@_tracked_tool()
async def memory_contradictions(topic: str | None = None) -> str:
    """List detected contradictions from the audit log.

    Shows cases where knowledge records contradicted each other during
    consolidation, including both the old and new content and how it
    was resolved. Use this to review belief changes over time.

    Args:
        topic: Optional topic filename or title to filter results.
    """
    try:
        client = await _get_client_with_timeout()
        result = await _run_blocking(client.contradictions, topic)
        return json.dumps(dataclasses.asdict(result), default=str)
    except Exception as e:
        logger.exception("memory_contradictions failed")
        return json.dumps({"error": str(e)})


@_tracked_tool()
async def memory_browse() -> str:
    """Browse all knowledge topics with summaries and metadata.

    Returns a list of all knowledge topics including titles, summaries,
    record counts by type, confidence scores, and file paths. Use this
    to see what the memory system has learned and consolidated.
    """
    try:
        client = await _get_client_with_timeout()
        result = await _run_blocking(client.browse)
        return json.dumps(dataclasses.asdict(result), default=str)
    except Exception as e:
        logger.exception("memory_browse failed")
        return json.dumps({"error": str(e)})


@_tracked_tool()
async def memory_read_topic(filename: str) -> str:
    """Read the full markdown content of a knowledge topic.

    Use memory_browse first to see available topics, then read specific
    ones to see the full details including all extracted facts, solutions,
    preferences, and procedures.

    Args:
        filename: The filename of the knowledge topic (e.g., 'python_setup.md').
    """
    try:
        client = await _get_client_with_timeout()
        import re as _re
        if _re.search(r"[/\\]|\.\.", filename):
            return json.dumps({"error": "Invalid filename: must not contain '/', '\\', or '..'."})
        result = await _run_blocking(client.read_topic, filename)
        return json.dumps(dataclasses.asdict(result), default=str)
    except Exception as e:
        logger.exception("memory_read_topic failed")
        return json.dumps({"error": str(e)})


# ── Entry point ──────────────────────────────────────────────────────────────

def run_server():
    """Run the MCP server on stdio transport."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    run_server()

