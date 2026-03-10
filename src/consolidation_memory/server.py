"""Consolidation Memory MCP server.

The MCP surface is intentionally thin. Shared lifecycle ownership lives in
MemoryRuntime, canonical tool dispatch lives in tool_dispatch, and this module
keeps only MCP-specific timeout and fallback behavior.
"""

from __future__ import annotations

import asyncio
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

from consolidation_memory.runtime import MemoryRuntime
from consolidation_memory.tool_dispatch import execute_tool_call
from consolidation_memory.types import DriftOutput

# Configure logging to stderr (stdout is the MCP JSON-RPC channel).
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("consolidation_memory")

_T = TypeVar("_T")

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
_WARMUP_START_DELAY_SECONDS = float(
    os.environ.get("CONSOLIDATION_MEMORY_WARMUP_START_DELAY_SECONDS", "0.25")
)

_runtime = MemoryRuntime(max_workers=_MCP_BLOCKING_WORKERS)
_warmup_task: asyncio.Task | None = None
_idle_task: asyncio.Task | None = None
_active_tool_calls = 0
_last_activity_monotonic = time.monotonic()


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


def _warmup_start_delay_seconds() -> float:
    configured = _WARMUP_START_DELAY_SECONDS
    return configured if configured > 0 else 0.0


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
    """Run blocking work on the runtime-owned executor."""
    _touch_activity()
    return await _runtime.run_blocking(func, *args, timeout=timeout, **kwargs)


def _get_client():
    """Return the runtime-owned client, creating it lazily once."""
    return _runtime.get_client()


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
        client = await _run_blocking(_get_client, timeout=timeout_seconds)
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
            logger.error("Client init timeout thread dump:%s", _format_thread_stacks())
        raise TimeoutError(
            f"MemoryClient initialization timed out after {timeout_seconds:g}s. "
            "Retry in a few seconds, or increase "
            "CONSOLIDATION_MEMORY_CLIENT_INIT_TIMEOUT_SECONDS."
        ) from exc


def _preload_numeric_backends() -> None:
    """Preload numpy/faiss on the main thread to avoid worker-thread import stalls."""
    if not _PRELOAD_NUMERIC_BACKENDS_ON_START:
        return
    started = time.monotonic()
    try:
        import faiss  # noqa: F401
        import numpy  # noqa: F401
    except Exception as exc:
        logger.warning("Numeric backend preload failed: %s", exc)
        return
    logger.info("Preloaded numpy/faiss in %.3fs", time.monotonic() - started)


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


async def _warm_client_background() -> None:
    try:
        delay_seconds = _warmup_start_delay_seconds()
        if delay_seconds > 0:
            await asyncio.sleep(delay_seconds)
        if _runtime.shutting_down:
            return
        client = await _get_client_with_timeout()
        await _run_blocking(_warm_recall_caches, client)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning("Background client warmup failed: %s", exc)


def _warm_recall_caches(client=None) -> None:
    """Prime recall caches so first user recall avoids bulk embedding work."""
    from consolidation_memory import claim_cache, record_cache, topic_cache
    from consolidation_memory.client import _resolved_scope_to_query_filter
    from consolidation_memory.config import get_config

    cfg = get_config()
    topic_cache.get_topic_vecs()
    record_cache.get_record_vecs(include_expired=False)
    if client is not None:
        try:
            default_scope_filter = _resolved_scope_to_query_filter(client.resolve_scope())
            record_cache.get_record_vecs(include_expired=False, scope=default_scope_filter)
        except Exception as exc:
            logger.debug("Scoped warmup skipped: %s", exc)
    warmed_claims = claim_cache.warm_active_claim_vecs(
        limit=max(cfg.RECORDS_MAX_RESULTS * 10, cfg.RECALL_MAX_N * 5)
    )
    logger.info("Warmup complete (claims cached=%d)", warmed_claims)


async def _call_tool_payload(
    name: str,
    arguments: dict[str, object],
    *,
    timeout: float | None = None,
) -> dict[str, object]:
    client = await _get_client_with_timeout()
    return await _run_blocking(
        execute_tool_call,
        name,
        arguments,
        client=client,
        timeout=timeout,
    )


async def _call_tool_json(
    name: str,
    arguments: dict[str, object],
    *,
    timeout: float | None = None,
) -> str:
    try:
        result = await _call_tool_payload(name, arguments, timeout=timeout)
        return json.dumps(result, default=str)
    except Exception as exc:
        logger.exception("%s failed", name)
        return json.dumps({"error": str(exc)})


def _run_detect_drift(
    *,
    base_ref: str | None = None,
    repo_path: str | None = None,
) -> DriftOutput:
    from consolidation_memory.drift import detect_code_drift

    result = detect_code_drift(base_ref=base_ref, repo_path=repo_path)
    logger.info(
        "Detect drift base_ref=%r repo_path=%r impacted=%d challenged=%d",
        base_ref,
        repo_path,
        len(result.get("impacted_claim_ids", [])),
        len(result.get("challenged_claim_ids", [])),
    )
    return result


@asynccontextmanager
async def lifespan(server: FastMCP):
    """Start and stop the runtime-owned MCP server resources."""
    del server
    global _warmup_task, _idle_task

    from consolidation_memory import __version__
    from consolidation_memory.config import get_active_project

    logger.info("Starting consolidation_memory MCP server v%s...", __version__)
    _runtime.startup()
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

    _runtime.shutdown()
    logger.info("Shutting down consolidation_memory MCP server.")


mcp = FastMCP("consolidation_memory", lifespan=lifespan)


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


@_tracked_tool()
async def memory_store(
    content: str,
    content_type: str = "exchange",
    tags: list[str] | None = None,
    surprise: float = 0.5,
    scope: dict[str, object] | None = None,
) -> str:
    """Store a memory episode in the episodic buffer."""
    return await _call_tool_json(
        "memory_store",
        {
            "content": content,
            "content_type": content_type,
            "tags": tags,
            "surprise": surprise,
            "scope": scope,
        },
    )


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
    """Retrieve relevant memories by semantic similarity."""
    try:
        client = await _get_client_with_timeout()
        bounded_n_results = max(1, min(n_results, 50))
        arguments = {
            "query": query,
            "n_results": bounded_n_results,
            "include_knowledge": include_knowledge,
            "content_types": content_types,
            "tags": tags,
            "after": after,
            "before": before,
            "include_expired": include_expired,
            "as_of": as_of,
            "scope": scope,
        }
        recall_timeout = _recall_timeout_seconds()

        try:
            result = await _run_blocking(
                execute_tool_call,
                "memory_recall",
                arguments,
                client=client,
                timeout=recall_timeout,
            )
        except asyncio.TimeoutError:
            fallback_timeout = _recall_fallback_timeout_seconds()
            logger.warning(
                "memory_recall timed out after %.2fs; retrying keyword episodes-only fallback",
                recall_timeout,
            )
            try:
                keyword_result = await _run_blocking(
                    execute_tool_call,
                    "memory_search",
                    {
                        "query": query,
                        "content_types": content_types,
                        "tags": tags,
                        "after": after,
                        "before": before,
                        "limit": bounded_n_results,
                        "scope": scope,
                    },
                    client=client,
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

            warnings = [
                f"Recall timed out after {recall_timeout:g}s; returned episodes-only fallback."
            ]
            if include_knowledge:
                warnings.append("Knowledge retrieval skipped in fallback mode.")

            payload = {
                "episodes": list(keyword_result.get("episodes", [])),
                "knowledge": [],
                "records": [],
                "claims": [],
                "total_episodes": int(
                    keyword_result.get(
                        "total_matches",
                        len(keyword_result.get("episodes", [])),
                    )
                ),
                "total_knowledge_topics": 0,
                "message": "Semantic recall timed out; returned keyword episodes-only fallback.",
                "warnings": warnings,
            }
            return json.dumps(payload, default=str)

        return json.dumps(result, default=str)
    except Exception as exc:
        logger.exception("memory_recall failed")
        return json.dumps({"error": str(exc)})


@_tracked_tool()
async def memory_store_batch(
    episodes: list[dict],
    scope: dict[str, object] | None = None,
) -> str:
    """Store multiple memory episodes in a single operation."""
    return await _call_tool_json(
        "memory_store_batch",
        {"episodes": episodes, "scope": scope},
    )


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
    """Keyword/metadata search over episodes."""
    return await _call_tool_json(
        "memory_search",
        {
            "query": query,
            "content_types": content_types,
            "tags": tags,
            "after": after,
            "before": before,
            "limit": limit,
            "scope": scope,
        },
    )


@_tracked_tool()
async def memory_claim_browse(
    claim_type: str | None = None,
    as_of: str | None = None,
    limit: int = 50,
    scope: dict[str, object] | None = None,
) -> str:
    """Browse claims from the claim graph."""
    return await _call_tool_json(
        "memory_claim_browse",
        {
            "claim_type": claim_type,
            "as_of": as_of,
            "limit": limit,
            "scope": scope,
        },
    )


@_tracked_tool()
async def memory_claim_search(
    query: str,
    claim_type: str | None = None,
    as_of: str | None = None,
    limit: int = 50,
    scope: dict[str, object] | None = None,
) -> str:
    """Search claims by text with optional temporal snapshot filtering."""
    return await _call_tool_json(
        "memory_claim_search",
        {
            "query": query,
            "claim_type": claim_type,
            "as_of": as_of,
            "limit": limit,
            "scope": scope,
        },
    )


@_tracked_tool()
async def memory_detect_drift(
    base_ref: str | None = None,
    repo_path: str | None = None,
) -> str:
    """Detect code drift and challenge impacted claims."""
    try:
        timeout_seconds = _drift_timeout_seconds()
        result = await _run_blocking(
            _run_detect_drift,
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
    except Exception as exc:
        logger.exception("memory_detect_drift failed")
        return json.dumps({"error": str(exc)})


@_tracked_tool()
async def memory_status() -> str:
    """Show memory system statistics."""
    return await _call_tool_json("memory_status", {})


@_tracked_tool()
async def memory_forget(episode_id: str) -> str:
    """Mark an episode for removal from the memory system."""
    return await _call_tool_json("memory_forget", {"episode_id": episode_id})


@_tracked_tool()
async def memory_export() -> str:
    """Export all episodes and knowledge to a JSON snapshot."""
    return await _call_tool_json("memory_export", {})


@_tracked_tool()
async def memory_correct(topic_filename: str, correction: str) -> str:
    """Correct a knowledge document with new information."""
    return await _call_tool_json(
        "memory_correct",
        {"topic_filename": topic_filename, "correction": correction},
    )


@_tracked_tool()
async def memory_compact() -> str:
    """Compact the FAISS index by removing tombstoned vectors."""
    return await _call_tool_json("memory_compact", {})


@_tracked_tool()
async def memory_consolidate() -> str:
    """Manually trigger a consolidation run."""
    try:
        result = await _call_tool_payload("memory_consolidate", {})
        if result.get("status") == "already_running":
            result.setdefault("message", "A consolidation run is already in progress")
        return json.dumps(result, default=str)
    except Exception as exc:
        logger.exception("memory_consolidate failed")
        return json.dumps({"error": str(exc)})


@_tracked_tool()
async def memory_consolidation_log(last_n: int = 5) -> str:
    """Show recent consolidation activity as a human-readable changelog."""
    return await _call_tool_json("memory_consolidation_log", {"last_n": last_n})


@_tracked_tool()
async def memory_decay_report() -> str:
    """Show what would be forgotten if pruning ran right now."""
    return await _call_tool_json("memory_decay_report", {})


@_tracked_tool()
async def memory_protect(
    episode_id: str | None = None,
    tag: str | None = None,
) -> str:
    """Mark episodes as immune to pruning."""
    return await _call_tool_json(
        "memory_protect",
        {"episode_id": episode_id, "tag": tag},
    )


@_tracked_tool()
async def memory_timeline(topic: str) -> str:
    """Show how understanding of a topic has changed over time."""
    return await _call_tool_json("memory_timeline", {"topic": topic})


@_tracked_tool()
async def memory_contradictions(topic: str | None = None) -> str:
    """List detected contradictions from the audit log."""
    return await _call_tool_json("memory_contradictions", {"topic": topic})


@_tracked_tool()
async def memory_browse() -> str:
    """Browse all knowledge topics with summaries and metadata."""
    return await _call_tool_json("memory_browse", {})


@_tracked_tool()
async def memory_read_topic(filename: str) -> str:
    """Read the full markdown content of a knowledge topic."""
    return await _call_tool_json("memory_read_topic", {"filename": filename})


def run_server() -> None:
    """Run the MCP server on stdio transport."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    run_server()
