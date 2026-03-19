"""Consolidation Memory MCP server.

The MCP surface is intentionally thin. Shared lifecycle ownership lives in
MemoryRuntime, canonical tool dispatch lives in tool_dispatch, and this module
keeps only MCP-specific timeout and fallback behavior.
"""

from __future__ import annotations

import asyncio
import functools
import gc
import hashlib
import json
import logging
import math
import os
import signal
import sys
import tempfile
import threading
import time
import traceback
from contextlib import asynccontextmanager
from typing import Any, Awaitable, Callable, TypeAlias, TypeVar

from mcp.server.fastmcp import FastMCP

from consolidation_memory.drift_subprocess import run_detect_drift_subprocess
from consolidation_memory.runtime import MemoryRuntime
from consolidation_memory.tool_dispatch import execute_tool_call

# Configure logging to stderr (stdout is the MCP JSON-RPC channel).
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("consolidation_memory")

_T = TypeVar("_T")

_MAX_BATCH_SIZE = 100
_WARMUP_ON_START = os.environ.get(
    "CONSOLIDATION_MEMORY_WARMUP_ON_START",
    "1",
).strip().lower() not in {"0", "false", "no", "off"}
_DUMP_STACKS_ON_CLIENT_INIT_TIMEOUT = os.environ.get(
    "CONSOLIDATION_MEMORY_DUMP_STACKS_ON_CLIENT_INIT_TIMEOUT",
    "0",
).strip().lower() in {"1", "true", "yes", "on"}
_PRELOAD_NUMERIC_BACKENDS_ON_START = os.environ.get(
    "CONSOLIDATION_MEMORY_PRELOAD_NUMERIC_BACKENDS_ON_START",
    "1",
).strip().lower() not in {"0", "false", "no", "off"}
_STDIO_SINGLETON_ENABLED = os.environ.get(
    "CONSOLIDATION_MEMORY_STDIO_SINGLETON",
    "1",
).strip().lower() not in {"0", "false", "no", "off"}


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


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    token = raw.strip()
    if not token:
        return default
    try:
        return int(token)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    token = raw.strip().lower()
    if not token:
        return default
    if token in {"1", "true", "yes", "on"}:
        return True
    if token in {"0", "false", "no", "off"}:
        return False
    return default


_MEMORY_DETECT_DRIFT_TIMEOUT_SECONDS = _env_float(
    "CONSOLIDATION_MEMORY_DRIFT_TIMEOUT_SECONDS",
    90.0,
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
_MCP_BLOCKING_WORKERS = max(
    1,
    _env_int("CONSOLIDATION_MEMORY_MCP_BLOCKING_WORKERS", 16),
)
_IDLE_TIMEOUT_SECONDS = _env_float(
    "CONSOLIDATION_MEMORY_IDLE_TIMEOUT_SECONDS",
    900.0,
)
_IDLE_CHECK_INTERVAL_SECONDS = _env_float(
    "CONSOLIDATION_MEMORY_IDLE_CHECK_INTERVAL_SECONDS",
    15.0,
)
_WARMUP_START_DELAY_SECONDS = _env_float(
    "CONSOLIDATION_MEMORY_WARMUP_START_DELAY_SECONDS",
    0.25,
)
_STDIO_SINGLETON_TAKEOVER_TIMEOUT_SECONDS = _env_float(
    "CONSOLIDATION_MEMORY_STDIO_SINGLETON_TAKEOVER_TIMEOUT_SECONDS",
    10.0,
)
_MCP_AUTO_CONSOLIDATE = _env_bool(
    "CONSOLIDATION_MEMORY_MCP_AUTO_CONSOLIDATE",
    False,
)


def _mcp_client_factory():
    """Create the MCP-owned client with MCP-safe lifecycle defaults."""
    from consolidation_memory.client import MemoryClient

    return MemoryClient(auto_consolidate=_MCP_AUTO_CONSOLIDATE)


_runtime = MemoryRuntime(
    client_factory=_mcp_client_factory,
    max_workers=_MCP_BLOCKING_WORKERS,
)
_warmup_task: asyncio.Task | None = None
_idle_task: asyncio.Task | None = None
_active_tool_calls = 0
_last_activity_monotonic = time.monotonic()
_runtime_started = False
_startup_error: Exception | None = None
_runtime_start_lock = threading.Lock()
_stdio_singleton_guard = None
ScopeInput: TypeAlias = dict[str, object] | str | None

if os.name == "nt":
    import msvcrt
    _msvcrt_locking: Callable[[int, int, int], Any] = getattr(msvcrt, "locking")
    _msvcrt_lk_nblck = int(getattr(msvcrt, "LK_NBLCK"))
    _msvcrt_lk_unlck = int(getattr(msvcrt, "LK_UNLCK"))
else:
    import fcntl
    _fcntl_flock: Callable[[int, int], Any] = getattr(fcntl, "flock")
    _fcntl_lock_ex = int(getattr(fcntl, "LOCK_EX"))
    _fcntl_lock_nb = int(getattr(fcntl, "LOCK_NB"))
    _fcntl_lock_un = int(getattr(fcntl, "LOCK_UN"))


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


def _stdio_singleton_takeover_timeout_seconds() -> float:
    configured = _STDIO_SINGLETON_TAKEOVER_TIMEOUT_SECONDS
    return configured if configured > 0 else 10.0


def _touch_activity() -> None:
    global _last_activity_monotonic
    _last_activity_monotonic = time.monotonic()


def _format_startup_error(exc: Exception) -> str:
    return (
        "MCP runtime startup failed: "
        f"{exc}. Fix the underlying environment/config issue and retry the tool call."
    )


def _ensure_runtime_started() -> None:
    """Start the shared runtime on demand without killing the MCP transport on failure."""
    global _runtime_started, _startup_error

    if _runtime_started and _startup_error is None:
        return

    with _runtime_start_lock:
        if _runtime_started and _startup_error is None:
            return
        try:
            _runtime.startup()
        except Exception as exc:
            _runtime_started = False
            _startup_error = exc
            raise RuntimeError(_format_startup_error(exc)) from exc

        _runtime_started = True
        _startup_error = None


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
    _ensure_runtime_started()
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


class _StdioSingletonGuard:
    """Hold a parent-scoped inter-process lock for stdio MCP servers."""

    def __init__(
        self,
        *,
        path: str,
        handle,
        metadata: dict[str, object],
    ) -> None:
        self.path = path
        self._handle = handle
        self.metadata = metadata
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        try:
            _unlock_singleton_handle(self._handle)
        finally:
            self._handle.close()


def _safe_process_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return None
        if not cleaned.isdigit():
            return None
        try:
            parsed = int(cleaned)
        except ValueError:
            return None
        return parsed if parsed > 0 else None
    return None


def _singleton_lock_path(*, project: str, parent_pid: int) -> str:
    safe_project = "".join(
        ch if ch.isalnum() or ch in {"-", "_", "."} else "_"
        for ch in project
    ).strip("._")
    if not safe_project:
        safe_project = "default"
    safe_project = safe_project[:48]
    fingerprint = hashlib.sha256(f"{project}:{parent_pid}".encode("utf-8")).hexdigest()[:12]
    filename = (
        f"consolidation_memory_stdio_{safe_project}_{parent_pid}_{fingerprint}.lock"
    )
    return os.path.join(tempfile.gettempdir(), filename)


def _open_singleton_lock_handle(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, "a+", encoding="utf-8")


def _try_lock_singleton_handle(handle) -> bool:
    handle.seek(0, os.SEEK_SET)
    try:
        if os.name == "nt":
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(" ")
                handle.flush()
            handle.seek(0, os.SEEK_SET)
            _msvcrt_locking(handle.fileno(), _msvcrt_lk_nblck, 1)
        else:
            _fcntl_flock(handle.fileno(), _fcntl_lock_ex | _fcntl_lock_nb)
        return True
    except OSError:
        return False


def _unlock_singleton_handle(handle) -> None:
    handle.seek(0, os.SEEK_SET)
    if os.name == "nt":
        try:
            _msvcrt_locking(handle.fileno(), _msvcrt_lk_unlck, 1)
        except OSError:
            return
    else:
        _fcntl_flock(handle.fileno(), _fcntl_lock_un)


def _read_singleton_metadata(handle) -> dict[str, object]:
    handle.seek(0, os.SEEK_SET)
    raw = handle.read().strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _write_singleton_metadata(handle, metadata: dict[str, object]) -> None:
    handle.seek(0, os.SEEK_SET)
    handle.truncate()
    handle.write(json.dumps(metadata, sort_keys=True))
    handle.flush()
    try:
        os.fsync(handle.fileno())
    except OSError:
        pass


def _process_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _terminate_process(pid: int) -> None:
    if pid <= 1 or pid in {os.getpid(), os.getppid()}:
        return
    os.kill(pid, signal.SIGTERM)


def _wait_for_process_exit(pid: int, *, timeout: float) -> bool:
    deadline = time.monotonic() + max(0.0, timeout)
    while time.monotonic() < deadline:
        if not _process_exists(pid):
            return True
        time.sleep(0.1)
    return not _process_exists(pid)


def _runtime_has_background_activity() -> bool:
    if _warmup_task is not None and not _warmup_task.done():
        return True
    if _runtime.client_initializing:
        return True
    client = _runtime.client
    if client is None:
        return False

    future = getattr(client, "_consolidation_future", None)
    if future is not None and not future.done():
        return True

    consolidation_lock = getattr(client, "_consolidation_lock", None)
    if consolidation_lock is not None and hasattr(consolidation_lock, "locked"):
        try:
            if consolidation_lock.locked():
                return True
        except Exception:
            return True
    return False


def _recycle_idle_runtime() -> None:
    """Release heavyweight runtime state without closing the stdio transport.

    Exiting the MCP server process on idle makes the transport unusable until the
    client explicitly reconnects. Instead, drop the client/executor state and let
    the next tool call lazily restart the runtime in-process.
    """
    global _runtime_started, _startup_error

    if not _runtime_started and _runtime.client is None:
        _touch_activity()
        return

    logger.info("Recycling idle MCP runtime state without exiting the process")
    _runtime.shutdown()
    _runtime_started = False
    _startup_error = None
    _touch_activity()
    gc.collect()


def _acquire_parent_scoped_stdio_singleton_guard(project: str) -> _StdioSingletonGuard | None:
    """Ensure a parent process owns at most one stdio MCP server per project."""
    if not _STDIO_SINGLETON_ENABLED:
        return None

    parent_pid = os.getppid()
    lock_path = _singleton_lock_path(project=project, parent_pid=parent_pid)
    deadline = time.monotonic() + _stdio_singleton_takeover_timeout_seconds()

    while True:
        handle = _open_singleton_lock_handle(lock_path)
        if _try_lock_singleton_handle(handle):
            metadata = {
                "pid": os.getpid(),
                "parent_pid": parent_pid,
                "project": project,
                "acquired_at": time.time(),
            }
            _write_singleton_metadata(handle, metadata)
            return _StdioSingletonGuard(path=lock_path, handle=handle, metadata=metadata)

        owner = _read_singleton_metadata(handle)
        handle.close()

        owner_pid = _safe_process_int(owner.get("pid"))
        owner_parent_pid = _safe_process_int(owner.get("parent_pid"))
        owner_project = owner.get("project")

        if owner_pid is None or not _process_exists(owner_pid):
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"Timed out recovering stale stdio singleton lock at {lock_path}."
                )
            time.sleep(0.05)
            continue

        if owner_parent_pid != parent_pid or owner_project != project:
            raise RuntimeError(
                "Found a conflicting stdio singleton lock with mismatched owner metadata."
            )

        raise RuntimeError(
            "Another MCP stdio server is already running for this parent process and project. "
            "Wait for it to exit, or set CONSOLIDATION_MEMORY_STDIO_SINGLETON=0 to disable "
            "the singleton guard."
        )


async def _get_client_with_timeout():
    timeout_seconds = _client_init_timeout_seconds()
    try:
        _ensure_runtime_started()
        started = time.monotonic()
        client = await _runtime.get_client_with_timeout(timeout=timeout_seconds)
        elapsed = time.monotonic() - started
        if elapsed > timeout_seconds * 0.8:
            logger.warning(
                "MemoryClient initialization was slow (%.2fs, budget %.2fs)",
                elapsed,
                timeout_seconds,
            )
        return client
    except TimeoutError as exc:
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
    """Recycle long-idle runtime state without breaking the stdio transport."""
    timeout_seconds = _idle_timeout_seconds()
    if timeout_seconds <= 0:
        return

    check_seconds = _idle_check_interval_seconds()
    while True:
        await asyncio.sleep(check_seconds)
        if _active_tool_calls > 0:
            continue
        if _runtime_has_background_activity():
            continue
        idle_for = time.monotonic() - _last_activity_monotonic
        if idle_for < timeout_seconds:
            continue
        logger.info(
            "MCP server idle for %.1fs (threshold %.1fs); recycling runtime state",
            idle_for,
            timeout_seconds,
        )
        _recycle_idle_runtime()


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


def _degraded_drift_output(*, message: str) -> dict[str, object]:
    return {
        "checked_anchors": [],
        "impacted_claim_ids": [],
        "challenged_claim_ids": [],
        "impacts": [],
        "message": message,
    }


@asynccontextmanager
async def lifespan(server: FastMCP):
    """Start and stop the runtime-owned MCP server resources."""
    del server
    global _warmup_task, _idle_task, _runtime_started, _startup_error

    from consolidation_memory import __version__
    from consolidation_memory.config import get_active_project

    logger.info("Starting consolidation_memory MCP server v%s...", __version__)
    logger.info("Active project: %s", get_active_project())
    _preload_numeric_backends()
    if _warmup_on_start() and _runtime_started and _startup_error is None:
        _warmup_task = asyncio.create_task(_warm_client_background())
    elif _warmup_on_start():
        logger.debug(
            "Deferring MemoryClient warmup until the first tool call so MCP initialize stays lazy"
        )
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
    _runtime_started = False
    _startup_error = None
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
    scope: ScopeInput = None,
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
    scope: ScopeInput = None,
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
        except (TimeoutError, asyncio.TimeoutError):
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
            except (TimeoutError, asyncio.TimeoutError):
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
    scope: ScopeInput = None,
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
    scope: ScopeInput = None,
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
    scope: ScopeInput = None,
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
    scope: ScopeInput = None,
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
async def memory_outcome_record(
    action_summary: str,
    outcome_type: str,
    source_claim_ids: list[str] | None = None,
    source_record_ids: list[str] | None = None,
    source_episode_ids: list[str] | None = None,
    code_anchors: list[dict[str, str]] | None = None,
    issue_ids: list[str] | None = None,
    pr_ids: list[str] | None = None,
    action_key: str | None = None,
    summary: str | None = None,
    details: dict[str, Any] | str | None = None,
    confidence: float = 0.8,
    provenance: dict[str, Any] | str | None = None,
    observed_at: str | None = None,
    scope: ScopeInput = None,
) -> str:
    """Record an action outcome observation with provenance links."""
    return await _call_tool_json(
        "memory_outcome_record",
        {
            "action_summary": action_summary,
            "outcome_type": outcome_type,
            "source_claim_ids": source_claim_ids,
            "source_record_ids": source_record_ids,
            "source_episode_ids": source_episode_ids,
            "code_anchors": code_anchors,
            "issue_ids": issue_ids,
            "pr_ids": pr_ids,
            "action_key": action_key,
            "summary": summary,
            "details": details,
            "confidence": confidence,
            "provenance": provenance,
            "observed_at": observed_at,
            "scope": scope,
        },
    )


@_tracked_tool()
async def memory_outcome_browse(
    outcome_type: str | None = None,
    action_key: str | None = None,
    source_claim_id: str | None = None,
    source_record_id: str | None = None,
    source_episode_id: str | None = None,
    as_of: str | None = None,
    limit: int = 50,
    scope: ScopeInput = None,
) -> str:
    """Browse recorded action outcomes with optional filters."""
    return await _call_tool_json(
        "memory_outcome_browse",
        {
            "outcome_type": outcome_type,
            "action_key": action_key,
            "source_claim_id": source_claim_id,
            "source_record_id": source_record_id,
            "source_episode_id": source_episode_id,
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
    timeout_seconds = _drift_timeout_seconds()
    try:
        result = await run_detect_drift_subprocess(
            base_ref=base_ref,
            repo_path=repo_path,
            timeout_seconds=timeout_seconds,
        )
        return json.dumps(result, default=str)
    except (TimeoutError, asyncio.TimeoutError):
        fallback_timeout = max(5.0, min(20.0, timeout_seconds * 0.2))
        if base_ref:
            logger.warning(
                "memory_detect_drift timed out after %.2fs with base_ref=%r; retrying fallback without base_ref",
                timeout_seconds,
                base_ref,
            )
            try:
                fallback_result = await run_detect_drift_subprocess(
                    base_ref=None,
                    repo_path=repo_path,
                    timeout_seconds=fallback_timeout,
                )
                payload: dict[str, object] = dict(fallback_result)
                payload["message"] = (
                    f"memory_detect_drift timed out after {timeout_seconds:g}s using base_ref={base_ref!r}; "
                    "returned fallback scan without base_ref."
                )
                return json.dumps(payload, default=str)
            except (TimeoutError, asyncio.TimeoutError):
                logger.error(
                    "memory_detect_drift fallback without base_ref timed out after %.2fs",
                    fallback_timeout,
                )
            except Exception:
                logger.exception("memory_detect_drift fallback without base_ref failed")

        message = (
            f"memory_detect_drift timed out after {timeout_seconds:g}s. "
            "Returned a degraded empty result instead of failing."
        )
        logger.error(message)
        return json.dumps(_degraded_drift_output(message=message), default=str)
    except Exception as exc:
        logger.exception("memory_detect_drift failed")
        return json.dumps({"error": str(exc)})


@_tracked_tool()
async def memory_status() -> str:
    """Show memory system statistics."""
    return await _call_tool_json("memory_status", {})


@_tracked_tool()
async def memory_forget(
    episode_id: str,
    scope: ScopeInput = None,
) -> str:
    """Mark an episode for removal from the memory system."""
    return await _call_tool_json("memory_forget", {"episode_id": episode_id, "scope": scope})


@_tracked_tool()
async def memory_export(scope: ScopeInput = None) -> str:
    """Export all episodes and knowledge to a JSON snapshot."""
    return await _call_tool_json("memory_export", {"scope": scope})


@_tracked_tool()
async def memory_correct(
    topic_filename: str,
    correction: str,
    scope: ScopeInput = None,
) -> str:
    """Correct a knowledge document with new information."""
    return await _call_tool_json(
        "memory_correct",
        {"topic_filename": topic_filename, "correction": correction, "scope": scope},
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
    scope: ScopeInput = None,
) -> str:
    """Mark episodes as immune to pruning."""
    return await _call_tool_json(
        "memory_protect",
        {"episode_id": episode_id, "tag": tag, "scope": scope},
    )


@_tracked_tool()
async def memory_timeline(topic: str, scope: ScopeInput = None) -> str:
    """Show how understanding of a topic has changed over time."""
    return await _call_tool_json("memory_timeline", {"topic": topic, "scope": scope})


@_tracked_tool()
async def memory_contradictions(topic: str | None = None) -> str:
    """List detected contradictions from the audit log."""
    return await _call_tool_json("memory_contradictions", {"topic": topic})


@_tracked_tool()
async def memory_browse(scope: ScopeInput = None) -> str:
    """Browse all knowledge topics with summaries and metadata."""
    return await _call_tool_json("memory_browse", {"scope": scope})


@_tracked_tool()
async def memory_read_topic(
    filename: str,
    scope: ScopeInput = None,
) -> str:
    """Read the full markdown content of a knowledge topic."""
    return await _call_tool_json("memory_read_topic", {"filename": filename, "scope": scope})


def run_server() -> None:
    """Run the MCP server on stdio transport."""
    from consolidation_memory.config import get_active_project

    global _stdio_singleton_guard

    project = get_active_project()
    _stdio_singleton_guard = _acquire_parent_scoped_stdio_singleton_guard(project)
    try:
        mcp.run(transport="stdio")
    finally:
        guard = _stdio_singleton_guard
        _stdio_singleton_guard = None
        if guard is not None:
            guard.release()


if __name__ == "__main__":
    run_server()
