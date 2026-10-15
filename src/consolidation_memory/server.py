"""Consolidation Memory MCP server.

The MCP surface is intentionally thin. Shared lifecycle ownership lives in
MemoryRuntime, canonical tool dispatch lives in tool_dispatch, and this module
keeps only MCP-specific timeout and fallback behavior.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import functools
import gc
import hashlib
import importlib
import inspect
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
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager, suppress
from typing import Annotated, Any, TypeAlias, TypeVar, cast

from mcp.server import MCPServer
from mcp.types import CallToolResult, TextContent
from pydantic import ConfigDict, Field

from consolidation_memory import mcp_compat
from consolidation_memory.drift_subprocess import run_detect_drift_subprocess
from consolidation_memory.runtime import MemoryRuntime
from consolidation_memory.tool_adapter import (
    build_recall_search_arguments,
    build_recall_timeout_fallback_result,
    recall_result_needs_background_warm,
    resolve_recall_deadline,
    warm_recall_caches,
)
from consolidation_memory.tool_contracts import (
    AskOutput,
    BatchStoreOutput,
    BrowseOutput,
    ClaimBrowseOutput,
    ClaimSearchOutput,
    CompactOutput,
    ConsolidationLogOutput,
    ConsolidationOutput,
    ContradictionOutput,
    CorrectOutput,
    DecayReportOutput,
    DriftScanOutput,
    ExportOutput,
    ForgetOutput,
    HygieneApplyOutput,
    HygieneScanOutput,
    OutcomeBrowseOutput,
    OutcomeRecordOutput,
    PolicyGrantOutput,
    PolicyListOutput,
    ProtectOutput,
    RecallOutput,
    ScopeListOutput,
    SearchOutput,
    StatusOutput,
    StoreOutput,
    TimelineOutput,
    TopicDetailOutput,
)
from consolidation_memory.tool_dispatch import (
    ToolContractError,
    accepted_argument_names,
    execute_tool_call,
    reject_unknown_arguments,
    tool_requires_client,
)

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
_PRELOAD_SCIPY_ON_START = os.environ.get(
    "CONSOLIDATION_MEMORY_PRELOAD_SCIPY_ON_START",
    "1",
).strip().lower() not in {"0", "false", "no", "off"}
_STDIO_SINGLETON_ENABLED = os.environ.get(
    "CONSOLIDATION_MEMORY_STDIO_SINGLETON",
    "1",
).strip().lower() not in {"0", "false", "no", "off"}
_SIMPLE_MCP_TOOL_NAMES = frozenset({
    "memory_recall",
    "memory_remember",
    "memory_ask",
})
# Default budgets for MCP tools that go through ``_call_tool_result`` / ``_call_tool_payload``.
# Recall/drift keep dedicated handlers; consolidate is intentionally long.
_DEFAULT_TOOL_TIMEOUTS_SECONDS: dict[str, float] = {
    "memory_store": 30.0,
    "memory_store_batch": 60.0,
    "memory_remember": 30.0,
    "memory_search": 30.0,
    "memory_claim_browse": 30.0,
    "memory_claim_search": 45.0,
    "memory_outcome_record": 30.0,
    "memory_outcome_browse": 30.0,
    "memory_status": 30.0,
    "memory_forget": 30.0,
    "memory_export": 180.0,
    "memory_correct": 60.0,
    "memory_compact": 120.0,
    "memory_consolidate": 600.0,
    "memory_consolidation_log": 30.0,
    "memory_decay_report": 45.0,
    "memory_protect": 30.0,
    "memory_timeline": 45.0,
    "memory_contradictions": 30.0,
    "memory_browse": 45.0,
    "memory_read_topic": 30.0,
    "memory_hygiene_scan": 60.0,
    "memory_hygiene_apply": 180.0,
    "memory_policy_list": 20.0,
    "memory_policy_grant": 20.0,
    "memory_ask": 60.0,
}


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


def _mcp_tool_profile() -> str:
    """Return ``simple`` (3 tools) or ``full`` (default, all tools)."""
    raw = os.environ.get("CONSOLIDATION_MEMORY_MCP_TOOL_PROFILE", "full").strip().lower()
    if raw in {"simple", "full"}:
        return raw
    return "full"


def _mcp_tool_allowed(tool_name: str) -> bool:
    if _mcp_tool_profile() == "simple":
        return tool_name in _SIMPLE_MCP_TOOL_NAMES
    return True


_MEMORY_DETECT_DRIFT_TIMEOUT_SECONDS = _env_float(
    "CONSOLIDATION_MEMORY_DRIFT_TIMEOUT_SECONDS",
    90.0,
)
_MEMORY_RECALL_TIMEOUT_SECONDS = _env_float(
    "CONSOLIDATION_MEMORY_RECALL_TIMEOUT_SECONDS",
    60.0,
)
_MEMORY_RECALL_FALLBACK_TIMEOUT_SECONDS = _env_float(
    "CONSOLIDATION_MEMORY_RECALL_FALLBACK_TIMEOUT_SECONDS",
    10.0,
)
_STATUS_LIGHTWEIGHT_DEFAULT = _env_bool(
    "CONSOLIDATION_MEMORY_STATUS_LIGHTWEIGHT",
    True,
)
_CLIENT_INIT_TIMEOUT_SECONDS = _env_float(
    "CONSOLIDATION_MEMORY_CLIENT_INIT_TIMEOUT_SECONDS",
    30.0,
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
_WARMUP_AWAIT_SECONDS = _env_float(
    "CONSOLIDATION_MEMORY_WARMUP_AWAIT_SECONDS",
    0.0,
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
_warmup_executor: concurrent.futures.ThreadPoolExecutor | None = None
_idle_task: asyncio.Task | None = None
_active_tool_calls = 0
_last_activity_monotonic = time.monotonic()
_runtime_started = False
_startup_error: Exception | None = None
_runtime_start_lock = threading.Lock()
_stdio_singleton_guard = None
ScopeInput: TypeAlias = dict[str, object] | str | None

# sys.platform (not os.name) so type checkers skip the Windows-only branch.
if sys.platform == "win32":
    import msvcrt
    _msvcrt_locking: Callable[[int, int, int], Any] = msvcrt.locking
    _msvcrt_lk_nblck = int(msvcrt.LK_NBLCK)
    _msvcrt_lk_unlck = int(msvcrt.LK_UNLCK)
else:
    import fcntl
    _fcntl_flock: Callable[[int, int], Any] = fcntl.flock
    _fcntl_lock_ex = int(fcntl.LOCK_EX)
    _fcntl_lock_nb = int(fcntl.LOCK_NB)
    _fcntl_lock_un = int(fcntl.LOCK_UN)


def _drift_timeout_seconds() -> float:
    configured = _MEMORY_DETECT_DRIFT_TIMEOUT_SECONDS
    return configured if configured > 0 else 180.0


def _recall_timeout_seconds() -> float:
    configured = _MEMORY_RECALL_TIMEOUT_SECONDS
    return configured if configured > 0 else 90.0


def _recall_fallback_timeout_seconds() -> float:
    configured = _MEMORY_RECALL_FALLBACK_TIMEOUT_SECONDS
    return configured if configured > 0 else 20.0


def _default_tool_timeout_seconds() -> float:
    """Fallback timeout when a tool has no dedicated budget."""
    configured = _env_float("CONSOLIDATION_MEMORY_TOOL_TIMEOUT_SECONDS", 60.0)
    return configured if configured > 0 else 60.0


def _tool_timeout_seconds(name: str) -> float:
    """Resolve bounded timeout for a named MCP tool.

    Order: ``CONSOLIDATION_MEMORY_TIMEOUT_<TOOL>`` (e.g. MEMORY_STATUS),
    then per-tool defaults, then ``CONSOLIDATION_MEMORY_TOOL_TIMEOUT_SECONDS``.
    """
    env_key = f"CONSOLIDATION_MEMORY_TIMEOUT_{name.upper()}"
    override = _env_float(env_key, 0.0)
    if override > 0:
        return override
    default = _DEFAULT_TOOL_TIMEOUTS_SECONDS.get(name)
    if default is not None and default > 0:
        return default
    return _default_tool_timeout_seconds()


def _client_init_timeout_seconds() -> float:
    configured = _CLIENT_INIT_TIMEOUT_SECONDS
    return configured if configured > 0 else 90.0


def _warmup_on_start() -> bool:
    return _WARMUP_ON_START


def _warmup_start_delay_seconds() -> float:
    configured = _WARMUP_START_DELAY_SECONDS
    return configured if configured > 0 else 0.0


def _warmup_await_seconds() -> float:
    configured = _WARMUP_AWAIT_SECONDS
    return configured if configured > 0 else 0.0


async def _await_warmup_ready() -> bool:
    """Wait briefly for background cache warmup before interactive recall."""
    if _warmup_task is None or _warmup_task.done():
        return True

    wait_seconds = _warmup_await_seconds()
    if wait_seconds <= 0:
        return False

    try:
        await asyncio.wait_for(asyncio.shield(_warmup_task), timeout=wait_seconds)
        return True
    except asyncio.TimeoutError:
        logger.debug(
            "Warmup still running after %.2fs; proceeding with bounded recall",
            wait_seconds,
        )
        return False


def _schedule_deferred_cache_warm(client) -> None:
    """Finish embedding cache priming in the background after a bounded recall."""
    loop = asyncio.get_running_loop()
    loop.run_in_executor(_get_warmup_executor(), warm_recall_caches, client)


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
    fingerprint = hashlib.sha256(f"{project}:{parent_pid}".encode()).hexdigest()[:12]
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
        if sys.platform == "win32":
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
    if sys.platform == "win32":
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
    # Best-effort durability: fsync is unsupported on some filesystems.
    with suppress(OSError):
        os.fsync(handle.fileno())


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


def _preload_scipy_clustering() -> None:
    """Load SciPy clustering on the main thread (Windows worker-thread DLL hangs)."""
    if not _PRELOAD_SCIPY_ON_START:
        return
    if "scipy.cluster.hierarchy" in sys.modules:
        return
    started = time.monotonic()
    try:
        from scipy.cluster.hierarchy import fcluster, linkage  # noqa: F401
    except Exception as exc:
        logger.warning("SciPy clustering preload failed: %s", exc)
        return
    logger.info("Preloaded scipy clustering in %.3fs", time.monotonic() - started)


def _ensure_scipy_for_consolidate() -> None:
    """Guarantee SciPy is imported before consolidate runs on a worker thread."""
    if "scipy.cluster.hierarchy" in sys.modules:
        return
    started = time.monotonic()
    try:
        from scipy.cluster.hierarchy import fcluster, linkage  # noqa: F401
    except Exception as exc:
        logger.warning("SciPy import before consolidate failed: %s", exc)
        return
    logger.info(
        "Loaded scipy clustering on main thread before consolidate (%.3fs)",
        time.monotonic() - started,
    )


def _preload_numeric_backends() -> None:
    """Preload heavy native deps on the main thread to avoid worker-thread stalls."""
    if _PRELOAD_NUMERIC_BACKENDS_ON_START:
        started = time.monotonic()
        try:
            importlib.import_module("faiss")
            importlib.import_module("numpy")
        except Exception as exc:
            logger.warning("Numeric backend preload failed: %s", exc)
        else:
            logger.info("Preloaded numpy/faiss in %.3fs", time.monotonic() - started)
    _preload_scipy_clustering()


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


def _get_warmup_executor() -> concurrent.futures.ThreadPoolExecutor:
    """Dedicated pool so bulk cache warmup cannot starve interactive MCP tools."""
    global _warmup_executor
    if _warmup_executor is None:
        _warmup_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="consolidation_memory_warmup",
        )
    return _warmup_executor


def _shutdown_warmup_executor() -> None:
    global _warmup_executor
    executor = _warmup_executor
    _warmup_executor = None
    if executor is not None:
        executor.shutdown(wait=False, cancel_futures=True)


async def _warm_client_background() -> None:
    try:
        delay_seconds = _warmup_start_delay_seconds()
        if delay_seconds > 0:
            await asyncio.sleep(delay_seconds)
        if _runtime.shutting_down:
            return
        client = await _get_client_with_timeout()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(_get_warmup_executor(), warm_recall_caches, client)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning("Background client warmup failed: %s", exc)


async def _call_tool_payload(
    name: str,
    arguments: dict[str, object],
    *,
    timeout: float | None = None,
    recall_deadline_monotonic: float | None = None,
) -> dict[str, object]:
    client = None
    if tool_requires_client(name):
        client = await _get_client_with_timeout()
    effective_timeout = _tool_timeout_seconds(name) if timeout is None else timeout
    return await _run_blocking(
        execute_tool_call,
        name,
        arguments,
        client=client,
        timeout=effective_timeout,
        recall_deadline_monotonic=recall_deadline_monotonic,
    )


def _json_payload(value: Mapping[str, object]) -> _T:  # type: ignore[type-var]
    """Coerce a tool result to plain JSON types (anything else goes through str)."""
    return cast(_T, json.loads(json.dumps(value, default=str)))


def _tool_error_result(
    message: str,
    *,
    structured: Mapping[str, object] | None = None,
) -> _T:  # type: ignore[type-var]
    """Report a tool execution error as isError=true with actionable text (MCP spec).

    Handlers stay annotated with their output contract: FastMCP passes a
    CallToolResult through untouched and skips output validation when
    ``is_error`` is set. The TypeVar only defers the annotation; runtime
    validation against the contract happens for successful payloads.
    """
    payload: dict[str, Any] = _json_payload(structured if structured is not None else {"error": message})
    return cast(
        _T,
        CallToolResult(
            content=[TextContent(type="text", text=message)],
            structured_content=payload,
            is_error=True,
        ),
    )


async def _call_tool_result(
    name: str,
    arguments: dict[str, object],
    *,
    timeout: float | None = None,
    recall_deadline_monotonic: float | None = None,
) -> _T:  # type: ignore[type-var]
    try:
        result = await _call_tool_payload(
            name,
            arguments,
            timeout=timeout,
            recall_deadline_monotonic=recall_deadline_monotonic,
        )
        return _json_payload(result)
    except (TimeoutError, asyncio.TimeoutError):
        # Name the variable that actually governs this call: a caller-supplied
        # timeout bypasses _tool_timeout_seconds entirely, so the per-tool
        # CONSOLIDATION_MEMORY_TIMEOUT_<TOOL> override would be advice the
        # caller cannot act on.
        if timeout is None:
            hint = (
                f"Raise CONSOLIDATION_MEMORY_TIMEOUT_{name.upper()} or "
                "CONSOLIDATION_MEMORY_TOOL_TIMEOUT_SECONDS."
            )
        else:
            hint = (
                "The caller fixed this budget, so no environment variable "
                "applies; raise it where the call is made."
            )
        message = f"{name} timed out after {(timeout if timeout is not None else _tool_timeout_seconds(name)):g}s. {hint}"
        logger.error(message)
        return _tool_error_result(message)
    except Exception as exc:
        logger.exception("%s failed", name)
        return _tool_error_result(str(exc))


def _degraded_drift_output(*, message: str) -> dict[str, object]:
    """Degraded drift payload for a timed-out scan.

    Carries the full `DriftScanOutput` shape with empty lists *and* an `error`
    key. The shape alone matched the success arm of the published
    `anyOf[success, error]`, so a client validating `structuredContent` read a
    timeout as a successful empty scan; the `error` key is what the error arm
    requires, so the same payload now validates as the failure it is.
    """
    return {
        "checked_anchors": [],
        "impacted_claim_ids": [],
        "challenged_claim_ids": [],
        "impacts": [],
        "message": message,
        "error": message,
    }


@asynccontextmanager
async def lifespan(server: MCPServer):
    """Start and stop the runtime-owned MCP server resources."""
    del server
    global _warmup_task, _idle_task, _runtime_started, _startup_error

    from consolidation_memory import __version__
    from consolidation_memory.config import get_active_project

    logger.info("Starting consolidation_memory MCP server v%s...", __version__)
    logger.info("Active project: %s", get_active_project())
    # Refuse to serve if any registered tool lost its two-arm outputSchema
    # (late registration, or a private mcp SDK surface moving). Cheap: the
    # already-published tools are cache hits.
    _publish_and_verify_output_schemas()
    # Re-check that the argument set the SDK enforces is still the published one,
    # in case a tool was registered after the import-time check.
    _verify_published_argument_contract()
    # A tool registered after the import-time check must still publish the same
    # description the OpenAI surface does.
    _verify_tool_descriptions()
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
        # Awaiting a task we just cancelled: the cancellation is the expected outcome.
        with suppress(asyncio.CancelledError):
            await _idle_task
        _idle_task = None

    if _warmup_task is not None:
        _warmup_task.cancel()
        # Awaiting a task we just cancelled: the cancellation is the expected outcome.
        with suppress(asyncio.CancelledError):
            await _warmup_task
        _warmup_task = None

    _shutdown_warmup_executor()
    _runtime.shutdown()
    _runtime_started = False
    _startup_error = None
    logger.info("Shutting down consolidation_memory MCP server.")


# Unknown tool arguments must fail as input validation errors instead of being
# silently dropped (pydantic's default extra="ignore"); "forbid" also publishes
# additionalProperties: false in every input schema. Set before tools register,
# because each tool's arg model is derived from ArgModelBase at registration
# time. mcp_compat owns every private mcp import; see its module docstring for
# the surfaces used and why a minor mcp bump can break them.
#
# This is the MCP fast path, not a second contract: the SDK validates arguments
# from the handler signature before any body runs, so a rejected key never
# reaches tool_dispatch.reject_unknown_arguments. "Which arguments are allowed"
# therefore has two implementations that must agree — the signature-derived arg
# model below and the published contract the shared helper reads. Startup
# enforces that agreement in _verify_published_argument_contract(), which asks
# the shared helper about every registered tool and fails on any divergence.
mcp_compat.ArgModelBase.model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

mcp = MCPServer("consolidation_memory", lifespan=lifespan)


@functools.lru_cache(maxsize=1)
def _published_tool_descriptions() -> dict[str, str]:
    """Tool descriptions as published to OpenAI-compatible callers.

    ``schemas.openai_tools`` is the single source for what each tool says it
    does. The MCP surface would otherwise publish the handler docstring, which
    is a one-line summary: the decision guidance a host needs to pick between
    ``memory_search`` and ``memory_recall``, or the fact that
    ``memory_scope_list`` is a deployment-wide listing outside the read filter,
    never reaches an MCP host. Sharing the string removes the second copy
    instead of maintaining two.
    """
    from consolidation_memory.schemas import openai_tools

    return {
        str(tool["function"]["name"]): str(tool["function"]["description"])
        for tool in openai_tools
    }


def _verify_tool_descriptions() -> None:
    """Fail at import when a tool's docstring contradicts its published text.

    The description is load-bearing: it is what a host reads to decide whether
    and how to call a tool, and it is the only place a caller learns that
    ``memory_scope_list`` and ``memory_policy_list`` are not read-visibility
    filtered. Drift between the two must not reach a client.
    """
    published = _published_tool_descriptions()
    for name, description in sorted(published.items()):
        handler = globals().get(name)
        if handler is None:
            raise mcp_compat.MCPCompatError(
                f"Tool {name} is published in schemas.openai_tools but no handler is "
                "defined in server.py, so the two surfaces cannot agree on it."
            )
        docstring = inspect.getdoc(handler)
        if not docstring:
            raise mcp_compat.MCPCompatError(
                f"Tool {name} has no docstring; the MCP surface would publish no "
                "description at all while the OpenAI surface publishes one."
            )
        if " ".join(docstring.split()) != " ".join(description.split()):
            raise mcp_compat.MCPCompatError(
                f"Tool {name} docstring diverges from its published description.\n"
                f"  docstring:  {' '.join(docstring.split())[:200]}\n"
                f"  published:  {' '.join(description.split())[:200]}\n"
                "Both surfaces must publish the same text; fix the docstring or the "
                "schemas.openai_tools entry."
            )


def _tracked_tool() -> Callable[[Callable[..., Awaitable[_T]]], Callable[..., Awaitable[_T]]]:
    """Wrap MCP tools with lightweight activity accounting for idle shutdown."""

    def _decorator(func: Callable[..., Awaitable[_T]]) -> Callable[..., Awaitable[_T]]:
        if not _mcp_tool_allowed(func.__name__):
            return func

        @mcp.tool(description=_published_tool_descriptions().get(func.__name__))
        @functools.wraps(func)
        async def _wrapped(*args: object, **kwargs: object) -> _T:
            _begin_tool_call()
            try:
                return await func(*args, **kwargs)
            finally:
                _end_tool_call()

        return _wrapped

    return _decorator


@_tracked_tool()
async def memory_store(
    content: Annotated[
        str,
        Field(description="The text content to store. Include relevant context."),
    ],
    content_type: Annotated[
        str,
        Field(description="Category of the memory. 'exchange' for conversation, 'fact' for learned info, 'solution' for problem+fix (use Problem:/Fix: lines, structured JSON type=solution, or path anchors — see docs/FAST_PATH_EPISODES.md), 'preference' for user preference, 'procedure' for repeatable workflows (see docs/FAST_PATH_EPISODES.md). Post-consolidation 'strategy' records use structured JSON with any ingest content_type."),
    ] = "exchange",
    tags: Annotated[
        list[str] | None,
        Field(description="Optional topic tags for organization (e.g., ['python', 'debugging'])."),
    ] = None,
    surprise: Annotated[
        float,
        Field(description="How novel this is, 0.0 (routine) to 1.0 (very surprising)."),
    ] = 0.5,
    scope: Annotated[
        ScopeInput,
        Field(description="Optional scope input. Use a canonical scope object, or pass a string shorthand that auto-maps to project scope (path-like values -> project.root_uri, otherwise -> project.slug)."),
    ] = None,
) -> StoreOutput:
    """Store a memory episode in the episodic buffer. Always store memories when you learn
   something new about the user, solve a problem, discover a preference, or encounter something
   surprising. Write content as a self-contained note that future-you can understand without
   context. Include both the problem AND solution for solution-type memories. Do NOT store
   trivial exchanges like greetings."""
    return await _call_tool_result(
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
    query: Annotated[str, Field(description="Natural language description of what to recall.")],
    n_results: Annotated[int, Field(description="Maximum number of episode results.")] = 10,
    include_knowledge: Annotated[
        bool,
        Field(description="Whether to include consolidated knowledge documents."),
    ] = True,
    content_types: Annotated[
        list[str] | None,
        Field(description="Filter to specific content types (e.g. ['solution', 'fact'])."),
    ] = None,
    tags: Annotated[
        list[str] | None,
        Field(description="Filter to episodes with at least one matching tag."),
    ] = None,
    after: Annotated[
        str | None,
        Field(description="Only episodes created after this ISO date (e.g. '2025-01-01')."),
    ] = None,
    before: Annotated[
        str | None,
        Field(description="Only episodes created before this ISO date."),
    ] = None,
    include_expired: Annotated[
        bool,
        Field(description="Include temporally expired knowledge records. Default False."),
    ] = False,
    as_of: Annotated[
        str | None,
        Field(description="ISO datetime for temporal belief queries. Returns knowledge state at that point in time, including records that have since been superseded (e.g. '2025-06-15T00:00:00+00:00')."),
    ] = None,
    entity: Annotated[
        str | None,
        Field(description="Optional file path, module, or subject token to boost entity-linked memories (e.g. 'context_assembler.py' or 'memory_status')."),
    ] = None,
    hypothesis_competition: Annotated[
        bool,
        Field(description="When true, include expired or challenged contradicting partner claims as competing hypotheses with lowered ranking."),
    ] = False,
    scope: Annotated[
        ScopeInput,
        Field(description="Optional scope input. Use a canonical scope object, or pass a string shorthand that auto-maps to project scope (path-like values -> project.root_uri, otherwise -> project.slug)."),
    ] = None,
) -> RecallOutput:
    """Retrieve relevant memories by semantic similarity. Returns episodes, knowledge documents,
   and individual knowledge records (facts, solutions, preferences, procedures, strategies).
   Call this at the start of every new conversation and when context about the user's setup or
   preferences would improve your response. This is your persistent memory."""
    try:
        await _await_warmup_ready()
        client = await _get_client_with_timeout()
        bounded_n_results = max(1, min(n_results, 50))
        recall_timeout = _recall_timeout_seconds()
        arguments: dict[str, object] = {
            "query": query,
            "n_results": bounded_n_results,
            "include_knowledge": include_knowledge,
            "content_types": content_types,
            "tags": tags,
            "after": after,
            "before": before,
            "include_expired": include_expired,
            "as_of": as_of,
            "entity": entity,
            "hypothesis_competition": hypothesis_competition,
            "scope": scope,
        }

        try:
            result = await _run_blocking(
                execute_tool_call,
                "memory_recall",
                arguments,
                client=client,
                timeout=recall_timeout,
                recall_deadline_monotonic=resolve_recall_deadline(recall_timeout),
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
                    build_recall_search_arguments(arguments),
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
                return _tool_error_result(message)
            except Exception as fallback_error:
                message = (
                    f"memory_recall timed out after {recall_timeout:g}s and keyword fallback "
                    f"failed: {fallback_error}"
                )
                logger.error(message)
                return _tool_error_result(message)

            payload = build_recall_timeout_fallback_result(
                keyword_result,
                recall_timeout_seconds=recall_timeout,
                include_knowledge=include_knowledge,
            )
            return _json_payload(payload)

        result_warnings_raw = result.get("warnings")
        result_warnings: list[str] = (
            list(result_warnings_raw)
            if isinstance(result_warnings_raw, list)
            else []
        )
        if recall_result_needs_background_warm(result_warnings):
            _schedule_deferred_cache_warm(client)

        return _json_payload(result)
    except Exception as exc:
        logger.exception("memory_recall failed")
        return _tool_error_result(str(exc))


@_tracked_tool()
async def memory_remember(
    content: Annotated[
        str,
        Field(description="What to remember — include problem and fix for kind=fix."),
    ],
    kind: Annotated[
        str,
        Field(description="note=general, fix=solution, fact=fact, preference=user preference."),
    ] = "note",
    tags: Annotated[list[str] | None, Field(description="Optional topic tags.")] = None,
    scope: Annotated[
        ScopeInput,
        Field(description="Optional scope input. Use a canonical scope object, or pass a string shorthand that auto-maps to project scope (path-like values -> project.root_uri, otherwise -> project.slug)."),
    ] = None,
) -> StoreOutput:
    """Save something to memory using plain language. Prefer this over memory_store when the simple
   kind vocabulary is enough; memory_store is the one that takes an explicit content_type. Use
   kind=fix for debugging solutions (problem + what worked)."""
    return await _call_tool_result(
        "memory_remember",
        {
            "content": content,
            "kind": kind,
            "tags": tags,
            "scope": scope,
        },
    )


@_tracked_tool()
async def memory_ask(
    query: Annotated[
        str,
        Field(description="Natural-language question about prior work or fixes."),
    ],
    n_results: Annotated[int, Field(description="Maximum matches per section (default 8).")] = 8,
    scope: Annotated[
        ScopeInput,
        Field(description="Optional scope input. Use a canonical scope object, or pass a string shorthand that auto-maps to project scope (path-like values -> project.root_uri, otherwise -> project.slug)."),
    ] = None,
) -> AskOutput:
    """Search memory with a plain-language question. Returns a compact summary of matching
   episodes, records, claims, and topics. Call at conversation start (or use memory_recall when
   hooks require it). Prefer this over memory_recall for everyday retrieval."""
    try:
        await _await_warmup_ready()
        recall_timeout = _recall_timeout_seconds()
        bounded_n_results = max(1, min(n_results, 20))
        arguments: dict[str, object] = {
            "query": query,
            "n_results": bounded_n_results,
            "scope": scope,
        }
        return await _call_tool_result(
            "memory_ask",
            arguments,
            timeout=recall_timeout,
            recall_deadline_monotonic=resolve_recall_deadline(recall_timeout),
        )
    except Exception as exc:
        logger.exception("memory_ask failed")
        return _tool_error_result(str(exc))


@_tracked_tool()
async def memory_store_batch(
    episodes: Annotated[list[dict], Field(description="List of episode objects to store.")],
    scope: Annotated[
        ScopeInput,
        Field(description="Optional scope input. Use a canonical scope object, or pass a string shorthand that auto-maps to project scope (path-like values -> project.root_uri, otherwise -> project.slug)."),
    ] = None,
) -> BatchStoreOutput:
    """Store multiple memory episodes in a single operation. More efficient than calling
   memory_store repeatedly."""
    return await _call_tool_result(
        "memory_store_batch",
        {"episodes": episodes, "scope": scope},
    )


@_tracked_tool()
async def memory_search(
    query: Annotated[
        str | None,
        Field(description="Text substring to search for in episode content (case-insensitive)."),
    ] = None,
    content_types: Annotated[
        list[str] | None,
        Field(description="Filter to specific content types."),
    ] = None,
    tags: Annotated[
        list[str] | None,
        Field(description="Filter to episodes with at least one matching tag."),
    ] = None,
    after: Annotated[
        str | None,
        Field(description="Only episodes created after this ISO date (e.g. '2025-01-01')."),
    ] = None,
    before: Annotated[
        str | None,
        Field(description="Only episodes created before this ISO date."),
    ] = None,
    limit: Annotated[int, Field(description="Maximum results to return.")] = 20,
    scope: Annotated[
        ScopeInput,
        Field(description="Optional scope input. Use a canonical scope object, or pass a string shorthand that auto-maps to project scope (path-like values -> project.root_uri, otherwise -> project.slug)."),
    ] = None,
) -> SearchOutput:
    """Keyword/metadata search over episodes. Works without embedding backend. Unlike memory_recall
   (semantic similarity), this does plain text matching. Use when the embedding backend is
   down, or for exact substring searches."""
    return await _call_tool_result(
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
    claim_type: Annotated[
        str | None,
        Field(description="Optional claim type filter (e.g. 'fact', 'solution', 'strategy')."),
    ] = None,
    as_of: Annotated[
        str | None,
        Field(description="Optional ISO datetime for temporal claim queries. When set, returns claims valid at that point in time."),
    ] = None,
    limit: Annotated[int, Field(description="Maximum claims to return.")] = 50,
    scope: Annotated[
        ScopeInput,
        Field(description="Optional scope input. Use a canonical scope object, or pass a string shorthand that auto-maps to project scope (path-like values -> project.root_uri, otherwise -> project.slug)."),
    ] = None,
) -> ClaimBrowseOutput:
    """Browse claims from the claim graph. Supports optional type filtering and temporal snapshot
   queries via as_of."""
    return await _call_tool_result(
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
    query: Annotated[
        str,
        Field(description="Search text to match against claim canonical text and payload."),
    ],
    claim_type: Annotated[
        str | None,
        Field(description="Optional claim type filter (e.g. 'fact', 'solution', 'strategy')."),
    ] = None,
    as_of: Annotated[
        str | None,
        Field(description="Optional ISO datetime for temporal claim queries. When set, searches claims valid at that point in time."),
    ] = None,
    limit: Annotated[int, Field(description="Maximum matched claims to return.")] = 50,
    scope: Annotated[
        ScopeInput,
        Field(description="Optional scope input. Use a canonical scope object, or pass a string shorthand that auto-maps to project scope (path-like values -> project.root_uri, otherwise -> project.slug)."),
    ] = None,
) -> ClaimSearchOutput:
    """Search claims using deterministic phrase and keyword matching. Supports optional claim type
   filtering and temporal as_of queries."""
    return await _call_tool_result(
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
    action_summary: Annotated[
        str,
        Field(description="Concise description of the attempted strategy or action."),
    ],
    outcome_type: Annotated[str, Field(description="Outcome classification for this observation.")],
    source_claim_ids: Annotated[
        list[str] | None,
        Field(description="Claim IDs this outcome is linked to."),
    ] = None,
    source_record_ids: Annotated[
        list[str] | None,
        Field(description="Knowledge record IDs this outcome is linked to."),
    ] = None,
    source_episode_ids: Annotated[
        list[str] | None,
        Field(description="Episode IDs this outcome is linked to."),
    ] = None,
    code_anchors: Annotated[
        list[dict[str, str]] | None,
        Field(description="Optional code anchors associated with the outcome."),
    ] = None,
    issue_ids: Annotated[
        list[str] | None,
        Field(description="Optional issue identifiers associated with the outcome."),
    ] = None,
    pr_ids: Annotated[
        list[str] | None,
        Field(description="Optional pull request identifiers associated with the outcome."),
    ] = None,
    action_key: Annotated[
        str | None,
        Field(description="Optional stable strategy key. Auto-derived when omitted."),
    ] = None,
    summary: Annotated[
        str | None,
        Field(description="Optional short human summary of the observed outcome."),
    ] = None,
    details: Annotated[
        dict[str, Any] | str | None,
        Field(description="Optional structured details payload for this outcome."),
    ] = None,
    confidence: Annotated[float, Field(description="Confidence in this observation.")] = 0.8,
    provenance: Annotated[
        dict[str, Any] | str | None,
        Field(description="Optional provenance metadata (agent/tool/run identifiers, etc.)."),
    ] = None,
    observed_at: Annotated[
        str | None,
        Field(description="Optional ISO datetime when the outcome was observed."),
    ] = None,
    scope: Annotated[
        ScopeInput,
        Field(description="Optional scope input. Use a canonical scope object, or pass a string shorthand that auto-maps to project scope (path-like values -> project.root_uri, otherwise -> project.slug)."),
    ] = None,
) -> OutcomeRecordOutput:
    """Record whether a strategy/action worked. Outcome observations are durable evidence for trust
   scoring and can be linked to claim/record/episode provenance, code anchors, and issue/PR
   identifiers."""
    return await _call_tool_result(
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
    outcome_type: Annotated[str | None, Field(description="Optional outcome type filter.")] = None,
    action_key: Annotated[
        str | None,
        Field(description="Optional action/strategy key filter."),
    ] = None,
    source_claim_id: Annotated[
        str | None,
        Field(description="Optional claim source filter."),
    ] = None,
    source_record_id: Annotated[
        str | None,
        Field(description="Optional record source filter."),
    ] = None,
    source_episode_id: Annotated[
        str | None,
        Field(description="Optional episode source filter."),
    ] = None,
    as_of: Annotated[
        str | None,
        Field(description="Optional ISO datetime upper bound for observed outcomes."),
    ] = None,
    limit: Annotated[int, Field(description="Maximum outcomes to return.")] = 50,
    scope: Annotated[
        ScopeInput,
        Field(description="Optional scope input. Use a canonical scope object, or pass a string shorthand that auto-maps to project scope (path-like values -> project.root_uri, otherwise -> project.slug)."),
    ] = None,
) -> OutcomeBrowseOutput:
    """Browse recorded outcome observations over time with optional source and temporal filters."""
    return await _call_tool_result(
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
    base_ref: Annotated[
        str | None,
        Field(description="Optional git base ref for comparison (e.g. 'origin/main')."),
    ] = None,
    repo_path: Annotated[
        str | None,
        Field(description="Optional repository path (defaults to current working directory)."),
    ] = None,
) -> DriftScanOutput:
    """Detect code drift by checking changed files and challenge impacted claims. Use after
   substantial file edits."""
    timeout_seconds = _drift_timeout_seconds()
    try:
        result = await run_detect_drift_subprocess(
            base_ref=base_ref,
            repo_path=repo_path,
            timeout_seconds=timeout_seconds,
        )
        return _json_payload(result)
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
                return _json_payload(payload)
            except (TimeoutError, asyncio.TimeoutError):
                logger.error(
                    "memory_detect_drift fallback without base_ref timed out after %.2fs",
                    fallback_timeout,
                )
            except Exception:
                logger.exception("memory_detect_drift fallback without base_ref failed")

        message = (
            f"memory_detect_drift timed out after {timeout_seconds:g}s. "
            "Returned a degraded empty result; structuredContent carries the "
            "DriftScanOutput shape with empty lists plus an `error` key."
        )
        logger.error(message)
        return _tool_error_result(message, structured=_degraded_drift_output(message=message))
    except Exception as exc:
        logger.exception("memory_detect_drift failed")
        return _tool_error_result(str(exc))


@_tracked_tool()
async def memory_status(
    lightweight: Annotated[
        bool | None,
        Field(description="When true, skip markdown consistency scans and return a cached snapshot when available (faster for routine checks)."),
    ] = None,
    scope: Annotated[
        ScopeInput,
        Field(description="Optional scope input. Use a canonical scope object, or pass a string shorthand that auto-maps to project scope (path-like values -> project.root_uri, otherwise -> project.slug)."),
    ] = None,
    global_scope: Annotated[
        bool,
        Field(description="When true, return corpus-wide audit stats ignoring resolved default scope. Default false — audit reads use the same resolved scope as recall/browse."),
    ] = False,
) -> StatusOutput:
    """Show memory system statistics, including trust posture, claim coverage, provenance coverage,
   drift-watch pressure, episode counts, backend info, and fast-path consolidation metrics
   (fast_path_hits / llm_fallbacks from the last run)."""
    payload: dict[str, object] = {}
    if lightweight is not None:
        payload["lightweight"] = lightweight
    elif _STATUS_LIGHTWEIGHT_DEFAULT:
        payload["lightweight"] = _STATUS_LIGHTWEIGHT_DEFAULT
    if scope is not None:
        payload["scope"] = scope
    if global_scope:
        payload["global_scope"] = True
    return await _call_tool_result(
        "memory_status",
        payload,
        timeout=_tool_timeout_seconds("memory_status"),
    )


@_tracked_tool()
async def memory_forget(
    episode_id: Annotated[str, Field(description="The UUID of the episode to forget.")],
    scope: Annotated[
        ScopeInput,
        Field(description="Optional scope input. Use a canonical scope object, or pass a string shorthand that auto-maps to project scope (path-like values -> project.root_uri, otherwise -> project.slug)."),
    ] = None,
) -> ForgetOutput:
    """Mark an episode for removal from the memory system. Use to forget specific memories that are
   incorrect, outdated, or that the user wants removed."""
    return await _call_tool_result("memory_forget", {"episode_id": episode_id, "scope": scope})


@_tracked_tool()
async def memory_export(scope: Annotated[
    ScopeInput,
    Field(description="Optional scope input. Use a canonical scope object, or pass a string shorthand that auto-maps to project scope (path-like values -> project.root_uri, otherwise -> project.slug)."),
] = None) -> ExportOutput:
    """Export all episodes and knowledge to a JSON snapshot. Creates a timestamped backup file and
   returns the file path."""
    return await _call_tool_result("memory_export", {"scope": scope})


@_tracked_tool()
async def memory_correct(
    topic_filename: Annotated[
        str,
        Field(description="The filename of the knowledge topic (e.g., 'vr_setup.md')."),
    ],
    correction: Annotated[
        str,
        Field(description="Description of what needs to be corrected and the correct information."),
    ],
    scope: Annotated[
        ScopeInput,
        Field(description="Optional scope input. Use a canonical scope object, or pass a string shorthand that auto-maps to project scope (path-like values -> project.root_uri, otherwise -> project.slug)."),
    ] = None,
) -> CorrectOutput:
    """Correct a knowledge document with new information. Use when you discover that a knowledge
   document contains outdated or incorrect information and needs to be updated."""
    return await _call_tool_result(
        "memory_correct",
        {"topic_filename": topic_filename, "correction": correction, "scope": scope},
    )


@_tracked_tool()
async def memory_compact() -> CompactOutput:
    """Compact the FAISS index by removing tombstoned vectors. Call when memory_status shows high
   tombstone count or ratio."""
    return await _call_tool_result("memory_compact", {})


@_tracked_tool()
async def memory_consolidate() -> ConsolidationOutput:
    """Manually trigger a consolidation run. Clusters unconsolidated episodes, synthesizes
   knowledge, prunes old episodes, and compacts FAISS. Can take several minutes."""
    try:
        # Import SciPy on the asyncio main thread before worker execution.
        _ensure_scipy_for_consolidate()
        result = await _call_tool_payload(
            "memory_consolidate",
            {},
            timeout=_tool_timeout_seconds("memory_consolidate"),
        )
        if result.get("status") == "already_running":
            result.setdefault("message", "A consolidation run is already in progress")
        return _json_payload(result)
    except (TimeoutError, asyncio.TimeoutError):
        budget = _tool_timeout_seconds("memory_consolidate")
        message = (
            f"memory_consolidate timed out after {budget:g}s. "
            "Raise CONSOLIDATION_MEMORY_TIMEOUT_MEMORY_CONSOLIDATE."
        )
        logger.error(message)
        return _tool_error_result(message)
    except Exception as exc:
        logger.exception("memory_consolidate failed")
        return _tool_error_result(str(exc))


@_tracked_tool()
async def memory_consolidation_log(
    last_n: Annotated[
        int,
        Field(description="Number of recent runs to show (1-20, default 5)."),
    ] = 5,
    scope: Annotated[
        ScopeInput,
        Field(description="Optional scope input. Use a canonical scope object, or pass a string shorthand that auto-maps to project scope (path-like values -> project.root_uri, otherwise -> project.slug)."),
    ] = None,
    global_scope: Annotated[
        bool,
        Field(description="When true, return corpus-wide audit stats ignoring resolved default scope. Default false — audit reads use the same resolved scope as recall/browse."),
    ] = False,
) -> ConsolidationLogOutput:
    """Show recent consolidation activity as a human-readable changelog. Returns summaries of
   recent runs: topics created/updated, contradictions detected, episodes pruned."""
    payload: dict[str, object] = {"last_n": last_n, "scope": scope}
    if global_scope:
        payload["global_scope"] = True
    return await _call_tool_result("memory_consolidation_log", payload)


@_tracked_tool()
async def memory_decay_report(
    scope: Annotated[
        ScopeInput,
        Field(description="Optional scope input. Use a canonical scope object, or pass a string shorthand that auto-maps to project scope (path-like values -> project.root_uri, otherwise -> project.slug)."),
    ] = None,
    global_scope: Annotated[
        bool,
        Field(description="When true, return corpus-wide audit stats ignoring resolved default scope. Default false — audit reads use the same resolved scope as recall/browse."),
    ] = False,
) -> DecayReportOutput:
    """Show what would be forgotten if pruning ran right now. Reports prunable episodes,
   low-confidence records, and protected episode counts. Does NOT actually delete anything."""
    payload: dict[str, object] = {"scope": scope}
    if global_scope:
        payload["global_scope"] = True
    return await _call_tool_result("memory_decay_report", payload)


@_tracked_tool()
async def memory_protect(
    episode_id: Annotated[
        str | None,
        Field(description="Protect a specific episode by its UUID."),
    ] = None,
    tag: Annotated[str | None, Field(description="Protect all episodes with this tag.")] = None,
    scope: Annotated[
        ScopeInput,
        Field(description="Optional scope input. Use a canonical scope object, or pass a string shorthand that auto-maps to project scope (path-like values -> project.root_uri, otherwise -> project.slug)."),
    ] = None,
) -> ProtectOutput:
    """Mark episodes as immune to pruning. Protect specific episodes or all episodes with a given
   tag from being pruned during consolidation."""
    return await _call_tool_result(
        "memory_protect",
        {"episode_id": episode_id, "tag": tag, "scope": scope},
    )


@_tracked_tool()
async def memory_timeline(topic: Annotated[
    str,
    Field(description="Natural language topic to query (e.g., 'frontend framework preference')."),
], scope: Annotated[
    ScopeInput,
    Field(description="Optional scope input. Use a canonical scope object, or pass a string shorthand that auto-maps to project scope (path-like values -> project.root_uri, otherwise -> project.slug)."),
] = None) -> TimelineOutput:
    """Show how understanding of a topic has changed over time. Returns all knowledge records
   matching the topic sorted chronologically, including expired/superseded records."""
    return await _call_tool_result("memory_timeline", {"topic": topic, "scope": scope})


@_tracked_tool()
async def memory_contradictions(
    topic: Annotated[
        str | None,
        Field(description="Optional topic filename or title to filter results."),
    ] = None,
    scope: Annotated[
        ScopeInput,
        Field(description="Optional scope input. Use a canonical scope object, or pass a string shorthand that auto-maps to project scope (path-like values -> project.root_uri, otherwise -> project.slug)."),
    ] = None,
    global_scope: Annotated[
        bool,
        Field(description="When true, return corpus-wide audit stats ignoring resolved default scope. Default false — audit reads use the same resolved scope as recall/browse."),
    ] = False,
) -> ContradictionOutput:
    """List detected contradictions from the audit log. Shows cases where knowledge records
   contradicted each other during consolidation, including both the old and new content and how
   it was resolved."""
    payload: dict[str, object] = {"topic": topic, "scope": scope}
    if global_scope:
        payload["global_scope"] = True
    return await _call_tool_result("memory_contradictions", payload)


@_tracked_tool()
async def memory_browse(scope: Annotated[
    ScopeInput,
    Field(description="Optional scope input. Use a canonical scope object, or pass a string shorthand that auto-maps to project scope (path-like values -> project.root_uri, otherwise -> project.slug)."),
] = None) -> BrowseOutput:
    """Browse all knowledge topics with summaries and metadata. Returns titles, summaries, record
   counts by type, confidence scores, and file paths."""
    return await _call_tool_result("memory_browse", {"scope": scope})


@_tracked_tool()
async def memory_read_topic(
    filename: Annotated[
        str,
        Field(description="The filename of the knowledge topic (e.g., 'python_setup.md')."),
    ],
    scope: Annotated[
        ScopeInput,
        Field(description="Optional scope input. Use a canonical scope object, or pass a string shorthand that auto-maps to project scope (path-like values -> project.root_uri, otherwise -> project.slug)."),
    ] = None,
) -> TopicDetailOutput:
    """Read the full markdown content of a knowledge topic. Use memory_browse first to see
   available topics."""
    return await _call_tool_result("memory_read_topic", {"filename": filename, "scope": scope})


@_tracked_tool()
async def memory_hygiene_scan() -> HygieneScanOutput:
    """Read-only corpus hygiene report: noisy episode candidates, orphaned active claims, and stale
   episode sources. Use before batch cleanup."""
    return await _call_tool_result("memory_hygiene_scan", {})


@_tracked_tool()
async def memory_hygiene_apply(
    episode_ids: Annotated[
        list[str] | None,
        Field(description="Explicit episode UUIDs to forget."),
    ] = None,
    use_recommended: Annotated[
        bool,
        Field(description="Forget all episodes flagged by the latest hygiene scan (temp/test, exchange, noise journal)."),
    ] = False,
    expire_orphans: Annotated[
        bool,
        Field(description="Expire active claims that lost all provenance."),
    ] = False,
    dry_run: Annotated[
        bool,
        Field(description="Preview actions without mutating the corpus."),
    ] = False,
) -> HygieneApplyOutput:
    """Apply corpus hygiene: forget selected or recommended noisy episodes and optionally expire
   orphaned claims. Irreversible unless dry_run=true."""
    payload: dict[str, object] = {
        "use_recommended": use_recommended,
        "expire_orphans": expire_orphans,
        "dry_run": dry_run,
    }
    if episode_ids is not None:
        payload["episode_ids"] = episode_ids
    return await _call_tool_result("memory_hygiene_apply", payload)


@_tracked_tool()
async def memory_policy_list() -> PolicyListOutput:
    """List persisted namespace/project access policies and ACL bindings. Use to inspect who can read or write memory in self-hosted deployments. Like scope discovery, this listing is deployment-wide: it is NOT filtered by read_visibility or the caller's scope, so it names every persisted binding, including principals the caller cannot read into."""
    return await _call_tool_result("memory_policy_list", {})


@_tracked_tool()
async def memory_scope_list(
    limit: Annotated[
        int,
        Field(description="Maximum number of scopes to return (1-1000)."),
    ] = 100,
    offset: Annotated[
        int,
        Field(description="0-based offset into the ordered scope list for page iteration; combine with limit."),
    ] = 0,
) -> ScopeListOutput:
    """Discover which scopes exist: lists every scope that has stored data, with per-table usage counts and a canonical scope envelope you can pass back as the scope argument of other tools. Counts cover live rows only. This listing is deployment-wide: it is a topology audit, intentionally NOT filtered by read_visibility, policy or the caller's scope, so it names every scope that holds data. No content is read through it. Keep the MCP subprocess off untrusted multi-tenant deployments without OS-level isolation."""
    return await _call_tool_result("memory_scope_list", {"limit": limit, "offset": offset})


@_tracked_tool()
async def memory_policy_grant(
    principal_type: Annotated[
        str,
        Field(description="Principal type (e.g. app_client, agent_name)."),
    ],
    principal_key: Annotated[
        str,
        Field(description="Principal key (e.g. python_sdk:legacy_client)."),
    ],
    namespace: Annotated[
        str | None,
        Field(description="Namespace slug selector (wildcard when omitted)."),
    ] = None,
    project: Annotated[
        str | None,
        Field(description="Project slug selector (wildcard when omitted)."),
    ] = None,
    write_mode: Annotated[str | None, Field(description="Write policy for the principal.")] = None,
    read_visibility: Annotated[
        str | None,
        Field(description="Read visibility policy for the principal."),
    ] = None,
) -> PolicyGrantOutput:
    """Create or update a persisted ACL binding for a principal. Omitted namespace or project
   selectors act as wildcards. Provide at least one of write_mode or read_visibility."""
    payload: dict[str, object] = {
        "principal_type": principal_type,
        "principal_key": principal_key,
    }
    if namespace is not None:
        payload["namespace"] = namespace
    if project is not None:
        payload["project"] = project
    if write_mode is not None:
        payload["write_mode"] = write_mode
    if read_visibility is not None:
        payload["read_visibility"] = read_visibility
    return await _call_tool_result("memory_policy_grant", payload)


_ERROR_OUTPUT_ARM: dict[str, Any] = {
    "type": "object",
    "title": "ToolErrorPayload",
    "description": "Tool execution error payload published alongside isError=true results.",
    "properties": {"error": {"type": "string", "description": "Actionable error message."}},
    "required": ["error"],
    "additionalProperties": True,
}

# Tool name -> the exact dict object published for it. Identity is the cache key:
# a re-registered tool gets a fresh schema object, so it is republished instead of
# being skipped as "already done".
_PUBLISHED_OUTPUT_SCHEMAS: dict[str, dict[str, Any]] = {}


def _publish_output_schemas() -> list[str]:
    """Publish each tool's output schema as anyOf[success contract, error payload].

    Success payloads are validated against the contract model itself; error
    results skip output validation and carry ``{"error": ...}`` instead, so the
    published schema has to accept both shapes (MCP spec: clients validate
    structured results against outputSchema).

    Idempotent and self-healing: called at import, on every tool listing, and at
    lifespan startup. A tool registered after import (plugins, a future dynamic
    tool) is published on the next listing instead of serving a success-only
    schema forever. The cache check is the published ``anyOf`` marker itself, so
    repeat calls rebuild nothing and can never re-wrap an already-wrapped arm.
    Returns the problems found (a tool with no output schema at all) for the
    caller to report.
    """
    problems: list[str] = []
    for name, tool in mcp_compat.registered_tools(mcp).items():
        schema = mcp_compat.output_schema_dict(tool)
        if schema is None:
            problems.append(f"{name}: handler has no annotated return contract, so no outputSchema")
            continue
        if _PUBLISHED_OUTPUT_SCHEMAS.get(name) is schema:
            continue
        if "anyOf" in schema:
            # Already carries the two-arm contract (published here or by a
            # caller that owns it); adopt it instead of nesting a second pair.
            _PUBLISHED_OUTPUT_SCHEMAS[name] = schema
            continue
        success = dict(schema)
        defs = success.pop("$defs", None)
        # Root "type" keeps pre-2026 protocol validators (which require it) happy;
        # both anyOf arms are objects, so the conjunction stays accurate.
        published: dict[str, Any] = {"type": "object", "anyOf": [success, _ERROR_OUTPUT_ARM]}
        if defs is not None:
            published["$defs"] = defs
        mcp_compat.set_output_schema(tool, published)
        _PUBLISHED_OUTPUT_SCHEMAS[name] = published
    return problems


def _verify_published_output_schemas(unrepairable: Sequence[str] = ()) -> None:
    """Fail startup unless every registered tool publishes the two-arm contract.

    This is the guard the previous one-shot import-time publication lacked: a
    private-SDK rename used to strip ``outputSchema`` from every tool (or leave
    a success-only one) with no error anywhere. A client that trusted the
    contract would then reject every ``{"error": ...}`` failure payload. Raise
    at startup instead of degrading silently.

    Pure verification — it never repairs, so it can actually fail. Callers run
    :func:`_publish_output_schemas` first and pass the problems it could not fix
    (a tool with no annotated return contract) in as ``unrepairable``.
    """
    problems = list(unrepairable)
    for name, tool in mcp_compat.registered_tools(mcp).items():
        schema = mcp_compat.output_schema_dict(tool) or {}
        arms = schema.get("anyOf")
        if not isinstance(arms, list) or len(arms) != 2:
            problems.append(f"{name}: published outputSchema is not anyOf[success, error]")
            continue
        if arms[1].get("title") != "ToolErrorPayload" or arms[1].get("required") != ["error"]:
            problems.append(f"{name}: published outputSchema is missing the ToolErrorPayload arm")
    if not problems:
        return
    raise mcp_compat.MCPCompatError(
        "MCP outputSchema contract is incomplete at startup: "
        + "; ".join(problems)
        + f". Installed mcp: {mcp_compat.installed_mcp_version()}; supported range: "
        f"mcp {mcp_compat.MCP_COMPAT_REQUIREMENT}. See consolidation_memory/mcp_compat.py for the "
        "private SDK surfaces this depends on."
    )


def _publish_and_verify_output_schemas() -> None:
    """Repair what can be repaired, then fail on whatever is still broken."""
    _verify_published_output_schemas(_publish_output_schemas())


# Sentinel argument name probed against the shared contract. It is not a
# published key of any tool, so accepting it is a contract failure.
_ARGUMENT_DRIFT_PROBE = "undeclared_contract_probe"


def _verify_published_argument_contract() -> None:
    """Fail startup unless the SDK's enforced arguments are the dispatch contract.

    The two surfaces cannot share one code path: the MCP SDK validates a request
    against the per-tool arg model derived from the handler signature *before*
    any body runs, so ``reject_unknown_arguments`` is unreachable for MCP calls
    and the SDK check stays as the fast path. What this guard adds is the
    delegation that is possible — every registered tool is looked up in
    :func:`tool_dispatch.accepted_argument_names` (the published contract, read
    from ``schemas.openai_tools``) and compared with the argument set the SDK
    actually enforces, in both directions: the shared helper must accept every
    name the SDK accepts, and must refuse an undeclared key.

    A silent divergence here is the exact failure this PR exists to prevent: MCP
    would reject a key the other three surfaces run happily, or accept one they
    refuse. Raise at startup instead.
    """
    problems: list[str] = []
    for name, tool in mcp_compat.registered_tools(mcp).items():
        enforced = frozenset((getattr(tool, "parameters", None) or {}).get("properties") or {})
        shared = accepted_argument_names(name)
        if shared is None:
            problems.append(
                f"{name}: registered on MCP but absent from schemas.openai_tools, so the "
                "dispatch seam enforces no contract for it"
            )
            continue
        if enforced != shared:
            problems.append(
                f"{name}: MCP accepts {sorted(enforced) or '[]'} but the published inputSchema "
                f"declares {sorted(shared) or '[]'}"
            )
            continue
        parameters = getattr(tool, "parameters", None) or {}
        if parameters.get("additionalProperties") is not False:
            problems.append(
                f"{name}: MCP inputSchema does not publish additionalProperties: false, so "
                "the SDK lost its strict extra-key check"
            )
        try:
            reject_unknown_arguments(name, {key: None for key in enforced})
        except ToolContractError as exc:
            problems.append(f"{name}: the SDK accepts arguments the shared contract refuses ({exc})")
        try:
            reject_unknown_arguments(name, {_ARGUMENT_DRIFT_PROBE: None})
        except ToolContractError:
            continue
        problems.append(
            f"{name}: the shared contract accepts {_ARGUMENT_DRIFT_PROBE!r}, which MCP rejects"
        )
    if not problems:
        return
    raise mcp_compat.MCPCompatError(
        "MCP tool-argument contract has drifted from the published inputSchema: "
        + "; ".join(problems)
        + ". Keep the handler signature and the published schema in sync; both surfaces enforce "
        "the published inputSchema and must accept exactly the same argument names."
    )


_publish_and_verify_output_schemas()
_verify_published_argument_contract()
_verify_tool_descriptions()


def _verify_every_contract() -> None:
    """Repair and verify all three contracts on a tool listing.

    The heal wrapper used to call ``_publish_output_schemas`` alone, so a tool
    registered at runtime was repaired but never checked: its argument set and its
    description were never compared against the published contract. Point the
    wrapper at the same three checks the import-time path runs, so late
    registration is as strictly served as import-time registration.
    """
    _publish_and_verify_output_schemas()
    _verify_published_argument_contract()
    _verify_tool_descriptions()


mcp_compat.install_list_tools_heal(mcp, _verify_every_contract)


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
