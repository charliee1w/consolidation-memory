"""Shared runtime owner for MemoryClient lifecycle and blocking execution."""

from __future__ import annotations

import asyncio
import concurrent.futures
import functools
import logging
import threading
import time
from collections.abc import Callable
from typing import Any, TypeVar, cast

from consolidation_memory.database import ensure_schema

_T = TypeVar("_T")
logger = logging.getLogger(__name__)


def _default_client_factory():
    from consolidation_memory.client import MemoryClient

    return MemoryClient()


class MemoryRuntime:
    """Own the process-local MemoryClient and blocking executor lifecycle."""

    def __init__(
        self,
        *,
        client_factory: Callable[[], object] | None = None,
        max_workers: int = 16,
    ) -> None:
        self._client_factory = client_factory or _default_client_factory
        self._max_workers = max(1, int(max_workers))

        self._client: object | None = None
        self._client_lock = threading.Lock()
        self._client_initializing = False
        self._client_init_owner_thread_id: int | None = None
        self._client_init_error: Exception | None = None
        self._client_init_thread: threading.Thread | None = None
        self._client_init_cond = threading.Condition(self._client_lock)
        self._shutting_down = False
        self._lifecycle_epoch = 0

        self._blocking_executor: concurrent.futures.ThreadPoolExecutor | None = None
        self._blocking_executor_lock = threading.Lock()

    @property
    def client(self):
        return self._client

    @property
    def client_initializing(self) -> bool:
        return self._client_initializing

    @property
    def client_init_owner_thread_id(self) -> int | None:
        return self._client_init_owner_thread_id

    @property
    def client_init_error(self) -> Exception | None:
        return self._client_init_error

    @property
    def client_init_cond(self) -> threading.Condition:
        return self._client_init_cond

    @property
    def shutting_down(self) -> bool:
        return self._shutting_down

    @property
    def lifecycle_epoch(self) -> int:
        return self._lifecycle_epoch

    @property
    def blocking_executor(self) -> concurrent.futures.ThreadPoolExecutor | None:
        return self._blocking_executor

    def startup(self) -> None:
        with self._client_init_cond:
            self._shutting_down = False
            self._lifecycle_epoch += 1
            self._client_init_error = None
        ensure_schema()

    def shutdown(self) -> None:
        with self._client_init_cond:
            self._shutting_down = True
            self._lifecycle_epoch += 1
            self._client_initializing = False
            self._client_init_owner_thread_id = None
            self._client_init_error = None
            self._client_init_thread = None
            self._client_init_cond.notify_all()

        client = self._client
        self._client = None
        if client is not None:
            cast(Any, client).close()

        with self._blocking_executor_lock:
            executor = self._blocking_executor
            self._blocking_executor = None
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

        with self._client_init_cond:
            self._client_initializing = False
            self._client_init_owner_thread_id = None
            self._client_init_error = None
            self._shutting_down = False
            self._client_init_cond.notify_all()

    def _get_blocking_executor(self) -> concurrent.futures.ThreadPoolExecutor:
        with self._blocking_executor_lock:
            if self._blocking_executor is None:
                self._blocking_executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=self._max_workers,
                    thread_name_prefix="consolidation_memory_mcp",
                )
            return self._blocking_executor

    async def run_blocking(
        self,
        func: Callable[..., _T],
        /,
        *args: object,
        timeout: float | None = None,
        **kwargs: object,
    ) -> _T:
        loop = asyncio.get_running_loop()
        work = functools.partial(func, *args, **kwargs)
        future = loop.run_in_executor(self._get_blocking_executor(), work)
        if timeout is None:
            return await future
        return await asyncio.wait_for(future, timeout=timeout)

    def get_client(self, *, wait_timeout: float | None = None):
        """Return the process-local client, initializing it lazily once."""
        current_thread_id = threading.get_ident()
        with self._client_init_cond:
            if self._client is not None:
                return self._client
            if self._shutting_down:
                raise RuntimeError("MemoryClient initialization aborted: runtime is shutting down")
            lifecycle_epoch = self._lifecycle_epoch
            if self._client_initializing and self._client_init_owner_thread_id == current_thread_id:
                raise RuntimeError(
                    "Re-entrant MemoryClient initialization detected. "
                    "Avoid calling memory tools/hooks during client startup."
                )
            if not self._client_initializing:
                self._start_client_init_locked(lifecycle_epoch=lifecycle_epoch)
            return self._wait_for_client_locked(
                lifecycle_epoch=lifecycle_epoch,
                wait_timeout=wait_timeout,
            )

    async def get_client_with_timeout(self, timeout: float | None = None):
        return await self.run_blocking(self.get_client, wait_timeout=timeout)

    def _start_client_init_locked(self, *, lifecycle_epoch: int) -> None:
        """Start a detached client initializer under ``_client_init_cond``."""
        self._client_initializing = True
        self._client_init_owner_thread_id = None
        self._client_init_error = None
        init_thread = threading.Thread(
            target=self._initialize_client_background,
            args=(lifecycle_epoch,),
            name="consolidation_memory_client_init",
            daemon=True,
        )
        self._client_init_thread = init_thread
        init_thread.start()

    def _wait_for_client_locked(
        self,
        *,
        lifecycle_epoch: int,
        wait_timeout: float | None,
    ):
        """Wait under ``_client_init_cond`` for client initialization to finish."""
        deadline = None if wait_timeout is None else time.monotonic() + max(0.0, wait_timeout)
        while True:
            if self._client is not None:
                return self._client
            if self._shutting_down or self._lifecycle_epoch != lifecycle_epoch:
                raise RuntimeError("MemoryClient initialization aborted: runtime lifecycle changed")
            if self._client_init_error is not None:
                raise RuntimeError(
                    f"MemoryClient initialization failed: {self._client_init_error}"
                ) from self._client_init_error
            if not self._client_initializing:
                raise RuntimeError("MemoryClient initialization did not complete")

            wait_seconds = 0.5
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("MemoryClient initialization timed out")
                wait_seconds = min(wait_seconds, remaining)
            self._client_init_cond.wait(timeout=wait_seconds)

    def _initialize_client_background(self, lifecycle_epoch: int) -> None:
        """Run client construction on a daemon thread so callers can time out safely."""
        current_thread_id = threading.get_ident()
        with self._client_init_cond:
            if self._client_init_thread is None:
                return
            self._client_init_owner_thread_id = current_thread_id
            self._client_init_cond.notify_all()

        initialized_client = None
        init_error: Exception | None = None
        try:
            initialized_client = self._client_factory()
            if initialized_client is None:
                init_error = RuntimeError("MemoryClient factory returned None")
        except Exception as exc:
            init_error = exc

        abort_error: RuntimeError | None = None
        client_to_close: object | None = None
        with self._client_init_cond:
            if self._client_init_owner_thread_id != current_thread_id:
                client_to_close = initialized_client
            elif init_error is not None:
                self._client_initializing = False
                self._client_init_owner_thread_id = None
                self._client_init_thread = None
                self._client_init_error = init_error
                self._client_init_cond.notify_all()
                return
            elif self._shutting_down or self._lifecycle_epoch != lifecycle_epoch or self._client is not None:
                abort_error = RuntimeError(
                    "MemoryClient initialization aborted: runtime lifecycle changed during startup"
                )
                self._client_initializing = False
                self._client_init_owner_thread_id = None
                self._client_init_thread = None
                self._client_init_error = abort_error
                self._client_init_cond.notify_all()
                client_to_close = initialized_client
            else:
                self._client = initialized_client
                self._client_initializing = False
                self._client_init_owner_thread_id = None
                self._client_init_thread = None
                self._client_init_error = None
                self._client_init_cond.notify_all()
                return

        if client_to_close is not None:
            try:
                cast(Any, client_to_close).close()
            except Exception:
                logger.warning("Failed to close aborted MemoryClient initialization cleanly", exc_info=True)
            if abort_error is None and init_error is None:
                logger.warning("Discarded late MemoryClient initialization result after lifecycle change")
