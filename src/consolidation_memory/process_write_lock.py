"""Cross-process exclusive file lease for shared on-disk mutation paths."""

from __future__ import annotations

import logging
import os
import sys
import time
from collections.abc import Callable
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

if sys.platform == "win32":
    import msvcrt

    _msvcrt_locking: Callable[[int, int, int], Any] = msvcrt.locking
    _msvcrt_lk_nblck = int(msvcrt.LK_NBLCK)
    _msvcrt_lk_unlck = int(msvcrt.LK_UNLCK)
else:  # pragma: no cover - exercised on non-Windows CI
    import fcntl


def _try_lock_file(handle: Any) -> None:
    """Attempt non-blocking exclusive lock of a lockfile handle."""
    handle.seek(0)
    if sys.platform == "win32":
        _msvcrt_locking(handle.fileno(), _msvcrt_lk_nblck, 1)
    else:  # pragma: no cover - exercised on non-Windows CI
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)  # type: ignore[attr-defined]


def _unlock_file(handle: Any) -> None:
    """Release exclusive lock for a lockfile handle."""
    handle.seek(0)
    if sys.platform == "win32":
        _msvcrt_locking(handle.fileno(), _msvcrt_lk_unlck, 1)
    else:  # pragma: no cover - exercised on non-Windows CI
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)  # type: ignore[attr-defined]


class ProcessWriteLease:
    """Cross-process lock guarding shared storage mutation paths."""

    def __init__(self, lock_path: Path, timeout_seconds: float) -> None:
        self._lock_path = lock_path
        self._timeout_seconds = max(0.1, float(timeout_seconds))

    @contextmanager
    def acquire(self) -> Any:
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        with open(self._lock_path, "a+b") as handle:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"0")
                handle.flush()

            deadline = started + self._timeout_seconds
            while True:
                try:
                    _try_lock_file(handle)
                    break
                except OSError as err:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            "Timed out waiting for write lease at "
                            f"{self._lock_path} after {self._timeout_seconds:.1f}s"
                        ) from err
                    time.sleep(0.05)

            waited = time.monotonic() - started
            if waited > 0.25:
                logger.info("Waited %.3fs for write lease at %s", waited, self._lock_path)
            try:
                handle.seek(0)
                payload = f"pid={os.getpid()} acquired_at={time.time():.6f}"
                handle.truncate()
                handle.write(payload.encode("utf-8"))
                handle.flush()
                # Best-effort durability: fsync is unsupported on some filesystems.
                with suppress(OSError):
                    os.fsync(handle.fileno())
                yield
            finally:
                _unlock_file(handle)