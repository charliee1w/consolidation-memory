"""Cross-process write lease tests."""

from __future__ import annotations

import multiprocessing as mp
import time
from pathlib import Path

from consolidation_memory.process_write_lock import ProcessWriteLease


def _hold_lock_then_release(
    lock_path: str,
    hold_seconds: float,
    order: mp.Queue,
    holder_ready: mp.Event,
) -> None:
    lease = ProcessWriteLease(Path(lock_path), timeout_seconds=10.0)
    with lease.acquire():
        order.put("held")
        holder_ready.set()
        time.sleep(hold_seconds)
    order.put("released")


def _acquire_after_peer(
    lock_path: str,
    order: mp.Queue,
    holder_ready: mp.Event,
) -> None:
    assert holder_ready.wait(timeout=10.0)
    lease = ProcessWriteLease(Path(lock_path), timeout_seconds=10.0)
    with lease.acquire():
        order.put("acquired")


class TestProcessWriteLease:
    def test_serializes_cross_process_access(self, tmp_path):
        lock_path = tmp_path / "cross_process.lock"
        order: mp.Queue = mp.Queue()
        holder_ready = mp.Event()
        holder = mp.Process(
            target=_hold_lock_then_release,
            args=(str(lock_path), 0.35, order, holder_ready),
        )
        waiter = mp.Process(
            target=_acquire_after_peer,
            args=(str(lock_path), order, holder_ready),
        )

        holder.start()
        waiter.start()
        holder.join(timeout=15)
        waiter.join(timeout=15)

        assert holder.exitcode == 0
        assert waiter.exitcode == 0

        events = [order.get(timeout=5) for _ in range(3)]
        assert events[0] == "held"
        assert events[1] == "released"
        assert events[2] == "acquired"