"""Tests for MemoryRuntime lifecycle ownership and initialization hardening."""

from __future__ import annotations

import concurrent.futures
import threading
import time
from unittest.mock import MagicMock, patch

import pytest


class TestMemoryRuntimeLifecycle:
    def test_shutdown_preserves_other_runtime_clients(self):
        from consolidation_memory.client import MemoryClient
        from consolidation_memory.database import ensure_schema
        from consolidation_memory.runtime import MemoryRuntime

        ensure_schema()
        runtime_a = MemoryRuntime(client_factory=lambda: MemoryClient(auto_consolidate=False))
        runtime_b = MemoryRuntime(client_factory=lambda: MemoryClient(auto_consolidate=False))
        runtime_a.startup()
        runtime_b.startup()

        with patch("consolidation_memory.database.close_all_connections") as mock_close_all:
            client_a = runtime_a.get_client(wait_timeout=1.0)
            client_b = runtime_b.get_client(wait_timeout=1.0)
            assert runtime_a.client is client_a

            runtime_a.shutdown()
            assert mock_close_all.call_count == 0
            assert client_b.status().version != ""

            runtime_b.shutdown()

    def test_timed_out_client_init_does_not_wedge_shutdown(self):
        from consolidation_memory.runtime import MemoryRuntime

        entered = threading.Event()
        release = threading.Event()
        mock_client = MagicMock()

        def _slow_factory():
            entered.set()
            release.wait(timeout=5.0)
            return mock_client

        runtime = MemoryRuntime(client_factory=_slow_factory)
        runtime.startup()

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(runtime.get_client, wait_timeout=0.05)
            assert entered.wait(timeout=1.0)
            with pytest.raises(TimeoutError, match="timed out"):
                future.result(timeout=1.0)

        started = time.monotonic()
        runtime.shutdown()
        elapsed = time.monotonic() - started
        assert elapsed < 1.0

        release.set()
        time.sleep(0.05)
        mock_client.close.assert_called_once()
