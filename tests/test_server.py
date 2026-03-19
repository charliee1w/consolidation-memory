"""Tests for MCP server tool wrappers."""

from __future__ import annotations

import asyncio
import concurrent.futures
import importlib
import inspect
import json
import threading
import time
from contextlib import contextmanager
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest


@contextmanager
def _patched_server_runtime():
    import consolidation_memory.server as server
    from consolidation_memory.runtime import MemoryRuntime

    runtime = MemoryRuntime(max_workers=2)
    with (
        patch.object(server, "_runtime", runtime),
        patch.object(server, "_runtime_started", False),
        patch.object(server, "_startup_error", None),
    ):
        try:
            yield server, runtime
        finally:
            runtime.shutdown()


class TestServerEnvParsing:
    def test_server_module_uses_defaults_when_numeric_env_values_are_invalid(self, monkeypatch):
        monkeypatch.setenv("CONSOLIDATION_MEMORY_DRIFT_TIMEOUT_SECONDS", "nan")
        monkeypatch.setenv("CONSOLIDATION_MEMORY_RECALL_TIMEOUT_SECONDS", "not-a-float")
        monkeypatch.setenv("CONSOLIDATION_MEMORY_RECALL_FALLBACK_TIMEOUT_SECONDS", "inf")
        monkeypatch.setenv("CONSOLIDATION_MEMORY_CLIENT_INIT_TIMEOUT_SECONDS", "")
        monkeypatch.setenv("CONSOLIDATION_MEMORY_MCP_BLOCKING_WORKERS", "many")
        monkeypatch.setenv("CONSOLIDATION_MEMORY_IDLE_TIMEOUT_SECONDS", "nan")
        monkeypatch.setenv("CONSOLIDATION_MEMORY_IDLE_CHECK_INTERVAL_SECONDS", "inf")
        monkeypatch.setenv("CONSOLIDATION_MEMORY_WARMUP_START_DELAY_SECONDS", "whoops")
        monkeypatch.setenv(
            "CONSOLIDATION_MEMORY_STDIO_SINGLETON_TAKEOVER_TIMEOUT_SECONDS",
            "",
        )

        import consolidation_memory.server as server

        server = importlib.reload(server)

        assert server._MEMORY_DETECT_DRIFT_TIMEOUT_SECONDS == 90.0
        assert server._MEMORY_RECALL_TIMEOUT_SECONDS == 45.0
        assert server._MEMORY_RECALL_FALLBACK_TIMEOUT_SECONDS == 10.0
        assert server._CLIENT_INIT_TIMEOUT_SECONDS == 45.0
        assert server._MCP_BLOCKING_WORKERS == 16
        assert server._IDLE_TIMEOUT_SECONDS == 900.0
        assert server._IDLE_CHECK_INTERVAL_SECONDS == 15.0
        assert server._WARMUP_START_DELAY_SECONDS == 0.25
        assert server._STDIO_SINGLETON_TAKEOVER_TIMEOUT_SECONDS == 10.0

    def test_mcp_client_factory_defaults_auto_consolidate_to_false(self, monkeypatch):
        monkeypatch.delenv("CONSOLIDATION_MEMORY_MCP_AUTO_CONSOLIDATE", raising=False)

        import consolidation_memory.server as server

        server = importlib.reload(server)
        with patch("consolidation_memory.client.MemoryClient") as mock_client:
            server._mcp_client_factory()

        mock_client.assert_called_once_with(auto_consolidate=False)

    def test_mcp_client_factory_allows_auto_consolidate_override(self, monkeypatch):
        monkeypatch.setenv("CONSOLIDATION_MEMORY_MCP_AUTO_CONSOLIDATE", "true")

        import consolidation_memory.server as server

        server = importlib.reload(server)
        with patch("consolidation_memory.client.MemoryClient") as mock_client:
            server._mcp_client_factory()

        mock_client.assert_called_once_with(auto_consolidate=True)


class TestMCPDetectDriftTool:
    def test_memory_detect_drift_signature(self):
        from consolidation_memory.server import memory_detect_drift

        sig = inspect.signature(memory_detect_drift)
        assert "base_ref" in sig.parameters
        assert "repo_path" in sig.parameters
        assert sig.parameters["base_ref"].default is None
        assert sig.parameters["repo_path"].default is None

    def test_memory_detect_drift_calls_drift_engine(self):
        from consolidation_memory.server import memory_detect_drift

        expected = {
            "checked_anchors": [{"anchor_type": "path", "anchor_value": "src/app.py"}],
            "impacted_claim_ids": ["claim-1"],
            "challenged_claim_ids": ["claim-1"],
            "impacts": [{
                "claim_id": "claim-1",
                "previous_status": "active",
                "new_status": "challenged",
                "matched_anchors": [{"anchor_type": "path", "anchor_value": "src/app.py"}],
            }],
        }

        with patch(
            "consolidation_memory.server.run_detect_drift_subprocess",
            new_callable=AsyncMock,
            return_value=expected,
        ) as mock_detect:
            output = asyncio.run(
                memory_detect_drift(base_ref="origin/main", repo_path="C:/repo")
            )

        assert json.loads(output) == expected
        mock_detect.assert_awaited_once_with(
            base_ref="origin/main",
            repo_path="C:/repo",
            timeout_seconds=ANY,
        )

    def test_memory_detect_drift_returns_error_json(self):
        from consolidation_memory.server import memory_detect_drift

        with patch(
            "consolidation_memory.server.run_detect_drift_subprocess",
            new_callable=AsyncMock,
            side_effect=RuntimeError("git diff failed"),
        ):
            output = asyncio.run(memory_detect_drift())

        data = json.loads(output)
        assert "error" in data
        assert "git diff failed" in data["error"]

    def test_memory_detect_drift_timeout_returns_fallback_json(self):
        from consolidation_memory.server import memory_detect_drift

        async def _detect(*args, **kwargs):
            del args
            if kwargs.get("base_ref") is not None:
                raise asyncio.TimeoutError()
            return {
                "checked_anchors": [{"anchor_type": "path", "anchor_value": "src/fallback.py"}],
                "impacted_claim_ids": [],
                "challenged_claim_ids": [],
                "impacts": [],
            }

        with (
            patch(
                "consolidation_memory.server.run_detect_drift_subprocess",
                new_callable=AsyncMock,
                side_effect=_detect,
            ),
            patch("consolidation_memory.server._MEMORY_DETECT_DRIFT_TIMEOUT_SECONDS", 0.01),
        ):
            output = asyncio.run(memory_detect_drift(base_ref="origin/main"))

        data = json.loads(output)
        assert data["checked_anchors"] == [
            {"anchor_type": "path", "anchor_value": "src/fallback.py"}
        ]
        assert "message" in data
        assert "timed out after" in data["message"]

    def test_memory_detect_drift_timeout_without_base_ref_returns_degraded_json(self):
        from consolidation_memory.server import memory_detect_drift

        async def _slow_detect(*args, **kwargs):
            del args, kwargs
            raise asyncio.TimeoutError()

        with (
            patch(
                "consolidation_memory.server.run_detect_drift_subprocess",
                new_callable=AsyncMock,
                side_effect=_slow_detect,
            ),
            patch("consolidation_memory.server._MEMORY_DETECT_DRIFT_TIMEOUT_SECONDS", 0.01),
        ):
            output = asyncio.run(memory_detect_drift())

        data = json.loads(output)
        assert data["checked_anchors"] == []
        assert data["impacted_claim_ids"] == []
        assert data["challenged_claim_ids"] == []
        assert data["impacts"] == []
        assert "timed out after" in data["message"]

    def test_memory_detect_drift_does_not_require_memory_client_init(self):
        from consolidation_memory.server import memory_detect_drift

        expected = {
            "checked_anchors": [],
            "impacted_claim_ids": [],
            "challenged_claim_ids": [],
            "impacts": [],
        }

        with (
            patch(
                "consolidation_memory.server._get_client_with_timeout",
                side_effect=AssertionError("memory client init should not be used"),
            ),
            patch(
                "consolidation_memory.server.run_detect_drift_subprocess",
                new_callable=AsyncMock,
                return_value=expected,
            ),
        ):
            output = asyncio.run(memory_detect_drift())

        assert json.loads(output) == expected


class TestMCPStoreTools:
    def test_memory_store_signature_supports_scope(self):
        from consolidation_memory.server import memory_store

        sig = inspect.signature(memory_store)
        assert "scope" in sig.parameters
        assert sig.parameters["scope"].default is None

    def test_memory_store_with_scope_calls_scoped_client_method(self):
        from consolidation_memory.server import memory_store
        from consolidation_memory.types import StoreResult

        scoped_payload = {
            "namespace": {"slug": "team-a"},
            "policy": {"write_mode": "deny"},
        }
        mock_client = MagicMock()
        mock_client.store_with_scope.return_value = StoreResult(status="stored", id="scoped-id")

        with patch("consolidation_memory.server._get_client_with_timeout", return_value=mock_client):
            output = asyncio.run(
                memory_store(
                    content="scoped write",
                    scope=scoped_payload,
                )
            )

        assert json.loads(output)["id"] == "scoped-id"
        mock_client.store_with_scope.assert_called_once_with(
            content="scoped write",
            content_type="exchange",
            tags=None,
            surprise=0.5,
            scope=scoped_payload,
        )
        mock_client.store.assert_not_called()

    def test_memory_store_batch_signature_supports_scope(self):
        from consolidation_memory.server import memory_store_batch

        sig = inspect.signature(memory_store_batch)
        assert "scope" in sig.parameters
        assert sig.parameters["scope"].default is None

    def test_memory_store_batch_with_scope_calls_scoped_client_method(self):
        from consolidation_memory.server import memory_store_batch
        from consolidation_memory.types import BatchStoreResult

        scoped_payload = {"project": {"slug": "repo-a"}, "policy": {"write_mode": "allow"}}
        mock_client = MagicMock()
        mock_client.store_batch_with_scope.return_value = BatchStoreResult(
            status="stored",
            stored=1,
            duplicates=0,
        )

        with patch("consolidation_memory.server._get_client_with_timeout", return_value=mock_client):
            output = asyncio.run(
                memory_store_batch(
                    episodes=[{"content": "x"}],
                    scope=scoped_payload,
                )
            )

        assert json.loads(output)["stored"] == 1
        mock_client.store_batch_with_scope.assert_called_once_with(
            episodes=[{"content": "x", "content_type": "exchange", "tags": None, "surprise": 0.5}],
            scope=scoped_payload,
        )
        mock_client.store_batch.assert_not_called()


class TestMCPServerLifecycle:
    def test_get_client_initializes_lazily(self):
        with _patched_server_runtime() as (server, runtime):
            mock_client = MagicMock()
            with patch("consolidation_memory.client.MemoryClient", return_value=mock_client):
                resolved = server._get_client()
            assert resolved is mock_client
            assert runtime.client is mock_client

    def test_lifespan_does_not_eagerly_construct_client(self):
        with _patched_server_runtime() as (server, runtime):
            async def _enter_and_exit_lifespan():
                async with server.lifespan(server.mcp):
                    assert runtime.client is None

            with (
                patch("consolidation_memory.server._WARMUP_ON_START", False),
                patch.object(runtime, "startup") as mock_startup,
                patch("consolidation_memory.client.MemoryClient") as mock_ctor,
            ):
                asyncio.run(_enter_and_exit_lifespan())
                mock_startup.assert_not_called()
                mock_ctor.assert_not_called()

    def test_lifespan_with_warmup_enabled_still_keeps_runtime_lazy(self):
        with _patched_server_runtime() as (server, runtime):
            async def _enter_and_exit_lifespan():
                async with server.lifespan(server.mcp):
                    assert runtime.client is None
                    assert server._warmup_task is None

            with (
                patch("consolidation_memory.server._WARMUP_ON_START", True),
                patch.object(runtime, "startup") as mock_startup,
                patch("consolidation_memory.client.MemoryClient") as mock_ctor,
            ):
                asyncio.run(_enter_and_exit_lifespan())
                mock_startup.assert_not_called()
                mock_ctor.assert_not_called()

    def test_lifespan_closes_connections_and_blocking_executor(self):
        with _patched_server_runtime() as (server, runtime):
            mock_client = MagicMock()
            runtime._client = mock_client

            async def _enter_and_exit_lifespan():
                async with server.lifespan(server.mcp):
                    await server._run_blocking(lambda: 1)

            with patch("consolidation_memory.server._WARMUP_ON_START", False):
                asyncio.run(_enter_and_exit_lifespan())

            mock_client.close.assert_called_once()
            assert runtime.blocking_executor is None

    def test_lifespan_survives_startup_failure_and_tools_return_error_json(self):
        from consolidation_memory.server import memory_status

        with _patched_server_runtime() as (server, runtime):
            async def _enter_and_call():
                async with server.lifespan(server.mcp):
                    return json.loads(await memory_status())

            with (
                patch("consolidation_memory.server._WARMUP_ON_START", False),
                patch.object(runtime, "startup", side_effect=RuntimeError("schema boom")),
            ):
                result = asyncio.run(_enter_and_call())

        assert "error" in result
        assert "MCP runtime startup failed" in result["error"]
        assert "schema boom" in result["error"]

    def test_get_client_with_timeout_raises_timeout_error(self):
        import consolidation_memory.server as server

        with (
            patch("consolidation_memory.server._ensure_runtime_started"),
            patch(
                "consolidation_memory.server._runtime.get_client_with_timeout",
                side_effect=TimeoutError("MemoryClient initialization timed out"),
            ),
            patch("consolidation_memory.server._CLIENT_INIT_TIMEOUT_SECONDS", 0.01),
        ):
            try:
                asyncio.run(server._get_client_with_timeout())
                assert False, "Expected TimeoutError"
            except TimeoutError as exc:
                assert "MemoryClient initialization timed out" in str(exc)

    def test_get_client_reentrant_init_fails_fast_without_deadlock(self):
        with _patched_server_runtime() as (server, runtime):
            mock_client = MagicMock()

            def _constructor():
                try:
                    server._get_client()
                    assert False, "Expected re-entrant init to fail fast"
                except RuntimeError as exc:
                    assert "Re-entrant MemoryClient initialization detected" in str(exc)
                return mock_client

            with (
                patch("consolidation_memory.client.MemoryClient", side_effect=_constructor),
                concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool,
            ):
                resolved = pool.submit(server._get_client).result(timeout=1.0)
                assert resolved is mock_client
                assert runtime.client is mock_client

    def test_get_client_concurrent_calls_share_single_initialization(self):
        with _patched_server_runtime() as (server, runtime):
            mock_client = MagicMock()
            entered = threading.Event()
            release = threading.Event()

            def _constructor():
                entered.set()
                # Keep initialization in-flight long enough for second caller to wait.
                release.wait(timeout=1.0)
                return mock_client

            with (
                patch("consolidation_memory.client.MemoryClient", side_effect=_constructor),
                concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool,
            ):
                fut1 = pool.submit(server._get_client)
                assert entered.wait(timeout=1.0)
                fut2 = pool.submit(server._get_client)
                release.set()
                assert fut1.result(timeout=1.0) is mock_client
                assert fut2.result(timeout=1.0) is mock_client
            assert runtime.client is mock_client

    def test_get_client_aborts_when_lifecycle_changes_mid_init(self):
        with _patched_server_runtime() as (server, runtime):
            runtime._lifecycle_epoch = 7
            mock_client = MagicMock()
            entered = threading.Event()
            release = threading.Event()

            def _constructor():
                entered.set()
                release.wait(timeout=1.0)
                return mock_client

            with (
                patch("consolidation_memory.client.MemoryClient", side_effect=_constructor),
                concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool,
            ):
                future = pool.submit(server._get_client)
                assert entered.wait(timeout=1.0)
                with runtime.client_init_cond:
                    runtime._shutting_down = True
                    runtime._lifecycle_epoch += 1
                    runtime.client_init_cond.notify_all()
                release.set()
                try:
                    future.result(timeout=1.0)
                    assert False, "Expected initialization abort after lifecycle change"
                except RuntimeError as exc:
                    assert "lifecycle changed" in str(exc)

            assert runtime.client is None
            mock_client.close.assert_called_once()

    def test_warm_client_background_skips_init_when_shutdown_flag_set(self):
        with _patched_server_runtime() as (server, runtime):
            runtime._shutting_down = True
            with patch(
                "consolidation_memory.server._get_client_with_timeout",
                side_effect=AssertionError("warmup should not initialize during shutdown"),
            ):
                with patch("consolidation_memory.server._WARMUP_START_DELAY_SECONDS", 0.0):
                    asyncio.run(server._warm_client_background())

    def test_run_server_acquires_and_releases_stdio_singleton_guard(self):
        import consolidation_memory.server as server

        guard = MagicMock()
        with (
            patch("consolidation_memory.config.get_active_project", return_value="default"),
            patch(
                "consolidation_memory.server._acquire_parent_scoped_stdio_singleton_guard",
                return_value=guard,
            ) as mock_acquire,
            patch.object(server.mcp, "run") as mock_run,
        ):
            server.run_server()

        mock_acquire.assert_called_once_with("default")
        mock_run.assert_called_once_with(transport="stdio")
        guard.release.assert_called_once_with()

    def test_run_server_releases_stdio_singleton_guard_on_transport_error(self):
        import consolidation_memory.server as server

        guard = MagicMock()
        with (
            patch("consolidation_memory.config.get_active_project", return_value="default"),
            patch(
                "consolidation_memory.server._acquire_parent_scoped_stdio_singleton_guard",
                return_value=guard,
            ),
            patch.object(server.mcp, "run", side_effect=RuntimeError("transport boom")),
        ):
            try:
                server.run_server()
                assert False, "Expected stdio transport failure to propagate"
            except RuntimeError as exc:
                assert "transport boom" in str(exc)

        guard.release.assert_called_once_with()

    def test_stdio_singleton_guard_rejects_duplicate_same_parent_server(self):
        import consolidation_memory.server as server

        locked_handle = MagicMock()
        with (
            patch("consolidation_memory.server.os.getpid", return_value=222),
            patch("consolidation_memory.server.os.getppid", return_value=111),
            patch(
                "consolidation_memory.server._open_singleton_lock_handle",
                return_value=locked_handle,
            ),
            patch(
                "consolidation_memory.server._try_lock_singleton_handle",
                return_value=False,
            ),
            patch(
                "consolidation_memory.server._read_singleton_metadata",
                return_value={"pid": 333, "parent_pid": 111, "project": "default"},
            ),
            patch(
                "consolidation_memory.server._process_exists",
                return_value=True,
            ),
        ):
            with pytest.raises(RuntimeError, match="already running"):
                server._acquire_parent_scoped_stdio_singleton_guard("default")

        locked_handle.close.assert_called_once_with()

    def test_safe_process_int_rejects_non_pid_values(self):
        import consolidation_memory.server as server

        assert server._safe_process_int(True) is None
        assert server._safe_process_int(0) is None
        assert server._safe_process_int(-5) is None
        assert server._safe_process_int(12.5) is None
        assert server._safe_process_int("  ") is None
        assert server._safe_process_int("12.5") is None
        assert server._safe_process_int("0") is None
        assert server._safe_process_int("123") == 123

    def test_terminate_process_skips_reserved_and_parent_pids(self):
        import consolidation_memory.server as server

        with (
            patch("consolidation_memory.server.os.getpid", return_value=50),
            patch("consolidation_memory.server.os.getppid", return_value=40),
            patch("consolidation_memory.server.os.kill") as mock_kill,
        ):
            server._terminate_process(1)
            server._terminate_process(50)
            server._terminate_process(40)
            server._terminate_process(99)

        mock_kill.assert_called_once_with(99, server.signal.SIGTERM)

    def test_runtime_has_background_activity_detects_consolidation_future(self):
        import consolidation_memory.server as server

        client = MagicMock()
        future = MagicMock()
        future.done.return_value = False
        client._consolidation_future = future
        client._consolidation_lock = threading.Lock()

        with (
            patch.object(server, "_warmup_task", None),
            patch.object(server._runtime, "_client", client),
            patch.object(server._runtime, "_client_initializing", False),
        ):
            assert server._runtime_has_background_activity() is True

    def test_runtime_has_background_activity_returns_false_when_idle(self):
        import consolidation_memory.server as server

        client = MagicMock()
        future = MagicMock()
        future.done.return_value = True
        client._consolidation_future = future
        client._consolidation_lock = threading.Lock()

        with (
            patch.object(server, "_warmup_task", None),
            patch.object(server._runtime, "_client", client),
            patch.object(server._runtime, "_client_initializing", False),
        ):
            assert server._runtime_has_background_activity() is False

    def test_recycle_idle_runtime_shuts_down_runtime_without_exiting_transport(self):
        import consolidation_memory.server as server_module

        with _patched_server_runtime() as (server, runtime):
            mock_client = MagicMock()
            runtime._client = mock_client
            runtime._blocking_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            with patch.object(server_module, "_runtime_started", True):
                server._recycle_idle_runtime()

            mock_client.close.assert_called_once_with()
            assert runtime.client is None
            assert runtime.blocking_executor is None
            assert server._runtime_started is False
            assert server._startup_error is None

    def test_idle_shutdown_monitor_recycles_runtime_instead_of_exiting(self):
        import consolidation_memory.server as server

        sleep_calls = 0

        async def _fake_sleep(_seconds):
            nonlocal sleep_calls
            sleep_calls += 1
            if sleep_calls > 1:
                raise asyncio.CancelledError()

        with (
            patch("consolidation_memory.server._IDLE_TIMEOUT_SECONDS", 1.0),
            patch("consolidation_memory.server._IDLE_CHECK_INTERVAL_SECONDS", 0.0),
            patch.object(server, "_active_tool_calls", 0),
            patch.object(server, "_last_activity_monotonic", time.monotonic() - 5.0),
            patch("consolidation_memory.server._runtime_has_background_activity", return_value=False),
            patch("consolidation_memory.server._recycle_idle_runtime") as mock_recycle,
            patch("consolidation_memory.server.asyncio.sleep", side_effect=_fake_sleep),
        ):
            with pytest.raises(asyncio.CancelledError):
                asyncio.run(server._idle_shutdown_monitor())

        mock_recycle.assert_called_once_with()


class TestMCPRecallTool:
    def test_memory_recall_calls_canonical_query_service(self):
        from consolidation_memory.server import memory_recall
        from consolidation_memory.types import RecallResult

        mock_client = MagicMock()
        mock_client.query_recall.return_value = RecallResult()

        with patch("consolidation_memory.server._get_client_with_timeout", return_value=mock_client):
            output = asyncio.run(
                memory_recall(
                    query="python runtime",
                    as_of="2025-06-01T00:00:00+00:00",
                    scope={"project": {"slug": "repo-a"}},
                )
            )

        data = json.loads(output)
        assert data["total_episodes"] == 0
        mock_client.query_recall.assert_called_once_with(
            query="python runtime",
            n_results=10,
            include_knowledge=True,
            content_types=None,
            tags=None,
            after=None,
            before=None,
            include_expired=False,
            as_of="2025-06-01T00:00:00+00:00",
            scope={"project": {"slug": "repo-a"}},
        )

    def test_memory_recall_returns_client_init_error_json(self):
        from consolidation_memory.server import memory_recall

        with patch(
            "consolidation_memory.server._get_client_with_timeout",
            side_effect=RuntimeError("client init failed"),
        ):
            output = asyncio.run(memory_recall(query="test"))

        data = json.loads(output)
        assert "error" in data
        assert "client init failed" in data["error"]

    def test_memory_recall_accepts_project_string_scope_shorthand(self):
        from consolidation_memory.server import memory_recall
        from consolidation_memory.types import RecallResult

        mock_client = MagicMock()
        mock_client.query_recall.return_value = RecallResult()

        with patch("consolidation_memory.server._get_client_with_timeout", return_value=mock_client):
            output = asyncio.run(
                memory_recall(
                    query="python runtime",
                    scope={"project": "repo-a"},
                )
            )

        data = json.loads(output)
        assert data["total_episodes"] == 0
        mock_client.query_recall.assert_called_once_with(
            query="python runtime",
            n_results=10,
            include_knowledge=True,
            content_types=None,
            tags=None,
            after=None,
            before=None,
            include_expired=False,
            as_of=None,
            scope={"project": "repo-a"},
        )

    def test_memory_recall_accepts_scope_string_path_and_auto_coerces(self):
        from consolidation_memory.server import memory_recall
        from consolidation_memory.types import RecallResult

        mock_client = MagicMock()
        mock_client.query_recall.return_value = RecallResult()

        with patch("consolidation_memory.server._get_client_with_timeout", return_value=mock_client):
            output = asyncio.run(
                memory_recall(
                    query="python runtime",
                    scope=r"C:\\Users\\gore\\consolidation-memory",
                )
            )

        data = json.loads(output)
        assert data["total_episodes"] == 0
        mock_client.query_recall.assert_called_once_with(
            query="python runtime",
            n_results=10,
            include_knowledge=True,
            content_types=None,
            tags=None,
            after=None,
            before=None,
            include_expired=False,
            as_of=None,
            scope={"project": {"root_uri": r"C:\\Users\\gore\\consolidation-memory"}},
        )

    def test_memory_store_rejects_invalid_scope_enum(self):
        from consolidation_memory.server import memory_store

        output = asyncio.run(
            memory_store(
                content="test",
                scope={"policy": {"write_mode": "invalid"}},
            )
        )

        data = json.loads(output)
        assert data == {"error": "scope.policy.write_mode must be one of: allow, deny"}

    def test_memory_claim_search_rejects_oversized_query(self):
        from consolidation_memory.server import memory_claim_search

        output = asyncio.run(memory_claim_search(query="x" * 10_001))

        data = json.loads(output)
        assert "error" in data
        assert "Maximum is 10000 characters" in data["error"]


class TestMCPScopeForwarding:
    def test_memory_forget_signature_supports_scope(self):
        from consolidation_memory.server import memory_forget

        sig = inspect.signature(memory_forget)
        assert "scope" in sig.parameters
        assert sig.parameters["scope"].default is None

    def test_memory_forget_with_scope_calls_client_method(self):
        from consolidation_memory.server import memory_forget
        from consolidation_memory.types import ForgetResult

        scope = {"project": {"slug": "repo-a"}}
        mock_client = MagicMock()
        mock_client.forget.return_value = ForgetResult(status="forgotten", id="ep-1")

        with patch("consolidation_memory.server._get_client_with_timeout", return_value=mock_client):
            output = asyncio.run(memory_forget(episode_id="ep-1", scope=scope))

        assert json.loads(output)["status"] == "forgotten"
        mock_client.forget.assert_called_once_with(episode_id="ep-1", scope=scope)

    def test_memory_export_signature_supports_scope(self):
        from consolidation_memory.server import memory_export

        sig = inspect.signature(memory_export)
        assert "scope" in sig.parameters
        assert sig.parameters["scope"].default is None

    def test_memory_export_with_scope_calls_client_method(self):
        from consolidation_memory.server import memory_export
        from consolidation_memory.types import ExportResult

        scope = {"namespace": {"slug": "team-a"}}
        mock_client = MagicMock()
        mock_client.export.return_value = ExportResult(status="exported", path="/tmp/export.json")

        with patch("consolidation_memory.server._get_client_with_timeout", return_value=mock_client):
            output = asyncio.run(memory_export(scope=scope))

        assert json.loads(output)["status"] == "exported"
        mock_client.export.assert_called_once_with(scope=scope)

    def test_memory_correct_with_scope_calls_client_method(self):
        from consolidation_memory.server import memory_correct
        from consolidation_memory.types import CorrectResult

        scope = {"project": {"slug": "repo-a"}}
        mock_client = MagicMock()
        mock_client.correct.return_value = CorrectResult(status="write_denied", filename="topic.md")

        with patch("consolidation_memory.server._get_client_with_timeout", return_value=mock_client):
            output = asyncio.run(
                memory_correct(topic_filename="topic.md", correction="fix", scope=scope)
            )

        assert json.loads(output)["status"] == "write_denied"
        mock_client.correct.assert_called_once_with(
            topic_filename="topic.md",
            correction="fix",
            scope=scope,
        )

    def test_memory_protect_with_scope_calls_client_method(self):
        from consolidation_memory.server import memory_protect
        from consolidation_memory.types import ProtectResult

        scope = {"namespace": {"slug": "team-a"}}
        mock_client = MagicMock()
        mock_client.protect.return_value = ProtectResult(status="protected", protected_count=1)

        with patch("consolidation_memory.server._get_client_with_timeout", return_value=mock_client):
            output = asyncio.run(memory_protect(episode_id="ep-1", scope=scope))

        assert json.loads(output)["status"] == "protected"
        mock_client.protect.assert_called_once_with(episode_id="ep-1", tag=None, scope=scope)

    def test_memory_browse_read_topic_and_timeline_with_scope_call_client_methods(self):
        from consolidation_memory.server import memory_browse, memory_read_topic, memory_timeline
        from consolidation_memory.types import BrowseResult, TimelineResult, TopicDetailResult

        scope = {"project": {"slug": "repo-a"}}
        mock_client = MagicMock()
        mock_client.browse.return_value = BrowseResult(topics=[], total=0)
        mock_client.read_topic.return_value = TopicDetailResult(status="ok", filename="topic.md")
        mock_client.timeline.return_value = TimelineResult(query="python", entries=[], total=0)

        with patch("consolidation_memory.server._get_client_with_timeout", return_value=mock_client):
            browse_output = asyncio.run(memory_browse(scope=scope))
            read_output = asyncio.run(memory_read_topic(filename="topic.md", scope=scope))
            timeline_output = asyncio.run(memory_timeline(topic="python", scope=scope))

        assert json.loads(browse_output)["total"] == 0
        assert json.loads(read_output)["status"] == "ok"
        assert json.loads(timeline_output)["query"] == "python"
        mock_client.browse.assert_called_once_with(scope=scope)
        mock_client.read_topic.assert_called_once_with(filename="topic.md", scope=scope)
        mock_client.timeline.assert_called_once_with(topic="python", scope=scope)

    def test_memory_browse_accepts_flat_scope_row(self):
        from consolidation_memory.server import memory_browse
        from consolidation_memory.types import BrowseResult

        flat_scope = {
            "namespace_slug": "team-a",
            "app_client_name": "desktop",
            "app_client_type": "mcp",
            "project_slug": "repo-a",
        }
        mock_client = MagicMock()
        mock_client.browse.return_value = BrowseResult(topics=[], total=0)

        with patch("consolidation_memory.server._get_client_with_timeout", return_value=mock_client):
            browse_output = asyncio.run(memory_browse(scope=flat_scope))

        assert json.loads(browse_output)["total"] == 0
        mock_client.browse.assert_called_once_with(scope=flat_scope)

    def test_memory_recall_timeout_falls_back_to_episodes_only(self):
        from consolidation_memory.server import memory_recall
        from consolidation_memory.types import RecallResult, SearchResult

        def _query_recall(*args, **kwargs):
            include_knowledge = bool(kwargs["include_knowledge"])
            if include_knowledge:
                time.sleep(0.05)
                return RecallResult()

        mock_client = MagicMock()
        mock_client.query_recall.side_effect = _query_recall
        mock_client.query_search.return_value = SearchResult(
            episodes=[{"id": "ep-1"}],
            total_matches=1,
            query="python runtime",
        )

        with (
            patch("consolidation_memory.server._get_client_with_timeout", return_value=mock_client),
            patch("consolidation_memory.server._MEMORY_RECALL_TIMEOUT_SECONDS", 0.01),
            patch("consolidation_memory.server._MEMORY_RECALL_FALLBACK_TIMEOUT_SECONDS", 0.2),
        ):
            output = asyncio.run(
                memory_recall(
                    query="python runtime",
                    include_knowledge=True,
                )
            )

        data = json.loads(output)
        assert data["episodes"] == [{"id": "ep-1"}]
        assert any("episodes-only fallback" in msg for msg in data.get("warnings", []))
        mock_client.query_search.assert_called_once_with(
            query="python runtime",
            content_types=None,
            tags=None,
            after=None,
            before=None,
            limit=10,
            scope=None,
        )

    def test_memory_recall_timeout_returns_error_without_fallback(self):
        from consolidation_memory.server import memory_recall

        def _slow_query_recall(*args, **kwargs):
            del args, kwargs
            time.sleep(0.05)
            return MagicMock()

        def _slow_query_search(*args, **kwargs):
            del args, kwargs
            time.sleep(0.05)
            return MagicMock()

        mock_client = MagicMock()
        mock_client.query_recall.side_effect = _slow_query_recall
        mock_client.query_search.side_effect = _slow_query_search

        with (
            patch("consolidation_memory.server._get_client_with_timeout", return_value=mock_client),
            patch("consolidation_memory.server._MEMORY_RECALL_TIMEOUT_SECONDS", 0.01),
            patch("consolidation_memory.server._MEMORY_RECALL_FALLBACK_TIMEOUT_SECONDS", 0.01),
        ):
            output = asyncio.run(
                memory_recall(
                    query="python runtime",
                    include_knowledge=False,
                )
            )

        data = json.loads(output)
        assert "error" in data
        assert "memory_recall timed out after" in data["error"]
        assert "keyword fallback timed out after" in data["error"]


class TestMCPClaimTools:
    def test_memory_claim_browse_calls_canonical_query_service(self):
        from consolidation_memory.server import memory_claim_browse
        from consolidation_memory.types import ClaimBrowseResult

        mock_client = MagicMock()
        mock_client.query_browse_claims.return_value = ClaimBrowseResult(claims=[], total=0)

        with patch("consolidation_memory.server._get_client_with_timeout", return_value=mock_client):
            output = asyncio.run(
                memory_claim_browse(
                    claim_type="fact",
                    scope={"project": {"slug": "repo-a"}},
                )
            )

        assert json.loads(output)["total"] == 0
        mock_client.query_browse_claims.assert_called_once_with(
            claim_type="fact",
            as_of=None,
            limit=50,
            scope={"project": {"slug": "repo-a"}},
        )

    def test_memory_claim_search_calls_canonical_query_service(self):
        from consolidation_memory.server import memory_claim_search
        from consolidation_memory.types import ClaimSearchResult

        mock_client = MagicMock()
        mock_client.query_search_claims.return_value = ClaimSearchResult(
            claims=[],
            total_matches=0,
            query="python",
        )

        with patch("consolidation_memory.server._get_client_with_timeout", return_value=mock_client):
            output = asyncio.run(
                memory_claim_search(
                    query="python",
                    scope={"namespace": {"slug": "team-a"}},
                )
            )

        assert json.loads(output)["total_matches"] == 0
        mock_client.query_search_claims.assert_called_once_with(
            query="python",
            claim_type=None,
            as_of=None,
            limit=50,
            scope={"namespace": {"slug": "team-a"}},
        )


class TestMCPOutcomeTools:
    def test_memory_outcome_record_calls_canonical_client_method(self):
        from consolidation_memory.server import memory_outcome_record
        from consolidation_memory.types import OutcomeRecordResult

        mock_client = MagicMock()
        mock_client.record_outcome.return_value = OutcomeRecordResult(
            status="recorded",
            id="outcome-1",
            action_key="act_abc",
            outcome_type="success",
        )

        with patch("consolidation_memory.server._get_client_with_timeout", return_value=mock_client):
            output = asyncio.run(
                memory_outcome_record(
                    action_summary="Run targeted pytest",
                    outcome_type="success",
                    source_claim_ids=["claim-1"],
                    scope={"project": {"slug": "repo-a"}},
                )
            )

        assert json.loads(output)["status"] == "recorded"
        mock_client.record_outcome.assert_called_once_with(
            action_summary="Run targeted pytest",
            outcome_type="success",
            source_claim_ids=["claim-1"],
            source_record_ids=None,
            source_episode_ids=None,
            code_anchors=None,
            issue_ids=None,
            pr_ids=None,
            action_key=None,
            summary=None,
            details=None,
            confidence=0.8,
            provenance=None,
            observed_at=None,
            scope={"project": {"slug": "repo-a"}},
        )

    def test_memory_outcome_browse_calls_canonical_query_service(self):
        from consolidation_memory.server import memory_outcome_browse
        from consolidation_memory.types import OutcomeBrowseResult

        mock_client = MagicMock()
        mock_client.query_browse_outcomes.return_value = OutcomeBrowseResult(outcomes=[], total=0)

        with patch("consolidation_memory.server._get_client_with_timeout", return_value=mock_client):
            output = asyncio.run(
                memory_outcome_browse(
                    outcome_type="failure",
                    source_claim_id="claim-1",
                    scope={"namespace": {"slug": "team-a"}},
                )
            )

        assert json.loads(output)["total"] == 0
        mock_client.query_browse_outcomes.assert_called_once_with(
            outcome_type="failure",
            action_key=None,
            source_claim_id="claim-1",
            source_record_id=None,
            source_episode_id=None,
            as_of=None,
            limit=50,
            scope={"namespace": {"slug": "team-a"}},
        )


class TestMCPConsolidateTool:
    def test_memory_consolidate_passthrough_status(self):
        from consolidation_memory.server import memory_consolidate

        mock_client = MagicMock()
        mock_client.consolidate.return_value = {"status": "error", "message": "boom"}

        with patch("consolidation_memory.server._get_client_with_timeout", return_value=mock_client):
            output = asyncio.run(memory_consolidate())

        assert json.loads(output) == {"status": "error", "message": "boom"}

    def test_memory_consolidate_already_running_message(self):
        from consolidation_memory.server import memory_consolidate

        mock_client = MagicMock()
        mock_client.consolidate.return_value = {"status": "already_running"}

        with patch("consolidation_memory.server._get_client_with_timeout", return_value=mock_client):
            output = asyncio.run(memory_consolidate())

        assert json.loads(output) == {
            "status": "already_running",
            "message": "A consolidation run is already in progress",
        }
