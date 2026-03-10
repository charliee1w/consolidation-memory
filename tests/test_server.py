"""Tests for MCP server tool wrappers."""

from __future__ import annotations

import asyncio
import concurrent.futures
import inspect
import json
import threading
import time
from unittest.mock import MagicMock, patch


class TestMCPDetectDriftTool:
    def test_memory_detect_drift_signature(self):
        from consolidation_memory.server import memory_detect_drift

        sig = inspect.signature(memory_detect_drift)
        assert "base_ref" in sig.parameters
        assert "repo_path" in sig.parameters
        assert sig.parameters["base_ref"].default is None
        assert sig.parameters["repo_path"].default is None

    def test_memory_detect_drift_calls_client(self):
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
        mock_client = MagicMock()
        mock_client.query_detect_drift.return_value = expected

        with patch("consolidation_memory.server._get_client", return_value=mock_client):
            output = asyncio.run(
                memory_detect_drift(base_ref="origin/main", repo_path="C:/repo")
            )

        assert json.loads(output) == expected
        mock_client.query_detect_drift.assert_called_once_with(
            base_ref="origin/main",
            repo_path="C:/repo",
        )

    def test_memory_detect_drift_returns_error_json(self):
        from consolidation_memory.server import memory_detect_drift

        mock_client = MagicMock()
        mock_client.query_detect_drift.side_effect = RuntimeError("git diff failed")

        with patch("consolidation_memory.server._get_client", return_value=mock_client):
            output = asyncio.run(memory_detect_drift())

        data = json.loads(output)
        assert "error" in data
        assert "git diff failed" in data["error"]

    def test_memory_detect_drift_timeout_returns_error_json(self):
        from consolidation_memory.server import memory_detect_drift

        def _slow_detect(*args, **kwargs):
            del args, kwargs
            time.sleep(0.05)
            return {}

        mock_client = MagicMock()
        mock_client.query_detect_drift.side_effect = _slow_detect

        with (
            patch("consolidation_memory.server._get_client", return_value=mock_client),
            patch("consolidation_memory.server._MEMORY_DETECT_DRIFT_TIMEOUT_SECONDS", 0.01),
        ):
            output = asyncio.run(memory_detect_drift())

        data = json.loads(output)
        assert "error" in data
        assert "timed out after" in data["error"]


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
            "scoped write",
            "exchange",
            None,
            0.5,
            scoped_payload,
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
            [{"content": "x"}],
            scoped_payload,
        )
        mock_client.store_batch.assert_not_called()


class TestMCPServerLifecycle:
    def test_get_client_initializes_lazily(self):
        import consolidation_memory.server as server

        original_client = server._client
        try:
            server._client = None
            mock_client = MagicMock()
            with patch("consolidation_memory.client.MemoryClient", return_value=mock_client):
                resolved = server._get_client()
            assert resolved is mock_client
            assert server._client is mock_client
        finally:
            server._client = original_client

    def test_lifespan_does_not_eagerly_construct_client(self):
        import consolidation_memory.server as server

        original_client = server._client
        try:
            server._client = None

            async def _enter_and_exit_lifespan():
                async with server.lifespan(server.mcp):
                    assert server._client is None

            with (
                patch("consolidation_memory.server._WARMUP_ON_START", False),
                patch("consolidation_memory.client.MemoryClient") as mock_ctor,
            ):
                asyncio.run(_enter_and_exit_lifespan())
                mock_ctor.assert_not_called()
        finally:
            server._client = original_client

    def test_get_client_with_timeout_raises_timeout_error(self):
        import consolidation_memory.server as server

        def _slow_get_client():
            time.sleep(0.05)
            return MagicMock()

        with (
            patch("consolidation_memory.server._get_client", side_effect=_slow_get_client),
            patch("consolidation_memory.server._CLIENT_INIT_TIMEOUT_SECONDS", 0.01),
        ):
            try:
                asyncio.run(server._get_client_with_timeout())
                assert False, "Expected TimeoutError"
            except TimeoutError as exc:
                assert "MemoryClient initialization timed out" in str(exc)

    def test_get_client_reentrant_init_fails_fast_without_deadlock(self):
        import consolidation_memory.server as server

        original_client = server._client
        original_initializing = server._client_initializing
        original_owner_thread_id = server._client_init_owner_thread_id
        original_init_error = server._client_init_error
        try:
            server._client = None
            server._client_initializing = False
            server._client_init_owner_thread_id = None
            server._client_init_error = None

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
                assert server._client is mock_client
        finally:
            server._client = original_client
            server._client_initializing = original_initializing
            server._client_init_owner_thread_id = original_owner_thread_id
            server._client_init_error = original_init_error

    def test_get_client_concurrent_calls_share_single_initialization(self):
        import consolidation_memory.server as server

        original_client = server._client
        original_initializing = server._client_initializing
        original_owner_thread_id = server._client_init_owner_thread_id
        original_init_error = server._client_init_error
        try:
            server._client = None
            server._client_initializing = False
            server._client_init_owner_thread_id = None
            server._client_init_error = None

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
        finally:
            server._client = original_client
            server._client_initializing = original_initializing
            server._client_init_owner_thread_id = original_owner_thread_id
            server._client_init_error = original_init_error


class TestMCPRecallTool:
    def test_memory_recall_calls_canonical_query_service(self):
        from consolidation_memory.server import memory_recall
        from consolidation_memory.types import RecallResult

        mock_client = MagicMock()
        mock_client.query_recall.return_value = RecallResult()

        with patch("consolidation_memory.server._get_client", return_value=mock_client):
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
            "python runtime",
            10,
            True,
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
            "consolidation_memory.server._get_client",
            side_effect=RuntimeError("client init failed"),
        ):
            output = asyncio.run(memory_recall(query="test"))

        data = json.loads(output)
        assert "error" in data
        assert "client init failed" in data["error"]

    def test_memory_recall_timeout_falls_back_to_episodes_only(self):
        from consolidation_memory.server import memory_recall
        from consolidation_memory.types import RecallResult

        def _query_recall(*args, **kwargs):
            include_knowledge = bool(args[2])
            if include_knowledge:
                time.sleep(0.05)
                return RecallResult()

        mock_client = MagicMock()
        mock_client.query_recall.side_effect = _query_recall
        mock_client.query_search.return_value = MagicMock(
            episodes=[{"id": "ep-1"}],
            total_matches=1,
        )

        with (
            patch("consolidation_memory.server._get_client", return_value=mock_client),
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
            patch("consolidation_memory.server._get_client", return_value=mock_client),
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

        with patch("consolidation_memory.server._get_client", return_value=mock_client):
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

        with patch("consolidation_memory.server._get_client", return_value=mock_client):
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


class TestMCPConsolidateTool:
    def test_memory_consolidate_passthrough_status(self):
        from consolidation_memory.server import memory_consolidate

        mock_client = MagicMock()
        mock_client.consolidate.return_value = {"status": "error", "message": "boom"}

        with patch("consolidation_memory.server._get_client", return_value=mock_client):
            output = asyncio.run(memory_consolidate())

        assert json.loads(output) == {"status": "error", "message": "boom"}

    def test_memory_consolidate_already_running_message(self):
        from consolidation_memory.server import memory_consolidate

        mock_client = MagicMock()
        mock_client.consolidate.return_value = {"status": "already_running"}

        with patch("consolidation_memory.server._get_client", return_value=mock_client):
            output = asyncio.run(memory_consolidate())

        assert json.loads(output) == {
            "status": "already_running",
            "message": "A consolidation run is already in progress",
        }
