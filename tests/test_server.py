"""Tests for MCP server tool wrappers."""

from __future__ import annotations

import asyncio
import inspect
import json
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
