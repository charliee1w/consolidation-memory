"""Cross-surface contract tests (OpenAI dispatch, MCP wrapper, REST)."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest

from consolidation_memory.schemas import dispatch_tool_call
from consolidation_memory.types import ClaimSearchResult, RecallResult

try:
    from fastapi.testclient import TestClient
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


pytestmark = pytest.mark.skipif(not HAS_FASTAPI, reason="fastapi not installed")


def _normalize_server_recall_call(mock_client: MagicMock) -> dict:
    args, kwargs = mock_client.query_recall.call_args
    return {
        "query": args[0],
        "n_results": args[1],
        "include_knowledge": args[2],
        "content_types": kwargs["content_types"],
        "tags": kwargs["tags"],
        "after": kwargs["after"],
        "before": kwargs["before"],
        "include_expired": kwargs["include_expired"],
        "as_of": kwargs["as_of"],
        "scope": kwargs["scope"],
    }


class TestSurfaceRecallContract:
    def test_recall_defaults_and_output_match_across_surfaces(self):
        from consolidation_memory.rest import create_app
        from consolidation_memory.server import memory_recall

        scope = {"project": {"slug": "repo-a"}}
        expected = RecallResult(
            episodes=[{"id": "ep-1", "content": "hello"}],
            knowledge=[],
            records=[],
            claims=[],
            total_episodes=1,
            total_knowledge_topics=0,
        )
        expected_args = {
            "query": "hello",
            "n_results": 10,
            "include_knowledge": True,
            "content_types": None,
            "tags": None,
            "after": None,
            "before": None,
            "include_expired": False,
            "as_of": None,
            "scope": scope,
        }

        # OpenAI dispatch surface
        dispatch_client = MagicMock()
        dispatch_client.query_recall.return_value = expected
        dispatch_out = dispatch_tool_call(
            dispatch_client,
            "memory_recall",
            {"query": "hello", "scope": scope},
        )
        dispatch_client.query_recall.assert_called_once_with(**expected_args)

        # MCP server surface
        mcp_client = MagicMock()
        mcp_client.query_recall.return_value = expected
        with patch("consolidation_memory.server._get_client_with_timeout", return_value=mcp_client):
            mcp_out = json.loads(asyncio.run(memory_recall(query="hello", scope=scope)))
        assert _normalize_server_recall_call(mcp_client) == expected_args

        # REST surface
        with patch("consolidation_memory.client.MemoryClient.query_recall", return_value=expected) as rest_call:
            app = create_app()
            with TestClient(app) as client:
                rest_resp = client.post("/memory/recall", json={"query": "hello", "scope": scope})
            rest_out = rest_resp.json()
        assert rest_resp.status_code == 200
        rest_call.assert_called_once_with(**expected_args)

        assert dispatch_out == mcp_out == rest_out


class TestSurfaceClaimSearchContract:
    def test_claim_search_defaults_and_output_match_across_surfaces(self):
        from consolidation_memory.rest import create_app
        from consolidation_memory.server import memory_claim_search

        scope = {"namespace": {"slug": "team-a"}}
        expected = ClaimSearchResult(
            claims=[{"id": "claim-1", "canonical_text": "python version is 3.12"}],
            total_matches=1,
            query="python",
        )
        expected_args = {
            "query": "python",
            "claim_type": None,
            "as_of": None,
            "limit": 50,
            "scope": scope,
        }

        # OpenAI dispatch surface
        dispatch_client = MagicMock()
        dispatch_client.query_search_claims.return_value = expected
        dispatch_out = dispatch_tool_call(
            dispatch_client,
            "memory_claim_search",
            {"query": "python", "scope": scope},
        )
        dispatch_client.query_search_claims.assert_called_once_with(**expected_args)

        # MCP server surface
        mcp_client = MagicMock()
        mcp_client.query_search_claims.return_value = expected
        with patch("consolidation_memory.server._get_client_with_timeout", return_value=mcp_client):
            mcp_out = json.loads(asyncio.run(memory_claim_search(query="python", scope=scope)))
        mcp_client.query_search_claims.assert_called_once_with(**expected_args)

        # REST surface
        with patch(
            "consolidation_memory.client.MemoryClient.query_search_claims",
            return_value=expected,
        ) as rest_call:
            app = create_app()
            with TestClient(app) as client:
                rest_resp = client.post(
                    "/memory/claims/search",
                    json={"query": "python", "scope": scope},
                )
            rest_out = rest_resp.json()
        assert rest_resp.status_code == 200
        rest_call.assert_called_once_with(**expected_args)

        assert dispatch_out == mcp_out == rest_out
