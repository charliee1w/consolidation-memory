"""Tests for memory_remember / memory_ask simple MCP tools."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from consolidation_memory.simple_api import (
    build_ask_recall_arguments,
    build_remember_store_arguments,
    map_simple_kind,
    simplify_recall_result,
)
from consolidation_memory.tool_dispatch import execute_tool_call
from tests.surface_contract_helpers import invoke_surfaces_with_execute_tool_call

try:
    import fastapi  # noqa: F401

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


class TestSimpleApiHelpers:
    def test_remember_maps_fix_to_solution(self):
        args = build_remember_store_arguments(
            {"content": "Fixed auth timeout", "kind": "fix", "tags": ["auth"]}
        )
        assert args["content_type"] == "solution"
        assert args["tags"] == ["auth"]

    def test_ask_builds_recall_payload(self):
        args = build_ask_recall_arguments({"query": "auth fix", "n_results": 5})
        assert args["query"] == "auth fix"
        assert args["n_results"] == 5
        assert args["include_knowledge"] is True


class TestSimpleToolDispatch:
    @patch("consolidation_memory.backends.encode_documents")
    def test_memory_remember_stores_with_mapped_type(self, mock_embed, tmp_data_dir):
        from tests.helpers import make_normalized_vec as _vec

        mock_embed.return_value = _vec(seed=3).reshape(1, -1)

        from consolidation_memory.client import MemoryClient

        with MemoryClient(auto_consolidate=False) as client:
            result = execute_tool_call(
                "memory_remember",
                {"content": "Simple remember test", "kind": "fact", "tags": ["simple"]},
                client=client,
            )
        assert result["status"] == "stored"
        assert result["content_type"] == "fact"

    @patch("consolidation_memory.backends.encode_documents")
    def test_memory_ask_returns_simplified_envelope(self, mock_embed, tmp_data_dir):
        from tests.helpers import make_normalized_vec as _vec

        mock_embed.return_value = _vec(seed=5).reshape(1, -1)

        from consolidation_memory.client import MemoryClient

        with MemoryClient(auto_consolidate=False) as client:
            execute_tool_call(
                "memory_remember",
                {"content": "Ask target content unique xyz", "kind": "note"},
                client=client,
            )
            result = execute_tool_call(
                "memory_ask",
                {"query": "Ask target content unique xyz", "n_results": 5},
                client=client,
            )
        assert result["query"] == "Ask target content unique xyz"
        assert "episodes" in result
        assert isinstance(result["episodes"], list)

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError, match="kind must be one of"):
            map_simple_kind("unknown")


@pytest.mark.skipif(not HAS_FASTAPI, reason="fastapi not installed")
class TestSimpleSurfaceContracts:
    def test_remember_matches_across_surfaces(self):
        expected = {"status": "stored", "id": "ep-simple", "content_type": "solution"}

        async def _mcp():
            from consolidation_memory.server import memory_remember

            return await memory_remember(
                content="MCP remember",
                kind="fix",
                tags=["contract"],
            )

        dispatch_out, mcp_out, rest_out, mock_execute = (
            invoke_surfaces_with_execute_tool_call(
                tool_name="memory_remember",
                tool_args={
                    "content": "OpenAI remember",
                    "kind": "fix",
                    "tags": ["contract"],
                },
                expected_result=expected,
                mcp_coro_factory=_mcp,
                rest_path="/memory/remember",
                rest_json={
                    "content": "REST remember",
                    "kind": "fix",
                    "tags": ["contract"],
                },
            )
        )

        assert dispatch_out == expected
        assert mcp_out == expected
        assert rest_out == expected
        store_calls = [
            args
            for tool_name, args in mock_execute.recorded_tool_calls
            if tool_name == "memory_store"
        ]
        assert store_calls
        assert store_calls[0]["content_type"] == "solution"

    def test_ask_matches_across_surfaces(self):
        recall_payload = {
            "episodes": [
                {
                    "id": "1",
                    "content": "hello",
                    "content_type": "fact",
                    "tags": [],
                }
            ],
            "knowledge": [],
            "records": [],
            "claims": [],
            "warnings": [],
        }
        expected = simplify_recall_result(recall_payload)
        expected["query"] = "test query"

        async def _mcp():
            from consolidation_memory.server import memory_ask

            return await memory_ask(query="test query", n_results=5)

        dispatch_out, mcp_out, rest_out, mock_execute = (
            invoke_surfaces_with_execute_tool_call(
                tool_name="memory_ask",
                tool_args={"query": "test query", "n_results": 5},
                expected_result=recall_payload,
                mcp_coro_factory=_mcp,
                rest_path="/memory/ask",
                rest_json={"query": "test query", "n_results": 5},
            )
        )

        assert dispatch_out == expected
        assert mcp_out == expected
        assert rest_out == expected
        assert mock_execute.recorded_tool_calls[0][0] == "memory_ask"