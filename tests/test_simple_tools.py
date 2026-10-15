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
from consolidation_memory.tool_dispatch import (
    ToolContractError,
    accepted_argument_names,
    execute_tool_call,
)
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

    def test_ask_translation_sets_include_knowledge_rather_than_reading_it(self):
        """Why the web UI may pass its own arguments through untouched.

        The UI used to translate before calling ``memory_ask``, and the value it
        forwarded was this constant. Removing the double translation therefore
        changed nothing: the flag is set here, inside the tool that owns it.
        """
        assert build_ask_recall_arguments(
            {"query": "auth fix", "include_knowledge": False}
        )["include_knowledge"] is True

    def test_translation_outputs_only_published_target_arguments(self):
        """Both translators feed the dispatch seam, which rejects unknown keys."""
        store_published = accepted_argument_names("memory_store") or frozenset()
        recall_published = accepted_argument_names("memory_recall") or frozenset()

        store_args = build_remember_store_arguments(
            {"content": "c", "kind": "fact", "tags": ["a"], "scope": {"namespace": {"slug": "n"}}}
        )
        recall_args = build_ask_recall_arguments({"query": "q", "n_results": 3})

        assert set(store_args) <= store_published
        assert set(recall_args) <= recall_published

    def test_remember_does_not_forward_an_unpublished_surprise(self):
        """``surprise`` is a memory_store parameter, not a memory_remember one.

        A branch that forwarded it was unreachable: dispatch rejects the key
        before the translation runs, and no surface (MCP, REST, OpenAI, the
        browser UI) ever published it for the simple tool.
        """
        assert "surprise" not in (accepted_argument_names("memory_remember") or frozenset())
        args = build_remember_store_arguments({"content": "c", "kind": "note", "surprise": 0.9})
        assert "surprise" not in args

    def test_surprise_is_rejected_before_the_translation_runs(self):
        with pytest.raises(ToolContractError) as excinfo:
            execute_tool_call("memory_remember", {"content": "c", "surprise": 0.9})
        assert "surprise" in str(excinfo.value)
        assert "Extra inputs are not permitted" in str(excinfo.value)

    def test_non_string_kind_is_a_type_error(self):
        """A type problem is TypeError; an unknown kind string is ValueError."""
        with pytest.raises(TypeError, match="kind must be a string"):
            build_remember_store_arguments({"content": "c", "kind": 7})
        with pytest.raises(ValueError, match="kind must be one of"):
            build_remember_store_arguments({"content": "c", "kind": "nope"})


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