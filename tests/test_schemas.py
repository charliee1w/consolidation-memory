"""Tests for OpenAI function calling schemas and dispatch.

Run with: python -m pytest tests/test_schemas.py -v
"""

import pytest
from unittest.mock import MagicMock

from consolidation_memory.schemas import openai_tools, dispatch_tool_call
from consolidation_memory.types import StoreResult, RecallResult, ForgetResult, StatusResult


class TestSchemaStructure:
    def test_all_tools_present(self):
        names = {t["function"]["name"] for t in openai_tools}
        assert names == {
            "memory_store",
            "memory_recall",
            "memory_status",
            "memory_forget",
            "memory_export",
            "memory_correct",
        }

    def test_schemas_have_required_fields(self):
        for tool in openai_tools:
            assert tool["type"] == "function"
            func = tool["function"]
            assert "name" in func
            assert "description" in func
            assert "parameters" in func
            assert func["parameters"]["type"] == "object"

    def test_store_schema_requires_content(self):
        store = next(t for t in openai_tools if t["function"]["name"] == "memory_store")
        assert "content" in store["function"]["parameters"]["required"]

    def test_recall_schema_requires_query(self):
        recall = next(t for t in openai_tools if t["function"]["name"] == "memory_recall")
        assert "query" in recall["function"]["parameters"]["required"]

    def test_forget_schema_requires_episode_id(self):
        forget = next(t for t in openai_tools if t["function"]["name"] == "memory_forget")
        assert "episode_id" in forget["function"]["parameters"]["required"]

    def test_correct_schema_requires_fields(self):
        correct = next(t for t in openai_tools if t["function"]["name"] == "memory_correct")
        required = correct["function"]["parameters"]["required"]
        assert "topic_filename" in required
        assert "correction" in required


class TestDispatch:
    def test_dispatch_store(self):
        client = MagicMock()
        client.store.return_value = StoreResult(status="stored", id="abc", content_type="fact")

        result = dispatch_tool_call(client, "memory_store", {"content": "test", "content_type": "fact"})
        assert result["status"] == "stored"
        assert result["id"] == "abc"
        client.store.assert_called_once_with(
            content="test", content_type="fact", tags=None, surprise=0.5,
        )

    def test_dispatch_recall(self):
        client = MagicMock()
        client.recall.return_value = RecallResult(
            episodes=[{"id": "1", "content": "test"}],
            knowledge=[],
            total_episodes=1,
            total_knowledge_topics=0,
        )

        result = dispatch_tool_call(client, "memory_recall", {"query": "test"})
        assert result["total_episodes"] == 1
        client.recall.assert_called_once_with(query="test", n_results=10, include_knowledge=True)

    def test_dispatch_status(self):
        client = MagicMock()
        client.status.return_value = StatusResult(version="0.1.0", embedding_backend="fastembed")

        result = dispatch_tool_call(client, "memory_status", {})
        assert result["version"] == "0.1.0"

    def test_dispatch_forget(self):
        client = MagicMock()
        client.forget.return_value = ForgetResult(status="forgotten", id="abc")

        result = dispatch_tool_call(client, "memory_forget", {"episode_id": "abc"})
        assert result["status"] == "forgotten"
        client.forget.assert_called_once_with(episode_id="abc")

    def test_dispatch_unknown_raises(self):
        client = MagicMock()
        with pytest.raises(ValueError, match="Unknown tool"):
            dispatch_tool_call(client, "nonexistent_tool", {})

    def test_dispatch_defaults_applied(self):
        """Dispatch should apply default values for optional arguments."""
        client = MagicMock()
        client.store.return_value = StoreResult(status="stored", id="x")

        dispatch_tool_call(client, "memory_store", {"content": "test"})
        client.store.assert_called_once_with(
            content="test",
            content_type="exchange",
            tags=None,
            surprise=0.5,
        )
