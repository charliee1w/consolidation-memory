"""Cross-surface contract tests (OpenAI dispatch, MCP wrapper, REST)."""

from __future__ import annotations

import asyncio
import dataclasses
import json
from unittest.mock import MagicMock, patch

import pytest

from consolidation_memory.schemas import dispatch_tool_call
from consolidation_memory.types import (
    BrowseResult,
    ClaimSearchResult,
    OutcomeBrowseResult,
    OutcomeRecordResult,
    RecallResult,
    TimelineResult,
    TopicDetailResult,
)
from tests.surface_contract_helpers import (
    assert_recall_deadline_injected,
    invoke_surfaces_with_execute_tool_call,
)

try:
    from fastapi.testclient import TestClient
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


pytestmark = pytest.mark.skipif(not HAS_FASTAPI, reason="fastapi not installed")


class TestSurfaceRecallContract:
    def test_recall_defaults_and_output_match_across_surfaces(self):
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
        tool_args = {"query": "hello", "scope": scope}
        expected_payload = dataclasses.asdict(expected)

        dispatch_out, mcp_out, rest_out, mock_execute = invoke_surfaces_with_execute_tool_call(
            tool_name="memory_recall",
            tool_args=tool_args,
            expected_result=expected_payload,
            mcp_coro_factory=lambda: memory_recall(query="hello", scope=scope),
            rest_path="/memory/recall",
            rest_json=tool_args,
        )

        assert mock_execute.call_count >= 1
        transport_args = assert_recall_deadline_injected(mock_execute.call_args.args[1])
        assert transport_args["query"] == "hello"
        assert transport_args["scope"] == scope
        assert dispatch_out == mcp_out == rest_out == expected_payload

    def test_recall_scope_string_path_auto_coerces_across_surfaces(self):
        from consolidation_memory.rest import create_app
        from consolidation_memory.server import memory_recall

        scope = r"C:\\Users\\gore\\consolidation-memory"
        expected_scope = {"project": {"root_uri": scope}}
        expected = RecallResult(
            episodes=[],
            knowledge=[],
            records=[],
            claims=[],
            total_episodes=0,
            total_knowledge_topics=0,
        )
        tool_args = {"query": "hello", "scope": scope}
        expected_payload = dataclasses.asdict(expected)
        from consolidation_memory.tool_dispatch import execute_tool_call as real_execute_tool_call

        mock_client = MagicMock()
        mock_client.query_recall.return_value = expected

        def execute_with_client(name, arguments, client=None, **_kwargs):
            return real_execute_tool_call(name, arguments, client=client or mock_client)

        mock_execute = MagicMock(side_effect=execute_with_client)

        with (
            patch("consolidation_memory.tool_dispatch.execute_tool_call", mock_execute),
            patch("consolidation_memory.server.execute_tool_call", mock_execute),
            patch("consolidation_memory.rest.execute_tool_call", mock_execute),
            patch(
                "consolidation_memory.server._get_client_with_timeout",
                return_value=mock_client,
            ),
            patch("consolidation_memory.server._await_warmup_ready", return_value=True),
            patch(
                "consolidation_memory.rest.MemoryRuntime.get_client_with_timeout",
                return_value=mock_client,
            ),
        ):
            dispatch_out = dispatch_tool_call(mock_client, "memory_recall", tool_args)
            mcp_out = json.loads(asyncio.run(memory_recall(query="hello", scope=scope)))
            app = create_app()
            with TestClient(app) as client:
                rest_resp = client.post("/memory/recall", json=tool_args)
            rest_out = rest_resp.json()

        assert rest_resp.status_code == 200
        mock_client.query_recall.assert_called()
        assert mock_client.query_recall.call_args.kwargs["scope"] == expected_scope
        assert dispatch_out == mcp_out == rest_out == expected_payload


class TestSurfaceClaimSearchContract:
    def test_claim_search_defaults_and_output_match_across_surfaces(self):
        from consolidation_memory.server import memory_claim_search

        scope = {"namespace": {"slug": "team-a"}}
        expected = ClaimSearchResult(
            claims=[{"id": "claim-1", "canonical_text": "python version is 3.12"}],
            total_matches=1,
            query="python",
        )
        tool_args = {"query": "python", "scope": scope}
        expected_payload = dataclasses.asdict(expected)

        dispatch_out, mcp_out, rest_out, mock_execute = invoke_surfaces_with_execute_tool_call(
            tool_name="memory_claim_search",
            tool_args=tool_args,
            expected_result=expected_payload,
            mcp_coro_factory=lambda: memory_claim_search(query="python", scope=scope),
            rest_path="/memory/claims/search",
            rest_json=tool_args,
        )

        transport_args = assert_recall_deadline_injected(mock_execute.call_args.args[1])
        assert transport_args["query"] == "python"
        assert transport_args["scope"] == scope
        assert dispatch_out == mcp_out == rest_out == expected_payload


class TestSurfaceOutcomeContract:
    def test_outcome_browse_defaults_and_output_match_across_surfaces(self):
        from consolidation_memory.rest import create_app
        from consolidation_memory.server import memory_outcome_browse

        scope = {"project": {"slug": "repo-a"}}
        expected = OutcomeBrowseResult(
            outcomes=[{"id": "outcome-1", "outcome_type": "success"}],
            total=1,
            outcome_type="success",
        )
        expected_args = {
            "outcome_type": "success",
            "action_key": None,
            "source_claim_id": None,
            "source_record_id": None,
            "source_episode_id": None,
            "as_of": None,
            "limit": 50,
            "scope": scope,
        }

        dispatch_client = MagicMock()
        dispatch_client.query_browse_outcomes.return_value = expected
        dispatch_out = dispatch_tool_call(
            dispatch_client,
            "memory_outcome_browse",
            {"outcome_type": "success", "scope": scope},
        )
        dispatch_client.query_browse_outcomes.assert_called_once_with(
            **expected_args,
            recall_deadline_monotonic=None,
        )

        mcp_client = MagicMock()
        mcp_client.query_browse_outcomes.return_value = expected
        with patch("consolidation_memory.server._get_client_with_timeout", return_value=mcp_client):
            mcp_out = json.loads(asyncio.run(memory_outcome_browse(outcome_type="success", scope=scope)))
        mcp_client.query_browse_outcomes.assert_called_once_with(
            **expected_args,
            recall_deadline_monotonic=None,
        )

        with patch(
            "consolidation_memory.client.MemoryClient.query_browse_outcomes",
            return_value=expected,
        ) as rest_call:
            app = create_app()
            with TestClient(app) as client:
                rest_resp = client.post(
                    "/memory/outcomes/browse",
                    json={"outcome_type": "success", "scope": scope},
                )
            rest_out = rest_resp.json()
        assert rest_resp.status_code == 200
        normalized_rest = dict(rest_call.call_args.kwargs)
        deadline = normalized_rest.pop("recall_deadline_monotonic", None)
        assert isinstance(deadline, float)
        assert normalized_rest == expected_args

        assert dispatch_out == mcp_out == rest_out

    def test_outcome_record_defaults_and_output_match_across_surfaces(self):
        from consolidation_memory.rest import create_app
        from consolidation_memory.server import memory_outcome_record

        scope = {"project": {"slug": "repo-a"}}
        expected = OutcomeRecordResult(
            status="recorded",
            id="outcome-1",
            action_key="act_abc",
            outcome_type="success",
        )
        expected_args = {
            "action_summary": "Run targeted tests",
            "outcome_type": "success",
            "source_claim_ids": ["claim-1"],
            "source_record_ids": None,
            "source_episode_ids": None,
            "code_anchors": None,
            "issue_ids": None,
            "pr_ids": None,
            "action_key": None,
            "summary": None,
            "details": None,
            "confidence": 0.8,
            "provenance": None,
            "observed_at": None,
            "scope": scope,
        }

        dispatch_client = MagicMock()
        dispatch_client.record_outcome.return_value = expected
        dispatch_out = dispatch_tool_call(
            dispatch_client,
            "memory_outcome_record",
            {
                "action_summary": "Run targeted tests",
                "outcome_type": "success",
                "source_claim_ids": ["claim-1"],
                "scope": scope,
            },
        )
        dispatch_client.record_outcome.assert_called_once_with(
            **expected_args,
            recall_deadline_monotonic=None,
        )

        mcp_client = MagicMock()
        mcp_client.record_outcome.return_value = expected
        with patch("consolidation_memory.server._get_client_with_timeout", return_value=mcp_client):
            mcp_out = json.loads(
                asyncio.run(
                    memory_outcome_record(
                        action_summary="Run targeted tests",
                        outcome_type="success",
                        source_claim_ids=["claim-1"],
                        scope=scope,
                    )
                )
            )
        mcp_client.record_outcome.assert_called_once_with(
            **expected_args,
            recall_deadline_monotonic=None,
        )

        with patch(
            "consolidation_memory.client.MemoryClient.record_outcome",
            return_value=expected,
        ) as rest_call:
            app = create_app()
            with TestClient(app) as client:
                rest_resp = client.post(
                    "/memory/outcomes/record",
                    json={
                        "action_summary": "Run targeted tests",
                        "outcome_type": "success",
                        "source_claim_ids": ["claim-1"],
                        "scope": scope,
                    },
                )
            rest_out = rest_resp.json()
        assert rest_resp.status_code == 200
        normalized_rest = dict(rest_call.call_args.kwargs)
        deadline = normalized_rest.pop("recall_deadline_monotonic", None)
        assert isinstance(deadline, float)
        assert normalized_rest == expected_args

        assert dispatch_out == mcp_out == rest_out


class TestSurfaceKnowledgeContracts:
    def test_browse_scope_and_output_match_across_surfaces(self):
        from consolidation_memory.rest import create_app
        from consolidation_memory.server import memory_browse

        scope = {"project": {"slug": "repo-a"}, "policy": {"read_visibility": "project"}}
        expected = BrowseResult(
            topics=[{"filename": "topic.md", "title": "Topic"}],
            total=1,
        )

        dispatch_client = MagicMock()
        dispatch_client.browse.return_value = expected
        dispatch_out = dispatch_tool_call(
            dispatch_client,
            "memory_browse",
            {"scope": scope},
        )
        dispatch_client.browse.assert_called_once_with(scope=scope)

        mcp_client = MagicMock()
        mcp_client.browse.return_value = expected
        with patch("consolidation_memory.server._get_client_with_timeout", return_value=mcp_client):
            mcp_out = json.loads(asyncio.run(memory_browse(scope=scope)))
        mcp_client.browse.assert_called_once_with(scope=scope)

        with patch("consolidation_memory.client.MemoryClient.browse", return_value=expected) as rest_call:
            app = create_app()
            with TestClient(app) as client:
                rest_resp = client.post("/memory/browse", json={"scope": scope})
            rest_out = rest_resp.json()
        assert rest_resp.status_code == 200
        rest_call.assert_called_once_with(scope=scope)

        assert dispatch_out == mcp_out == rest_out

    def test_read_topic_scope_and_output_match_across_surfaces(self):
        from consolidation_memory.rest import create_app
        from consolidation_memory.server import memory_read_topic

        scope = {"namespace": {"slug": "team-a"}}
        expected = TopicDetailResult(
            status="ok",
            filename="topic.md",
            content="# topic\n",
        )

        dispatch_client = MagicMock()
        dispatch_client.read_topic.return_value = expected
        dispatch_out = dispatch_tool_call(
            dispatch_client,
            "memory_read_topic",
            {"filename": "topic.md", "scope": scope},
        )
        dispatch_client.read_topic.assert_called_once_with(filename="topic.md", scope=scope)

        mcp_client = MagicMock()
        mcp_client.read_topic.return_value = expected
        with patch("consolidation_memory.server._get_client_with_timeout", return_value=mcp_client):
            mcp_out = json.loads(asyncio.run(memory_read_topic(filename="topic.md", scope=scope)))
        mcp_client.read_topic.assert_called_once_with(filename="topic.md", scope=scope)

        with patch("consolidation_memory.client.MemoryClient.read_topic", return_value=expected) as rest_call:
            app = create_app()
            with TestClient(app) as client:
                rest_resp = client.post(
                    "/memory/topics/read",
                    json={"filename": "topic.md", "scope": scope},
                )
            rest_out = rest_resp.json()
        assert rest_resp.status_code == 200
        rest_call.assert_called_once_with(filename="topic.md", scope=scope)

        assert dispatch_out == mcp_out == rest_out

    def test_timeline_scope_and_output_match_across_surfaces(self):
        from consolidation_memory.rest import create_app
        from consolidation_memory.server import memory_timeline

        scope = {"project": {"slug": "repo-a"}}
        expected = TimelineResult(
            query="python",
            entries=[{"topic_filename": "topic.md", "embedding_text": "python"}],
            total=1,
        )

        dispatch_client = MagicMock()
        dispatch_client.timeline.return_value = expected
        dispatch_out = dispatch_tool_call(
            dispatch_client,
            "memory_timeline",
            {"topic": "python", "scope": scope},
        )
        dispatch_client.timeline.assert_called_once_with(topic="python", scope=scope)

        mcp_client = MagicMock()
        mcp_client.timeline.return_value = expected
        with patch("consolidation_memory.server._get_client_with_timeout", return_value=mcp_client):
            mcp_out = json.loads(asyncio.run(memory_timeline(topic="python", scope=scope)))
        mcp_client.timeline.assert_called_once_with(topic="python", scope=scope)

        with patch("consolidation_memory.client.MemoryClient.timeline", return_value=expected) as rest_call:
            app = create_app()
            with TestClient(app) as client:
                rest_resp = client.post(
                    "/memory/timeline",
                    json={"topic": "python", "scope": scope},
                )
            rest_out = rest_resp.json()
        assert rest_resp.status_code == 200
        rest_call.assert_called_once_with(topic="python", scope=scope)

        assert dispatch_out == mcp_out == rest_out
