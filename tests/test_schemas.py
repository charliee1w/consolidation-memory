"""Tests for OpenAI function calling schemas and dispatch.

Run with: python -m pytest tests/test_schemas.py -v
"""

from unittest.mock import MagicMock

from consolidation_memory.schemas import openai_tools, dispatch_tool_call
from consolidation_memory.types import (
    BrowseResult,
    StoreResult,
    BatchStoreResult,
    CorrectResult,
    ExportResult,
    RecallResult,
    SearchResult,
    ClaimBrowseResult,
    ClaimSearchResult,
    ForgetResult,
    ProtectResult,
    StatusResult,
    TimelineResult,
    TopicDetailResult,
)


class TestSchemaStructure:
    def test_all_tools_present(self):
        names = {t["function"]["name"] for t in openai_tools}
        assert names == {
            "memory_store",
            "memory_store_batch",
            "memory_recall",
            "memory_search",
            "memory_claim_browse",
            "memory_claim_search",
            "memory_detect_drift",
            "memory_status",
            "memory_forget",
            "memory_export",
            "memory_correct",
            "memory_compact",
            "memory_consolidate",
            "memory_protect",
            "memory_timeline",
            "memory_contradictions",
            "memory_browse",
            "memory_read_topic",
            "memory_decay_report",
            "memory_consolidation_log",
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

    def test_scope_schema_present_for_all_scoped_tools(self):
        store = next(t for t in openai_tools if t["function"]["name"] == "memory_store")
        store_batch = next(t for t in openai_tools if t["function"]["name"] == "memory_store_batch")
        recall = next(t for t in openai_tools if t["function"]["name"] == "memory_recall")
        search = next(t for t in openai_tools if t["function"]["name"] == "memory_search")
        claim_browse = next(t for t in openai_tools if t["function"]["name"] == "memory_claim_browse")
        claim_search = next(t for t in openai_tools if t["function"]["name"] == "memory_claim_search")
        forget = next(t for t in openai_tools if t["function"]["name"] == "memory_forget")
        export = next(t for t in openai_tools if t["function"]["name"] == "memory_export")
        correct = next(t for t in openai_tools if t["function"]["name"] == "memory_correct")
        protect = next(t for t in openai_tools if t["function"]["name"] == "memory_protect")
        timeline = next(t for t in openai_tools if t["function"]["name"] == "memory_timeline")
        browse = next(t for t in openai_tools if t["function"]["name"] == "memory_browse")
        read_topic = next(t for t in openai_tools if t["function"]["name"] == "memory_read_topic")
        assert "scope" in store["function"]["parameters"]["properties"]
        assert "scope" in store_batch["function"]["parameters"]["properties"]
        assert "scope" in recall["function"]["parameters"]["properties"]
        assert "scope" in search["function"]["parameters"]["properties"]
        assert "scope" in claim_browse["function"]["parameters"]["properties"]
        assert "scope" in claim_search["function"]["parameters"]["properties"]
        assert "scope" in forget["function"]["parameters"]["properties"]
        assert "scope" in export["function"]["parameters"]["properties"]
        assert "scope" in correct["function"]["parameters"]["properties"]
        assert "scope" in protect["function"]["parameters"]["properties"]
        assert "scope" in timeline["function"]["parameters"]["properties"]
        assert "scope" in browse["function"]["parameters"]["properties"]
        assert "scope" in read_topic["function"]["parameters"]["properties"]
        scope_schema = store["function"]["parameters"]["properties"]["scope"]
        assert "policy" in scope_schema["properties"]
        assert scope_schema["properties"]["policy"]["properties"]["read_visibility"]["enum"] == [
            "private",
            "namespace",
            "project",
        ]
        assert scope_schema["properties"]["policy"]["properties"]["write_mode"]["enum"] == [
            "allow",
            "deny",
        ]


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

    def test_dispatch_store_with_scope(self):
        client = MagicMock()
        client.store_with_scope.return_value = StoreResult(status="stored", id="scoped-store")

        result = dispatch_tool_call(
            client,
            "memory_store",
            {
                "content": "test",
                "content_type": "fact",
                "scope": {
                    "namespace": {"slug": "team-a"},
                    "policy": {"write_mode": "deny"},
                },
            },
        )

        assert result["id"] == "scoped-store"
        client.store_with_scope.assert_called_once_with(
            content="test",
            content_type="fact",
            tags=None,
            surprise=0.5,
            scope={
                "namespace": {"slug": "team-a"},
                "policy": {"write_mode": "deny"},
            },
        )
        client.store.assert_not_called()

    def test_dispatch_recall(self):
        client = MagicMock()
        client.query_recall.return_value = RecallResult(
            episodes=[{"id": "1", "content": "test"}],
            knowledge=[],
            total_episodes=1,
            total_knowledge_topics=0,
        )

        result = dispatch_tool_call(client, "memory_recall", {"query": "test"})
        assert result["total_episodes"] == 1
        client.query_recall.assert_called_once_with(
            query="test", n_results=10, include_knowledge=True,
            content_types=None, tags=None, after=None, before=None,
            include_expired=False, as_of=None, scope=None,
        )

    def test_dispatch_recall_with_scope(self):
        client = MagicMock()
        client.query_recall.return_value = RecallResult(episodes=[], knowledge=[])

        result = dispatch_tool_call(
            client,
            "memory_recall",
            {"query": "test", "scope": {"project": {"slug": "repo-a"}}},
        )
        assert "episodes" in result
        client.query_recall.assert_called_once_with(
            query="test",
            n_results=10,
            include_knowledge=True,
            content_types=None,
            tags=None,
            after=None,
            before=None,
            include_expired=False,
            as_of=None,
            scope={"project": {"slug": "repo-a"}},
        )

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
        client.forget.assert_called_once_with(episode_id="abc", scope=None)

    def test_dispatch_forget_with_scope(self):
        client = MagicMock()
        client.forget.return_value = ForgetResult(status="write_denied", id="abc")

        result = dispatch_tool_call(
            client,
            "memory_forget",
            {"episode_id": "abc", "scope": {"project": {"slug": "repo-a"}}},
        )
        assert result["status"] == "write_denied"
        client.forget.assert_called_once_with(
            episode_id="abc",
            scope={"project": {"slug": "repo-a"}},
        )

    def test_dispatch_export_with_scope(self):
        client = MagicMock()
        client.export.return_value = ExportResult(status="exported", path="/tmp/export.json")

        result = dispatch_tool_call(
            client,
            "memory_export",
            {"scope": {"namespace": {"slug": "team-a"}}},
        )
        assert result["status"] == "exported"
        client.export.assert_called_once_with(scope={"namespace": {"slug": "team-a"}})

    def test_dispatch_correct_with_scope(self):
        client = MagicMock()
        client.correct.return_value = CorrectResult(status="write_denied", filename="topic.md")

        result = dispatch_tool_call(
            client,
            "memory_correct",
            {
                "topic_filename": "topic.md",
                "correction": "fix",
                "scope": {"project": {"slug": "repo-a"}},
            },
        )
        assert result["status"] == "write_denied"
        client.correct.assert_called_once_with(
            topic_filename="topic.md",
            correction="fix",
            scope={"project": {"slug": "repo-a"}},
        )

    def test_dispatch_search(self):
        client = MagicMock()
        client.query_search.return_value = SearchResult(
            episodes=[{"id": "1", "content": "test"}],
            total_matches=1,
            query="test",
        )

        result = dispatch_tool_call(client, "memory_search", {"query": "test"})
        assert result["total_matches"] == 1
        client.query_search.assert_called_once_with(
            query="test",
            content_types=None,
            tags=None,
            after=None,
            before=None,
            limit=20,
            scope=None,
        )

    def test_dispatch_store_batch_with_scope(self):
        client = MagicMock()
        client.store_batch_with_scope.return_value = BatchStoreResult(
            status="stored",
            stored=1,
            duplicates=0,
        )

        result = dispatch_tool_call(
            client,
            "memory_store_batch",
            {
                "episodes": [{"content": "x"}],
                "scope": {"namespace": {"slug": "team-a"}},
            },
        )

        assert result["status"] == "stored"
        client.store_batch_with_scope.assert_called_once_with(
            episodes=[{"content": "x"}],
            scope={"namespace": {"slug": "team-a"}},
        )
        client.store_batch.assert_not_called()

    def test_dispatch_search_with_scope(self):
        client = MagicMock()
        client.query_search.return_value = SearchResult(
            episodes=[],
            total_matches=0,
            query="test",
        )

        result = dispatch_tool_call(
            client,
            "memory_search",
            {"query": "test", "scope": {"project": {"slug": "repo-a"}}},
        )
        assert result["total_matches"] == 0
        client.query_search.assert_called_once_with(
            query="test",
            content_types=None,
            tags=None,
            after=None,
            before=None,
            limit=20,
            scope={"project": {"slug": "repo-a"}},
        )

    def test_dispatch_claim_browse(self):
        client = MagicMock()
        client.query_browse_claims.return_value = ClaimBrowseResult(
            claims=[{"id": "claim-1", "canonical_text": "python version is 3.12"}],
            total=1,
            claim_type="fact",
        )

        result = dispatch_tool_call(
            client,
            "memory_claim_browse",
            {"claim_type": "fact", "as_of": "2026-01-01T00:00:00+00:00", "limit": 25},
        )
        assert result["total"] == 1
        client.query_browse_claims.assert_called_once_with(
            claim_type="fact",
            as_of="2026-01-01T00:00:00+00:00",
            limit=25,
            scope=None,
        )

    def test_dispatch_claim_search(self):
        client = MagicMock()
        client.query_search_claims.return_value = ClaimSearchResult(
            claims=[{"id": "claim-2", "canonical_text": "uses uvicorn"}],
            total_matches=1,
            query="uvicorn",
        )

        result = dispatch_tool_call(
            client,
            "memory_claim_search",
            {"query": "uvicorn", "claim_type": "procedure", "as_of": "2026-01-01T00:00:00+00:00"},
        )
        assert result["total_matches"] == 1
        client.query_search_claims.assert_called_once_with(
            query="uvicorn",
            claim_type="procedure",
            as_of="2026-01-01T00:00:00+00:00",
            limit=50,
            scope=None,
        )

    def test_dispatch_claim_browse_with_scope(self):
        client = MagicMock()
        client.query_browse_claims.return_value = ClaimBrowseResult(claims=[], total=0)

        dispatch_tool_call(
            client,
            "memory_claim_browse",
            {"claim_type": "fact", "scope": {"project": {"slug": "repo-a"}}},
        )
        client.query_browse_claims.assert_called_once_with(
            claim_type="fact",
            as_of=None,
            limit=50,
            scope={"project": {"slug": "repo-a"}},
        )

    def test_dispatch_claim_search_with_scope(self):
        client = MagicMock()
        client.query_search_claims.return_value = ClaimSearchResult(claims=[], total_matches=0, query="python")

        dispatch_tool_call(
            client,
            "memory_claim_search",
            {"query": "python", "scope": {"namespace": {"slug": "team-a"}}},
        )
        client.query_search_claims.assert_called_once_with(
            query="python",
            claim_type=None,
            as_of=None,
            limit=50,
            scope={"namespace": {"slug": "team-a"}},
        )

    def test_dispatch_protect_with_scope(self):
        client = MagicMock()
        client.protect.return_value = ProtectResult(status="protected", protected_count=1)

        result = dispatch_tool_call(
            client,
            "memory_protect",
            {"episode_id": "abc", "scope": {"namespace": {"slug": "team-a"}}},
        )
        assert result["status"] == "protected"
        client.protect.assert_called_once_with(
            episode_id="abc",
            tag=None,
            scope={"namespace": {"slug": "team-a"}},
        )

    def test_dispatch_browse_with_scope(self):
        client = MagicMock()
        client.browse.return_value = BrowseResult(topics=[], total=0)

        result = dispatch_tool_call(
            client,
            "memory_browse",
            {"scope": {"project": {"slug": "repo-a"}}},
        )
        assert result["total"] == 0
        client.browse.assert_called_once_with(scope={"project": {"slug": "repo-a"}})

    def test_dispatch_read_topic_with_scope(self):
        client = MagicMock()
        client.read_topic.return_value = TopicDetailResult(status="ok", filename="topic.md", content="# hi")

        result = dispatch_tool_call(
            client,
            "memory_read_topic",
            {"filename": "topic.md", "scope": {"namespace": {"slug": "team-a"}}},
        )
        assert result["status"] == "ok"
        client.read_topic.assert_called_once_with(
            filename="topic.md",
            scope={"namespace": {"slug": "team-a"}},
        )

    def test_dispatch_timeline_with_scope(self):
        client = MagicMock()
        client.timeline.return_value = TimelineResult(query="python", entries=[], total=0)

        result = dispatch_tool_call(
            client,
            "memory_timeline",
            {"topic": "python", "scope": {"project": {"slug": "repo-a"}}},
        )
        assert result["query"] == "python"
        client.timeline.assert_called_once_with(
            topic="python",
            scope={"project": {"slug": "repo-a"}},
        )

    def test_dispatch_detect_drift(self):
        client = MagicMock()
        client.query_detect_drift.return_value = {
            "checked_anchors": [{"anchor_type": "path", "anchor_value": "src/app.py"}],
            "impacted_claim_ids": ["claim-1"],
            "challenged_claim_ids": ["claim-1"],
            "impacts": [],
        }

        result = dispatch_tool_call(
            client,
            "memory_detect_drift",
            {"base_ref": "origin/main", "repo_path": "C:/repo"},
        )
        assert result["challenged_claim_ids"] == ["claim-1"]
        client.query_detect_drift.assert_called_once_with(
            base_ref="origin/main",
            repo_path="C:/repo",
        )

    def test_dispatch_unknown_returns_error(self):
        client = MagicMock()
        result = dispatch_tool_call(client, "nonexistent_tool", {})
        assert "error" in result
        assert "Unknown tool" in result["error"]

    def test_dispatch_exception_returns_error(self):
        """Any exception raised during dispatch should be caught and returned as error dict."""
        client = MagicMock()
        client.store.side_effect = RuntimeError("embedding backend down")
        result = dispatch_tool_call(
            client, "memory_store", {"content": "test"},
        )
        assert result == {"error": "embedding backend down"}

    def test_dispatch_store_content_too_long(self):
        """Content exceeding 50,000 chars should be rejected before calling client."""
        client = MagicMock()
        long_content = "x" * 50_001
        result = dispatch_tool_call(
            client, "memory_store", {"content": long_content},
        )
        assert "error" in result
        assert "50001" in result["error"]
        assert "50000" in result["error"]
        client.store.assert_not_called()

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
