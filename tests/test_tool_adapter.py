"""Tests for shared transport adapter logic."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from consolidation_memory.tool_adapter import (
    _DEFERRED_KNOWLEDGE_WARNING,
    append_deferred_knowledge_warning,
    build_recall_search_arguments,
    build_recall_timeout_fallback_result,
    deferred_knowledge_requested,
    effective_include_knowledge,
    inject_recall_deadline,
    maybe_complete_deferred_recall,
    recall_deadline_monotonic,
    result_has_deferred_knowledge_warning,
)
from consolidation_memory.tool_dispatch import execute_tool_call
from consolidation_memory.types import RecallResult


class TestRecallAdapterSemantics:
    def test_deferred_knowledge_when_cache_cold(self):
        with patch(
            "consolidation_memory.tool_adapter.recall_knowledge_cache_ready",
            return_value=False,
        ):
            assert deferred_knowledge_requested(True) is True
            assert effective_include_knowledge(True) is False
            assert effective_include_knowledge(False) is False

    def test_full_knowledge_when_cache_warm(self):
        with patch(
            "consolidation_memory.tool_adapter.recall_knowledge_cache_ready",
            return_value=True,
        ):
            assert deferred_knowledge_requested(True) is False
            assert effective_include_knowledge(True) is True

    def test_append_deferred_knowledge_warning(self):
        updated = append_deferred_knowledge_warning({"warnings": ["existing"]})
        assert updated["warnings"][0].startswith("Knowledge/records/claims deferred")
        assert updated["warnings"][1] == "existing"

    def test_inject_recall_deadline(self):
        import time

        payload: dict[str, object] = {"query": "hello"}
        inject_recall_deadline(payload, timeout_seconds=60.0)
        deadline = payload["_recall_deadline_monotonic"]
        assert isinstance(deadline, float)
        assert deadline >= time.monotonic()
        assert deadline <= recall_deadline_monotonic(60.0) + 0.05

    def test_build_recall_search_arguments_from_recall_payload(self):
        search_args = build_recall_search_arguments(
            {
                "query": "python",
                "n_results": 25,
                "content_types": ["fact"],
                "tags": ["tag-a"],
                "after": "2026-01-01",
                "before": "2026-02-01",
                "scope": {"project": {"slug": "repo-a"}},
            }
        )
        assert search_args == {
            "query": "python",
            "content_types": ["fact"],
            "tags": ["tag-a"],
            "after": "2026-01-01",
            "before": "2026-02-01",
            "limit": 25,
            "scope": {"project": {"slug": "repo-a"}},
        }

    def test_build_recall_timeout_fallback_result(self):
        payload = build_recall_timeout_fallback_result(
            {"episodes": [{"id": "ep-1"}], "total_matches": 1},
            recall_timeout_seconds=12.5,
            include_knowledge=True,
        )
        assert payload["episodes"] == [{"id": "ep-1"}]
        assert payload["total_episodes"] == 1
        assert payload["knowledge"] == []
        assert any("episodes-only fallback" in item for item in payload["warnings"])
        assert any("Knowledge retrieval skipped" in item for item in payload["warnings"])


class TestDeferredRecallCompletion:
    def test_result_has_deferred_knowledge_warning(self):
        assert result_has_deferred_knowledge_warning(
            {"warnings": [_DEFERRED_KNOWLEDGE_WARNING]}
        )
        assert not result_has_deferred_knowledge_warning({"warnings": ["other"]})
        assert not result_has_deferred_knowledge_warning({})

    def test_maybe_complete_deferred_recall_retries_when_cache_warms(self):
        deferred = append_deferred_knowledge_warning({"episodes": [{"id": "ep-1"}]})
        completed = {
            "episodes": [{"id": "ep-1"}],
            "knowledge": [{"filename": "topic.md"}],
            "warnings": [],
        }
        calls = {"count": 0}

        def recall_again() -> dict[str, object]:
            calls["count"] += 1
            return completed

        with (
            patch(
                "consolidation_memory.tool_adapter.recall_knowledge_cache_ready",
                side_effect=[False, True, True],
            ),
            patch("consolidation_memory.tool_adapter.warm_recall_caches"),
            patch("consolidation_memory.tool_adapter.time.sleep"),
        ):
            result = maybe_complete_deferred_recall(
                deferred,
                include_knowledge=True,
                recall_executor=recall_again,
                retry_seconds=1.0,
            )

        assert calls["count"] == 1
        assert result == completed

    def test_maybe_complete_deferred_recall_skips_when_retry_disabled(self):
        deferred = append_deferred_knowledge_warning({"episodes": []})
        calls = {"count": 0}

        def recall_again() -> dict[str, object]:
            calls["count"] += 1
            return {"episodes": [], "knowledge": [{"filename": "topic.md"}]}

        with patch("consolidation_memory.tool_adapter.warm_recall_caches") as warm:
            result = maybe_complete_deferred_recall(
                deferred,
                include_knowledge=True,
                recall_executor=recall_again,
                retry_seconds=0.0,
            )

        assert result == deferred
        assert calls["count"] == 0
        warm.assert_not_called()

class TestDispatchUsesAdapterSemantics:
    def test_execute_tool_call_defers_knowledge_when_cache_cold(self):
        client = MagicMock()
        client.query_recall.return_value = RecallResult()

        with (
            patch(
                "consolidation_memory.tool_adapter.recall_knowledge_cache_ready",
                return_value=False,
            ),
            patch(
                "consolidation_memory.tool_adapter.deferred_knowledge_retry_seconds",
                return_value=0.0,
            ),
        ):
            result = execute_tool_call(
                "memory_recall",
                {"query": "hello", "include_knowledge": True},
                client=client,
            )

        client.query_recall.assert_called_once()
        assert client.query_recall.call_args.kwargs["include_knowledge"] is False
        assert result["warnings"][0].startswith("Knowledge/records/claims deferred")

    def test_execute_tool_call_completes_deferred_recall_when_cache_warms(self):
        client = MagicMock()
        deferred = RecallResult(episodes=[{"id": "ep-1"}])
        completed = RecallResult(
            episodes=[{"id": "ep-1"}],
            knowledge=[{"filename": "topic.md"}],
        )
        client.query_recall.side_effect = [deferred, completed]
        cache_checks = {"count": 0}

        def cache_ready() -> bool:
            cache_checks["count"] += 1
            return cache_checks["count"] >= 4

        with (
            patch(
                "consolidation_memory.tool_adapter.recall_knowledge_cache_ready",
                side_effect=cache_ready,
            ),
            patch("consolidation_memory.tool_adapter.warm_recall_caches"),
            patch("consolidation_memory.tool_adapter.time.sleep"),
        ):
            result = execute_tool_call(
                "memory_recall",
                {"query": "hello", "include_knowledge": True},
                client=client,
            )

        assert client.query_recall.call_count == 2
        assert result["knowledge"] == [{"filename": "topic.md"}]
        assert not any(
            str(item).startswith("Knowledge/records/claims deferred")
            for item in (result.get("warnings") or [])
        )