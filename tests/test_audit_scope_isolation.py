"""Scope isolation tests for audit/ops read paths."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

_SCOPE_A = {
    "namespace": {"slug": "team-a"},
    "project": {"slug": "repo-a"},
    "app_client": {"name": "legacy_client", "app_type": "python_sdk"},
}

_SCOPE_B = {
    "namespace": {"slug": "team-b"},
    "project": {"slug": "repo-b"},
    "app_client": {"name": "legacy_client", "app_type": "python_sdk"},
}


def _scope_filter(scope: dict) -> dict[str, str]:
    return {
        "namespace_slug": scope["namespace"]["slug"],
        "project_slug": scope["project"]["slug"],
        "app_client_name": scope["app_client"]["name"],
        "app_client_type": scope["app_client"]["app_type"],
    }


class TestAuditScopeIsolation:
    def test_contradictions_respect_scope(self, tmp_data_dir):
        from consolidation_memory.client import MemoryClient
        from consolidation_memory.database import (
            ensure_schema,
            insert_contradiction,
            upsert_knowledge_topic,
        )

        ensure_schema()
        topic_a = upsert_knowledge_topic(
            filename="scope-a.md",
            title="Scope A",
            summary="A",
            source_episodes=[],
            scope=_scope_filter(_SCOPE_A),
        )
        topic_b = upsert_knowledge_topic(
            filename="scope-b.md",
            title="Scope B",
            summary="B",
            source_episodes=[],
            scope=_scope_filter(_SCOPE_B),
        )
        insert_contradiction(topic_a, None, None, "old-a", "new-a")
        insert_contradiction(topic_b, None, None, "old-b", "new-b")

        with MemoryClient(auto_consolidate=False) as client:
            scoped = client.contradictions(scope=_SCOPE_A)
            assert scoped.total == 1
            assert scoped.contradictions[0]["topic_id"] == topic_a

    def test_decay_report_respects_scope(self, tmp_data_dir):
        from consolidation_memory.client import MemoryClient
        from consolidation_memory.database import ensure_schema, insert_episode, protect_episode

        ensure_schema()
        old = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
        prunable_a = insert_episode(
            content="prunable in scope a",
            consolidated=1,
            consolidated_at=old,
            indexed=1,
            scope=_scope_filter(_SCOPE_A),
        )
        insert_episode(
            content="prunable in scope b",
            consolidated=1,
            consolidated_at=old,
            indexed=1,
            scope=_scope_filter(_SCOPE_B),
        )
        protected_b = insert_episode(
            content="protected in scope b",
            scope=_scope_filter(_SCOPE_B),
        )
        protect_episode(protected_b, scope=_scope_filter(_SCOPE_B))

        with MemoryClient(auto_consolidate=False) as client:
            report = client.decay_report(scope=_SCOPE_A)
            assert report.prunable_episodes == 1
            assert report.protected_episodes == 0
            assert report.details["prunable_episodes"][0]["id"] == prunable_a

    def test_status_stats_respect_scope(self, tmp_data_dir):
        from consolidation_memory.client import MemoryClient
        from consolidation_memory.database import ensure_schema, insert_episode

        ensure_schema()
        insert_episode(content="episode a", scope=_scope_filter(_SCOPE_A))
        insert_episode(content="episode b", scope=_scope_filter(_SCOPE_B))

        with MemoryClient(auto_consolidate=False) as client:
            scoped = client.status(lightweight=True, scope=_SCOPE_A)
            assert scoped.episodic_buffer is not None
            assert scoped.episodic_buffer["total"] == 1

    def test_invalid_content_type_rejected_by_python_sdk(self, tmp_data_dir):
        from consolidation_memory.client import MemoryClient
        from consolidation_memory.database import ensure_schema

        ensure_schema()
        with MemoryClient(auto_consolidate=False) as client:
            with pytest.raises(ValueError, match="content_type must be one of"):
                client.store("bad type", content_type="not-a-real-type")