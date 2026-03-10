"""Tests for MemoryClient — the pure Python API.

Run with: python -m pytest tests/test_client.py -v
"""

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from tests.helpers import make_normalized_vec as _make_normalized_vec
from tests.helpers import mock_encode as _mock_encode


class TestClientLifecycle:
    def test_construct_and_close(self):
        from consolidation_memory.database import ensure_schema
        ensure_schema()

        from consolidation_memory.client import MemoryClient
        client = MemoryClient(auto_consolidate=False)
        assert client._vector_store is not None
        client.close()

    def test_context_manager(self):
        from consolidation_memory.database import ensure_schema
        ensure_schema()

        from consolidation_memory.client import MemoryClient
        with MemoryClient(auto_consolidate=False) as client:
            assert client._vector_store is not None
        # after exit, thread should be stopped
        assert client._consolidation_thread is None

    def test_multiple_instances_safe(self):
        from consolidation_memory.database import ensure_schema
        ensure_schema()

        from consolidation_memory.client import MemoryClient
        c1 = MemoryClient(auto_consolidate=False)
        c2 = MemoryClient(auto_consolidate=False)
        c1.close()
        c2.close()


class TestClientScopeModel:
    def test_resolve_scope_defaults_to_legacy_single_project(self):
        from consolidation_memory.database import ensure_schema
        from consolidation_memory.client import MemoryClient

        ensure_schema()
        client = MemoryClient(auto_consolidate=False)
        try:
            with patch("consolidation_memory.config.get_active_project", return_value="legacy-project"):
                resolved = client.resolve_scope()

            assert resolved.namespace.slug == "default"
            assert resolved.app_client.name == "legacy_client"
            assert resolved.project.slug == "legacy-project"
            assert resolved.project.display_name == "legacy-project"
        finally:
            client.close()

    def test_resolve_scope_uses_explicit_canonical_values(self):
        from consolidation_memory.database import ensure_schema
        from consolidation_memory.client import MemoryClient
        from consolidation_memory.types import (
            AgentScope,
            AppClientScope,
            NamespaceScope,
            PolicyScope,
            ProjectRepoScope,
            ScopeEnvelope,
            SessionScope,
        )

        ensure_schema()
        client = MemoryClient(auto_consolidate=False)
        try:
            scope = ScopeEnvelope(
                namespace=NamespaceScope(slug="shared", display_name="Shared Team", sharing_mode="team"),
                app_client=AppClientScope(app_type="rest", name="gateway", provider="internal"),
                agent=AgentScope(name="triage-agent", model_provider="openai", model_name="gpt-5"),
                session=SessionScope(external_key="thread-42", session_kind="thread"),
                project=ProjectRepoScope(slug="repo-a", repo_remote="https://example.com/repo-a.git"),
                policy=PolicyScope(read_visibility="project", write_mode="allow"),
            )

            resolved = client.resolve_scope(scope)
            assert resolved.namespace.slug == "shared"
            assert resolved.namespace.display_name == "Shared Team"
            assert resolved.app_client.app_type == "rest"
            assert resolved.app_client.name == "gateway"
            assert resolved.agent is not None
            assert resolved.agent.name == "triage-agent"
            assert resolved.session is not None
            assert resolved.session.external_key == "thread-42"
            assert resolved.project.slug == "repo-a"
            assert resolved.policy.read_visibility == "project"
            assert resolved.policy.write_mode == "allow"
        finally:
            client.close()

    def test_resolve_scope_defaults_policy_to_private_allow(self):
        from consolidation_memory.database import ensure_schema
        from consolidation_memory.client import MemoryClient

        ensure_schema()
        client = MemoryClient(auto_consolidate=False)
        try:
            resolved = client.resolve_scope()
            assert resolved.policy.read_visibility == "private"
            assert resolved.policy.write_mode == "allow"
        finally:
            client.close()

    def test_store_with_scope_keeps_existing_store_behavior(self):
        from consolidation_memory.database import ensure_schema
        from consolidation_memory.client import MemoryClient
        from consolidation_memory.types import StoreResult

        ensure_schema()
        client = MemoryClient(auto_consolidate=False)
        try:
            with patch.object(
                client,
                "_store_internal",
                return_value=StoreResult(status="stored", id="scope-store-id"),
            ) as mock_store:
                result = client.store_with_scope(
                    content="scope-aware write",
                    content_type="fact",
                    tags=["scope"],
                    surprise=0.7,
                    scope={"namespace": {"slug": "team-a"}, "project": {"slug": "repo-a"}},
                )

            assert result.status == "stored"
            assert result.id == "scope-store-id"
            assert mock_store.call_count == 1
            kwargs = mock_store.call_args.kwargs
            assert kwargs["content"] == "scope-aware write"
            assert kwargs["content_type"] == "fact"
            assert kwargs["tags"] == ["scope"]
            assert kwargs["surprise"] == 0.7
            assert kwargs["resolved_scope"].namespace.slug == "team-a"
            assert kwargs["resolved_scope"].project.slug == "repo-a"
        finally:
            client.close()

    def test_store_with_scope_write_deny_policy_returns_write_denied(self):
        from consolidation_memory.database import ensure_schema
        from consolidation_memory.client import MemoryClient

        ensure_schema()
        client = MemoryClient(auto_consolidate=False)
        try:
            with patch("consolidation_memory.backends.encode_documents") as mock_embed:
                result = client.store_with_scope(
                    content="blocked write",
                    scope={
                        "namespace": {"slug": "team-a"},
                        "project": {"slug": "repo-a"},
                        "policy": {"write_mode": "deny"},
                    },
                )

            assert result.status == "write_denied"
            assert "write_mode='deny'" in (result.message or "")
            mock_embed.assert_not_called()
        finally:
            client.close()

    def test_store_batch_with_scope_write_deny_policy_returns_write_denied(self):
        from consolidation_memory.database import ensure_schema
        from consolidation_memory.client import MemoryClient

        ensure_schema()
        client = MemoryClient(auto_consolidate=False)
        try:
            with patch("consolidation_memory.backends.encode_documents") as mock_embed:
                result = client.store_batch_with_scope(
                    episodes=[{"content": "blocked write"}],
                    scope={
                        "namespace": {"slug": "team-a"},
                        "project": {"slug": "repo-a"},
                        "policy": {"write_mode": "deny"},
                    },
                )

            assert result.status == "write_denied"
            assert result.stored == 0
            assert result.results and result.results[0]["status"] == "write_denied"
            mock_embed.assert_not_called()
        finally:
            client.close()

    def test_read_visibility_namespace_widens_scope_filter(self):
        from consolidation_memory.database import ensure_schema
        from consolidation_memory.client import MemoryClient, _resolved_scope_to_query_filter

        ensure_schema()
        client = MemoryClient(auto_consolidate=False)
        try:
            resolved = client.resolve_scope(
                {
                    "namespace": {"slug": "team-a", "sharing_mode": "private"},
                    "app_client": {"name": "app-a", "app_type": "rest"},
                    "project": {"slug": "repo-a"},
                    "policy": {"read_visibility": "namespace"},
                }
            )
            scope_filter = _resolved_scope_to_query_filter(resolved)

            assert scope_filter["namespace_slug"] == "team-a"
            assert "project_slug" not in scope_filter
            assert "app_client_name" not in scope_filter
            assert "app_client_type" not in scope_filter
        finally:
            client.close()

    def test_recall_with_scope_keeps_existing_recall_behavior(self):
        from consolidation_memory.database import ensure_schema
        from consolidation_memory.client import MemoryClient
        from consolidation_memory.types import RecallResult

        ensure_schema()
        client = MemoryClient(auto_consolidate=False)
        try:
            with patch.object(
                client,
                "_recall_internal",
                return_value=RecallResult(episodes=[], knowledge=[]),
            ) as mock_recall:
                result = client.recall_with_scope(
                    query="scope read",
                    n_results=3,
                    include_knowledge=False,
                    scope={"namespace": {"slug": "team-a"}, "session": {"external_key": "thread-1"}},
                )

            assert result.episodes == []
            assert mock_recall.call_count == 1
            kwargs = mock_recall.call_args.kwargs
            assert kwargs["query"] == "scope read"
            assert kwargs["n_results"] == 3
            assert kwargs["include_knowledge"] is False
            assert kwargs["resolved_scope"].namespace.slug == "team-a"
            assert kwargs["resolved_scope"].session is not None
            assert kwargs["resolved_scope"].session.external_key == "thread-1"
        finally:
            client.close()

    @patch("consolidation_memory.backends.encode_query")
    @patch("consolidation_memory.backends.encode_documents")
    def test_scope_isolation_and_explicit_shared_namespace(self, mock_embed_docs, mock_embed_query):
        from consolidation_memory.database import ensure_schema
        from consolidation_memory.client import MemoryClient

        ensure_schema()
        mock_embed_docs.side_effect = _mock_encode
        mock_embed_query.side_effect = lambda text: _mock_encode([text])

        client = MemoryClient(auto_consolidate=False)
        try:
            legacy = client.store("legacy-only memory")
            scoped = client.store_with_scope(
                content="team shared memory",
                content_type="fact",
                scope={
                    "namespace": {"slug": "team-a", "sharing_mode": "shared"},
                    "app_client": {"name": "app-a", "app_type": "rest"},
                    "project": {"slug": "repo-a"},
                },
            )
            assert legacy.status == "stored"
            assert scoped.status == "stored"

            legacy_recall = client.recall("memory", include_knowledge=False)
            assert any(ep["content"] == "legacy-only memory" for ep in legacy_recall.episodes)
            assert all(ep["content"] != "team shared memory" for ep in legacy_recall.episodes)

            shared_recall = client.recall_with_scope(
                query="memory",
                include_knowledge=False,
                scope={
                    "namespace": {"slug": "team-a", "sharing_mode": "shared"},
                    "app_client": {"name": "app-b", "app_type": "rest"},
                    "project": {"slug": "repo-a"},
                },
            )
            assert any(ep["content"] == "team shared memory" for ep in shared_recall.episodes)
        finally:
            client.close()

    @patch("consolidation_memory.backends.encode_documents")
    def test_scope_aware_dedup_isolated_for_private_apps(self, mock_embed_docs):
        from consolidation_memory.database import ensure_schema
        from consolidation_memory.client import MemoryClient

        ensure_schema()
        mock_embed_docs.side_effect = _mock_encode
        client = MemoryClient(auto_consolidate=False)
        try:
            first = client.store_with_scope(
                content="same text",
                scope={
                    "namespace": {"slug": "team-b", "sharing_mode": "private"},
                    "app_client": {"name": "app-a", "app_type": "rest"},
                    "project": {"slug": "repo-b"},
                },
            )
            second = client.store_with_scope(
                content="same text",
                scope={
                    "namespace": {"slug": "team-b", "sharing_mode": "private"},
                    "app_client": {"name": "app-b", "app_type": "rest"},
                    "project": {"slug": "repo-b"},
                },
            )
            assert first.status == "stored"
            assert second.status == "stored"
        finally:
            client.close()

    @patch("consolidation_memory.backends.encode_documents")
    def test_scope_aware_dedup_shared_namespace_cross_app(self, mock_embed_docs):
        from consolidation_memory.database import ensure_schema
        from consolidation_memory.client import MemoryClient

        ensure_schema()
        mock_embed_docs.side_effect = _mock_encode
        client = MemoryClient(auto_consolidate=False)
        try:
            first = client.store_with_scope(
                content="duplicate in shared namespace",
                scope={
                    "namespace": {"slug": "team-c", "sharing_mode": "shared"},
                    "app_client": {"name": "app-a", "app_type": "rest"},
                    "project": {"slug": "repo-c"},
                },
            )
            second = client.store_with_scope(
                content="duplicate in shared namespace",
                scope={
                    "namespace": {"slug": "team-c", "sharing_mode": "shared"},
                    "app_client": {"name": "app-b", "app_type": "rest"},
                    "project": {"slug": "repo-c"},
                },
            )
            assert first.status == "stored"
            assert second.status == "duplicate_detected"
        finally:
            client.close()


class TestClientStore:
    @patch("consolidation_memory.backends.encode_documents")
    def test_basic_store(self, mock_embed):
        from consolidation_memory.database import ensure_schema
        from consolidation_memory.client import MemoryClient

        ensure_schema()
        client = MemoryClient(auto_consolidate=False)

        vec = _make_normalized_vec(seed=42)
        mock_embed.return_value = vec.reshape(1, -1)

        result = client.store("test content", content_type="fact", tags=["python"])
        assert result.status == "stored"
        assert result.id is not None
        assert result.content_type == "fact"
        assert result.tags == ["python"]

        client.close()


class TestClientAutoConsolidation:
    @patch("consolidation_memory.backends.encode_documents")
    def test_store_triggers_auto_consolidation_tick(self, mock_embed):
        from consolidation_memory.client import MemoryClient
        from consolidation_memory.database import ensure_schema

        ensure_schema()
        client = MemoryClient(auto_consolidate=False)
        try:
            vec = _make_normalized_vec(seed=42)
            mock_embed.return_value = vec.reshape(1, -1)
            with patch.object(client, "_maybe_auto_consolidate", return_value=False) as mock_tick:
                result = client.store("auto consolidate trigger", content_type="fact")

            assert result.status == "stored"
            mock_tick.assert_called_once_with(trigger_source="store")
        finally:
            client.close()

    def test_maybe_auto_consolidate_submits_when_due_and_lease_acquired(self):
        from consolidation_memory.client import MemoryClient
        from consolidation_memory.database import ensure_schema

        ensure_schema()
        client = MemoryClient(auto_consolidate=False)
        try:
            client._auto_consolidate_enabled = True
            fake_future = MagicMock()
            fake_pool = MagicMock()
            fake_pool.submit.return_value = fake_future
            client._consolidation_pool = fake_pool

            past_due = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
            utility_state = {
                "score": 0.95,
                "normalized_signals": {
                    "unconsolidated_backlog": 0.0,
                    "recall_miss_fallback": 0.0,
                    "contradiction_spike": 0.0,
                    "challenged_claim_backlog": 0.0,
                },
                "weighted_components": {
                    "unconsolidated_backlog": 0.0,
                    "recall_miss_fallback": 0.0,
                    "contradiction_spike": 0.0,
                    "challenged_claim_backlog": 0.0,
                },
                "raw_signals": {
                    "unconsolidated_backlog": 0,
                    "recall_miss_count": 0,
                    "recall_fallback_count": 0,
                    "contradiction_count": 0,
                    "challenged_claim_backlog": 0,
                    "lookback_seconds": 3600.0,
                },
            }

            with (
                patch.object(client, "_compute_consolidation_utility", return_value=utility_state),
                patch(
                    "consolidation_memory.database.get_consolidation_scheduler_state",
                    return_value={"next_due_at": past_due},
                ),
                patch(
                    "consolidation_memory.database.try_acquire_consolidation_lease",
                    return_value=True,
                ),
                patch("consolidation_memory.database.mark_consolidation_scheduler_started") as mock_started,
            ):
                started = client._maybe_auto_consolidate(trigger_source="store")

            assert started is True
            assert fake_pool.submit.call_count == 1
            assert fake_future.add_done_callback.call_count == 1
            assert mock_started.call_count == 1
        finally:
            client.close()

    @patch("consolidation_memory.backends.encode_documents")
    def test_surprise_clamping(self, mock_embed):
        from consolidation_memory.database import ensure_schema, get_episode
        from consolidation_memory.client import MemoryClient

        ensure_schema()
        client = MemoryClient(auto_consolidate=False)

        vec = _make_normalized_vec(seed=42)
        mock_embed.return_value = vec.reshape(1, -1)

        # Surprise > 1.0 should clamp to 1.0
        r1 = client.store("high surprise", surprise=5.0)
        ep = get_episode(r1.id)
        assert ep["surprise_score"] == 1.0

        # Negative should clamp to 0.0
        vec2 = _make_normalized_vec(seed=99)
        mock_embed.return_value = vec2.reshape(1, -1)
        r2 = client.store("low surprise", surprise=-1.0)
        ep2 = get_episode(r2.id)
        assert ep2["surprise_score"] == 0.0

        client.close()


class TestClientRecall:
    @patch("consolidation_memory.backends.encode_query")
    @patch("consolidation_memory.backends.encode_documents")
    def test_basic_recall(self, mock_embed_docs, mock_embed_query):
        from consolidation_memory.database import ensure_schema
        from consolidation_memory.client import MemoryClient

        ensure_schema()
        client = MemoryClient(auto_consolidate=False)

        vec = _make_normalized_vec(seed=42)
        mock_embed_docs.return_value = vec.reshape(1, -1)
        mock_embed_query.return_value = vec.reshape(1, -1)

        client.store("recall test content", content_type="fact")

        result = client.recall("recall test", n_results=5, include_knowledge=False)
        assert result.total_episodes == 1
        assert len(result.episodes) >= 1
        assert result.episodes[0]["content"] == "recall test content"

        client.close()

    @patch("consolidation_memory.backends.encode_query")
    @patch("consolidation_memory.backends.encode_documents")
    def test_empty_recall(self, mock_embed_docs, mock_embed_query):
        from consolidation_memory.database import ensure_schema
        from consolidation_memory.client import MemoryClient

        ensure_schema()
        client = MemoryClient(auto_consolidate=False)

        vec = _make_normalized_vec(seed=42)
        mock_embed_query.return_value = vec.reshape(1, -1)

        result = client.recall("nothing here")
        assert result.total_episodes == 0
        assert len(result.episodes) == 0

        client.close()

    @patch("consolidation_memory.backends.encode_documents")
    def test_store_batch_all_invalid_returns_empty_result(self, mock_embed):
        from consolidation_memory.database import ensure_schema
        from consolidation_memory.client import MemoryClient

        ensure_schema()
        client = MemoryClient(auto_consolidate=False)
        try:
            result = client.store_batch([{}, {"content_type": "fact"}])
            assert result.status == "stored"
            assert result.stored == 0
            assert result.duplicates == 0
            mock_embed.assert_not_called()
        finally:
            client.close()

    def test_recall_surfaces_claims_field(self):
        from consolidation_memory.database import ensure_schema
        from consolidation_memory.client import MemoryClient

        ensure_schema()
        client = MemoryClient(auto_consolidate=False)
        try:
            fake_recall = {
                "episodes": [],
                "knowledge": [],
                "records": [],
                "claims": [{"id": "claim-1", "canonical_text": "python runtime is 3.12"}],
                "warnings": [],
            }
            fake_stats = {
                "episodic_buffer": {"total": 0},
                "knowledge_base": {"total_topics": 0},
            }

            with (
                patch("consolidation_memory.context_assembler.recall", return_value=fake_recall),
                patch("consolidation_memory.database.get_stats", return_value=fake_stats),
            ):
                result = client.recall("python runtime")

            assert result.claims == fake_recall["claims"]
            assert result.episodes == []
            assert result.knowledge == []
            assert result.records == []
        finally:
            client.close()


class TestClientClaims:
    def test_browse_claims_supports_as_of(self):
        from consolidation_memory.database import ensure_schema, upsert_claim
        from consolidation_memory.client import MemoryClient

        ensure_schema()
        upsert_claim(
            claim_id="claim-client-old",
            claim_type="fact",
            canonical_text="python is 3.11",
            payload={"subject": "python", "info": "3.11"},
            valid_from="2025-01-01T00:00:00+00:00",
            valid_until="2025-06-01T00:00:00+00:00",
        )
        upsert_claim(
            claim_id="claim-client-new",
            claim_type="fact",
            canonical_text="python is 3.12",
            payload={"subject": "python", "info": "3.12"},
            valid_from="2025-07-01T00:00:00+00:00",
        )

        client = MemoryClient(auto_consolidate=False)
        try:
            result = client.browse_claims(
                claim_type="fact",
                as_of="2025-03-01T00:00:00+00:00",
            )
            ids = {claim["id"] for claim in result.claims}
            assert "claim-client-old" in ids
            assert "claim-client-new" not in ids
        finally:
            client.close()

    def test_search_claims_matches_canonical_text(self):
        from consolidation_memory.database import ensure_schema, upsert_claim
        from consolidation_memory.client import MemoryClient

        ensure_schema()
        upsert_claim(
            claim_id="claim-client-search",
            claim_type="procedure",
            canonical_text="start API with uvicorn main:app",
            payload={"trigger": "run api", "steps": "uvicorn main:app --reload"},
            valid_from="2025-01-01T00:00:00+00:00",
        )

        client = MemoryClient(auto_consolidate=False)
        try:
            result = client.search_claims(query="uvicorn", claim_type="procedure")
            assert result.total_matches >= 1
            assert any(c["id"] == "claim-client-search" for c in result.claims)
            assert "relevance" in result.claims[0]
        finally:
            client.close()


class TestClientDrift:
    def test_detect_drift_delegates_to_drift_module(self):
        from consolidation_memory.database import ensure_schema
        from consolidation_memory.client import MemoryClient

        ensure_schema()
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

        client = MemoryClient(auto_consolidate=False)
        try:
            with patch("consolidation_memory.drift.detect_code_drift", return_value=expected) as mock_detect:
                result = client.detect_drift(base_ref="origin/main", repo_path="C:/repo")
            assert result == expected
            mock_detect.assert_called_once_with(base_ref="origin/main", repo_path="C:/repo")
        finally:
            client.close()


class TestClientForget:
    @patch("consolidation_memory.backends.encode_documents")
    def test_forget_existing(self, mock_embed):
        from consolidation_memory.database import ensure_schema
        from consolidation_memory.client import MemoryClient

        ensure_schema()
        client = MemoryClient(auto_consolidate=False)

        vec = _make_normalized_vec(seed=42)
        mock_embed.return_value = vec.reshape(1, -1)

        stored = client.store("forgettable content")
        result = client.forget(stored.id)
        assert result.status == "forgotten"
        assert result.id == stored.id

        client.close()

    def test_forget_nonexistent(self):
        from consolidation_memory.database import ensure_schema
        from consolidation_memory.client import MemoryClient

        ensure_schema()
        client = MemoryClient(auto_consolidate=False)

        result = client.forget("nonexistent-uuid")
        assert result.status == "not_found"

        client.close()


class TestClientStatus:
    @patch("consolidation_memory.backends.encode_documents")
    def test_status_counts(self, mock_embed):
        from consolidation_memory.database import ensure_schema
        from consolidation_memory.client import MemoryClient

        ensure_schema()
        client = MemoryClient(auto_consolidate=False)

        vec = _make_normalized_vec(seed=42)
        mock_embed.return_value = vec.reshape(1, -1)

        client.store("status test episode")

        status = client.status()
        assert status.episodic_buffer["total"] == 1
        assert status.faiss_index_size == 1
        from consolidation_memory import __version__
        assert status.version == __version__
        assert status.embedding_backend != ""
        assert status.knowledge_consistency is not None
        assert "consistency_ratio" in status.knowledge_consistency
        assert status.scaling is not None
        assert "index_type" in status.scaling

        client.close()


class TestClientExport:
    @patch("consolidation_memory.backends.encode_documents")
    def test_export_round_trip(self, mock_embed):
        from consolidation_memory.database import (
            ensure_schema,
            insert_claim_edge,
            insert_claim_event,
            insert_claim_sources,
            insert_episode_anchors,
            upsert_claim,
        )
        from consolidation_memory.client import MemoryClient

        ensure_schema()
        client = MemoryClient(auto_consolidate=False)

        vec = _make_normalized_vec(seed=42)
        mock_embed.return_value = vec.reshape(1, -1)

        store_result = client.store("export test", content_type="fact", tags=["test"])
        assert store_result.id is not None

        upsert_claim(
            claim_id="export-claim-a",
            claim_type="fact",
            canonical_text="src/app.py sets APP_MODE=legacy",
            payload={"path": "src/app.py", "app_mode": "legacy"},
            valid_from="2026-01-01T00:00:00+00:00",
        )
        upsert_claim(
            claim_id="export-claim-b",
            claim_type="fact",
            canonical_text="src/app.py sets APP_MODE=modern",
            payload={"path": "src/app.py", "app_mode": "modern"},
            status="challenged",
            valid_from="2026-01-01T00:00:00+00:00",
        )
        insert_claim_sources("export-claim-a", [{"source_episode_id": store_result.id}])
        insert_claim_event("export-claim-a", event_type="create", details={"source": "test"})
        insert_claim_edge(
            from_claim_id="export-claim-a",
            to_claim_id="export-claim-b",
            edge_type="contradicts",
            details={"reason": "mode changed"},
        )
        insert_episode_anchors(
            store_result.id,
            [{"anchor_type": "path", "anchor_value": "src/app.py"}],
        )

        result = client.export()
        assert result.status == "exported"
        assert result.episodes == 1
        assert result.claims == 2
        assert result.claim_edges == 1
        assert result.claim_sources == 1
        assert result.claim_events == 1
        assert result.episode_anchors >= 1

        from pathlib import Path
        export_data = json.loads(Path(result.path).read_text(encoding="utf-8"))
        assert export_data["stats"]["episode_count"] == 1
        assert export_data["stats"]["claim_count"] == 2
        assert export_data["stats"]["claim_edge_count"] == 1
        assert export_data["stats"]["claim_source_count"] == 1
        assert export_data["stats"]["claim_event_count"] == 1
        assert export_data["stats"]["episode_anchor_count"] >= 1
        assert len(export_data["claims"]) == 2
        assert len(export_data["claim_edges"]) == 1
        assert len(export_data["claim_sources"]) == 1
        assert len(export_data["claim_events"]) == 1
        assert len(export_data["episode_anchors"]) >= 1

        client.close()


class TestClientCompact:
    @patch("consolidation_memory.backends.encode_documents")
    def test_compact_with_tombstones(self, mock_embed):
        from consolidation_memory.database import ensure_schema
        from consolidation_memory.client import MemoryClient

        ensure_schema()
        client = MemoryClient(auto_consolidate=False)

        vec = _make_normalized_vec(seed=42)
        mock_embed.return_value = vec.reshape(1, -1)

        stored = client.store("tombstone test")
        client.forget(stored.id)

        result = client.compact()
        assert result.status == "compacted"
        assert result.tombstones_removed == 1
        assert result.index_size == 0

        client.close()

    def test_compact_no_tombstones(self):
        from consolidation_memory.database import ensure_schema
        from consolidation_memory.client import MemoryClient

        ensure_schema()
        client = MemoryClient(auto_consolidate=False)

        result = client.compact()
        assert result.status == "no_tombstones"
        assert result.tombstones_removed == 0

        client.close()

    @patch("consolidation_memory.backends.encode_documents")
    def test_compact_preserves_live_vectors(self, mock_embed):
        from consolidation_memory.database import ensure_schema
        from consolidation_memory.client import MemoryClient

        ensure_schema()
        client = MemoryClient(auto_consolidate=False)

        vec1 = _make_normalized_vec(seed=42)
        vec2 = _make_normalized_vec(seed=99)

        mock_embed.return_value = vec1.reshape(1, -1)
        client.store("keep this")

        mock_embed.return_value = vec2.reshape(1, -1)
        s2 = client.store("forget this")

        client.forget(s2.id)

        result = client.compact()
        assert result.status == "compacted"
        assert result.tombstones_removed == 1
        assert result.index_size == 1

        client.close()


class TestClientConsolidate:
    def test_consolidate_lock(self):
        from consolidation_memory.database import ensure_schema
        from consolidation_memory.client import MemoryClient

        ensure_schema()
        client = MemoryClient(auto_consolidate=False)

        # Acquire lock externally
        client._consolidation_lock.acquire()

        result = client.consolidate()
        assert result["status"] == "already_running"

        client._consolidation_lock.release()
        client.close()


class TestClientCorrect:
    def test_correct_updates_structured_records(self, tmp_data_dir):
        from consolidation_memory.client import MemoryClient
        from consolidation_memory.config import get_config
        from consolidation_memory.database import (
            ensure_schema,
            get_records_by_topic,
            insert_knowledge_records,
            upsert_knowledge_topic,
        )

        ensure_schema()
        cfg = get_config()
        filename = "python.md"
        (cfg.KNOWLEDGE_DIR / filename).write_text(
            "---\ntitle: Python\nsummary: Python 3.12\n---\n\n## Facts\n- **Python**: 3.12\n",
            encoding="utf-8",
        )
        topic_id = upsert_knowledge_topic(
            filename=filename,
            title="Python",
            summary="Python 3.12",
            source_episodes=["ep1"],
            fact_count=1,
            confidence=0.8,
        )
        insert_knowledge_records(
            topic_id,
            records=[{
                "record_type": "fact",
                "content": {"type": "fact", "subject": "Python", "info": "3.12"},
                "embedding_text": "Python: 3.12",
                "confidence": 0.8,
            }],
            source_episodes=["ep1"],
        )

        corrected_md = (
            "---\n"
            "title: Python\n"
            "summary: Python 3.13\n"
            "tags: [python]\n"
            "confidence: 0.9\n"
            "---\n\n"
            "## Facts\n"
            "- **Python**: 3.13\n"
        )
        mock_llm = MagicMock()
        mock_llm.generate.return_value = corrected_md

        client = MemoryClient(auto_consolidate=False)
        try:
            with patch("consolidation_memory.backends.get_llm_backend", return_value=mock_llm):
                result = client.correct(filename, "Python version changed to 3.13")

            assert result.status == "corrected"

            active = get_records_by_topic(topic_id, include_expired=False)
            assert len(active) == 1
            active_content = json.loads(active[0]["content"])
            assert active_content["info"] == "3.13"
            assert active[0]["valid_from"] is not None

            all_records = get_records_by_topic(topic_id, include_expired=True)
            assert len(all_records) == 2
            old = next(r for r in all_records if json.loads(r["content"]).get("info") == "3.12")
            assert old["valid_until"] is not None
        finally:
            client.close()
