"""Tests for failure-outcome-driven consolidation cluster prioritization."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch


class TestFailureLinkedEpisodeIds:
    def test_collects_direct_and_claim_provenance(self, tmp_data_dir):
        from consolidation_memory.database import (
            ensure_schema,
            get_failure_linked_episode_ids_since,
            insert_episode,
            record_action_outcome,
            upsert_claim,
        )
        from consolidation_memory.database import insert_claim_sources

        ensure_schema()
        episode_direct = insert_episode(
            content="deploy failed directly",
            episode_id="episode-failure-direct",
        )
        episode_claim = insert_episode(
            content="claim provenance episode",
            episode_id="episode-failure-claim",
        )
        upsert_claim(
            claim_id="claim-failure-linked",
            claim_type="solution",
            canonical_text="Use retry backoff for deploy",
            payload={"problem": "deploy", "fix": "retry"},
            status="active",
            valid_from="2026-06-01T00:00:00+00:00",
        )
        insert_claim_sources(
            "claim-failure-linked",
            [{"source_episode_id": episode_claim}],
        )

        record_action_outcome(
            action_summary="deploy service",
            outcome_type="failure",
            observed_at="2026-06-02T12:00:00+00:00",
            source_episode_ids=[episode_direct],
        )
        record_action_outcome(
            action_summary="retry deploy",
            outcome_type="failure",
            observed_at="2026-06-03T12:00:00+00:00",
            source_claim_ids=["claim-failure-linked"],
        )
        record_action_outcome(
            action_summary="unrelated success",
            outcome_type="success",
            observed_at="2026-06-03T13:00:00+00:00",
            source_episode_ids=[insert_episode(content="success only", episode_id="episode-success")],
        )

        linked = get_failure_linked_episode_ids_since("2026-06-02T00:00:00+00:00")
        assert linked == frozenset({episode_direct, episode_claim})


class TestUnconsolidatedEpisodePriority:
    def test_priority_episodes_surface_first(self, tmp_data_dir):
        from consolidation_memory.database import ensure_schema, get_unconsolidated_episodes, insert_episode

        ensure_schema()
        insert_episode(
            content="older low-priority",
            episode_id="episode-old",
            created_at="2026-06-01T00:00:00+00:00",
            indexed=1,
        )
        insert_episode(
            content="newer low-priority",
            episode_id="episode-new",
            created_at="2026-06-05T00:00:00+00:00",
            indexed=1,
        )
        insert_episode(
            content="failure-linked but older",
            episode_id="episode-priority",
            created_at="2026-06-02T00:00:00+00:00",
            indexed=1,
        )

        rows = get_unconsolidated_episodes(
            limit=2,
            priority_episode_ids=["episode-priority"],
        )
        assert [row["id"] for row in rows] == ["episode-priority", "episode-new"]


class TestClusterPrioritization:
    def test_prioritize_valid_clusters_failure_linked_first(self):
        from consolidation_memory.consolidation.engine import _prioritize_valid_clusters

        ep_fail = {"id": "ep-fail"}
        ep_other = {"id": "ep-other"}
        clusters = {
            10: [(ep_other, 0), (ep_other, 1)],
            20: [(ep_fail, 2)],
            30: [(ep_other, 3), (ep_fail, 4)],
        }

        ordered = _prioritize_valid_clusters(clusters, frozenset({"ep-fail"}))
        assert list(ordered.keys()) == [20, 30, 10]

    def test_run_cluster_processing_loop_honors_priority_order(self):
        from consolidation_memory.consolidation.engine import _run_cluster_processing_loop
        from consolidation_memory.config import get_config

        processed: list[int] = []

        def _fake_process_cluster(cluster_id, cluster_items, sim_matrix, cluster_confidences):
            del cluster_items, sim_matrix, cluster_confidences
            processed.append(cluster_id)
            return {"status": "created", "api_calls": 0, "fast_path": True}

        ep_fail = {"id": "ep-fail"}
        ep_other = {"id": "ep-other"}
        clusters = {
            1: [(ep_other, 0)],
            2: [(ep_fail, 1)],
        }
        sim_matrix = [[1.0, 0.0], [0.0, 1.0]]

        with patch(
            "consolidation_memory.consolidation.engine._process_cluster",
            side_effect=_fake_process_cluster,
        ):
            _run_cluster_processing_loop(
                {2: clusters[2], 1: clusters[1]},
                sim_matrix,
                get_config(),
            )

        assert processed == [2, 1]


class TestRunConsolidationFailurePriority:
    def test_report_includes_failure_priority_metrics(self, tmp_data_dir):
        from consolidation_memory.client import MemoryClient
        from consolidation_memory.consolidation.engine import run_consolidation
        from consolidation_memory.config import override_config
        from consolidation_memory.database import ensure_schema, record_action_outcome
        from tests.helpers import mock_encode as _mock_encode

        ensure_schema()
        with (
            override_config(
                LLM_BACKEND="disabled",
                CONSOLIDATION_MIN_CLUSTER_SIZE=1,
                RENDER_MARKDOWN=False,
                CONTRADICTION_LLM_ENABLED=False,
            ),
            patch(
                "consolidation_memory.backends.encode_documents",
                side_effect=_mock_encode,
            ),
            patch(
                "consolidation_memory.backends.encode_query",
                side_effect=lambda q: _mock_encode([q])[0],
            ),
            patch("consolidation_memory.backends.get_dimension", return_value=384),
            patch(
                "consolidation_memory.consolidation.engine._llm_extract_with_validation",
                side_effect=AssertionError("LLM extraction must not run"),
            ),
        ):
            client = MemoryClient(auto_consolidate=False)
            try:
                failure_store = client.store(
                    "User prefers retry backoff of 5s after deploy failures.",
                    content_type="preference",
                    tags=["deploy"],
                )
                client.store(
                    "User prefers strict lint rules in CI pipelines.",
                    content_type="preference",
                    tags=["ci"],
                )
                recent_failure_at = (
                    datetime.now(timezone.utc) - timedelta(hours=1)
                ).isoformat()
                record_action_outcome(
                    action_summary="deploy with retry policy",
                    outcome_type="failure",
                    observed_at=recent_failure_at,
                    source_episode_ids=[failure_store.id],
                )
                report = run_consolidation(vector_store=client._vector_store)
            finally:
                client.close()

        assert report.get("failure_linked_episodes_loaded", 0) >= 1
        assert report.get("clusters_prioritized_by_failures", 0) >= 1
        assert report.get("topics_created", 0) >= 1