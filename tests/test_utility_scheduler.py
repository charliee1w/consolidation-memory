"""Tests for utility-based consolidation scheduling."""

from __future__ import annotations

from unittest.mock import patch

import pytest


class TestTriggerExplanation:
    def test_build_consolidation_trigger_explanation_utility(self):
        from consolidation_memory.types import build_consolidation_trigger_explanation

        explanation = build_consolidation_trigger_explanation(
            trigger_reason="utility",
            utility_score=0.82,
            threshold=0.7,
            weighted_components={
                "outcome_failure_rate": 0.2,
                "unconsolidated_backlog": 0.1,
                "recall_miss_fallback": 0.0,
                "contradiction_spike": 0.0,
                "challenged_claim_backlog": 0.0,
            },
            normalized_signals={"outcome_failure_rate": 1.0, "unconsolidated_backlog": 0.5},
        )
        assert "utility score 0.820" in explanation
        assert "threshold 0.700" in explanation
        assert "action outcome failures" in explanation

    def test_build_consolidation_trigger_explanation_backlog_pressure(self):
        from consolidation_memory.types import build_consolidation_trigger_explanation

        explanation = build_consolidation_trigger_explanation(
            trigger_reason="backlog_pressure",
            raw_signals={"unconsolidated_backlog": 150},
            force_thresholds={"unconsolidated_backlog": 100},
        )
        assert "pending episode backlog (150)" in explanation
        assert "force threshold (100)" in explanation


class TestUtilityScore:
    def test_compute_utility_score_is_deterministic(self):
        from consolidation_memory.consolidation.utility_scheduler import compute_utility_score

        weights = {
            "unconsolidated_backlog": 0.35,
            "recall_miss_fallback": 0.15,
            "contradiction_spike": 0.15,
            "challenged_claim_backlog": 0.15,
            "outcome_failure_rate": 0.2,
        }
        result = compute_utility_score(
            unconsolidated_backlog=50,
            recall_miss_count=1,
            recall_fallback_count=1,  # weighted as 2 misses
            contradiction_count=2,
            challenged_claim_backlog=5,
            outcome_failure_rate=0.5,
            weights=weights,
            backlog_target=100,
            recall_signal_target=3,
            contradiction_target=4,
            challenged_claim_target=10,
        )

        assert result["normalized_signals"] == {
            "unconsolidated_backlog": 0.5,
            "recall_miss_fallback": 1.0,
            "contradiction_spike": 0.5,
            "challenged_claim_backlog": 0.5,
            "outcome_failure_rate": 0.5,
        }
        assert result["weighted_components"] == {
            "unconsolidated_backlog": 0.175,
            "recall_miss_fallback": 0.15,
            "contradiction_spike": 0.075,
            "challenged_claim_backlog": 0.075,
            "outcome_failure_rate": 0.1,
        }
        assert result["score"] == 0.575

    def test_compute_utility_score_clamps_to_zero_and_one(self):
        from consolidation_memory.consolidation.utility_scheduler import compute_utility_score

        weights = {
            "unconsolidated_backlog": 0.2,
            "recall_miss_fallback": 0.2,
            "contradiction_spike": 0.2,
            "challenged_claim_backlog": 0.2,
            "outcome_failure_rate": 0.2,
        }
        result = compute_utility_score(
            unconsolidated_backlog=-5,
            recall_miss_count=0,
            recall_fallback_count=5,
            contradiction_count=999,
            challenged_claim_backlog=-1,
            outcome_failure_rate=1.5,
            weights=weights,
            backlog_target=10,
            recall_signal_target=1,
            contradiction_target=2,
            challenged_claim_target=10,
        )

        assert result["normalized_signals"]["unconsolidated_backlog"] == 0.0
        assert result["normalized_signals"]["recall_miss_fallback"] == 1.0
        assert result["normalized_signals"]["contradiction_spike"] == 1.0
        assert result["normalized_signals"]["challenged_claim_backlog"] == 0.0
        assert result["normalized_signals"]["outcome_failure_rate"] == 1.0
        assert result["score"] == 0.6


class TestUtilitySignalQueries:
    def test_count_helpers(self, tmp_data_dir):
        from consolidation_memory.database import (
            count_active_challenged_claims,
            count_contradictions_since,
            ensure_schema,
            insert_contradiction,
            upsert_claim,
        )

        ensure_schema()
        insert_contradiction(
            topic_id=None,
            old_record_id="old-1",
            new_record_id="new-1",
            old_content="old",
            new_content="new",
        )
        upsert_claim(
            claim_id="challenged-1",
            claim_type="fact",
            canonical_text="challenged claim",
            payload={"k": "v"},
            status="challenged",
            valid_from="2025-01-01T00:00:00+00:00",
        )
        upsert_claim(
            claim_id="active-1",
            claim_type="fact",
            canonical_text="active claim",
            payload={"k": "v"},
            status="active",
            valid_from="2025-01-01T00:00:00+00:00",
        )

        assert count_contradictions_since("2000-01-01T00:00:00+00:00") == 1
        assert count_active_challenged_claims("2026-01-01T00:00:00+00:00") == 1

    def test_outcome_failure_rate_since(self, tmp_data_dir):
        from consolidation_memory.database import (
            ensure_schema,
            get_outcome_failure_rate_since,
            insert_episode,
            record_action_outcome,
        )

        ensure_schema()
        source_episode_id = insert_episode(
            content="deploy service episode",
            episode_id="episode-outcome-rate-1",
        )
        record_action_outcome(
            action_key="deploy",
            action_summary="deploy service",
            outcome_type="success",
            observed_at="2026-06-01T12:00:00+00:00",
            source_episode_ids=[source_episode_id],
        )
        record_action_outcome(
            action_key="deploy",
            action_summary="deploy service",
            outcome_type="failure",
            observed_at="2026-06-02T12:00:00+00:00",
            source_episode_ids=[source_episode_id],
        )
        record_action_outcome(
            action_key="deploy",
            action_summary="deploy service",
            outcome_type="failure",
            observed_at="2026-06-03T12:00:00+00:00",
            source_episode_ids=[source_episode_id],
        )

        stats = get_outcome_failure_rate_since("2026-06-02T00:00:00+00:00")
        assert stats == {
            "failure_count": 2,
            "total_count": 2,
            "failure_rate": 1.0,
        }

        empty_stats = get_outcome_failure_rate_since("2999-01-01T00:00:00+00:00")
        assert empty_stats == {
            "failure_count": 0,
            "total_count": 0,
            "failure_rate": 0.0,
        }


class TestClientUtilityScheduling:
    def test_compute_consolidation_utility_uses_signals(self):
        from consolidation_memory.client import MemoryClient
        from consolidation_memory.config import override_config
        from consolidation_memory.database import ensure_schema

        ensure_schema()
        client = MemoryClient(auto_consolidate=False)
        try:
            utility_weights = {
                "unconsolidated_backlog": 0.2,
                "recall_miss_fallback": 0.2,
                "contradiction_spike": 0.2,
                "challenged_claim_backlog": 0.2,
                "outcome_failure_rate": 0.2,
            }
            with (
                override_config(
                    CONSOLIDATION_UTILITY_WEIGHTS=utility_weights,
                    CONSOLIDATION_MAX_EPISODES_PER_RUN=100,
                ),
                patch(
                    "consolidation_memory.database.get_stats",
                    return_value={
                        "episodic_buffer": {"pending_consolidation": 20},
                        "knowledge_base": {},
                    },
                ),
                patch("consolidation_memory.database.count_contradictions_since", return_value=2),
                patch("consolidation_memory.database.count_active_challenged_claims", return_value=10),
                patch(
                    "consolidation_memory.database.get_outcome_failure_rate_since",
                    return_value={
                        "failure_count": 1,
                        "total_count": 4,
                        "failure_rate": 0.25,
                    },
                ),
            ):
                client._record_recall_signal(miss=True, timestamp_monotonic=10.0)
                client._record_recall_signal(fallback=True, timestamp_monotonic=10.0)
                utility = client._compute_consolidation_utility(now_monotonic=20.0)

            assert utility["score"] == pytest.approx(0.503333, abs=1e-6)
            assert utility["normalized_signals"] == {
                "unconsolidated_backlog": pytest.approx(0.2),
                "recall_miss_fallback": pytest.approx(1.0),
                "contradiction_spike": pytest.approx(2.0 / 3.0),
                "challenged_claim_backlog": pytest.approx(0.4),
                "outcome_failure_rate": pytest.approx(0.25),
            }
            assert utility["raw_signals"]["outcome_failure_count"] == 1
            assert utility["raw_signals"]["outcome_total_count"] == 4
            assert utility["raw_signals"]["outcome_failure_rate"] == pytest.approx(0.25)
        finally:
            client.close()

    def test_should_trigger_consolidation_by_utility_or_interval(self):
        from consolidation_memory.client import MemoryClient
        from consolidation_memory.config import override_config
        from consolidation_memory.database import ensure_schema

        ensure_schema()
        client = MemoryClient(auto_consolidate=False)
        try:
            with override_config(CONSOLIDATION_UTILITY_THRESHOLD=0.7):
                assert client._should_trigger_consolidation(
                    now_monotonic=10.0,
                    last_run_monotonic=0.0,
                    interval_seconds=30.0,
                    utility_score=0.2,
                ) == (False, "none")
                assert client._should_trigger_consolidation(
                    now_monotonic=10.0,
                    last_run_monotonic=0.0,
                    interval_seconds=30.0,
                    utility_score=0.8,
                ) == (True, "utility")
                assert client._should_trigger_consolidation(
                    now_monotonic=40.0,
                    last_run_monotonic=0.0,
                    interval_seconds=30.0,
                    utility_score=0.1,
                ) == (True, "interval")
        finally:
            client.close()

    def test_should_trigger_scheduler_run_backlog_pressure(self):
        from consolidation_memory.client import MemoryClient
        from consolidation_memory.config import override_config
        from consolidation_memory.database import ensure_schema

        ensure_schema()
        client = MemoryClient(auto_consolidate=False)
        try:
            with override_config(
                CONSOLIDATION_MAX_EPISODES_PER_RUN=100,
                CONSOLIDATION_UTILITY_THRESHOLD=0.95,
            ):
                should_run, reason = client._should_trigger_scheduler_run(
                    scheduler_state={"next_due_at": "2999-01-01T00:00:00+00:00"},
                    utility_score=0.1,
                    raw_signals={
                        "unconsolidated_backlog": 100,
                        "challenged_claim_backlog": 0,
                    },
                )

            assert should_run is True
            assert reason == "backlog_pressure"
        finally:
            client.close()

    def test_should_trigger_scheduler_run_challenged_backlog_pressure(self):
        from consolidation_memory.client import MemoryClient
        from consolidation_memory.config import override_config
        from consolidation_memory.database import ensure_schema

        ensure_schema()
        client = MemoryClient(auto_consolidate=False)
        try:
            with override_config(
                CONSOLIDATION_MAX_EPISODES_PER_RUN=60,
                CONSOLIDATION_UTILITY_THRESHOLD=0.95,
            ):
                should_run, reason = client._should_trigger_scheduler_run(
                    scheduler_state={"next_due_at": "2999-01-01T00:00:00+00:00"},
                    utility_score=0.1,
                    raw_signals={
                        "unconsolidated_backlog": 0,
                        "challenged_claim_backlog": 25,
                    },
                )

            assert should_run is True
            assert reason == "challenged_backlog_pressure"
        finally:
            client.close()

    def test_should_trigger_scheduler_run_challenged_pressure_scales_for_large_runs(self):
        from consolidation_memory.client import MemoryClient
        from consolidation_memory.config import override_config
        from consolidation_memory.database import ensure_schema

        ensure_schema()
        client = MemoryClient(auto_consolidate=False)
        try:
            with override_config(
                CONSOLIDATION_MAX_EPISODES_PER_RUN=200,
                CONSOLIDATION_UTILITY_THRESHOLD=0.95,
            ):
                should_run, reason = client._should_trigger_scheduler_run(
                    scheduler_state={"next_due_at": "2999-01-01T00:00:00+00:00"},
                    utility_score=0.1,
                    raw_signals={
                        "unconsolidated_backlog": 0,
                        "challenged_claim_backlog": 53,
                    },
                )

            assert should_run is True
            assert reason == "challenged_backlog_pressure"
        finally:
            client.close()
