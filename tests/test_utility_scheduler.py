"""Tests for utility-based consolidation scheduling."""

from __future__ import annotations

from unittest.mock import patch

import pytest


class TestUtilityScore:
    def test_compute_utility_score_is_deterministic(self):
        from consolidation_memory.consolidation.utility_scheduler import compute_utility_score

        weights = {
            "unconsolidated_backlog": 0.4,
            "recall_miss_fallback": 0.2,
            "contradiction_spike": 0.2,
            "challenged_claim_backlog": 0.2,
        }
        result = compute_utility_score(
            unconsolidated_backlog=50,
            recall_miss_count=1,
            recall_fallback_count=1,  # weighted as 2 misses
            contradiction_count=2,
            challenged_claim_backlog=5,
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
        }
        assert result["weighted_components"] == {
            "unconsolidated_backlog": 0.2,
            "recall_miss_fallback": 0.2,
            "contradiction_spike": 0.1,
            "challenged_claim_backlog": 0.1,
        }
        assert result["score"] == 0.6

    def test_compute_utility_score_clamps_to_zero_and_one(self):
        from consolidation_memory.consolidation.utility_scheduler import compute_utility_score

        weights = {
            "unconsolidated_backlog": 0.25,
            "recall_miss_fallback": 0.25,
            "contradiction_spike": 0.25,
            "challenged_claim_backlog": 0.25,
        }
        result = compute_utility_score(
            unconsolidated_backlog=-5,
            recall_miss_count=0,
            recall_fallback_count=5,
            contradiction_count=999,
            challenged_claim_backlog=-1,
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
        assert result["score"] == 0.5


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


class TestClientUtilityScheduling:
    def test_compute_consolidation_utility_uses_signals(self):
        from consolidation_memory.client import MemoryClient
        from consolidation_memory.config import override_config
        from consolidation_memory.database import ensure_schema

        ensure_schema()
        client = MemoryClient(auto_consolidate=False)
        try:
            utility_weights = {
                "unconsolidated_backlog": 0.25,
                "recall_miss_fallback": 0.25,
                "contradiction_spike": 0.25,
                "challenged_claim_backlog": 0.25,
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
            ):
                client._record_recall_signal(miss=True, timestamp_monotonic=10.0)
                client._record_recall_signal(fallback=True, timestamp_monotonic=10.0)
                utility = client._compute_consolidation_utility(now_monotonic=20.0)

            assert utility["score"] == pytest.approx(0.566667, abs=1e-6)
            assert utility["normalized_signals"] == {
                "unconsolidated_backlog": pytest.approx(0.2),
                "recall_miss_fallback": pytest.approx(1.0),
                "contradiction_spike": pytest.approx(2.0 / 3.0),
                "challenged_claim_backlog": pytest.approx(0.4),
            }
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
