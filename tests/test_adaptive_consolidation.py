"""Tests for adaptive consolidation scheduler behavior."""

from __future__ import annotations

from unittest.mock import patch


class _FakeFuture:
    def __init__(self, value: dict[str, object]) -> None:
        self._value = value

    def result(self, timeout: float | None = None):  # noqa: ANN001
        del timeout
        return self._value


class _FakePool:
    def __init__(self, result: dict[str, object] | None = None) -> None:
        self.submissions: list[tuple[object, tuple[object, ...], dict[str, object]]] = []
        self._result = result or {"status": "completed"}

    def submit(self, fn, *args, **kwargs):  # noqa: ANN001
        self.submissions.append((fn, args, kwargs))
        return _FakeFuture(self._result)

    def shutdown(self, wait: bool = True, cancel_futures: bool = True) -> None:
        del wait, cancel_futures


def _utility_state(score: float) -> dict[str, object]:
    return {
        "score": score,
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


class TestAdaptiveConsolidationLoop:
    def test_high_utility_triggers_consolidation(self):
        from consolidation_memory.client import MemoryClient
        from consolidation_memory.config import override_config
        from consolidation_memory.database import ensure_schema

        ensure_schema()
        client = MemoryClient(auto_consolidate=False)
        try:
            fake_pool = _FakePool()
            client._consolidation_pool = fake_pool

            with (
                override_config(
                    CONSOLIDATION_INTERVAL_HOURS=24.0,
                    CONSOLIDATION_UTILITY_THRESHOLD=0.7,
                ),
                patch.object(client, "_compute_consolidation_utility", return_value=_utility_state(0.9)),
                patch.object(client._consolidation_stop, "wait", side_effect=[False, True]),
            ):
                client._consolidation_loop()

            assert len(fake_pool.submissions) == 1
        finally:
            client.close()

    def test_low_utility_skips_consolidation(self):
        from consolidation_memory.client import MemoryClient
        from consolidation_memory.config import override_config
        from consolidation_memory.database import ensure_schema

        ensure_schema()
        client = MemoryClient(auto_consolidate=False)
        try:
            fake_pool = _FakePool()
            client._consolidation_pool = fake_pool

            with (
                override_config(
                    CONSOLIDATION_INTERVAL_HOURS=24.0,
                    CONSOLIDATION_UTILITY_THRESHOLD=0.8,
                ),
                patch.object(client, "_compute_consolidation_utility", return_value=_utility_state(0.1)),
                patch.object(client._consolidation_stop, "wait", side_effect=[False, True]),
            ):
                client._consolidation_loop()

            assert fake_pool.submissions == []
        finally:
            client.close()

    def test_interval_fallback_still_triggers(self):
        from consolidation_memory.client import MemoryClient
        from consolidation_memory.config import override_config
        from consolidation_memory.database import ensure_schema

        ensure_schema()
        client = MemoryClient(auto_consolidate=False)
        try:
            fake_pool = _FakePool()
            client._consolidation_pool = fake_pool

            with (
                override_config(
                    CONSOLIDATION_INTERVAL_HOURS=1.0,
                    CONSOLIDATION_UTILITY_THRESHOLD=0.95,
                ),
                patch.object(client, "_compute_consolidation_utility", return_value=_utility_state(0.0)),
                patch.object(client._consolidation_stop, "wait", side_effect=[False, True]),
                patch("consolidation_memory.client.time.monotonic", side_effect=[0.0, 7200.0, 7200.0]),
            ):
                client._consolidation_loop()

            assert len(fake_pool.submissions) == 1
        finally:
            client.close()


class TestStatusSchedulerState:
    def test_status_output_exposes_utility_scheduler_state(self):
        from consolidation_memory.client import MemoryClient
        from consolidation_memory.database import ensure_schema

        ensure_schema()
        client = MemoryClient(auto_consolidate=False)
        try:
            with patch.object(client, "_compute_consolidation_utility", return_value=_utility_state(0.42)):
                status = client.status()

            assert status.utility_scheduler is not None
            scheduler = status.utility_scheduler
            assert scheduler["score"] == 0.42
            assert "threshold" in scheduler
            assert "weights" in scheduler
            assert "normalized_signals" in scheduler
            assert "weighted_components" in scheduler
            assert "raw_signals" in scheduler
            assert "is_due" in scheduler
            assert "run_decision" in scheduler
            assert "force_thresholds" in scheduler
        finally:
            client.close()
