"""Tests for the circuit breaker pattern.

Run with: python -m pytest tests/test_circuit_breaker.py -v
"""

import threading
import time

import pytest

from consolidation_memory.circuit_breaker import CircuitBreaker, CircuitState


class TestCircuitBreakerStates:
    """Test state transitions."""

    def test_starts_closed(self):
        cb = CircuitBreaker(threshold=3, cooldown=1.0, name="test")
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0

    def test_closed_to_open_after_n_failures(self):
        cb = CircuitBreaker(threshold=3, cooldown=60.0, name="test")
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_open_to_half_open_after_cooldown(self):
        cb = CircuitBreaker(threshold=1, cooldown=0.05, name="test")
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        time.sleep(0.1)
        assert cb.state == CircuitState.HALF_OPEN

    def test_half_open_to_closed_on_success(self):
        cb = CircuitBreaker(threshold=1, cooldown=0.05, name="test")
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        time.sleep(0.1)
        assert cb.state == CircuitState.HALF_OPEN

        cb.record_success()
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0

    def test_half_open_to_open_on_failure(self):
        cb = CircuitBreaker(threshold=1, cooldown=0.05, name="test")
        cb.record_failure()
        time.sleep(0.1)
        assert cb.state == CircuitState.HALF_OPEN

        cb.record_failure()
        assert cb.state == CircuitState.OPEN


class TestCheck:
    """Test the check() method."""

    def test_check_passes_when_closed(self):
        cb = CircuitBreaker(threshold=3, cooldown=60.0, name="test")
        cb.check()  # Should not raise

    def test_check_raises_when_open(self):
        cb = CircuitBreaker(threshold=1, cooldown=60.0, name="test")
        cb.record_failure()
        with pytest.raises(ConnectionError, match="OPEN"):
            cb.check()

    def test_check_passes_when_half_open(self):
        cb = CircuitBreaker(threshold=1, cooldown=0.05, name="test")
        cb.record_failure()
        time.sleep(0.1)
        assert cb.state == CircuitState.HALF_OPEN
        cb.check()  # Should not raise (allows probe)


class TestReset:
    """Test reset() behavior."""

    def test_reset_from_open(self):
        cb = CircuitBreaker(threshold=1, cooldown=60.0, name="test")
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        cb.reset()
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0

    def test_reset_from_half_open(self):
        cb = CircuitBreaker(threshold=1, cooldown=0.05, name="test")
        cb.record_failure()
        time.sleep(0.1)
        assert cb.state == CircuitState.HALF_OPEN

        cb.reset()
        assert cb.state == CircuitState.CLOSED

    def test_reset_clears_failure_count(self):
        cb = CircuitBreaker(threshold=5, cooldown=60.0, name="test")
        cb.record_failure()
        cb.record_failure()
        assert cb.failure_count == 2

        cb.reset()
        assert cb.failure_count == 0


class TestThreadSafety:
    """Test thread safety under concurrent failures."""

    def test_concurrent_failures_reach_open(self):
        cb = CircuitBreaker(threshold=5, cooldown=60.0, name="test")
        barrier = threading.Barrier(10)

        def hammer():
            barrier.wait()
            for _ in range(3):
                cb.record_failure()

        threads = [threading.Thread(target=hammer) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
            assert not t.is_alive(), f"Thread {t.name} still alive"

        assert cb.state == CircuitState.OPEN
        assert cb.failure_count >= 5

    def test_concurrent_success_and_failure(self):
        cb = CircuitBreaker(threshold=10, cooldown=60.0, name="test")
        barrier = threading.Barrier(6)

        def fail_n():
            barrier.wait()
            for _ in range(5):
                cb.record_failure()

        def succeed_n():
            barrier.wait()
            for _ in range(5):
                cb.record_success()

        threads = [
            *[threading.Thread(target=fail_n) for _ in range(3)],
            *[threading.Thread(target=succeed_n) for _ in range(3)],
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
            assert not t.is_alive(), f"Thread {t.name} still alive"

        # Result is non-deterministic but should not crash
        assert cb.state in {CircuitState.CLOSED, CircuitState.OPEN}
