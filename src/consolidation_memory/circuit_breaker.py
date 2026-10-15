"""Circuit breaker for backend calls.

Prevents repeated timeout waits when a backend is down. Three states:
  CLOSED    — normal operation, calls pass through
  OPEN      — backend assumed down, calls fail immediately with ConnectionError
  HALF_OPEN — one probe call allowed; success closes, failure reopens

Usage::

    cb = CircuitBreaker(threshold=3, cooldown=60.0, name="embedding")
    cb.check()           # raises ConnectionError if OPEN
    try:
        result = do_work()
        cb.record_success()
    except ConnectionError:
        cb.record_failure()
        raise
"""

import logging
import threading
import time
from enum import Enum

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Thread-safe circuit breaker with configurable threshold and cooldown."""

    def __init__(self, threshold: int = 3, cooldown: float = 60.0, name: str = "backend"):
        self._threshold = threshold
        self._cooldown = cooldown
        self._name = name
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        """Current state, transitioning OPEN → HALF_OPEN if cooldown elapsed."""
        with self._lock:
            if (
                self._state == CircuitState.OPEN
                and time.time() - self._last_failure_time >= self._cooldown
            ):
                self._state = CircuitState.HALF_OPEN
                logger.info(
                    "Circuit breaker [%s]: OPEN -> HALF_OPEN (cooldown elapsed)", self._name
                )
            return self._state

    @property
    def failure_count(self) -> int:
        return self._failure_count

    def record_success(self) -> None:
        """Record a successful call. Closes the circuit if it was half-open."""
        with self._lock:
            if self._state != CircuitState.CLOSED:
                logger.info(
                    "Circuit breaker [%s]: %s -> CLOSED", self._name, self._state.value
                )
            self._failure_count = 0
            self._state = CircuitState.CLOSED

    def record_failure(self) -> None:
        """Record a failed call. Opens the circuit if threshold is reached."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            if self._failure_count >= self._threshold and self._state == CircuitState.CLOSED:
                self._state = CircuitState.OPEN
                logger.warning(
                    "Circuit breaker [%s]: CLOSED -> OPEN after %d consecutive failures "
                    "(cooldown: %.0fs)",
                    self._name,
                    self._failure_count,
                    self._cooldown,
                )
            elif self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                logger.warning(
                    "Circuit breaker [%s]: HALF_OPEN -> OPEN (probe failed)", self._name
                )

    def check(self) -> None:
        """Raise ConnectionError immediately if the circuit is OPEN.

        In HALF_OPEN state, allows the call through (the next success/failure
        will determine whether to close or reopen).

        Uses the lock to avoid TOCTOU races with state transitions.
        """
        with self._lock:
            if self._state == CircuitState.OPEN:
                if time.time() - self._last_failure_time >= self._cooldown:
                    self._state = CircuitState.HALF_OPEN
                    logger.info(
                        "Circuit breaker [%s]: OPEN -> HALF_OPEN (cooldown elapsed)", self._name
                    )
                else:
                    raise ConnectionError(
                        f"Circuit breaker [{self._name}] is OPEN — backend unavailable. "
                        f"Will probe again after {self._cooldown:.0f}s cooldown."
                    )

    def reset(self) -> None:
        """Force-reset to CLOSED state. Used in tests and backend reinitialization."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._last_failure_time = 0.0
