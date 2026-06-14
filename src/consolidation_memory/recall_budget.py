"""Time-budget helpers for bounded recall work.

When a recall deadline is active, embedding backends and cache warmers can
yield early so MCP/REST callers return episodes (and keyword-ranked knowledge)
instead of blocking until the client-side tool timeout fires.
"""

from __future__ import annotations

import contextlib
import contextvars
import time
from typing import Iterator

_recall_deadline: contextvars.ContextVar[float | None] = contextvars.ContextVar(
    "_recall_deadline",
    default=None,
)


class RecallBudgetExceeded(TimeoutError):
    """Raised when recall-scoped work exceeds the active deadline."""


def is_active() -> bool:
    return _recall_deadline.get() is not None


def deadline_monotonic() -> float | None:
    return _recall_deadline.get()


def remaining_seconds() -> float | None:
    deadline = _recall_deadline.get()
    if deadline is None:
        return None
    return max(0.0, deadline - time.monotonic())


def deadline_exceeded() -> bool:
    remaining = remaining_seconds()
    return remaining is not None and remaining <= 0.0


def check_before_expensive_work() -> None:
    if deadline_exceeded():
        raise RecallBudgetExceeded("recall deadline exceeded before expensive work")


@contextlib.contextmanager
def recall_deadline(deadline: float | None) -> Iterator[None]:
    if deadline is None:
        yield
        return
    token = _recall_deadline.set(float(deadline))
    try:
        yield
    finally:
        _recall_deadline.reset(token)