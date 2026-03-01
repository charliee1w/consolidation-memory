"""Tests for context_assembler scoring and ranking.

Run with: python -m pytest tests/test_context_assembler.py -v
"""

from datetime import datetime, timedelta, timezone

import pytest

from consolidation_memory.context_assembler import (
    _recency_decay,
    _priority_score,
    _TASK_INDICATORS,
)


class TestRecencyDecay:
    """Verify half-life decay formula (fix #2)."""

    def test_zero_age_returns_one(self):
        now = datetime.now(timezone.utc).isoformat()
        score = _recency_decay(now, half_life_days=7.0)
        assert score == pytest.approx(1.0, abs=0.02)

    def test_one_half_life_returns_half(self):
        t = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        score = _recency_decay(t, half_life_days=7.0)
        assert score == pytest.approx(0.5, abs=0.05)

    def test_two_half_lives_returns_quarter(self):
        t = (datetime.now(timezone.utc) - timedelta(days=14)).isoformat()
        score = _recency_decay(t, half_life_days=7.0)
        assert score == pytest.approx(0.25, abs=0.05)

    def test_naive_datetime_assumed_utc(self):
        """Naive datetime (no tzinfo) should not crash."""
        t = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%S")
        score = _recency_decay(t, half_life_days=7.0)
        assert 0.0 < score <= 1.0

    def test_invalid_date_returns_default(self):
        score = _recency_decay("not-a-date", half_life_days=7.0)
        assert score == 0.5

    def test_future_date_clamped_to_one(self):
        future = (datetime.now(timezone.utc) + timedelta(days=5)).isoformat()
        score = _recency_decay(future, half_life_days=7.0)
        assert score == pytest.approx(1.0, abs=0.01)


class TestPriorityScore:
    """Test the combined priority scoring."""

    def _episode(self, **overrides):
        now = datetime.now(timezone.utc).isoformat()
        ep = {
            "content": "test",
            "surprise_score": 0.5,
            "created_at": now,
            "access_count": 0,
        }
        ep.update(overrides)
        return ep

    def test_higher_similarity_higher_score(self):
        ep = self._episode()
        s1 = _priority_score(0.5, ep)
        s2 = _priority_score(0.9, ep)
        assert s2 > s1

    def test_higher_surprise_higher_score(self):
        s1 = _priority_score(0.8, self._episode(surprise_score=0.2))
        s2 = _priority_score(0.8, self._episode(surprise_score=0.9))
        assert s2 > s1

    def test_more_access_higher_score(self):
        s1 = _priority_score(0.8, self._episode(access_count=0))
        s2 = _priority_score(0.8, self._episode(access_count=10))
        assert s2 > s1


class TestTaskIndicators:
    """Test task-oriented query detection."""

    def test_indicators_is_frozenset(self):
        assert isinstance(_TASK_INDICATORS, frozenset)

    def test_known_task_words(self):
        for word in ["how", "deploy", "build", "setup", "run"]:
            assert word in _TASK_INDICATORS

    def test_non_task_words(self):
        for word in ["hello", "remember", "python"]:
            assert word not in _TASK_INDICATORS
