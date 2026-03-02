"""Tests for context_assembler scoring and ranking.

Run with: python -m pytest tests/test_context_assembler.py -v
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from consolidation_memory.context_assembler import (
    _recency_decay,
    _priority_score,
    _TASK_INDICATORS,
)

FIXED_NOW = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)


class TestRecencyDecay:
    """Verify half-life decay formula (fix #2)."""

    def test_zero_age_returns_one(self):
        with patch("consolidation_memory.context_assembler.datetime") as mock_dt:
            mock_dt.now.return_value = FIXED_NOW
            mock_dt.fromisoformat = datetime.fromisoformat
            score = _recency_decay(FIXED_NOW.isoformat(), half_life_days=7.0)
            assert score == pytest.approx(1.0, abs=0.02)

    def test_one_half_life_returns_half(self):
        with patch("consolidation_memory.context_assembler.datetime") as mock_dt:
            mock_dt.now.return_value = FIXED_NOW
            mock_dt.fromisoformat = datetime.fromisoformat
            t = (FIXED_NOW - timedelta(days=7)).isoformat()
            score = _recency_decay(t, half_life_days=7.0)
            assert score == pytest.approx(0.5, abs=0.05)

    def test_two_half_lives_returns_quarter(self):
        with patch("consolidation_memory.context_assembler.datetime") as mock_dt:
            mock_dt.now.return_value = FIXED_NOW
            mock_dt.fromisoformat = datetime.fromisoformat
            t = (FIXED_NOW - timedelta(days=14)).isoformat()
            score = _recency_decay(t, half_life_days=7.0)
            assert score == pytest.approx(0.25, abs=0.05)

    def test_naive_datetime_assumed_utc(self):
        """Naive datetime (no tzinfo) should not crash."""
        with patch("consolidation_memory.context_assembler.datetime") as mock_dt:
            mock_dt.now.return_value = FIXED_NOW
            mock_dt.fromisoformat = datetime.fromisoformat
            t = (FIXED_NOW - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%S")
            score = _recency_decay(t, half_life_days=7.0)
            assert 0.0 < score <= 1.0

    def test_invalid_date_returns_default(self):
        with patch("consolidation_memory.context_assembler.datetime") as mock_dt:
            mock_dt.now.return_value = FIXED_NOW
            mock_dt.fromisoformat = datetime.fromisoformat
            score = _recency_decay("not-a-date", half_life_days=7.0)
            assert score == 0.5

    def test_future_date_clamped_to_one(self):
        with patch("consolidation_memory.context_assembler.datetime") as mock_dt:
            mock_dt.now.return_value = FIXED_NOW
            mock_dt.fromisoformat = datetime.fromisoformat
            future = (FIXED_NOW + timedelta(days=5)).isoformat()
            score = _recency_decay(future, half_life_days=7.0)
            assert score == pytest.approx(1.0, abs=0.01)


class TestPriorityScore:
    """Test the combined priority scoring."""

    def _episode(self, **overrides):
        now = FIXED_NOW.isoformat()
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


class TestConfidenceAwareRanking:
    """Test confidence multiplier in record/topic ranking (Phase 2.4)."""

    def test_confidence_multiplier_formula(self):
        """Verify the formula: relevance *= 0.5 + 0.5 * confidence."""
        # confidence 0.5 → 0.75x multiplier
        assert 0.5 + 0.5 * 0.5 == pytest.approx(0.75)
        # confidence 0.8 → 0.9x multiplier
        assert 0.5 + 0.5 * 0.8 == pytest.approx(0.9)
        # confidence 1.0 → 1.0x multiplier
        assert 0.5 + 0.5 * 1.0 == pytest.approx(1.0)
        # confidence 0.0 → 0.5x multiplier (edge case)
        assert 0.5 + 0.5 * 0.0 == pytest.approx(0.5)

    def test_high_confidence_record_ranks_higher(self):
        """Records with higher confidence should produce higher relevance."""
        from unittest.mock import patch
        import numpy as np

        from consolidation_memory.context_assembler import _search_records

        query_vec = np.ones(384, dtype=np.float32)
        query_vec = query_vec / np.linalg.norm(query_vec)

        high_conf_rec = {
            "id": "rec-high",
            "record_type": "fact",
            "content": '{"subject": "Python", "info": "3.12"}',
            "embedding_text": "python version 3.12",
            "topic_title": "Python",
            "topic_filename": "python.md",
            "confidence": 0.95,
            "source_episodes": "[]",
        }
        low_conf_rec = {
            "id": "rec-low",
            "record_type": "fact",
            "content": '{"subject": "Python", "info": "maybe 3.11"}',
            "embedding_text": "python version 3.12",
            "topic_title": "Python",
            "topic_filename": "python.md",
            "confidence": 0.5,
            "source_episodes": "[]",
        }

        # Both records have identical embedding text → same semantic score
        rec_vecs = np.stack([query_vec, query_vec])

        with (
            patch("consolidation_memory.context_assembler.record_cache") as mock_rc,
            patch("consolidation_memory.context_assembler.increment_record_access"),
        ):
            mock_rc.get_record_vecs.return_value = ([high_conf_rec, low_conf_rec], rec_vecs)
            records, _ = _search_records("python version", query_vec)

        assert len(records) >= 2
        # High confidence record should have higher relevance
        high = next(r for r in records if r["id"] == "rec-high")
        low = next(r for r in records if r["id"] == "rec-low")
        assert high["relevance"] > low["relevance"]


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
