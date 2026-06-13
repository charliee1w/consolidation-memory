"""Tests for context_assembler scoring and ranking.

Run with: python -m pytest tests/test_context_assembler.py -v
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from consolidation_memory.context_assembler import (
    _distinctive_overlap_multiplier,
    _is_solution_shaped_query,
    _recency_decay,
    _priority_score,
    _recall_episode_score,
    _tag_overlap_multiplier,
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

    def test_high_precision_record_ranks_higher(self, tmp_data_dir):
        """Records linked to higher-precision claims should rank above peers."""
        import numpy as np

        from consolidation_memory.claim_graph import claim_from_record
        from consolidation_memory.context_assembler import _search_records
        from consolidation_memory.database import ensure_schema, insert_knowledge_records, upsert_claim, upsert_knowledge_topic

        ensure_schema()
        high_content = {"type": "fact", "subject": "Python runtime", "info": "3.12"}
        low_content = {"type": "fact", "subject": "Python runtime", "info": "3.11"}
        high_claim = claim_from_record(high_content)
        low_claim = claim_from_record(low_content)
        upsert_claim(
            claim_id=high_claim["id"],
            claim_type=high_claim["claim_type"],
            canonical_text=high_claim["canonical_text"],
            payload=high_claim["payload"],
            precision=1.0,
        )
        upsert_claim(
            claim_id=low_claim["id"],
            claim_type=low_claim["claim_type"],
            canonical_text=low_claim["canonical_text"],
            payload=low_claim["payload"],
            precision=0.8,
        )
        topic_id = upsert_knowledge_topic(
            filename="python-runtime.md",
            title="Python runtime",
            summary="Runtime facts",
            source_episodes=[],
            fact_count=2,
        )
        insert_knowledge_records(
            topic_id,
            [
                {
                    "record_type": "fact",
                    "content": high_content,
                    "embedding_text": "python runtime version",
                    "confidence": 0.8,
                },
                {
                    "record_type": "fact",
                    "content": low_content,
                    "embedding_text": "python runtime version",
                    "confidence": 0.8,
                },
            ],
            source_episodes=[],
        )

        query_vec = np.ones(384, dtype=np.float32)
        query_vec = query_vec / np.linalg.norm(query_vec)
        high_conf_rec = {
            "id": "rec-high-precision",
            "record_type": "fact",
            "content": high_content,
            "embedding_text": "python runtime version",
            "topic_title": "Python runtime",
            "topic_filename": "python-runtime.md",
            "confidence": 0.8,
            "source_episodes": "[]",
        }
        low_conf_rec = {
            "id": "rec-low-precision",
            "record_type": "fact",
            "content": low_content,
            "embedding_text": "python runtime version",
            "topic_title": "Python runtime",
            "topic_filename": "python-runtime.md",
            "confidence": 0.8,
            "source_episodes": "[]",
        }
        rec_vecs = np.stack([query_vec, query_vec])

        with (
            patch("consolidation_memory.context_assembler.record_cache") as mock_rc,
            patch("consolidation_memory.context_assembler.increment_record_access"),
        ):
            mock_rc.get_record_vecs.return_value = ([high_conf_rec, low_conf_rec], rec_vecs)
            records, _ = _search_records("python runtime version", query_vec)

        assert len(records) == 2
        high = next(r for r in records if r["id"] == "rec-high-precision")
        low = next(r for r in records if r["id"] == "rec-low-precision")
        assert high["relevance"] > low["relevance"]


class TestSolutionShapedQuery:
    def test_detects_debug_tokens(self):
        assert _is_solution_shaped_query("MCP recall timeout during health check")
        assert not _is_solution_shaped_query("project architecture overview")

    def test_detects_path_like_tokens(self):
        assert _is_solution_shaped_query("context_assembler.py recall scoring")
        assert _is_solution_shaped_query("src/consolidation_memory/client.py store path")

    def test_solution_shaped_query_prefers_semantic_match(self):
        now = FIXED_NOW.isoformat()
        high_access = {
            "content_type": "solution",
            "surprise_score": 0.9,
            "created_at": now,
            "access_count": 200,
            "content": "MCP recall slowness",
        }
        low_access = {
            "content_type": "solution",
            "surprise_score": 0.5,
            "created_at": now,
            "access_count": 0,
            "content": "MCP recall slowness fixed with cache",
        }
        query = "MCP recall slowness timeout fix"
        popular_score = _recall_episode_score(
            0.80, high_access, query=query, content_type_filter=None,
        )
        exact_score = _recall_episode_score(
            0.96, low_access, query=query, content_type_filter=None,
        )
        assert exact_score > popular_score

    def test_thin_solution_boosts_on_path_overlap_without_debug_query(self):
        now = FIXED_NOW.isoformat()
        thin_solution = {
            "content_type": "solution",
            "surprise_score": 0.4,
            "created_at": now,
            "access_count": 0,
            "content": "Fix in src/consolidation_memory/context_assembler.py for recall.",
            "tags": "[]",
        }
        generic = {
            "content_type": "solution",
            "surprise_score": 0.9,
            "created_at": now,
            "access_count": 300,
            "content": "General consolidation-memory architecture notes.",
            "tags": "[]",
        }
        query = "context_assembler.py recall ranking"
        thin_score = _recall_episode_score(
            0.70, thin_solution, query=query, content_type_filter=None,
        )
        generic_score = _recall_episode_score(
            0.72, generic, query=query, content_type_filter=None,
        )
        assert thin_score > generic_score


class TestDistinctiveOverlapMultiplier:
    def test_boosts_path_overlap(self):
        mult = _distinctive_overlap_multiplier(
            "context_assembler.py recall",
            "Updated src/consolidation_memory/context_assembler.py scoring.",
        )
        assert mult > 1.0

    def test_no_overlap_returns_one(self):
        assert _distinctive_overlap_multiplier("alpha beta", "gamma delta") == 1.0


class TestTagOverlapMultiplier:
    def test_boosts_matching_tags(self):
        episode = {"tags": '["benchmark", "recall"]'}
        mult = _tag_overlap_multiplier("benchmark recall metrics", episode)
        assert mult > 1.0

    def test_no_tag_overlap_returns_one(self):
        episode = {"tags": '["deployment"]'}
        assert _tag_overlap_multiplier("benchmark recall metrics", episode) == 1.0


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


class TestKnowledgePathTraversal:
    def test_search_knowledge_skips_outside_files(self, tmp_data_dir):
        import numpy as np

        from consolidation_memory.config import get_config
        from consolidation_memory.context_assembler import _search_knowledge

        cfg = get_config()
        outside = cfg.KNOWLEDGE_DIR.parent / "outside_secret.txt"
        outside.write_text("do-not-leak", encoding="utf-8")

        query_vec = np.ones(384, dtype=np.float32)
        query_vec = query_vec / np.linalg.norm(query_vec)
        topic = {
            "id": "topic-1",
            "title": "Secret Topic",
            "filename": "../outside_secret.txt",
            "summary": "secret test data",
            "confidence": 1.0,
            "source_episodes": "[]",
        }

        with (
            patch("consolidation_memory.context_assembler.topic_cache") as mock_tc,
            patch("consolidation_memory.context_assembler.increment_topic_access"),
            patch("consolidation_memory.context_assembler._apply_evolving_topic_signals"),
        ):
            mock_tc.get_topic_vecs.return_value = ([topic], np.stack([query_vec]))
            topics, _ = _search_knowledge("secret test", query_vec)

        assert len(topics) == 1
        assert topics[0]["content"] == ""
