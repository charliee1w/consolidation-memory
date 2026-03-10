"""Tests for consolidation clustering helper logic."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np

from consolidation_memory.config import override_config
from consolidation_memory.consolidation.clustering import (
    _compute_cluster_confidence,
    _find_similar_topic,
    _matches_scope,
)


class TestMatchesScope:
    def test_matches_when_scope_fields_equal(self):
        row = {"project_slug": "repo-a", "namespace_slug": "team-a"}
        assert _matches_scope(row, {"project_slug": "repo-a"}) is True

    def test_ignores_none_scope_values(self):
        row = {"project_slug": "repo-a"}
        assert _matches_scope(row, {"project_slug": "repo-a", "agent_name": None}) is True

    def test_rejects_missing_or_mismatched_values(self):
        row = {"project_slug": "repo-a"}
        assert _matches_scope(row, {"project_slug": "repo-b"}) is False
        assert _matches_scope(row, {"namespace_slug": "team-a"}) is False


class TestClusterConfidence:
    def test_computes_weighted_confidence_and_clamps(self):
        episodes = [{"surprise_score": 0.9}, {"surprise_score": 0.3}]
        sim = np.array([[1.0, 0.8], [0.8, 1.0]], dtype=np.float32)
        with override_config(
            CONSOLIDATION_CONFIDENCE_COHERENCE_W=0.7,
            CONSOLIDATION_CONFIDENCE_SURPRISE_W=0.3,
        ):
            confidence = _compute_cluster_confidence(episodes, sim, [0, 1])

        expected = round(max(0.5, min(0.95, (0.8 * 0.7) + (0.6 * 0.3))), 2)
        assert confidence == expected

    def test_single_item_cluster_uses_default_coherence(self):
        episodes = [{"surprise_score": 0.5}]
        sim = np.array([[1.0]], dtype=np.float32)
        with override_config(
            CONSOLIDATION_CONFIDENCE_COHERENCE_W=0.5,
            CONSOLIDATION_CONFIDENCE_SURPRISE_W=0.5,
        ):
            confidence = _compute_cluster_confidence(episodes, sim, [0])
        assert confidence == 0.65  # (0.8 * 0.5) + (0.5 * 0.5)


class TestFindSimilarTopic:
    @patch("consolidation_memory.backends.encode_documents")
    @patch("consolidation_memory.consolidation.clustering.topic_cache.get_topic_vecs")
    def test_returns_semantic_match_above_threshold(self, mock_topic_vecs, mock_encode_documents):
        mock_topic_vecs.return_value = (
            [{"id": "t1", "title": "Python Tips", "summary": "Tips", "project_slug": "repo-a"}],
            np.array([[1.0, 0.0]], dtype=np.float32),
        )
        mock_encode_documents.return_value = np.array([[0.95, 0.0]], dtype=np.float32)
        with override_config(CONSOLIDATION_TOPIC_SEMANTIC_THRESHOLD=0.9):
            result = _find_similar_topic("Python Tips", "Best practices", ["python"])

        assert result is not None
        assert result["id"] == "t1"

    @patch("consolidation_memory.backends.encode_documents")
    @patch("consolidation_memory.consolidation.clustering.topic_cache.get_topic_vecs")
    def test_filters_topics_by_scope_before_similarity(self, mock_topic_vecs, mock_encode_documents):
        mock_topic_vecs.return_value = (
            [
                {"id": "t1", "title": "Python Repo A", "project_slug": "repo-a"},
                {"id": "t2", "title": "Python Repo B", "project_slug": "repo-b"},
            ],
            np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
        )
        # Would match t2 if unscoped; scope filtering should force t1 candidate set only.
        mock_encode_documents.return_value = np.array([[0.0, 1.0]], dtype=np.float32)
        with override_config(CONSOLIDATION_TOPIC_SEMANTIC_THRESHOLD=0.1):
            result = _find_similar_topic(
                "Python Repo",
                "summary",
                [],
                scope={"project_slug": "repo-a"},
            )

        assert result is not None
        assert result["id"] == "t1"

    @patch("consolidation_memory.backends.encode_documents")
    @patch("consolidation_memory.consolidation.clustering.topic_cache.get_topic_vecs")
    def test_falls_back_to_word_overlap_when_embedding_fails(self, mock_topic_vecs, mock_encode_documents):
        mock_topic_vecs.return_value = (
            [{"id": "t1", "title": "Python Async Guide"}],
            np.array([[1.0, 0.0]], dtype=np.float32),
        )
        mock_encode_documents.side_effect = RuntimeError("embedding backend unavailable")
        with override_config(CONSOLIDATION_STOPWORDS=set()):
            result = _find_similar_topic("Python Async Tips", "summary", [])

        assert result is not None
        assert result["id"] == "t1"
