"""Tests for recall result deduplication (Phase 2.3).

When a knowledge record's source_episodes overlap with returned episode IDs,
the episode is redundant and should be removed — the record has higher signal
density.

Run with: python -m pytest tests/test_recall_dedup.py -v
"""

import pytest

from consolidation_memory.context_assembler import _deduplicate_episodes


class TestDeduplicateEpisodes:
    """Unit tests for _deduplicate_episodes()."""

    def _ep(self, ep_id: str, content: str = "test") -> dict:
        return {"id": ep_id, "content": content}

    def _rec(self, source_episodes: list[str] | None = None) -> dict:
        return {
            "id": "rec-1",
            "record_type": "fact",
            "source_episodes": source_episodes or [],
        }

    def test_no_records_returns_all_episodes(self):
        episodes = [self._ep("ep1"), self._ep("ep2")]
        result = _deduplicate_episodes(episodes, [])
        assert len(result) == 2

    def test_no_episodes_returns_empty(self):
        result = _deduplicate_episodes([], [self._rec(["ep1"])])
        assert result == []

    def test_removes_episode_covered_by_record(self):
        episodes = [self._ep("ep1"), self._ep("ep2"), self._ep("ep3")]
        records = [self._rec(["ep1", "ep3"])]
        result = _deduplicate_episodes(episodes, records)
        assert len(result) == 1
        assert result[0]["id"] == "ep2"

    def test_keeps_episode_not_covered(self):
        episodes = [self._ep("ep1"), self._ep("ep2")]
        records = [self._rec(["ep3"])]  # covers a different episode
        result = _deduplicate_episodes(episodes, records)
        assert len(result) == 2

    def test_multiple_records_cover_different_episodes(self):
        episodes = [self._ep("ep1"), self._ep("ep2"), self._ep("ep3")]
        records = [self._rec(["ep1"]), self._rec(["ep3"])]
        result = _deduplicate_episodes(episodes, records)
        assert len(result) == 1
        assert result[0]["id"] == "ep2"

    def test_record_with_empty_source_episodes(self):
        episodes = [self._ep("ep1"), self._ep("ep2")]
        records = [self._rec([])]
        result = _deduplicate_episodes(episodes, records)
        assert len(result) == 2

    def test_all_episodes_covered(self):
        episodes = [self._ep("ep1"), self._ep("ep2")]
        records = [self._rec(["ep1", "ep2"])]
        result = _deduplicate_episodes(episodes, records)
        assert len(result) == 0

    def test_record_missing_source_episodes_key(self):
        """Records without source_episodes key should not crash."""
        episodes = [self._ep("ep1")]
        records = [{"id": "rec-1", "record_type": "fact"}]  # no source_episodes
        result = _deduplicate_episodes(episodes, records)
        assert len(result) == 1


class TestDeduplicationInRecall:
    """Integration tests for dedup within the full recall pipeline."""

    @pytest.fixture
    def setup_db(self, tmp_data_dir):
        """Store episodes and create knowledge records that reference them."""
        import json
        import numpy as np

        from consolidation_memory.database import (
            ensure_schema,
            insert_episode,
            upsert_knowledge_topic,
            insert_knowledge_records,
        )
        from consolidation_memory.vector_store import VectorStore
        from consolidation_memory.config import get_config

        ensure_schema()
        cfg = get_config()
        dim = cfg.EMBEDDING_DIMENSION
        vs = VectorStore()

        rng = np.random.default_rng(99)

        # Store 3 episodes
        ep_ids = []
        for i, content in enumerate(["python setup guide", "python testing tips", "rust overview"]):
            eid = insert_episode(content, "fact", tags=["dev"], surprise_score=0.5)
            vec = rng.random(dim).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            vs.add(eid, vec)
            ep_ids.append(eid)

        # Create a knowledge topic + record that sources from ep_ids[0] and ep_ids[1]
        topic_id = upsert_knowledge_topic(
            filename="python_dev.md",
            title="Python Development",
            summary="Python setup and testing",
            source_episodes=[ep_ids[0], ep_ids[1]],
            fact_count=1,
            confidence=0.9,
        )
        insert_knowledge_records(
            topic_id,
            [{
                "record_type": "fact",
                "content": json.dumps({
                    "subject": "Python",
                    "info": "Uses pytest for testing",
                }),
                "embedding_text": "Python uses pytest for testing",
                "confidence": 0.9,
            }],
            source_episodes=[ep_ids[0], ep_ids[1]],
        )

        # Build query vec similar to all stored episodes
        query_vec = rng.random(dim).astype(np.float32)
        query_vec = query_vec / np.linalg.norm(query_vec)

        return vs, ep_ids, query_vec

    def test_recall_dedup_removes_covered_episodes(self, setup_db):
        """Episodes covered by knowledge records are removed from results."""
        from unittest.mock import patch

        from consolidation_memory.context_assembler import recall

        vs, ep_ids, query_vec = setup_db

        # Mock backends for both the main recall path and record_cache/topic_cache
        with (
            patch("consolidation_memory.context_assembler.backends") as mock_be,
            patch("consolidation_memory.context_assembler.record_cache") as mock_rc,
            patch("consolidation_memory.context_assembler.topic_cache") as mock_tc,
        ):
            mock_be.encode_query.return_value = query_vec
            mock_tc.get_topic_vecs.return_value = ([], None)

            # Return a record with source_episodes matching ep_ids[0] and ep_ids[1]
            mock_rc.get_record_vecs.return_value = ([
                {
                    "id": "rec-1",
                    "record_type": "fact",
                    "content": '{"subject": "Python", "info": "Uses pytest"}',
                    "embedding_text": "Python uses pytest for testing",
                    "topic_title": "Python Development",
                    "topic_filename": "python_dev.md",
                    "confidence": 0.9,
                    "source_episodes": f'["{ep_ids[0]}", "{ep_ids[1]}"]',
                },
            ], query_vec.reshape(1, -1))

            result = recall("python testing", n_results=10, vector_store=vs)

        # ep_ids[0] and ep_ids[1] should be deduped (covered by the record)
        returned_ep_ids = {ep["id"] for ep in result["episodes"]}
        assert ep_ids[0] not in returned_ep_ids, "Covered episode should be deduplicated"
        assert ep_ids[1] not in returned_ep_ids, "Covered episode should be deduplicated"
        # ep_ids[2] (rust overview) should remain
        assert ep_ids[2] in returned_ep_ids, "Uncovered episode should remain"
        # Records should still be returned
        assert len(result["records"]) == 1

    def test_recall_dedup_disabled(self, setup_db):
        """When RECALL_DEDUP_ENABLED=False, no dedup happens."""
        from unittest.mock import patch

        from consolidation_memory.context_assembler import recall
        from consolidation_memory.config import reset_config

        vs, ep_ids, query_vec = setup_db

        # Disable dedup
        reset_config(RECALL_DEDUP_ENABLED=False)

        with (
            patch("consolidation_memory.context_assembler.backends") as mock_be,
            patch("consolidation_memory.context_assembler.record_cache") as mock_rc,
            patch("consolidation_memory.context_assembler.topic_cache") as mock_tc,
        ):
            mock_be.encode_query.return_value = query_vec
            mock_tc.get_topic_vecs.return_value = ([], None)

            mock_rc.get_record_vecs.return_value = ([
                {
                    "id": "rec-1",
                    "record_type": "fact",
                    "content": '{"subject": "Python", "info": "Uses pytest"}',
                    "embedding_text": "Python uses pytest for testing",
                    "topic_title": "Python Development",
                    "topic_filename": "python_dev.md",
                    "confidence": 0.9,
                    "source_episodes": f'["{ep_ids[0]}", "{ep_ids[1]}"]',
                },
            ], query_vec.reshape(1, -1))

            result = recall("python testing", n_results=10, vector_store=vs)

        # With dedup disabled, covered episodes should still appear
        returned_ep_ids = {ep["id"] for ep in result["episodes"]}
        covered = {ep_ids[0], ep_ids[1]}
        assert covered & returned_ep_ids, "Covered episodes should remain when dedup is disabled"
