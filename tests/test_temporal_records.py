"""Tests for temporal fact tracking and automatic invalidation (v0.7.1)."""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import numpy as np

from consolidation_memory.database import (
    ensure_schema,
    expire_record,
    get_all_active_records,
    get_record_count,
    get_records_by_topic,
    insert_knowledge_records,
    soft_delete_records_by_ids,
    upsert_knowledge_topic,
)


# ── Schema migration ──────────────────────────────────────────────────────────

class TestSchemaMigrationV6:
    def test_migration_adds_temporal_columns(self, tmp_data_dir):
        ensure_schema()
        from consolidation_memory.database import get_connection
        with get_connection() as conn:
            cursor = conn.execute("PRAGMA table_info(knowledge_records)")
            columns = {row["name"] for row in cursor.fetchall()}
        assert "valid_from" in columns
        assert "valid_until" in columns

    def test_temporal_columns_are_nullable(self, tmp_data_dir):
        ensure_schema()
        tid = upsert_knowledge_topic(
            filename="temporal.md", title="Temporal", summary="S",
            source_episodes=["ep1"],
        )
        ids = insert_knowledge_records(tid, [
            {"record_type": "fact", "content": {"type": "fact", "subject": "X", "info": "Y"},
             "embedding_text": "X: Y"},
        ])
        recs = get_records_by_topic(tid, include_expired=True)
        assert len(recs) == 1
        assert recs[0]["valid_from"] is None
        assert recs[0]["valid_until"] is None


# ── expire_record ──────────────────────────────────────────────────────────────

class TestExpireRecord:
    def test_expire_sets_valid_until(self, tmp_data_dir):
        ensure_schema()
        tid = upsert_knowledge_topic(
            filename="exp.md", title="Exp", summary="S", source_episodes=[],
        )
        ids = insert_knowledge_records(tid, [
            {"record_type": "fact", "content": {}, "embedding_text": "a"},
        ])
        assert expire_record(ids[0])
        recs = get_records_by_topic(tid, include_expired=True)
        assert recs[0]["valid_until"] is not None

    def test_expire_with_custom_timestamp(self, tmp_data_dir):
        ensure_schema()
        tid = upsert_knowledge_topic(
            filename="exp2.md", title="Exp2", summary="S", source_episodes=[],
        )
        ids = insert_knowledge_records(tid, [
            {"record_type": "fact", "content": {}, "embedding_text": "b"},
        ])
        custom_ts = "2025-01-01T00:00:00+00:00"
        expire_record(ids[0], valid_until=custom_ts)
        recs = get_records_by_topic(tid, include_expired=True)
        assert recs[0]["valid_until"] == custom_ts

    def test_expire_nonexistent_returns_false(self, tmp_data_dir):
        ensure_schema()
        assert not expire_record("nonexistent-id")


# ── Temporal filtering ─────────────────────────────────────────────────────────

class TestTemporalFiltering:
    def _setup_topic_with_records(self):
        """Helper: create a topic with one current and one expired record."""
        tid = upsert_knowledge_topic(
            filename="filter.md", title="Filter", summary="S", source_episodes=[],
        )
        ids = insert_knowledge_records(tid, [
            {"record_type": "fact",
             "content": {"type": "fact", "subject": "Python", "info": "3.11"},
             "embedding_text": "Python: 3.11"},
            {"record_type": "fact",
             "content": {"type": "fact", "subject": "Python", "info": "3.12"},
             "embedding_text": "Python: 3.12"},
        ])
        # Expire the first record (old version)
        past_ts = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        expire_record(ids[0], valid_until=past_ts)
        return tid, ids

    def test_get_all_active_records_excludes_expired(self, tmp_data_dir):
        ensure_schema()
        tid, ids = self._setup_topic_with_records()
        records = get_all_active_records(include_expired=False)
        assert len(records) == 1
        assert records[0]["embedding_text"] == "Python: 3.12"

    def test_get_all_active_records_includes_expired(self, tmp_data_dir):
        ensure_schema()
        tid, ids = self._setup_topic_with_records()
        records = get_all_active_records(include_expired=True)
        assert len(records) == 2

    def test_get_records_by_topic_excludes_expired(self, tmp_data_dir):
        ensure_schema()
        tid, ids = self._setup_topic_with_records()
        records = get_records_by_topic(tid, include_expired=False)
        assert len(records) == 1

    def test_get_records_by_topic_includes_expired(self, tmp_data_dir):
        ensure_schema()
        tid, ids = self._setup_topic_with_records()
        records = get_records_by_topic(tid, include_expired=True)
        assert len(records) == 2

    def test_get_record_count_excludes_expired(self, tmp_data_dir):
        ensure_schema()
        tid, ids = self._setup_topic_with_records()
        assert get_record_count(include_expired=False) == 1
        assert get_record_count(include_expired=True) == 2

    def test_future_valid_until_not_filtered(self, tmp_data_dir):
        """Records with valid_until in the future should still be returned."""
        ensure_schema()
        tid = upsert_knowledge_topic(
            filename="future.md", title="Future", summary="S", source_episodes=[],
        )
        ids = insert_knowledge_records(tid, [
            {"record_type": "fact", "content": {}, "embedding_text": "z"},
        ])
        future_ts = (datetime.now(timezone.utc) + timedelta(days=365)).isoformat()
        expire_record(ids[0], valid_until=future_ts)
        # Should still appear since valid_until is in the future
        records = get_all_active_records(include_expired=False)
        assert len(records) == 1


# ── valid_from on insert ────────────────────────────────────────────────────

class TestValidFromInsert:
    def test_insert_with_valid_from(self, tmp_data_dir):
        ensure_schema()
        tid = upsert_knowledge_topic(
            filename="vf.md", title="VF", summary="S", source_episodes=[],
        )
        now_ts = datetime.now(timezone.utc).isoformat()
        ids = insert_knowledge_records(tid, [
            {"record_type": "fact", "content": {}, "embedding_text": "x",
             "valid_from": now_ts},
        ])
        recs = get_records_by_topic(tid, include_expired=True)
        assert recs[0]["valid_from"] == now_ts

    def test_insert_without_valid_from_is_null(self, tmp_data_dir):
        ensure_schema()
        tid = upsert_knowledge_topic(
            filename="nvf.md", title="NVF", summary="S", source_episodes=[],
        )
        ids = insert_knowledge_records(tid, [
            {"record_type": "fact", "content": {}, "embedding_text": "y"},
        ])
        recs = get_records_by_topic(tid, include_expired=True)
        assert recs[0]["valid_from"] is None


# ── soft_delete_records_by_ids ──────────────────────────────────────────────

class TestSoftDeleteByIds:
    def test_deletes_specific_records(self, tmp_data_dir):
        ensure_schema()
        tid = upsert_knowledge_topic(
            filename="sdbi.md", title="SDBI", summary="S", source_episodes=[],
        )
        ids = insert_knowledge_records(tid, [
            {"record_type": "fact", "content": {}, "embedding_text": "a"},
            {"record_type": "fact", "content": {}, "embedding_text": "b"},
            {"record_type": "fact", "content": {}, "embedding_text": "c"},
        ])
        deleted = soft_delete_records_by_ids([ids[0], ids[2]])
        assert deleted == 2
        remaining = get_records_by_topic(tid)
        assert len(remaining) == 1
        assert remaining[0]["embedding_text"] == "b"

    def test_empty_list_returns_zero(self, tmp_data_dir):
        ensure_schema()
        assert soft_delete_records_by_ids([]) == 0


# ── Contradiction detection ──────────────────────────────────────────────────

class TestContradictionDetection:
    def test_no_contradictions_when_empty(self, tmp_data_dir):
        from consolidation_memory.consolidation import _detect_contradictions
        assert _detect_contradictions([], []) == []
        assert _detect_contradictions([{"embedding_text": "x"}], []) == []
        assert _detect_contradictions([], [{"embedding_text": "y", "id": "1"}]) == []

    def test_skips_dissimilar_records(self, tmp_data_dir):
        """Records with low similarity should not trigger contradiction detection."""
        from consolidation_memory.consolidation import _detect_contradictions

        # Create vectors that are very different
        dim = 384
        new_vec = np.random.randn(1, dim).astype(np.float32)
        new_vec /= np.linalg.norm(new_vec)
        existing_vec = -new_vec  # Opposite direction = very dissimilar

        with patch("consolidation_memory.consolidation.encode_documents") as mock_encode:
            mock_encode.side_effect = [new_vec, existing_vec]

            result = _detect_contradictions(
                new_records=[{"type": "fact", "subject": "A", "info": "1", "embedding_text": "A: 1"}],
                existing_records=[{"id": "ex1", "content": '{"type": "fact", "subject": "B", "info": "2"}',
                                   "embedding_text": "B: 2"}],
            )
        assert result == []

    def test_detects_high_similarity_with_llm(self, tmp_data_dir):
        """High-similarity pairs should be sent to LLM for verification."""
        from consolidation_memory.consolidation import _detect_contradictions

        dim = 384
        # Create nearly identical vectors (high similarity)
        vec = np.random.randn(1, dim).astype(np.float32)
        vec /= np.linalg.norm(vec)

        with patch("consolidation_memory.consolidation.encode_documents") as mock_encode, \
             patch("consolidation_memory.consolidation._call_llm") as mock_llm:
            mock_encode.side_effect = [vec, vec]  # Same vector = similarity ~1.0
            mock_llm.return_value = '["CONTRADICTS"]'

            result = _detect_contradictions(
                new_records=[{"type": "fact", "subject": "Python", "info": "3.12",
                              "embedding_text": "Python: 3.12"}],
                existing_records=[{"id": "ex1",
                                   "content": '{"type": "fact", "subject": "Python", "info": "3.11"}',
                                   "embedding_text": "Python: 3.11"}],
            )
        assert len(result) == 1
        assert result[0] == (0, "ex1")

    def test_compatible_pair_not_flagged(self, tmp_data_dir):
        """LLM returning COMPATIBLE should not flag as contradiction."""
        from consolidation_memory.consolidation import _detect_contradictions

        dim = 384
        vec = np.random.randn(1, dim).astype(np.float32)
        vec /= np.linalg.norm(vec)

        with patch("consolidation_memory.consolidation.encode_documents") as mock_encode, \
             patch("consolidation_memory.consolidation._call_llm") as mock_llm:
            mock_encode.side_effect = [vec, vec]
            mock_llm.return_value = '["COMPATIBLE"]'

            result = _detect_contradictions(
                new_records=[{"type": "fact", "subject": "Python", "info": "has pip",
                              "embedding_text": "Python: has pip"}],
                existing_records=[{"id": "ex1",
                                   "content": '{"type": "fact", "subject": "Python", "info": "version 3.12"}',
                                   "embedding_text": "Python: version 3.12"}],
            )
        assert result == []

    def test_without_llm_treats_all_as_contradictions(self, tmp_data_dir):
        """When CONTRADICTION_LLM_ENABLED=False, all high-sim pairs are contradictions."""
        from consolidation_memory.consolidation import _detect_contradictions

        dim = 384
        vec = np.random.randn(1, dim).astype(np.float32)
        vec /= np.linalg.norm(vec)

        with patch("consolidation_memory.consolidation.encode_documents") as mock_encode, \
             patch("consolidation_memory.consolidation.CONTRADICTION_LLM_ENABLED", False):
            mock_encode.side_effect = [vec, vec]

            result = _detect_contradictions(
                new_records=[{"type": "fact", "subject": "X", "info": "1",
                              "embedding_text": "X: 1"}],
                existing_records=[{"id": "ex1",
                                   "content": '{"type": "fact", "subject": "X", "info": "2"}',
                                   "embedding_text": "X: 2"}],
            )
        assert len(result) == 1

    def test_llm_failure_returns_empty(self, tmp_data_dir):
        """LLM failures should gracefully return no contradictions."""
        from consolidation_memory.consolidation import _detect_contradictions

        dim = 384
        vec = np.random.randn(1, dim).astype(np.float32)
        vec /= np.linalg.norm(vec)

        with patch("consolidation_memory.consolidation.encode_documents") as mock_encode, \
             patch("consolidation_memory.consolidation._call_llm") as mock_llm:
            mock_encode.side_effect = [vec, vec]
            mock_llm.side_effect = RuntimeError("LLM down")

            result = _detect_contradictions(
                new_records=[{"type": "fact", "subject": "A", "info": "1",
                              "embedding_text": "A: 1"}],
                existing_records=[{"id": "ex1",
                                   "content": '{"type": "fact", "subject": "A", "info": "2"}',
                                   "embedding_text": "A: 2"}],
            )
        assert result == []

    def test_embedding_failure_returns_empty(self, tmp_data_dir):
        """Embedding failures should gracefully return no contradictions."""
        from consolidation_memory.consolidation import _detect_contradictions

        with patch("consolidation_memory.consolidation.encode_documents") as mock_encode:
            mock_encode.side_effect = RuntimeError("Backend down")

            result = _detect_contradictions(
                new_records=[{"type": "fact", "subject": "A", "info": "1",
                              "embedding_text": "A: 1"}],
                existing_records=[{"id": "ex1",
                                   "content": '{"type": "fact", "subject": "A", "info": "2"}',
                                   "embedding_text": "A: 2"}],
            )
        assert result == []


# ── Contradiction prompt ──────────────────────────────────────────────────────

class TestContradictionPrompt:
    def test_prompt_structure(self):
        from consolidation_memory.consolidation import _build_contradiction_prompt
        pairs = [
            ({"type": "fact", "subject": "Python", "info": "3.11"},
             {"type": "fact", "subject": "Python", "info": "3.12"}),
        ]
        prompt = _build_contradiction_prompt(pairs)
        assert "CONTRADICTS" in prompt
        assert "COMPATIBLE" in prompt
        assert "Pair 1:" in prompt
        assert "EXISTING:" in prompt
        assert "NEW:" in prompt
        assert "3.11" in prompt
        assert "3.12" in prompt
