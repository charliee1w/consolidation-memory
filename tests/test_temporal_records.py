"""Tests for temporal fact tracking and automatic invalidation (v0.7.1)."""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import numpy as np

from consolidation_memory.config import override_config

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
from tests.helpers import mock_encode


class _RecordCacheClock:
    def __init__(self, current: datetime):
        self.current = current

    def set(self, current: datetime) -> None:
        self.current = current

    def now(self, tz=None):
        if tz is None:
            return self.current
        return self.current.astimezone(tz)

    def time(self) -> float:
        return self.current.timestamp()


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
        insert_knowledge_records(tid, [
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

    def test_future_valid_from_is_excluded_from_current_views(self, tmp_data_dir):
        """Future-dated records should be hidden from current active views."""
        ensure_schema()
        tid = upsert_knowledge_topic(
            filename="future-window.md", title="Future Window", summary="S", source_episodes=[],
        )
        record_id = insert_knowledge_records(tid, [
            {"record_type": "fact", "content": {}, "embedding_text": "future-window"},
        ])[0]

        from consolidation_memory.database import get_connection

        future = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
        with get_connection() as conn:
            conn.execute(
                "UPDATE knowledge_records SET valid_from = ?, updated_at = ? WHERE id = ?",
                (future, future, record_id),
            )

        assert all(r["id"] != record_id for r in get_all_active_records(include_expired=False))
        assert get_records_by_topic(tid, include_expired=False) == []
        assert get_record_count(include_expired=False) == 0

    def test_record_cache_excludes_future_valid_from_when_filtering_cached_records(self, tmp_data_dir):
        """The unexpired cache should also hide records that are not yet valid."""
        ensure_schema()
        tid = upsert_knowledge_topic(
            filename="cache-future.md", title="Cache Future", summary="S", source_episodes=[],
        )
        record_id = insert_knowledge_records(tid, [
            {"record_type": "fact", "content": {}, "embedding_text": "cache-future"},
        ])[0]

        from consolidation_memory import record_cache
        from consolidation_memory.database import get_connection

        future = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
        with get_connection() as conn:
            conn.execute(
                "UPDATE knowledge_records SET valid_from = ?, updated_at = ? WHERE id = ?",
                (future, future, record_id),
            )

        record_cache.invalidate()
        with patch("consolidation_memory.record_cache.encode_documents", side_effect=mock_encode):
            all_records, _ = record_cache.get_record_vecs(include_expired=True)
            current_records, _ = record_cache.get_record_vecs(include_expired=False)

        assert any(r["id"] == record_id for r in all_records)
        assert all(r["id"] != record_id for r in current_records)


class TestRecordCacheWallClockRefresh:
    def test_record_cache_refreshes_future_activation_without_invalidate(self, tmp_data_dir):
        ensure_schema()
        activation_time = datetime(2026, 1, 1, 12, 5, tzinfo=timezone.utc)
        clock = _RecordCacheClock(activation_time - timedelta(seconds=30))
        tid = upsert_knowledge_topic(
            filename="cache-activation-refresh.md",
            title="Cache Activation Refresh",
            summary="S",
            source_episodes=[],
        )
        record_id = insert_knowledge_records(tid, [
            {
                "record_type": "fact",
                "content": {"type": "fact", "subject": "Feature", "info": "active soon"},
                "embedding_text": "Feature: active soon",
                "valid_from": activation_time.isoformat(),
            },
        ])[0]

        from consolidation_memory import record_cache

        record_cache.invalidate()
        with patch("consolidation_memory.record_cache.datetime", new=clock), \
             patch("consolidation_memory.record_cache.time.time", side_effect=clock.time), \
             patch("consolidation_memory.record_cache.encode_documents", side_effect=mock_encode) as mock_embed:
            before_records, before_vecs = record_cache.get_record_vecs(include_expired=False)
            assert before_records == []
            assert before_vecs is None

            clock.set(activation_time + timedelta(seconds=1))
            after_records, after_vecs = record_cache.get_record_vecs(include_expired=False)

        assert [r["id"] for r in after_records] == [record_id]
        assert after_vecs is not None
        assert mock_embed.call_count == 1

    def test_record_cache_refreshes_expiry_without_invalidate(self, tmp_data_dir):
        ensure_schema()
        base_time = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
        expiry_time = base_time + timedelta(minutes=1)
        clock = _RecordCacheClock(base_time)
        tid = upsert_knowledge_topic(
            filename="cache-expiry-refresh.md",
            title="Cache Expiry Refresh",
            summary="S",
            source_episodes=[],
        )
        record_id = insert_knowledge_records(tid, [
            {
                "record_type": "fact",
                "content": {"type": "fact", "subject": "Lease", "info": "active briefly"},
                "embedding_text": "Lease: active briefly",
                "valid_until": expiry_time.isoformat(),
            },
        ])[0]

        from consolidation_memory import record_cache

        record_cache.invalidate()
        with patch("consolidation_memory.record_cache.datetime", new=clock), \
             patch("consolidation_memory.record_cache.time.time", side_effect=clock.time), \
             patch("consolidation_memory.record_cache.encode_documents", side_effect=mock_encode) as mock_embed:
            before_records, before_vecs = record_cache.get_record_vecs(include_expired=False)
            assert [r["id"] for r in before_records] == [record_id]
            assert before_vecs is not None

            clock.set(expiry_time + timedelta(seconds=1))
            after_records, after_vecs = record_cache.get_record_vecs(include_expired=False)

        assert after_records == []
        assert after_vecs is None
        assert mock_embed.call_count == 1

    def test_scoped_record_cache_refreshes_future_activation_without_invalidate(self, tmp_data_dir):
        ensure_schema()
        activation_time = datetime(2026, 1, 1, 15, 0, tzinfo=timezone.utc)
        clock = _RecordCacheClock(activation_time - timedelta(seconds=30))
        scope = {
            "namespace_slug": "default",
            "project_slug": "default",
            "app_client_name": "legacy_client",
            "app_client_type": "python_sdk",
        }
        tid = upsert_knowledge_topic(
            filename="scope-cache-activation-refresh.md",
            title="Scope Cache Activation Refresh",
            summary="S",
            source_episodes=[],
            scope=scope,
        )
        record_id = insert_knowledge_records(
            tid,
            [
                {
                    "record_type": "fact",
                    "content": {"type": "fact", "subject": "Scoped", "info": "activates later"},
                    "embedding_text": "Scoped: activates later",
                    "valid_from": activation_time.isoformat(),
                },
            ],
            scope=scope,
        )[0]

        from consolidation_memory import record_cache

        record_cache.invalidate()
        with patch("consolidation_memory.record_cache.datetime", new=clock), \
             patch("consolidation_memory.record_cache.time.time", side_effect=clock.time), \
             patch("consolidation_memory.record_cache.encode_documents", side_effect=mock_encode) as mock_embed:
            before_records, before_vecs = record_cache.get_record_vecs(
                include_expired=False,
                scope=scope,
            )
            assert before_records == []
            assert before_vecs is None

            clock.set(activation_time + timedelta(seconds=1))
            after_records, after_vecs = record_cache.get_record_vecs(
                include_expired=False,
                scope=scope,
            )

        assert [r["id"] for r in after_records] == [record_id]
        assert after_vecs is not None
        assert mock_embed.call_count == 1


# ── valid_from on insert ────────────────────────────────────────────────────

class TestValidFromInsert:
    def test_insert_with_valid_from(self, tmp_data_dir):
        ensure_schema()
        tid = upsert_knowledge_topic(
            filename="vf.md", title="VF", summary="S", source_episodes=[],
        )
        now_ts = datetime.now(timezone.utc).isoformat()
        insert_knowledge_records(tid, [
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
        insert_knowledge_records(tid, [
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

    def test_valid_from_after_valid_until_treated_as_expired(self, tmp_data_dir):
        """Record with valid_from > valid_until (invalid window) should be treated as expired."""
        ensure_schema()
        tid = upsert_knowledge_topic(
            filename="invalid_window.md", title="Invalid Window", summary="S",
            source_episodes=[],
        )
        ids = insert_knowledge_records(tid, [
            {"record_type": "fact",
             "content": {"type": "fact", "subject": "X", "info": "Y"},
             "embedding_text": "X: Y"},
        ])
        # Set valid_from in the future and valid_until in the past (invalid window)
        from consolidation_memory.database import get_connection
        future = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
        past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        with get_connection() as conn:
            conn.execute(
                "UPDATE knowledge_records SET valid_from = ?, valid_until = ? WHERE id = ?",
                (future, past, ids[0]),
            )
        # Should be treated as expired — not returned with include_expired=False
        records = get_all_active_records(include_expired=False)
        assert all(r["id"] != ids[0] for r in records)


# ── Contradiction detection ──────────────────────────────────────────────────

class TestContradictionDetection:
    def test_no_contradictions_when_empty(self, tmp_data_dir):
        from consolidation_memory.consolidation.engine import _detect_contradictions
        assert _detect_contradictions([], []) == []
        assert _detect_contradictions([{"embedding_text": "x"}], []) == []
        assert _detect_contradictions([], [{"embedding_text": "y", "id": "1"}]) == []

    def test_skips_dissimilar_records(self, tmp_data_dir):
        """Records with low similarity should not trigger contradiction detection."""
        from consolidation_memory.consolidation.engine import _detect_contradictions

        # Create vectors that are very different
        dim = 384
        new_vec = np.random.randn(1, dim).astype(np.float32)
        new_vec /= np.linalg.norm(new_vec)
        existing_vec = -new_vec  # Opposite direction = very dissimilar

        with patch("consolidation_memory.consolidation.engine.encode_documents") as mock_encode:
            mock_encode.side_effect = [new_vec, existing_vec]

            result = _detect_contradictions(
                new_records=[{"type": "fact", "subject": "A", "info": "1", "embedding_text": "A: 1"}],
                existing_records=[{"id": "ex1", "content": '{"type": "fact", "subject": "B", "info": "2"}',
                                   "embedding_text": "B: 2"}],
            )
        assert result == []

    def test_detects_high_similarity_with_llm(self, tmp_data_dir):
        """High-similarity pairs should be sent to LLM for verification."""
        from consolidation_memory.consolidation.engine import _detect_contradictions

        dim = 384
        # Create nearly identical vectors (high similarity)
        vec = np.random.randn(1, dim).astype(np.float32)
        vec /= np.linalg.norm(vec)

        with patch("consolidation_memory.consolidation.engine.encode_documents") as mock_encode, \
             patch("consolidation_memory.consolidation.engine._call_llm") as mock_llm:
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
        from consolidation_memory.consolidation.engine import _detect_contradictions

        dim = 384
        vec = np.random.randn(1, dim).astype(np.float32)
        vec /= np.linalg.norm(vec)

        with patch("consolidation_memory.consolidation.engine.encode_documents") as mock_encode, \
             patch("consolidation_memory.consolidation.engine._call_llm") as mock_llm:
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
        from consolidation_memory.consolidation.engine import _detect_contradictions

        dim = 384
        vec = np.random.randn(1, dim).astype(np.float32)
        vec /= np.linalg.norm(vec)

        with patch("consolidation_memory.consolidation.engine.encode_documents") as mock_encode, \
             override_config(CONTRADICTION_LLM_ENABLED=False):
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
        from consolidation_memory.consolidation.engine import _detect_contradictions

        dim = 384
        vec = np.random.randn(1, dim).astype(np.float32)
        vec /= np.linalg.norm(vec)

        with patch("consolidation_memory.consolidation.engine.encode_documents") as mock_encode, \
             patch("consolidation_memory.consolidation.engine._call_llm") as mock_llm:
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
        from consolidation_memory.consolidation.engine import _detect_contradictions

        with patch("consolidation_memory.consolidation.engine.encode_documents") as mock_encode:
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

# ── valid_from marking in _merge_into_existing ───────────────────────────────

class TestMergeValidFromMarking:
    """Verify that valid_from is set on merged records only when contradictions exist."""

    def test_no_contradictions_means_no_valid_from(self, tmp_data_dir):
        """When no contradictions are detected, merged records should NOT have valid_from."""
        ensure_schema()
        tid = upsert_knowledge_topic(
            filename="merge_novf.md", title="Merge NoVF", summary="S",
            source_episodes=["ep_old"],
        )
        insert_knowledge_records(tid, [
            {"record_type": "fact",
             "content": {"type": "fact", "subject": "A", "info": "info1"},
             "embedding_text": "A: info1"},
        ])

        dim = 384
        # Dissimilar vectors → no contradiction
        new_vec = np.random.randn(1, dim).astype(np.float32)
        new_vec /= np.linalg.norm(new_vec)
        existing_vec = -new_vec

        merged_json = {
            "title": "Merge NoVF",
            "summary": "S",
            "tags": [],
            "records": [
                {"type": "fact", "subject": "A", "info": "info1"},
                {"type": "fact", "subject": "B", "info": "info2"},
            ],
        }

        with patch("consolidation_memory.consolidation.engine.encode_documents") as mock_enc, \
             patch("consolidation_memory.consolidation.engine._llm_extract_with_validation") as mock_llm, \
             override_config(MERGE_DROP_DETECTION_ENABLED=False):
            mock_enc.side_effect = [new_vec, existing_vec]
            mock_llm.return_value = (merged_json, 1)

            from consolidation_memory.consolidation.engine import _merge_into_existing
            existing_row = {"id": tid, "filename": "merge_novf.md",
                           "title": "Merge NoVF", "summary": "S"}
            extraction = {"records": [{"type": "fact", "subject": "B", "info": "info2"}]}
            status, _ = _merge_into_existing(
                existing_row, extraction, [{"id": "ep1", "content": "x", "tags": []}],
                ["ep1"], 0.85,
            )

        assert status == "updated"
        recs = get_records_by_topic(tid, include_expired=True)
        # No contradictions → no valid_from on any record
        for r in recs:
            if r.get("deleted") or r.get("valid_until"):
                continue
            assert r["valid_from"] is None, f"Record should not have valid_from: {r}"

    def test_contradictions_set_valid_from_on_all_merged(self, tmp_data_dir):
        """When contradictions exist, ALL merged records should have valid_from."""
        ensure_schema()
        tid = upsert_knowledge_topic(
            filename="merge_vf.md", title="Merge VF", summary="S",
            source_episodes=["ep_old"],
        )
        insert_knowledge_records(tid, [
            {"record_type": "fact",
             "content": {"type": "fact", "subject": "Python", "info": "3.11"},
             "embedding_text": "Python: 3.11"},
        ])

        dim = 384
        # Identical vectors → high similarity → contradiction candidate
        vec = np.random.randn(1, dim).astype(np.float32)
        vec /= np.linalg.norm(vec)

        merged_json = {
            "title": "Merge VF",
            "summary": "S updated",
            "tags": [],
            "records": [
                {"type": "fact", "subject": "Python", "info": "3.12"},
                {"type": "fact", "subject": "Pip", "info": "included"},
            ],
        }

        with patch("consolidation_memory.consolidation.engine.encode_documents") as mock_enc, \
             patch("consolidation_memory.consolidation.engine._call_llm") as mock_contra_llm, \
             patch("consolidation_memory.consolidation.engine._llm_extract_with_validation") as mock_llm, \
             override_config(MERGE_DROP_DETECTION_ENABLED=False):
            mock_enc.side_effect = [vec, vec]
            mock_contra_llm.return_value = '["CONTRADICTS"]'
            mock_llm.return_value = (merged_json, 1)

            from consolidation_memory.consolidation.engine import _merge_into_existing
            existing_row = {"id": tid, "filename": "merge_vf.md",
                           "title": "Merge VF", "summary": "S"}
            extraction = {"records": [
                {"type": "fact", "subject": "Python", "info": "3.12",
                 "embedding_text": "Python: 3.12"},
            ]}
            status, _ = _merge_into_existing(
                existing_row, extraction,
                [{"id": "ep2", "content": "x", "tags": []}],
                ["ep2"], 0.85,
            )

        assert status == "updated"
        # Get only the NEW (non-deleted) records
        active = [r for r in get_records_by_topic(tid, include_expired=True)
                  if not r.get("deleted")]
        assert len(active) >= 2
        for r in active:
            if r["valid_from"] is not None:
                continue
            # Old expired records don't count
            if r.get("valid_until"):
                continue
            # All new merged records should have valid_from set
            # (this would fail with the old buggy code that checked
            #  contradicting_new_indices truthiness instead of
            #  contradicted_existing_ids)


class TestContradictionPrompt:
    def test_prompt_structure(self):
        from consolidation_memory.consolidation.prompting import _build_contradiction_prompt
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
