"""Tests for contradiction audit log (Option A — truth maintenance)."""

from unittest.mock import patch

import numpy as np

from consolidation_memory.config import override_config
from consolidation_memory.database import (
    ensure_schema,
    get_contradictions,
    insert_contradiction,
    insert_knowledge_records,
    get_records_by_topic,
    upsert_knowledge_topic,
)


# ── Schema migration ──────────────────────────────────────────────────────────


class TestSchemaMigrationV8:
    def test_migration_creates_contradiction_log_table(self, tmp_data_dir):
        ensure_schema()
        from consolidation_memory.database import get_connection

        with get_connection() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='contradiction_log'"
            )
            assert cursor.fetchone() is not None

    def test_contradiction_log_columns(self, tmp_data_dir):
        ensure_schema()
        from consolidation_memory.database import get_connection

        with get_connection() as conn:
            cursor = conn.execute("PRAGMA table_info(contradiction_log)")
            columns = {row["name"] for row in cursor.fetchall()}

        expected = {
            "id", "topic_id", "old_record_id", "new_record_id",
            "old_content", "new_content", "resolution", "reason", "detected_at",
        }
        assert expected.issubset(columns)


# ── insert_contradiction ─────────────────────────────────────────────────────


class TestInsertContradiction:
    def test_basic_insert(self, tmp_data_dir):
        ensure_schema()
        tid = upsert_knowledge_topic(
            filename="test.md", title="Test", summary="S",
            source_episodes=["ep1"],
        )
        cid = insert_contradiction(
            topic_id=tid,
            old_record_id="old-rec-1",
            new_record_id="new-rec-1",
            old_content='{"type": "fact", "info": "Python 2 is latest"}',
            new_content='{"type": "fact", "info": "Python 3 is latest"}',
            resolution="expired_old",
            reason="Newer version available",
        )
        assert cid is not None

        rows = get_contradictions()
        assert len(rows) == 1
        assert rows[0]["old_record_id"] == "old-rec-1"
        assert rows[0]["resolution"] == "expired_old"
        assert rows[0]["reason"] == "Newer version available"

    def test_insert_without_new_record_id(self, tmp_data_dir):
        ensure_schema()
        insert_contradiction(
            topic_id=None,
            old_record_id="old-1",
            new_record_id=None,
            old_content="old",
            new_content="new",
        )
        rows = get_contradictions()
        assert len(rows) == 1
        assert rows[0]["new_record_id"] is None
        assert rows[0]["topic_id"] is None


# ── get_contradictions ──────────────────────────────────────────────────────


class TestGetContradictions:
    def test_returns_empty_when_none(self, tmp_data_dir):
        ensure_schema()
        assert get_contradictions() == []

    def test_filter_by_topic(self, tmp_data_dir):
        ensure_schema()
        tid1 = upsert_knowledge_topic(
            filename="t1.md", title="T1", summary="S", source_episodes=[],
        )
        tid2 = upsert_knowledge_topic(
            filename="t2.md", title="T2", summary="S", source_episodes=[],
        )

        insert_contradiction(
            topic_id=tid1, old_record_id="r1", new_record_id=None,
            old_content="a", new_content="b",
        )
        insert_contradiction(
            topic_id=tid2, old_record_id="r2", new_record_id=None,
            old_content="c", new_content="d",
        )
        insert_contradiction(
            topic_id=tid1, old_record_id="r3", new_record_id=None,
            old_content="e", new_content="f",
        )

        all_rows = get_contradictions()
        assert len(all_rows) == 3

        t1_rows = get_contradictions(topic_id=tid1)
        assert len(t1_rows) == 2
        assert all(r["topic_id"] == tid1 for r in t1_rows)

    def test_ordered_by_detected_at_desc(self, tmp_data_dir):
        ensure_schema()
        insert_contradiction(
            topic_id=None, old_record_id="r1", new_record_id=None,
            old_content="first", new_content="b",
        )
        insert_contradiction(
            topic_id=None, old_record_id="r2", new_record_id=None,
            old_content="second", new_content="d",
        )
        rows = get_contradictions()
        assert len(rows) == 2
        # Most recent first
        assert rows[0]["old_content"] == "second"

    def test_includes_topic_metadata(self, tmp_data_dir):
        ensure_schema()
        tid = upsert_knowledge_topic(
            filename="meta.md", title="Meta Topic", summary="S",
            source_episodes=[],
        )
        insert_contradiction(
            topic_id=tid, old_record_id="r1", new_record_id=None,
            old_content="old", new_content="new",
        )
        rows = get_contradictions(topic_id=tid)
        assert rows[0]["topic_title"] == "Meta Topic"
        assert rows[0]["topic_filename"] == "meta.md"

    def test_respects_limit(self, tmp_data_dir):
        ensure_schema()
        for i in range(5):
            insert_contradiction(
                topic_id=None, old_record_id=f"r{i}", new_record_id=None,
                old_content=f"old{i}", new_content=f"new{i}",
            )
        rows = get_contradictions(limit=3)
        assert len(rows) == 3


# ── Contradictions logged during consolidation ───────────────────────────────


class TestContradictionsDuringConsolidation:
    """Test that _merge_into_existing logs contradictions and reduces confidence."""

    def _make_vec(self, seed=42):
        rng = np.random.RandomState(seed)
        v = rng.randn(384).astype(np.float32)
        v /= np.linalg.norm(v)
        return v

    def test_contradictions_logged_during_merge(self, tmp_data_dir):
        """When _merge_into_existing detects contradictions, they should be
        logged to the contradiction_log table."""
        ensure_schema()

        tid = upsert_knowledge_topic(
            filename="merge_test.md", title="Merge Test", summary="S",
            source_episodes=["ep_old"],
        )

        # Insert an existing record
        old_rec_ids = insert_knowledge_records(tid, [
            {
                "record_type": "fact",
                "content": {"type": "fact", "subject": "Python", "info": "version 2"},
                "embedding_text": "Python version is 2",
                "confidence": 0.8,
            },
        ])

        # Simulate extraction data with a contradicting new record
        extraction_data = {
            "title": "Merge Test",
            "summary": "S",
            "tags": [],
            "records": [
                {"type": "fact", "subject": "Python", "info": "version 3"},
            ],
        }

        existing = {"id": tid, "filename": "merge_test.md", "title": "Merge Test", "summary": "S"}

        # Mock encode_documents to return controlled vectors
        def mock_encode(texts):
            # Return same vector for all — triggers high similarity
            return np.array([self._make_vec(42) for _ in texts])

        # Mock _llm_extract_with_validation to return merged data
        def mock_llm_extract(prompt, episodes):
            return {
                "title": "Merge Test",
                "summary": "S",
                "tags": [],
                "records": [
                    {"type": "fact", "subject": "Python", "info": "version 3"},
                ],
            }, 1

        with (
            patch("consolidation_memory.consolidation.engine.encode_documents", mock_encode),
            patch(
                "consolidation_memory.consolidation.engine._llm_extract_with_validation",
                mock_llm_extract,
            ),
            override_config(
                CONTRADICTION_LLM_ENABLED=False,
                CONTRADICTION_SIMILARITY_THRESHOLD=0.5,
                RENDER_MARKDOWN=False,
            ),
        ):
            from consolidation_memory.consolidation.engine import _merge_into_existing

            status, calls = _merge_into_existing(
                existing=existing,
                extraction_data=extraction_data,
                cluster_episodes=[{"id": "ep_new", "content": "new stuff", "tags": "[]"}],
                cluster_ep_ids=["ep_new"],
                confidence=0.8,
            )

        assert status == "updated"

        # Check contradictions were logged
        rows = get_contradictions(topic_id=tid)
        assert len(rows) >= 1
        assert rows[0]["resolution"] == "expired_old"
        assert rows[0]["old_record_id"] == old_rec_ids[0]

    def test_confidence_reduced_on_contradiction(self, tmp_data_dir):
        """When contradictions are detected, the merged topic's confidence
        should be reduced by 10%."""
        ensure_schema()

        tid = upsert_knowledge_topic(
            filename="conf_test.md", title="Conf Test", summary="S",
            source_episodes=["ep_old"],
        )

        insert_knowledge_records(tid, [
            {
                "record_type": "fact",
                "content": {"type": "fact", "subject": "X", "info": "old"},
                "embedding_text": "X is old",
                "confidence": 0.8,
            },
        ])

        extraction_data = {
            "title": "Conf Test",
            "summary": "S",
            "tags": [],
            "records": [
                {"type": "fact", "subject": "X", "info": "new"},
            ],
        }

        existing = {"id": tid, "filename": "conf_test.md", "title": "Conf Test", "summary": "S"}

        def mock_encode(texts):
            return np.array([self._make_vec(42) for _ in texts])

        def mock_llm_extract(prompt, episodes):
            return {
                "title": "Conf Test",
                "summary": "S",
                "tags": [],
                "records": [
                    {"type": "fact", "subject": "X", "info": "new"},
                ],
            }, 1

        with (
            patch("consolidation_memory.consolidation.engine.encode_documents", mock_encode),
            patch(
                "consolidation_memory.consolidation.engine._llm_extract_with_validation",
                mock_llm_extract,
            ),
            override_config(
                CONTRADICTION_LLM_ENABLED=False,
                CONTRADICTION_SIMILARITY_THRESHOLD=0.5,
                RENDER_MARKDOWN=False,
            ),
        ):
            from consolidation_memory.consolidation.engine import _merge_into_existing

            status, _ = _merge_into_existing(
                existing=existing,
                extraction_data=extraction_data,
                cluster_episodes=[{"id": "ep_new2", "content": "new", "tags": "[]"}],
                cluster_ep_ids=["ep_new2"],
                confidence=0.8,
            )

        assert status == "updated"

        # Check that new records have reduced confidence (0.8 * 0.9 = 0.72)
        new_recs = get_records_by_topic(tid)
        assert len(new_recs) >= 1
        for rec in new_recs:
            assert abs(rec["confidence"] - 0.72) < 0.01


# ── MCP tool ─────────────────────────────────────────────────────────────────


class TestContradictionsMCPTool:
    def test_memory_contradictions_tool(self, tmp_data_dir):
        """Test the memory_contradictions MCP tool returns results."""
        ensure_schema()
        tid = upsert_knowledge_topic(
            filename="mcp.md", title="MCP Test", summary="S",
            source_episodes=[],
        )
        insert_contradiction(
            topic_id=tid, old_record_id="r1", new_record_id=None,
            old_content="old fact", new_content="new fact",
            resolution="expired_old",
        )

        from consolidation_memory.client import MemoryClient

        client = MemoryClient(auto_consolidate=False)
        try:
            result = client.contradictions()
            assert result.total == 1
            assert len(result.contradictions) == 1
            assert result.contradictions[0]["old_content"] == "old fact"
        finally:
            client.close()

    def test_memory_contradictions_filtered_by_topic(self, tmp_data_dir):
        """Test filtering contradictions by topic name."""
        ensure_schema()
        tid = upsert_knowledge_topic(
            filename="filtered.md", title="Filtered", summary="S",
            source_episodes=[],
        )
        upsert_knowledge_topic(
            filename="other.md", title="Other", summary="S",
            source_episodes=[],
        )
        insert_contradiction(
            topic_id=tid, old_record_id="r1", new_record_id=None,
            old_content="old", new_content="new",
        )

        from consolidation_memory.client import MemoryClient

        client = MemoryClient(auto_consolidate=False)
        try:
            # Filter by filename
            result = client.contradictions(topic="filtered.md")
            assert result.total == 1

            # Filter by title
            result = client.contradictions(topic="Filtered")
            assert result.total == 1

            # Non-matching filter
            result = client.contradictions(topic="nonexistent")
            assert result.total == 0
        finally:
            client.close()
