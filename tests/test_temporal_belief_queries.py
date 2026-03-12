"""Tests for temporal belief queries (Phase 3.4).

Verifies the `as_of` parameter on recall, which returns knowledge state
at a specific point in time -- including records that have since been
superseded and excluding records that did not yet exist.

Run with: python -m pytest tests/test_temporal_belief_queries.py -v
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from consolidation_memory.database import (
    ensure_schema,
    get_records_as_of,
    insert_knowledge_records,
    upsert_knowledge_topic,
)
from consolidation_memory.types import RecallResult
from tests.helpers import mock_encode


# ── Database: get_records_as_of ──────────────────────────────────────────────

class TestGetRecordsAsOf:
    """Test the database query that returns records valid at a point in time."""

    def _setup_timeline(self):
        """Create a topic with records at different points in time.

        Timeline:
            T0: record_old created (Python 3.11)
            T1: record_old expired, record_new created (Python 3.12)
            T2: now (record_new is current)
        """
        ensure_schema()
        tid = upsert_knowledge_topic(
            filename="python_version.md",
            title="Python Version",
            summary="Tracks Python version",
            source_episodes=["ep1"],
        )

        # Insert record_old at T0 (using manual created_at)
        from consolidation_memory.database import get_connection
        import uuid

        t0 = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        t1 = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()

        old_id = str(uuid.uuid4())
        with get_connection() as conn:
            conn.execute(
                """INSERT INTO knowledge_records
                   (id, topic_id, record_type, content, embedding_text,
                    source_episodes, confidence, created_at, updated_at, valid_until)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (old_id, tid, "fact", '{"subject": "Python", "info": "3.11"}',
                 "Python: 3.11", "[]", 0.9, t0, t0, t1),
            )

        new_id = str(uuid.uuid4())
        with get_connection() as conn:
            conn.execute(
                """INSERT INTO knowledge_records
                   (id, topic_id, record_type, content, embedding_text,
                    source_episodes, confidence, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (new_id, tid, "fact", '{"subject": "Python", "info": "3.12"}',
                 "Python: 3.12", "[]", 0.9, t1, t1),
            )

        return tid, old_id, new_id, t0, t1

    def test_as_of_returns_old_record_at_early_time(self, tmp_data_dir):
        """Querying at T0 should return the old record only."""
        tid, old_id, new_id, t0, t1 = self._setup_timeline()

        # Query at T0 + 1 day (between T0 and T1)
        query_time = (datetime.now(timezone.utc) - timedelta(days=25)).isoformat()
        records = get_records_as_of(query_time)

        record_ids = {r["id"] for r in records}
        assert old_id in record_ids, "Old record should be present at T0"
        assert new_id not in record_ids, "New record should not exist yet at T0"

    def test_as_of_returns_new_record_at_later_time(self, tmp_data_dir):
        """Querying at T1 should return only the new record."""
        tid, old_id, new_id, t0, t1 = self._setup_timeline()

        # Query at T1 + 1 day (after T1)
        query_time = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()
        records = get_records_as_of(query_time)

        record_ids = {r["id"] for r in records}
        assert new_id in record_ids, "New record should be present after T1"
        assert old_id not in record_ids, "Old record should be expired by T1"

    def test_as_of_returns_both_at_transition(self, tmp_data_dir):
        """At the exact moment T1, old should be excluded (valid_until = T1 means expired at T1)."""
        tid, old_id, new_id, t0, t1 = self._setup_timeline()

        records = get_records_as_of(t1)
        record_ids = {r["id"] for r in records}
        # valid_until > as_of, so at exactly T1, old record (valid_until=T1) is NOT included
        assert old_id not in record_ids
        assert new_id in record_ids

    def test_as_of_treats_equivalent_offset_instants_equally(self, tmp_data_dir):
        """Equivalent timestamps with different offsets should match the same records."""
        ensure_schema()
        tid = upsert_knowledge_topic(
            filename="offsets.md",
            title="Offsets",
            summary="S",
            source_episodes=[],
        )
        rec_ids = insert_knowledge_records(
            tid,
            [{"record_type": "fact", "content": {"subject": "X", "info": "Y"}, "embedding_text": "X:Y"}],
        )
        rec_id = rec_ids[0]

        from consolidation_memory.database import get_connection

        created_utc = "2026-01-01T00:00:00+00:00"
        with get_connection() as conn:
            conn.execute(
                "UPDATE knowledge_records SET created_at = ?, updated_at = ? WHERE id = ?",
                (created_utc, created_utc, rec_id),
            )

        equivalent_as_of = "2025-12-31T16:00:00-08:00"
        records = get_records_as_of(equivalent_as_of)
        assert any(r["id"] == rec_id for r in records)

    def test_as_of_excludes_records_before_valid_from(self, tmp_data_dir):
        """Records should stay hidden until their valid_from time, even if created earlier."""
        ensure_schema()
        tid = upsert_knowledge_topic(
            filename="future-valid-from.md",
            title="Future Valid From",
            summary="S",
            source_episodes=[],
        )
        rec_id = insert_knowledge_records(
            tid,
            [{"record_type": "fact", "content": {"subject": "X", "info": "future"}, "embedding_text": "X: future"}],
        )[0]

        from consolidation_memory.database import get_connection

        created_at = "2026-01-01T00:00:00+00:00"
        valid_from = "2026-01-03T00:00:00+00:00"
        with get_connection() as conn:
            conn.execute(
                "UPDATE knowledge_records SET created_at = ?, updated_at = ?, valid_from = ? WHERE id = ?",
                (created_at, created_at, valid_from, rec_id),
            )

        early_records = get_records_as_of("2026-01-02T12:00:00+00:00")
        assert all(r["id"] != rec_id for r in early_records)

        active_records = get_records_as_of(valid_from)
        assert any(r["id"] == rec_id for r in active_records)

    def test_as_of_before_any_records(self, tmp_data_dir):
        """Querying before any records existed should return nothing."""
        tid, old_id, new_id, t0, t1 = self._setup_timeline()

        very_early = (datetime.now(timezone.utc) - timedelta(days=365)).isoformat()
        records = get_records_as_of(very_early)
        assert len(records) == 0

    def test_as_of_includes_topic_metadata(self, tmp_data_dir):
        """Records from as_of query should include joined topic metadata."""
        tid, old_id, new_id, t0, t1 = self._setup_timeline()

        query_time = (datetime.now(timezone.utc) - timedelta(days=25)).isoformat()
        records = get_records_as_of(query_time)

        assert len(records) >= 1
        for r in records:
            assert "topic_filename" in r
            assert "topic_title" in r
            assert r["topic_filename"] == "python_version.md"
            assert r["topic_title"] == "Python Version"

    def test_as_of_excludes_deleted_records(self, tmp_data_dir):
        """Soft-deleted records should not appear in as_of results."""
        ensure_schema()
        tid = upsert_knowledge_topic(
            filename="deleted_test.md", title="Deleted", summary="S",
            source_episodes=[],
        )
        ids = insert_knowledge_records(tid, [
            {"record_type": "fact", "content": {}, "embedding_text": "test"},
        ])

        # Soft-delete
        from consolidation_memory.database import soft_delete_records_by_ids
        soft_delete_records_by_ids(ids)

        now = datetime.now(timezone.utc).isoformat()
        records = get_records_as_of(now)
        assert all(r["id"] != ids[0] for r in records)

    def test_as_of_with_null_valid_until(self, tmp_data_dir):
        """Records with no valid_until (still current) should appear at any time after creation."""
        ensure_schema()
        tid = upsert_knowledge_topic(
            filename="current.md", title="Current", summary="S",
            source_episodes=[],
        )
        ids = insert_knowledge_records(tid, [
            {"record_type": "fact", "content": {}, "embedding_text": "current fact"},
        ])

        # Query at a future time -- should still appear
        future = (datetime.now(timezone.utc) + timedelta(days=365)).isoformat()
        records = get_records_as_of(future)
        assert any(r["id"] == ids[0] for r in records)


# ── Context assembler: _search_records with as_of ─────────────────────────

class TestSearchRecordsAsOf:
    """Test that _search_records uses temporal DB query when as_of is set."""

    def test_as_of_bypasses_cache(self, tmp_data_dir):
        """When as_of is set, should call get_records_as_of instead of record_cache."""
        from consolidation_memory.context_assembler import _search_records

        query_vec = np.ones(384, dtype=np.float32)
        query_vec = query_vec / np.linalg.norm(query_vec)

        mock_record = {
            "id": "rec-old",
            "record_type": "fact",
            "content": '{"subject": "X", "info": "old"}',
            "embedding_text": "X: old",
            "topic_title": "X",
            "topic_filename": "x.md",
            "confidence": 0.9,
            "source_episodes": "[]",
        }

        with (
            patch("consolidation_memory.context_assembler.get_records_as_of") as mock_db,
            patch("consolidation_memory.context_assembler.backends") as mock_backends,
            patch("consolidation_memory.context_assembler.increment_record_access"),
        ):
            mock_db.return_value = [mock_record]
            mock_backends.encode_documents.return_value = np.stack([query_vec])

            records, warnings = _search_records(
                "X", query_vec, as_of="2025-06-01T00:00:00+00:00",
            )

            mock_db.assert_called_once_with("2025-06-01T00:00:00+00:00")
            assert len(records) >= 1
            assert records[0]["id"] == "rec-old"

    def test_as_of_none_uses_cache(self, tmp_data_dir):
        """When as_of is not set, should use record_cache as before."""
        from consolidation_memory.context_assembler import _search_records

        query_vec = np.ones(384, dtype=np.float32)
        query_vec = query_vec / np.linalg.norm(query_vec)

        with (
            patch("consolidation_memory.context_assembler.record_cache") as mock_rc,
            patch("consolidation_memory.context_assembler.increment_record_access"),
        ):
            mock_rc.get_record_vecs.return_value = ([], None)
            records, warnings = _search_records("X", query_vec, as_of=None)
            mock_rc.get_record_vecs.assert_called_once()

    def test_as_of_embed_failure_returns_records_without_semantic(self, tmp_data_dir):
        """If embedding fails during temporal query, should degrade gracefully."""
        from consolidation_memory.context_assembler import _search_records

        query_vec = np.ones(384, dtype=np.float32)
        query_vec = query_vec / np.linalg.norm(query_vec)

        mock_record = {
            "id": "rec-old",
            "record_type": "fact",
            "content": '{"subject": "X", "info": "old"}',
            "embedding_text": "X old info",
            "topic_title": "X",
            "topic_filename": "x.md",
            "confidence": 0.9,
            "source_episodes": "[]",
        }

        with (
            patch("consolidation_memory.context_assembler.get_records_as_of") as mock_db,
            patch("consolidation_memory.context_assembler.backends") as mock_backends,
            patch("consolidation_memory.context_assembler.increment_record_access"),
        ):
            mock_db.return_value = [mock_record]
            mock_backends.encode_documents.side_effect = RuntimeError("backend down")

            # Should not raise -- degrades to keyword-only matching
            records, warnings = _search_records(
                "X old", query_vec, as_of="2025-06-01T00:00:00+00:00",
            )
            # With keyword matching, the record may or may not pass threshold,
            # but the function should not crash
            assert isinstance(records, list)


# ── Context assembler: _search_knowledge with as_of ──────────────────────

class TestSearchKnowledgeAsOf:
    """Test that _search_knowledge filters topics by as_of."""

    def test_as_of_filters_future_topics(self, tmp_data_dir):
        """Topics created after as_of should be excluded."""
        from consolidation_memory.context_assembler import _search_knowledge

        query_vec = np.ones(384, dtype=np.float32)
        query_vec = query_vec / np.linalg.norm(query_vec)

        old_topic = {
            "title": "Old Topic",
            "filename": "old.md",
            "summary": "test summary",
            "confidence": 0.9,
            "source_episodes": "[]",
            "created_at": "2025-01-01T00:00:00+00:00",
        }
        new_topic = {
            "title": "New Topic",
            "filename": "new.md",
            "summary": "test summary",
            "confidence": 0.9,
            "source_episodes": "[]",
            "created_at": "2025-07-01T00:00:00+00:00",
        }

        summary_vecs = np.stack([query_vec, query_vec])

        from consolidation_memory.config import get_config
        cfg = get_config()
        # Create dummy knowledge files
        (cfg.KNOWLEDGE_DIR / "old.md").write_text("old content", encoding="utf-8")
        (cfg.KNOWLEDGE_DIR / "new.md").write_text("new content", encoding="utf-8")

        with (
            patch("consolidation_memory.context_assembler.topic_cache") as mock_tc,
            patch("consolidation_memory.context_assembler.backends.encode_documents", return_value=np.stack([query_vec])),
            patch("consolidation_memory.context_assembler.increment_topic_access"),
            patch("consolidation_memory.context_assembler._apply_evolving_topic_signals"),
        ):
            mock_tc.get_topic_vecs.return_value = ([old_topic, new_topic], summary_vecs)

            # Query as of mid-2025 -- should exclude new_topic
            topics, warnings = _search_knowledge(
                "test", query_vec, as_of="2025-06-01T00:00:00+00:00",
            )

            titles = [t["title"] for t in topics]
            assert "Old Topic" in titles
            assert "New Topic" not in titles

    def test_as_of_none_returns_all_topics(self, tmp_data_dir):
        """When as_of is not set, all topics should be returned."""
        from consolidation_memory.context_assembler import _search_knowledge

        query_vec = np.ones(384, dtype=np.float32)
        query_vec = query_vec / np.linalg.norm(query_vec)

        topic = {
            "title": "Topic",
            "filename": "t.md",
            "summary": "test summary",
            "confidence": 0.9,
            "source_episodes": "[]",
            "created_at": "2025-01-01T00:00:00+00:00",
        }

        from consolidation_memory.config import get_config
        cfg = get_config()
        (cfg.KNOWLEDGE_DIR / "t.md").write_text("content", encoding="utf-8")

        with (
            patch("consolidation_memory.context_assembler.topic_cache") as mock_tc,
            patch("consolidation_memory.context_assembler.increment_topic_access"),
            patch("consolidation_memory.context_assembler._apply_evolving_topic_signals"),
        ):
            mock_tc.get_topic_vecs.return_value = ([topic], np.stack([query_vec]))

            topics, warnings = _search_knowledge("test", query_vec, as_of=None)
            assert len(topics) == 1

    def test_as_of_before_all_topics_returns_empty(self, tmp_data_dir):
        """Query before any topics existed should return empty."""
        from consolidation_memory.context_assembler import _search_knowledge

        query_vec = np.ones(384, dtype=np.float32)
        query_vec = query_vec / np.linalg.norm(query_vec)

        topic = {
            "title": "Topic",
            "filename": "t.md",
            "summary": "test",
            "confidence": 0.9,
            "source_episodes": "[]",
            "created_at": "2025-06-01T00:00:00+00:00",
        }

        with (
            patch("consolidation_memory.context_assembler.topic_cache") as mock_tc,
        ):
            mock_tc.get_topic_vecs.return_value = ([topic], np.stack([query_vec]))

            topics, warnings = _search_knowledge(
                "test", query_vec, as_of="2024-01-01T00:00:00+00:00",
            )
            assert topics == []

    def test_as_of_returns_historical_topic_revision(self, tmp_data_dir):
        """Topics updated after as_of should use the archived revision."""
        from consolidation_memory.consolidation.engine import _version_knowledge_file
        from consolidation_memory.context_assembler import _search_knowledge
        from consolidation_memory.database import ensure_schema, upsert_knowledge_topic
        from consolidation_memory.knowledge_paths import resolve_topic_path

        ensure_schema()
        query_vec = np.ones(384, dtype=np.float32)
        query_vec = query_vec / np.linalg.norm(query_vec)

        from consolidation_memory.config import get_config
        cfg = get_config()

        old_content = (
            "---\n"
            "title: Topic\n"
            "summary: old summary\n"
            "confidence: 0.8\n"
            "---\n\n"
            "old content\n"
        )
        new_content = (
            "---\n"
            "title: Topic\n"
            "summary: new summary\n"
            "confidence: 0.8\n"
            "---\n\n"
            "new content\n"
        )

        topic_id = upsert_knowledge_topic(
            filename="topic.md",
            title="Topic",
            summary="old summary",
            source_episodes=[],
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
        )
        topic_row = {
            "id": topic_id,
            "title": "Topic",
            "filename": "topic.md",
            "summary": "new summary",
            "confidence": 0.8,
            "source_episodes": "[]",
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-02-01T00:00:00+00:00",
        }

        filepath = resolve_topic_path(cfg.KNOWLEDGE_DIR, topic_row)
        filepath.write_text(old_content, encoding="utf-8")
        _version_knowledge_file(filepath)
        filepath.write_text(new_content, encoding="utf-8")
        upsert_knowledge_topic(
            filename="topic.md",
            title="Topic",
            summary="new summary",
            source_episodes=[],
            topic_id=topic_id,
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-02-01T00:00:00+00:00",
        )

        with (
            patch("consolidation_memory.context_assembler.topic_cache") as mock_tc,
            patch("consolidation_memory.context_assembler.backends.encode_documents", return_value=np.stack([query_vec])),
            patch("consolidation_memory.context_assembler.increment_topic_access"),
            patch("consolidation_memory.context_assembler._apply_evolving_topic_signals"),
        ):
            mock_tc.get_topic_vecs.return_value = ([topic_row], np.stack([query_vec]))

            topics, warnings = _search_knowledge(
                "topic",
                query_vec,
                as_of="2026-01-15T00:00:00+00:00",
            )

        assert len(topics) == 1
        assert topics[0]["summary"] == "old summary"
        assert "old content" in topics[0]["content"]
        assert "new content" not in topics[0]["content"]


# ── Context assembler: recall with as_of ─────────────────────────────────

class TestRecallAsOf:
    """Test the full recall pipeline with as_of."""

    def test_as_of_caps_episode_before_filter(self, tmp_data_dir):
        """as_of should act as a 'before' cap on episodes."""
        from consolidation_memory.context_assembler import recall
        from consolidation_memory.vector_store import VectorStore
        from consolidation_memory.database import ensure_schema, insert_episode

        ensure_schema()
        vs = VectorStore()

        # Create two episodes: one before as_of, one after
        ep1_id = insert_episode(
            content="old fact", content_type="fact", tags=[], surprise_score=0.5,
        )
        ep2_id = insert_episode(
            content="new fact", content_type="fact", tags=[], surprise_score=0.5,
        )

        vec1 = mock_encode(["old fact"])[0]
        vec2 = mock_encode(["new fact"])[0]
        vs.add(ep1_id, vec1)
        vs.add(ep2_id, vec2)

        # Backdate ep1 to 30 days ago
        from consolidation_memory.database import get_connection
        old_time = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        with get_connection() as conn:
            conn.execute(
                "UPDATE episodes SET created_at = ? WHERE id = ?",
                (old_time, ep1_id),
            )

        # Query with as_of = 15 days ago -- should only return ep1
        as_of_time = (datetime.now(timezone.utc) - timedelta(days=15)).isoformat()

        with patch("consolidation_memory.context_assembler.backends") as mock_be:
            mock_be.encode_query.return_value = mock_encode(["fact"])[0]

            result = recall(
                "fact", n_results=10, include_knowledge=False,
                vector_store=vs, as_of=as_of_time,
            )

        ep_ids = {e["id"] for e in result["episodes"]}
        assert ep1_id in ep_ids, "Old episode should appear"
        assert ep2_id not in ep_ids, "New episode (after as_of) should be excluded"

    def test_as_of_respects_explicit_before(self, tmp_data_dir):
        """If 'before' is explicitly set and earlier than as_of, use 'before'."""
        from consolidation_memory.context_assembler import recall
        from consolidation_memory.vector_store import VectorStore
        from consolidation_memory.database import ensure_schema, insert_episode

        ensure_schema()
        vs = VectorStore()

        ep_id = insert_episode(
            content="test", content_type="fact", tags=[], surprise_score=0.5,
        )
        vec = mock_encode(["test"])[0]
        vs.add(ep_id, vec)

        # Backdate episode
        from consolidation_memory.database import get_connection
        old_time = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        with get_connection() as conn:
            conn.execute(
                "UPDATE episodes SET created_at = ? WHERE id = ?",
                (old_time, ep_id),
            )

        # as_of = 10 days ago, but before = 20 days ago (tighter constraint)
        as_of_time = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        before_time = (datetime.now(timezone.utc) - timedelta(days=20)).isoformat()

        with patch("consolidation_memory.context_assembler.backends") as mock_be:
            mock_be.encode_query.return_value = mock_encode(["test"])[0]

            # before_time (20 days ago) is earlier than as_of (10 days ago)
            # and episode is from 30 days ago, so it should still appear
            # because created_at (30d ago) < before_time (20d ago)
            result = recall(
                "test", n_results=10, include_knowledge=False,
                vector_store=vs, before=before_time, as_of=as_of_time,
            )

        # Episode at 30d ago < before at 20d ago, so it should appear
        ep_ids = {e["id"] for e in result["episodes"]}
        assert ep_id in ep_ids

    def test_as_of_none_has_no_effect(self, tmp_data_dir):
        """When as_of is None, recall behaves normally."""
        from consolidation_memory.context_assembler import recall
        from consolidation_memory.vector_store import VectorStore
        from consolidation_memory.database import ensure_schema, insert_episode

        ensure_schema()
        vs = VectorStore()

        ep_id = insert_episode(
            content="normal recall", content_type="fact", tags=[], surprise_score=0.5,
        )
        vec = mock_encode(["normal recall"])[0]
        vs.add(ep_id, vec)

        with patch("consolidation_memory.context_assembler.backends") as mock_be:
            mock_be.encode_query.return_value = mock_encode(["normal"])[0]

            result = recall(
                "normal", n_results=10, include_knowledge=False,
                vector_store=vs, as_of=None,
            )

        assert len(result["episodes"]) >= 1


# ── Schema and dispatch ──────────────────────────────────────────────────

class TestSchemaAsOf:
    """Test that as_of is present in the OpenAI schema and dispatches correctly."""

    def test_recall_schema_has_as_of(self):
        from consolidation_memory.schemas import MEMORY_RECALL_SCHEMA
        props = MEMORY_RECALL_SCHEMA["function"]["parameters"]["properties"]
        assert "as_of" in props
        assert props["as_of"]["type"] == "string"

    def test_dispatch_passes_as_of(self):
        from consolidation_memory.schemas import dispatch_tool_call

        client = MagicMock()
        client.query_recall.return_value = RecallResult()

        dispatch_tool_call(client, "memory_recall", {
            "query": "test",
            "as_of": "2025-06-01T00:00:00+00:00",
        })

        client.query_recall.assert_called_once()
        call_kwargs = client.query_recall.call_args
        assert call_kwargs.kwargs.get("as_of") == "2025-06-01T00:00:00+00:00"

    def test_dispatch_as_of_defaults_to_none(self):
        from consolidation_memory.schemas import dispatch_tool_call

        client = MagicMock()
        client.query_recall.return_value = RecallResult()

        dispatch_tool_call(client, "memory_recall", {"query": "test"})

        call_kwargs = client.query_recall.call_args
        assert call_kwargs.kwargs.get("as_of") is None


# ── MCP server tool ──────────────────────────────────────────────────────

class TestMCPServerAsOf:
    """Test that the MCP server tool accepts as_of."""

    def test_memory_recall_signature_has_as_of(self):
        import inspect
        from consolidation_memory.server import memory_recall
        sig = inspect.signature(memory_recall)
        assert "as_of" in sig.parameters
        assert sig.parameters["as_of"].default is None


# ── REST API ─────────────────────────────────────────────────────────────

try:
    from consolidation_memory.rest import RecallRequest  # noqa: F401
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


@pytest.mark.skipif(not HAS_FASTAPI, reason="fastapi not installed")
class TestRESTAsOf:
    """Test that the REST API accepts as_of."""

    def test_recall_request_has_as_of(self):
        from consolidation_memory.rest import RecallRequest
        req = RecallRequest(query="test", as_of="2025-06-01T00:00:00+00:00")
        assert req.as_of == "2025-06-01T00:00:00+00:00"

    def test_recall_request_as_of_defaults_none(self):
        from consolidation_memory.rest import RecallRequest
        req = RecallRequest(query="test")
        assert req.as_of is None

    def test_recall_request_include_expired_defaults_false(self):
        from consolidation_memory.rest import RecallRequest
        req = RecallRequest(query="test")
        assert req.include_expired is False
