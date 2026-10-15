"""Tests for uncertainty signaling in recall results (Phase 3.3).

Tests that low-confidence records get flagged with warnings and that
topics with recent contradictions are marked as "evolving".
"""

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from consolidation_memory.config import reset_config
from consolidation_memory.context_assembler import (
    _EVOLVING_TOPIC_WARNING,
    _LOW_CONFIDENCE_THRESHOLD,
    _LOW_CONFIDENCE_WARNING,
    _apply_evolving_topic_signals,
    _apply_uncertainty_signals,
)
from consolidation_memory.database import (
    ensure_schema,
    get_recently_contradicted_topic_ids,
    insert_contradiction,
    upsert_knowledge_topic,
)

# ── Low-confidence record signaling ──────────────────────────────────────────


class TestApplyUncertaintySignals:
    """Test _apply_uncertainty_signals flags low-confidence records."""

    def test_no_records(self):
        records: list[dict] = []
        warnings: list[str] = []
        _apply_uncertainty_signals(records, warnings)
        assert warnings == []

    def test_all_high_confidence(self):
        records = [
            {"id": "1", "confidence": 0.9},
            {"id": "2", "confidence": 0.8},
            {"id": "3", "confidence": 0.7},
        ]
        warnings: list[str] = []
        _apply_uncertainty_signals(records, warnings)
        assert warnings == []
        assert all("uncertainty" not in r for r in records)

    def test_low_confidence_flagged(self):
        records = [
            {"id": "1", "confidence": 0.3},
            {"id": "2", "confidence": 0.5},
        ]
        warnings: list[str] = []
        _apply_uncertainty_signals(records, warnings)
        assert records[0]["uncertainty"] == _LOW_CONFIDENCE_WARNING
        assert records[1]["uncertainty"] == _LOW_CONFIDENCE_WARNING
        assert len(warnings) == 1
        assert "2 records" in warnings[0]
        assert "low confidence" in warnings[0]

    def test_mixed_confidence(self):
        records = [
            {"id": "1", "confidence": 0.9},
            {"id": "2", "confidence": 0.4},
            {"id": "3", "confidence": 0.8},
        ]
        warnings: list[str] = []
        _apply_uncertainty_signals(records, warnings)
        assert "uncertainty" not in records[0]
        assert records[1]["uncertainty"] == _LOW_CONFIDENCE_WARNING
        assert "uncertainty" not in records[2]
        assert len(warnings) == 1
        assert "1 record has" in warnings[0]

    def test_threshold_boundary(self):
        """Confidence exactly at threshold should NOT be flagged."""
        records = [{"id": "1", "confidence": _LOW_CONFIDENCE_THRESHOLD}]
        warnings: list[str] = []
        _apply_uncertainty_signals(records, warnings)
        assert "uncertainty" not in records[0]
        assert warnings == []

    def test_just_below_threshold(self):
        """Confidence just below threshold should be flagged."""
        records = [{"id": "1", "confidence": _LOW_CONFIDENCE_THRESHOLD - 0.01}]
        warnings: list[str] = []
        _apply_uncertainty_signals(records, warnings)
        assert records[0]["uncertainty"] == _LOW_CONFIDENCE_WARNING
        assert len(warnings) == 1

    def test_default_confidence_not_flagged(self):
        """Records with missing confidence default to 0.8 (not flagged)."""
        records = [{"id": "1"}]
        warnings: list[str] = []
        _apply_uncertainty_signals(records, warnings)
        assert "uncertainty" not in records[0]
        assert warnings == []


# ── Evolving topic signaling ─────────────────────────────────────────────────


class TestApplyEvolvingTopicSignals:
    """Test _apply_evolving_topic_signals flags topics with recent contradictions."""

    def test_no_topics(self):
        warnings: list[str] = []
        _apply_evolving_topic_signals([], warnings)
        assert warnings == []

    def test_no_contradictions(self):
        topics = [{"filename": "foo.md", "title": "Foo"}]
        warnings: list[str] = []
        with patch(
            "consolidation_memory.context_assembler.get_recently_contradicted_topic_ids",
            return_value=set(),
        ):
            _apply_evolving_topic_signals(topics, warnings)
        assert warnings == []
        assert "uncertainty" not in topics[0]

    def test_topic_with_contradictions_flagged(self):
        topic_id = "topic-123"
        topics = [{"filename": "foo.md", "title": "Foo"}]
        warnings: list[str] = []
        with patch(
            "consolidation_memory.context_assembler.get_recently_contradicted_topic_ids",
            return_value={topic_id},
        ), patch(
            "consolidation_memory.database.get_all_knowledge_topics",
            return_value=[{"filename": "foo.md", "id": topic_id, "title": "Foo"}],
        ):
            _apply_evolving_topic_signals(topics, warnings)
        assert topics[0]["uncertainty"] == _EVOLVING_TOPIC_WARNING
        assert len(warnings) == 1
        assert "evolving" in warnings[0]

    def test_unmatched_topic_not_flagged(self):
        """Topic whose ID is not in the contradicted set stays clean."""
        topics = [{"filename": "bar.md", "title": "Bar"}]
        warnings: list[str] = []
        with patch(
            "consolidation_memory.context_assembler.get_recently_contradicted_topic_ids",
            return_value={"other-topic-id"},
        ), patch(
            "consolidation_memory.database.get_all_knowledge_topics",
            return_value=[{"filename": "bar.md", "id": "bar-topic-id", "title": "Bar"}],
        ):
            _apply_evolving_topic_signals(topics, warnings)
        assert "uncertainty" not in topics[0]
        assert warnings == []

    def test_mixed_topics(self):
        """Only the contradicted topic gets flagged, not the other."""
        topics = [
            {"filename": "evolving.md", "title": "Evolving"},
            {"filename": "stable.md", "title": "Stable"},
        ]
        warnings: list[str] = []
        with patch(
            "consolidation_memory.context_assembler.get_recently_contradicted_topic_ids",
            return_value={"evolving-id"},
        ), patch(
            "consolidation_memory.database.get_all_knowledge_topics",
            return_value=[
                {"filename": "evolving.md", "id": "evolving-id", "title": "Evolving"},
                {"filename": "stable.md", "id": "stable-id", "title": "Stable"},
            ],
        ):
            _apply_evolving_topic_signals(topics, warnings)
        assert topics[0]["uncertainty"] == _EVOLVING_TOPIC_WARNING
        assert "uncertainty" not in topics[1]
        assert "1 topic" in warnings[0]

    def test_db_error_graceful(self):
        """If the contradiction lookup fails, no crash and no flags."""
        topics = [{"filename": "foo.md", "title": "Foo"}]
        warnings: list[str] = []
        with patch(
            "consolidation_memory.context_assembler.get_recently_contradicted_topic_ids",
            side_effect=OSError("db locked"),
        ):
            _apply_evolving_topic_signals(topics, warnings)
        assert "uncertainty" not in topics[0]
        assert warnings == []


# ── Database: get_recently_contradicted_topic_ids ────────────────────────────


class TestGetRecentlyContradictedTopicIds:
    """Test the DB function that finds topics with recent contradictions."""

    def test_empty_db(self, tmp_path):
        reset_config(_base_data_dir=tmp_path)
        ensure_schema()
        result = get_recently_contradicted_topic_ids(days=30)
        assert result == set()

    def test_recent_contradiction_found(self, tmp_path):
        reset_config(_base_data_dir=tmp_path)
        ensure_schema()
        topic_id = str(uuid.uuid4())
        upsert_knowledge_topic("test.md", "Test", "summary", [], confidence=0.8)
        insert_contradiction(
            topic_id=topic_id,
            old_record_id=str(uuid.uuid4()),
            new_record_id=None,
            old_content="old fact",
            new_content="new fact",
            resolution="replaced",
        )
        result = get_recently_contradicted_topic_ids(days=30)
        assert topic_id in result

    def test_old_contradiction_excluded(self, tmp_path):
        reset_config(_base_data_dir=tmp_path)
        ensure_schema()
        topic_id = str(uuid.uuid4())
        # Insert a contradiction with an old detected_at date
        cid = str(uuid.uuid4())
        old_date = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
        from consolidation_memory.database import get_connection
        with get_connection() as conn:
            conn.execute(
                """INSERT INTO contradiction_log
                   (id, topic_id, old_record_id, new_record_id,
                    old_content, new_content, resolution, reason, detected_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (cid, topic_id, str(uuid.uuid4()), None,
                 "old", "new", "replaced", "", old_date),
            )
        result = get_recently_contradicted_topic_ids(days=30)
        assert topic_id not in result

    def test_multiple_topics(self, tmp_path):
        reset_config(_base_data_dir=tmp_path)
        ensure_schema()
        tid1 = str(uuid.uuid4())
        tid2 = str(uuid.uuid4())
        insert_contradiction(
            topic_id=tid1,
            old_record_id=str(uuid.uuid4()),
            new_record_id=None,
            old_content="a", new_content="b", resolution="replaced",
        )
        insert_contradiction(
            topic_id=tid2,
            old_record_id=str(uuid.uuid4()),
            new_record_id=None,
            old_content="c", new_content="d", resolution="replaced",
        )
        result = get_recently_contradicted_topic_ids(days=30)
        assert tid1 in result
        assert tid2 in result
