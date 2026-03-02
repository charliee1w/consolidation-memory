"""Tests for source traceability in recall results (Phase 3.1).

Knowledge records and topics should include human-readable source summaries
indicating which episodes they were derived from and when.

Run with: python -m pytest tests/test_source_traceability.py -v
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from consolidation_memory.context_assembler import (
    _format_source_dates,
    _enrich_source_traceability,
)

FIXED_NOW = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)


class TestFormatSourceDates:
    """Unit tests for _format_source_dates()."""

    def test_empty_dates(self):
        assert _format_source_dates([]) == ""

    def test_single_date_current_year(self):
        with patch("consolidation_memory.context_assembler.datetime") as mock_dt:
            mock_dt.now.return_value = FIXED_NOW
            mock_dt.fromisoformat = datetime.fromisoformat
            iso = FIXED_NOW.isoformat()
            result = _format_source_dates([iso])
            assert result.startswith("Based on 1 conversation (")
            assert FIXED_NOW.strftime("%b %d") in result

    def test_multiple_dates(self):
        with patch("consolidation_memory.context_assembler.datetime") as mock_dt:
            mock_dt.now.return_value = FIXED_NOW
            mock_dt.fromisoformat = datetime.fromisoformat
            d1 = FIXED_NOW.isoformat()
            d2 = (FIXED_NOW - timedelta(days=5)).isoformat()
            result = _format_source_dates([d1, d2])
            assert result.startswith("Based on 2 conversations (")

    def test_past_year_includes_year(self):
        with patch("consolidation_memory.context_assembler.datetime") as mock_dt:
            mock_dt.now.return_value = FIXED_NOW
            mock_dt.fromisoformat = datetime.fromisoformat
            old = datetime(2023, 6, 15, tzinfo=timezone.utc).isoformat()
            result = _format_source_dates([old])
            assert "2023" in result

    def test_deduplicates_same_day(self):
        with patch("consolidation_memory.context_assembler.datetime") as mock_dt:
            mock_dt.now.return_value = FIXED_NOW
            mock_dt.fromisoformat = datetime.fromisoformat
            d1 = FIXED_NOW.isoformat()
            d2 = (FIXED_NOW + timedelta(hours=2)).isoformat()  # same day
            result = _format_source_dates([d1, d2])
            # Should say 2 conversations but only show 1 unique date
            assert "Based on 2 conversations" in result

    def test_invalid_dates_ignored(self):
        assert _format_source_dates(["not-a-date"]) == ""

    def test_mixed_valid_invalid(self):
        with patch("consolidation_memory.context_assembler.datetime") as mock_dt:
            mock_dt.now.return_value = FIXED_NOW
            mock_dt.fromisoformat = datetime.fromisoformat
            result = _format_source_dates([FIXED_NOW.isoformat(), "bad-date"])
            assert result.startswith("Based on 1 conversation (")

    def test_many_dates_truncated(self):
        """More than 5 unique dates should show (+N more)."""
        with patch("consolidation_memory.context_assembler.datetime") as mock_dt:
            mock_dt.now.return_value = FIXED_NOW
            mock_dt.fromisoformat = datetime.fromisoformat
            dates = [
                (FIXED_NOW - timedelta(days=i * 30)).isoformat()
                for i in range(8)
            ]
            result = _format_source_dates(dates)
            assert "+3 more" in result

    def test_naive_datetime_handled(self):
        """Naive datetime (no timezone) should not crash."""
        with patch("consolidation_memory.context_assembler.datetime") as mock_dt:
            mock_dt.now.return_value = FIXED_NOW
            mock_dt.fromisoformat = datetime.fromisoformat
            naive = "2025-03-15T10:30:00"
            result = _format_source_dates([naive])
            assert result.startswith("Based on 1 conversation")


class TestEnrichSourceTraceability:
    """Unit tests for _enrich_source_traceability()."""

    def test_no_source_episodes(self):
        records = [{"id": "r1", "source_episodes": []}]
        result = _enrich_source_traceability(records)
        assert result[0]["source_summary"] == ""
        assert result[0]["source_dates"] == []

    def test_missing_source_episodes_key(self):
        records = [{"id": "r1"}]
        result = _enrich_source_traceability(records)
        assert len(result) == 1

    def test_enriches_with_dates(self, tmp_data_dir):
        """When source episodes exist in DB, dates are populated."""
        from consolidation_memory.database import ensure_schema, insert_episode

        ensure_schema()
        eid = insert_episode("test content", "fact")

        records = [{"id": "r1", "source_episodes": [eid]}]
        result = _enrich_source_traceability(records)

        assert len(result[0]["source_dates"]) == 1
        assert result[0]["source_summary"].startswith("Based on 1 conversation")

    def test_missing_episode_graceful(self, tmp_data_dir):
        """Episodes that no longer exist should not crash."""
        from consolidation_memory.database import ensure_schema

        ensure_schema()
        records = [{"id": "r1", "source_episodes": ["nonexistent-id"]}]
        result = _enrich_source_traceability(records)
        assert result[0]["source_dates"] == []
        assert result[0]["source_summary"] == ""

    def test_multiple_records_batch_fetch(self, tmp_data_dir):
        """Multiple records should batch-fetch all source episodes."""
        from consolidation_memory.database import ensure_schema, insert_episode

        ensure_schema()
        eid1 = insert_episode("content 1", "fact")
        eid2 = insert_episode("content 2", "fact")

        records = [
            {"id": "r1", "source_episodes": [eid1]},
            {"id": "r2", "source_episodes": [eid2]},
        ]
        result = _enrich_source_traceability(records)

        assert len(result[0]["source_dates"]) == 1
        assert len(result[1]["source_dates"]) == 1
        assert result[0]["source_summary"].startswith("Based on 1")
        assert result[1]["source_summary"].startswith("Based on 1")
