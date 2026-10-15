"""Tests for the shared utils module.

Run with: python -m pytest tests/test_utils.py -v
"""

from datetime import datetime, timezone

import pytest

from consolidation_memory.utils import parse_datetime, parse_json_list

# ── parse_json_list ──────────────────────────────────────────────────────────


class TestParseJsonList:
    """Tests for parse_json_list — JSON string/list/None normalization."""

    def test_valid_json_string(self):
        assert parse_json_list('["a", "b", "c"]') == ["a", "b", "c"]

    def test_empty_json_array(self):
        assert parse_json_list("[]") == []

    def test_already_a_list(self):
        assert parse_json_list(["x", "y"]) == ["x", "y"]

    def test_empty_list_passthrough(self):
        assert parse_json_list([]) == []

    def test_none_returns_default(self):
        assert parse_json_list(None) == []

    def test_none_with_custom_default(self):
        sentinel = ["fallback"]
        assert parse_json_list(None, default=sentinel) is sentinel

    def test_invalid_json_returns_default(self):
        assert parse_json_list("not valid json") == []

    def test_invalid_json_with_custom_default(self):
        assert parse_json_list("{broken", default=["fb"]) == ["fb"]

    def test_json_object_returns_default(self):
        """A valid JSON dict is not a list — should return default."""
        assert parse_json_list('{"key": "value"}') == []

    def test_json_string_scalar_returns_default(self):
        """A bare JSON string is not a list."""
        assert parse_json_list('"hello"') == []

    def test_json_number_returns_default(self):
        """A bare JSON number is not a list."""
        assert parse_json_list("42") == []

    def test_nested_list(self):
        assert parse_json_list('[["a", "b"], "c"]') == [["a", "b"], "c"]

    def test_list_of_ints(self):
        assert parse_json_list("[1, 2, 3]") == [1, 2, 3]

    def test_default_none_gives_empty_list(self):
        """When default=None is passed explicitly, we still get []."""
        assert parse_json_list(None, default=None) == []


# ── parse_datetime ───────────────────────────────────────────────────────────


class TestParseDatetime:
    """Tests for parse_datetime — ISO datetime parsing with UTC normalization."""

    def test_naive_datetime_gets_utc(self):
        dt = parse_datetime("2025-01-15T10:30:00")
        assert dt.tzinfo is timezone.utc
        assert dt.year == 2025
        assert dt.month == 1
        assert dt.day == 15
        assert dt.hour == 10
        assert dt.minute == 30

    def test_aware_datetime_preserved(self):
        dt = parse_datetime("2025-01-15T10:30:00+00:00")
        assert dt.tzinfo is not None
        assert dt == datetime(2025, 1, 15, 10, 30, tzinfo=timezone.utc)

    def test_non_utc_offset_preserved(self):
        dt = parse_datetime("2025-01-15T10:30:00+05:00")
        assert dt.tzinfo is not None
        # The offset is preserved, not replaced with UTC
        assert dt.utcoffset().total_seconds() == 5 * 3600

    def test_date_only_string(self):
        dt = parse_datetime("2025-01-15")
        assert dt.tzinfo is timezone.utc
        assert dt.year == 2025
        assert dt.month == 1
        assert dt.day == 15

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError):
            parse_datetime("not-a-datetime")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            parse_datetime("")

    def test_microseconds_preserved(self):
        dt = parse_datetime("2025-06-15T12:00:00.123456")
        assert dt.microsecond == 123456
        assert dt.tzinfo is timezone.utc

    def test_z_suffix(self):
        """Python 3.11+ fromisoformat supports 'Z' suffix."""
        import sys

        if sys.version_info < (3, 11):
            pytest.skip("Python < 3.11 does not support Z suffix in fromisoformat")
        dt = parse_datetime("2025-06-15T12:00:00Z")
        assert dt.tzinfo is not None

    def test_result_is_timezone_aware(self):
        """The return value is always timezone-aware."""
        dt = parse_datetime("2024-12-25T00:00:00")
        assert dt.tzinfo is not None
        assert dt.utcoffset() is not None
