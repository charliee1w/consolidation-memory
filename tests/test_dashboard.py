"""Tests for dashboard helpers and no-textual fallback behavior."""

from __future__ import annotations

import pytest

import consolidation_memory.dashboard as dashboard


def test_fmt_ts_handles_none_and_iso_values():
    assert dashboard._fmt_ts(None) == "-"
    assert dashboard._fmt_ts("2026-03-08T12:34:56.999+00:00") == "2026-03-08 12:34:56"


def test_dashboard_app_fallback_raises_clear_import_error_without_textual():
    if getattr(dashboard, "_TEXTUAL_AVAILABLE", False):
        pytest.skip("textual installed; fallback path not active")
    with pytest.raises(ImportError, match="Dashboard requires textual"):
        dashboard.DashboardApp()
