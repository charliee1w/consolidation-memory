"""Tests for native desktop app backend and import guards."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from consolidation_memory.desktop_backend import DesktopBackend, build_health_snapshot


class TestBuildHealthSnapshot:
    def test_ok_when_no_last_run(self):
        health, note = build_health_snapshot(None)
        assert health == "ok"
        assert "ready" in note.lower()

    def test_warning_on_error_status(self):
        health, note = build_health_snapshot({"status": "error"})
        assert health == "warning"
        assert "error" in note.lower()


class TestDesktopBackend:
    def test_overview_shape(self):
        data = MagicMock()
        data.get_stats.return_value = {
            "total_episodes": 3,
            "last_consolidation": None,
        }
        data.get_faiss_stats.return_value = {"index_size": 2}
        backend = DesktopBackend(data=data)
        with patch(
            "consolidation_memory.desktop_backend.get_active_project",
            return_value="default",
        ):
            overview = backend.overview()
        assert overview["health"] == "ok"
        assert overview["stats"]["total_episodes"] == 3
        assert overview["faiss"]["index_size"] == 2

    def test_ask_delegates_to_tool_dispatch(self):
        backend = DesktopBackend(data=MagicMock())
        with patch(
            "consolidation_memory.desktop_backend.execute_tool_call",
            return_value={"episodes": []},
        ) as execute:
            result = backend.ask("auth bug", n_results=5)
        execute.assert_called_once_with(
            "memory_ask",
            {"query": "auth bug", "n_results": 5},
        )
        assert result == {"episodes": []}

    def test_remember_defaults_desktop_tag(self):
        backend = DesktopBackend(data=MagicMock())
        with patch(
            "consolidation_memory.desktop_backend.execute_tool_call",
            return_value={"episode_id": "ep-1"},
        ) as execute:
            backend.remember("saved note", kind="fact")
        execute.assert_called_once_with(
            "memory_remember",
            {"content": "saved note", "kind": "fact", "tags": ["desktop"]},
        )


class TestDesktopImportGuards:
    def test_run_desktop_app_raises_without_pyside6(self):
        import consolidation_memory.desktop_app as desktop_app

        original = desktop_app._PYSIDE_AVAILABLE
        desktop_app._PYSIDE_AVAILABLE = False
        try:
            with pytest.raises(ImportError, match="PySide6"):
                desktop_app.run_desktop_app()
        finally:
            desktop_app._PYSIDE_AVAILABLE = original

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("PySide6"),
        reason="PySide6 not installed",
    )
    def test_build_app_icon_with_pyside6(self):
        from consolidation_memory.desktop_app import build_app_icon

        icon = build_app_icon()
        assert not icon.isNull()