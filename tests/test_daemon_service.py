"""Tests for background maintenance daemon helpers."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch


class TestDaemonNaming:
    def test_task_name_sanitizes_project(self):
        from consolidation_memory.daemon_service import daemon_task_name

        assert daemon_task_name("universal") == "consolidation-memory-maintenance-universal"
        assert "my/project" not in daemon_task_name("my/project")
        assert daemon_task_name("my/project").endswith("my-project")


class TestDaemonWrapper:
    def test_write_wrapper_script_windows(self, tmp_path, monkeypatch):
        from consolidation_memory.config import reset_config
        from consolidation_memory.daemon_service import _write_wrapper_script

        cfg = reset_config(_base_data_dir=tmp_path / "data")
        cfg.LOG_DIR.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(os, "name", "nt", raising=False)

        wrapper = _write_wrapper_script(project="default")
        assert wrapper.suffix == ".cmd"
        assert wrapper.is_file()
        text = wrapper.read_text(encoding="utf-8")
        assert "consolidation_memory" in text
        assert "daemon" in text

    def test_write_wrapper_script_unix(self, tmp_path, monkeypatch):
        from consolidation_memory.config import reset_config
        from consolidation_memory.daemon_service import _write_wrapper_script

        cfg = reset_config(_base_data_dir=tmp_path / "data")
        cfg.LOG_DIR.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(os, "name", "posix", raising=False)

        wrapper = _write_wrapper_script(project="default")
        assert wrapper.suffix == ".sh"
        assert wrapper.is_file()
        text = wrapper.read_text(encoding="utf-8")
        assert text.startswith("#!/bin/sh")
        assert "consolidation_memory" in text


class TestDaemonStatus:
    def test_daemon_status_reports_not_installed(self, tmp_path):
        from consolidation_memory.config import reset_config
        from consolidation_memory.daemon_service import daemon_status

        reset_config(_base_data_dir=tmp_path / "data")
        with patch(
            "consolidation_memory.daemon_service._is_daemon_installed",
            return_value=False,
        ), patch(
            "consolidation_memory.daemon_service.is_daemon_running",
            return_value=False,
        ):
            status = daemon_status(project="default")

        assert status["installed"] is False
        assert status["running"] is False
        assert status["scheduler_enabled"] is True


class TestDaemonInstall:
    def test_install_windows_falls_back_to_startup_on_access_denied(self, tmp_path, monkeypatch):
        from consolidation_memory.config import reset_config
        from consolidation_memory.daemon_service import install_daemon

        reset_config(_base_data_dir=tmp_path / "data")
        monkeypatch.setattr(
            "consolidation_memory.daemon_service._platform_key",
            lambda: "windows",
        )
        startup_dir = tmp_path / "Startup"
        startup_dir.mkdir(parents=True)
        monkeypatch.setattr(
            "consolidation_memory.daemon_service._windows_startup_folder",
            lambda: startup_dir,
        )

        def fake_run(cmd, **kwargs):
            result = MagicMock()
            if cmd[:2] == ["schtasks", "/Create"]:
                result.returncode = 1
                result.stdout = ""
                result.stderr = "ERROR: Access is denied.\r\n"
            else:
                result.returncode = 0
                result.stdout = ""
                result.stderr = ""
            return result

        with patch("consolidation_memory.daemon_service.subprocess.run", side_effect=fake_run):
            result = install_daemon(project="default")

        assert result["status"] == "installed"
        assert result["install_method"] == "startup_folder"
        assert (startup_dir / "consolidation-memory-maintenance-default.cmd").is_file()

    def test_install_windows_success(self, tmp_path, monkeypatch):
        from consolidation_memory.config import reset_config
        from consolidation_memory.daemon_service import install_daemon

        reset_config(_base_data_dir=tmp_path / "data")
        monkeypatch.setattr(
            "consolidation_memory.daemon_service._platform_key",
            lambda: "windows",
        )

        def fake_run(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""
            return result

        with patch("consolidation_memory.daemon_service.subprocess.run", side_effect=fake_run):
            result = install_daemon(project="default")

        assert result["status"] == "installed"
        assert result["platform"] == "windows"
        assert "task_name" in result


class TestDaemonRun:
    def test_run_daemon_exits_when_lock_held(self, tmp_path):
        from consolidation_memory.config import reset_config
        from consolidation_memory.daemon_service import run_daemon

        reset_config(_base_data_dir=tmp_path / "data")

        from contextlib import contextmanager

        class _HeldLease:
            def __init__(self, *args, **kwargs):
                pass

            @contextmanager
            def acquire(self):
                raise TimeoutError("held")
                yield  # pragma: no cover

        with patch(
            "consolidation_memory.process_write_lock.ProcessWriteLease",
            _HeldLease,
        ):
            result = run_daemon()

        assert result["status"] == "already_running"


class TestDaemonCLI:
    def test_daemon_status_json(self, capsys, tmp_path):
        from consolidation_memory.cli import cmd_daemon_status
        from consolidation_memory.config import reset_config

        reset_config(_base_data_dir=tmp_path / "data")
        with patch(
            "consolidation_memory.daemon_service.daemon_status",
            return_value={
                "project": "default",
                "installed": False,
                "running": False,
                "scheduler_enabled": True,
                "message": "Maintenance daemon is not installed.",
            },
        ):
            cmd_daemon_status(as_json=True)

        captured = capsys.readouterr()
        assert '"installed": false' in captured.out

    def test_daemon_subcommand_registered(self):
        from consolidation_memory.cli import main

        with (
            patch("sys.argv", ["consolidation-memory", "daemon", "status", "--json"]),
            patch("consolidation_memory.cli.cmd_daemon_status") as mock_cmd,
        ):
            main()
            mock_cmd.assert_called_once_with(as_json=True)