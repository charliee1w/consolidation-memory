"""Tests for isolated drift detection subprocess execution."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest


class _FakeProcess:
    def __init__(
        self,
        *,
        returncode: int = 0,
        stdout: bytes = b"",
        stderr: bytes = b"",
        delay_seconds: float = 0.0,
    ) -> None:
        self.returncode = returncode
        self._stdout = stdout
        self._stderr = stderr
        self._delay_seconds = delay_seconds
        self.killed = False
        self.waited = False

    async def communicate(self) -> tuple[bytes, bytes]:
        if self._delay_seconds > 0:
            await asyncio.sleep(self._delay_seconds)
        return self._stdout, self._stderr

    def kill(self) -> None:
        self.killed = True

    async def wait(self) -> None:
        self.waited = True


def test_run_detect_drift_subprocess_success(monkeypatch, tmp_path):
    from consolidation_memory import drift_subprocess

    payload = {
        "checked_anchors": [{"anchor_type": "path", "anchor_value": "src/app.py"}],
        "impacted_claim_ids": ["claim-1"],
        "challenged_claim_ids": ["claim-1"],
        "impacts": [],
    }
    repo_path = str(tmp_path)
    fake_process = _FakeProcess(stdout=json.dumps(payload).encode("utf-8"))
    seen: dict[str, object] = {}

    async def _fake_create_subprocess_exec(*cmd, **kwargs):
        seen["cmd"] = cmd
        seen["kwargs"] = tuple(sorted(kwargs))
        seen["cwd"] = kwargs.get("cwd")
        return fake_process

    monkeypatch.setattr(drift_subprocess, "_resolve_python_executable", lambda: "C:/Python/python.exe")
    monkeypatch.setattr(
        drift_subprocess.asyncio,
        "create_subprocess_exec",
        _fake_create_subprocess_exec,
    )
    monkeypatch.setattr(
        "consolidation_memory.config.get_active_project",
        lambda: "universal",
    )

    result = asyncio.run(
        drift_subprocess.run_detect_drift_subprocess(
            base_ref="origin/main",
            repo_path=repo_path,
            timeout_seconds=1.0,
        )
    )

    assert result == payload
    assert seen["cmd"] == (
        "C:/Python/python.exe",
        "-m",
        "consolidation_memory.drift_worker",
        "--project",
        "universal",
        "--base-ref",
        "origin/main",
        "--repo-path",
        repo_path,
    )
    assert seen["cwd"] == str(Path(repo_path).expanduser().resolve())
    assert "stdin" in seen["kwargs"]
    assert "stdout" in seen["kwargs"]
    assert "stderr" in seen["kwargs"]


def test_run_detect_drift_subprocess_timeout_kills_process(monkeypatch):
    from consolidation_memory import drift_subprocess

    fake_process = _FakeProcess(stdout=b"{}", delay_seconds=0.05)

    async def _fake_create_subprocess_exec(*_cmd, **_kwargs):
        return fake_process

    monkeypatch.setattr(drift_subprocess, "_resolve_python_executable", lambda: "C:/Python/python.exe")
    monkeypatch.setattr(
        drift_subprocess.asyncio,
        "create_subprocess_exec",
        _fake_create_subprocess_exec,
    )
    monkeypatch.setattr(
        "consolidation_memory.config.get_active_project",
        lambda: "universal",
    )

    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(
            drift_subprocess.run_detect_drift_subprocess(
                base_ref=None,
                repo_path=None,
                timeout_seconds=0.001,
            )
        )

    assert fake_process.killed
    assert fake_process.waited


def test_run_detect_drift_subprocess_nonzero_exit_raises(monkeypatch):
    from consolidation_memory import drift_subprocess

    fake_process = _FakeProcess(returncode=2, stderr=b"git failed")

    async def _fake_create_subprocess_exec(*_cmd, **_kwargs):
        return fake_process

    monkeypatch.setattr(drift_subprocess, "_resolve_python_executable", lambda: "C:/Python/python.exe")
    monkeypatch.setattr(
        drift_subprocess.asyncio,
        "create_subprocess_exec",
        _fake_create_subprocess_exec,
    )
    monkeypatch.setattr(
        "consolidation_memory.config.get_active_project",
        lambda: "universal",
    )

    with pytest.raises(RuntimeError, match="Isolated drift detection failed: git failed"):
        asyncio.run(
            drift_subprocess.run_detect_drift_subprocess(
                base_ref=None,
                repo_path=None,
                timeout_seconds=1.0,
            )
        )
