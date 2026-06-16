"""Tests for scripts/sync_working_copy.py."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts import sync_working_copy as swc


@pytest.fixture
def repo_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(swc, "ROOT", tmp_path)
    return tmp_path


def _ok(stdout: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")


def _fail() -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="")


def test_sync_refuses_dirty_tree_without_stash(
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        if cmd[:3] == ["git", "status", "--porcelain"]:
            return _ok(" M README.md\n")
        if cmd[:3] == ["git", "rev-parse", "--abbrev-ref"]:
            return _ok("main")
        raise AssertionError(f"unexpected git call: {cmd}")

    monkeypatch.setattr(swc, "run", fake_run)

    assert swc.sync_working_copy() == 1
    assert calls[1] == ["git", "status", "--porcelain"]


def test_sync_fast_forwards_when_behind(
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        if cmd[:3] == ["git", "status", "--porcelain"]:
            return _ok("")
        if cmd[:3] == ["git", "rev-parse", "--abbrev-ref"] and cmd[3] == "HEAD":
            return _ok("main")
        if cmd[:4] == ["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name"]:
            return _ok("origin/main")
        if cmd[:3] == ["git", "rev-parse", "--short"]:
            return _ok("abc1234")
        if cmd[:2] == ["git", "fetch"]:
            return _ok("")
        if cmd[:4] == ["git", "rev-list", "--left-right", "--count"]:
            return _ok("0\t2\n")
        if cmd[:3] == ["git", "pull", "--rebase"]:
            return _ok("")
        raise AssertionError(f"unexpected git call: {cmd}")

    monkeypatch.setattr(swc, "run", fake_run)

    assert swc.sync_working_copy() == 0
    assert ["git", "pull", "--rebase", "origin", "main"] in calls


def test_sync_skips_pull_when_up_to_date(
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        if cmd[:3] == ["git", "status", "--porcelain"]:
            return _ok("")
        if cmd[:3] == ["git", "rev-parse", "--abbrev-ref"] and cmd[3] == "HEAD":
            return _ok("main")
        if cmd[:4] == ["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name"]:
            return _ok("origin/main")
        if cmd[:3] == ["git", "rev-parse", "--short"]:
            return _ok("deadbeef")
        if cmd[:2] == ["git", "fetch"]:
            return _ok("")
        if cmd[:4] == ["git", "rev-list", "--left-right", "--count"]:
            return _ok("0\t0\n")
        raise AssertionError(f"unexpected git call: {cmd}")

    monkeypatch.setattr(swc, "run", fake_run)

    assert swc.sync_working_copy() == 0
    assert not any(cmd[:3] == ["git", "pull", "--rebase"] for cmd in calls)