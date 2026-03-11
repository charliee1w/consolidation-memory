"""Tests for scripts/release.py automation helpers."""

from __future__ import annotations

import importlib.util
import re
import subprocess
from contextlib import contextmanager
from pathlib import Path

import pytest


def _load_release_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "release.py"
    spec = importlib.util.spec_from_file_location("release_script", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bump_semver_variants():
    module = _load_release_module()
    assert module.bump_semver("0.13.0", "patch") == "0.13.1"
    assert module.bump_semver("0.13.0", "minor") == "0.14.0"
    assert module.bump_semver("0.13.0", "major") == "1.0.0"


def test_resolve_target_version_prefers_explicit():
    module = _load_release_module()
    assert module.resolve_target_version("0.13.0", "0.14.2", "patch") == "0.14.2"


def test_collect_release_notes_filters_release_commits_and_duplicates(monkeypatch):
    module = _load_release_module()

    def fake_run(cmd, *, check=True, capture=False):
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout="feat: add scope model\nv0.13.0\nfeat: add scope model\nfix: patch release gate\n",
            stderr="",
        )

    monkeypatch.setattr(module, "run", fake_run)
    notes = module.collect_release_notes("v0.13.0")
    assert notes == ["fix: patch release gate", "feat: add scope model"]


def test_add_changelog_entry_inserts_rendered_notes(tmp_path, monkeypatch):
    module = _load_release_module()
    changelog_path = tmp_path / "CHANGELOG.md"
    changelog_path.write_text("# Changelog\n\n## 0.13.0 - 2026-03-07\n\n- Existing entry\n", encoding="utf-8")
    monkeypatch.setattr(module, "CHANGELOG", changelog_path)

    inserted = module.add_changelog_entry("0.13.1", ["fix: release automation", "docs: update flow"])
    text = changelog_path.read_text(encoding="utf-8")

    assert inserted is True
    assert re.search(r"^## 0\.13\.1 - \d{4}-\d{2}-\d{2}$", text, flags=re.MULTILINE)
    assert "- fix: release automation" in text
    assert "- docs: update flow" in text


def test_add_changelog_entry_noop_if_version_exists(tmp_path, monkeypatch):
    module = _load_release_module()
    changelog_path = tmp_path / "CHANGELOG.md"
    original = "# Changelog\n\n## 0.13.1 - 2026-03-07\n\n- Existing entry\n"
    changelog_path.write_text(original, encoding="utf-8")
    monkeypatch.setattr(module, "CHANGELOG", changelog_path)

    inserted = module.add_changelog_entry("0.13.1", ["new note"])

    assert inserted is False
    assert changelog_path.read_text(encoding="utf-8") == original


def test_ensure_release_branch_requires_main(monkeypatch):
    module = _load_release_module()
    monkeypatch.setattr(module, "get_current_branch", lambda: "release-fix")

    with pytest.raises(RuntimeError, match="must run from 'main'"):
        module.ensure_release_branch()


def test_get_quality_gate_commands_match_publish_contract(tmp_path):
    module = _load_release_module()

    commands = module.get_quality_gate_commands("python", tmp_path)

    assert [command.label for command in commands] == [
        "Install publish-grade release dependencies",
        "Run test suite with coverage",
        "Run builder baseline smoke",
        "Enforce ResourceWarning gate",
        "Run lint",
        "Run type checks",
        "Run security checks",
    ]
    assert commands[0].cmd == [
        "python",
        "-m",
        "pip",
        "install",
        "-e",
        ".[fastembed,rest,dev]",
        "--quiet",
    ]
    assert commands[1].env == {"COVERAGE_FILE": str(tmp_path / "coverage" / ".coverage")}
    assert "--cov=consolidation_memory" in commands[1].cmd
    assert any("error::ResourceWarning" in part for part in commands[3].cmd)
    assert commands[5].cmd == ["python", "-m", "mypy", "src/consolidation_memory/"]
    assert commands[6].cmd == ["python", "-m", "bandit", "-q", "-r", "src", "scripts"]


def test_verify_built_artifacts_requires_one_wheel_and_sdist(tmp_path):
    module = _load_release_module()
    artifact_dir = tmp_path / "dist"
    artifact_dir.mkdir()
    (artifact_dir / "consolidation_memory-0.13.6-py3-none-any.whl").write_text("", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Expected exactly one wheel and one sdist"):
        module.verify_built_artifacts(artifact_dir, "0.13.6")


def test_build_and_validate_artifacts_runs_build_then_twine(tmp_path, monkeypatch):
    module = _load_release_module()
    calls: list[list[str]] = []

    @contextmanager
    def fake_tempdir(prefix: str):
        del prefix
        yield str(tmp_path)

    def fake_run(cmd, *, check=True, capture=False, env=None):
        del check, capture, env
        calls.append(cmd)
        if cmd[:3] == ["python", "-m", "build"]:
            outdir = Path(cmd[-1])
            outdir.mkdir(parents=True, exist_ok=True)
            (outdir / "consolidation_memory-0.13.6-py3-none-any.whl").write_text(
                "",
                encoding="utf-8",
            )
            (outdir / "consolidation_memory-0.13.6.tar.gz").write_text("", encoding="utf-8")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(module.tempfile, "TemporaryDirectory", fake_tempdir)
    monkeypatch.setattr(module, "run", fake_run)

    module.build_and_validate_artifacts(
        "python",
        "0.13.6",
        pyproject_text='version = "0.13.5"\n',
        changelog_text="# Changelog\n",
    )

    assert calls[0] == ["python", "-m", "build", "--outdir", str(tmp_path / "dist")]
    assert calls[1] == [
        "python",
        "-m",
        "twine",
        "check",
        "--strict",
        str(tmp_path / "dist" / "consolidation_memory-0.13.6-py3-none-any.whl"),
        str(tmp_path / "dist" / "consolidation_memory-0.13.6.tar.gz"),
    ]
