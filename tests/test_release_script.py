"""Tests for scripts/release.py automation helpers."""

from __future__ import annotations

import importlib.util
import re
import subprocess
from pathlib import Path


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
