"""Tests for scripts/check_release_docs.py guard logic."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest


def _load_release_docs_check_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "check_release_docs.py"
    spec = importlib.util.spec_from_file_location("release_docs_check_script", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _release_doc_text(module) -> str:
    return "\n".join(
        [
            "RELEASE_AUTOMATION_PAT",
            "workflow_dispatch",
            "[skip release]",
            "[release major|minor|patch]",
        ]
    )


def test_evaluate_guard_noops_when_release_files_not_changed():
    module = _load_release_docs_check_module()
    errors = module.evaluate_release_docs_guard(
        changed_files={"docs/ROADMAP.md"},
        release_doc_text="",
        readme_text="",
    )
    assert errors == []


def test_evaluate_guard_requires_docs_update_when_release_files_change():
    module = _load_release_docs_check_module()
    errors = module.evaluate_release_docs_guard(
        changed_files={"scripts/release.py"},
        release_doc_text=_release_doc_text(module),
        readme_text=f"See {module.RELEASE_AUTOMATION_LINK}",
    )
    assert any("without docs updates" in error for error in errors)


def test_evaluate_guard_passes_when_release_and_docs_change_together():
    module = _load_release_docs_check_module()
    errors = module.evaluate_release_docs_guard(
        changed_files={"scripts/release.py", "docs/RELEASE_AUTOMATION.md"},
        release_doc_text=_release_doc_text(module),
        readme_text=f"See {module.RELEASE_AUTOMATION_LINK}",
    )
    assert errors == []


def test_evaluate_guard_requires_release_doc_markers():
    module = _load_release_docs_check_module()
    errors = module.evaluate_release_docs_guard(
        changed_files={"scripts/release.py", "docs/RELEASE_AUTOMATION.md"},
        release_doc_text="RELEASE_AUTOMATION_PAT",
        readme_text=f"See {module.RELEASE_AUTOMATION_LINK}",
    )
    assert any("missing required markers" in error for error in errors)


def test_evaluate_guard_requires_readme_link():
    module = _load_release_docs_check_module()
    errors = module.evaluate_release_docs_guard(
        changed_files={"scripts/release.py", "docs/RELEASE_AUTOMATION.md"},
        release_doc_text=_release_doc_text(module),
        readme_text="No release docs link here",
    )
    assert any("README.md is missing" in error for error in errors)


def test_run_git_raises_when_git_missing(monkeypatch):
    module = _load_release_docs_check_module()
    monkeypatch.setattr(module.shutil, "which", lambda _: None)

    with pytest.raises(RuntimeError, match="git executable not found"):
        module._run_git(["status"])


def test_run_git_resolves_absolute_git_path(monkeypatch):
    module = _load_release_docs_check_module()
    monkeypatch.setattr(module.shutil, "which", lambda _: "bin/git")

    captured: dict[str, object] = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        captured["kwargs"] = dict(kwargs)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    module._run_git(["status"])

    assert captured["cmd"][0] == str(Path("bin/git").resolve())
    assert captured["cmd"][1:] == ["status"]
