"""Tests for scripts/release_criteria.py decision logic."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_release_criteria_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "release_criteria.py"
    spec = importlib.util.spec_from_file_location("release_criteria_script", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _commit(module, subject: str, body: str = ""):
    return module.CommitMessage(subject=subject, body=body)


def test_decide_release_returns_no_release_with_no_commits():
    module = _load_release_criteria_module()
    result = module.decide_release([])
    assert result["should_release"] is False
    assert result["bump"] is None


def test_decide_release_prefers_highest_semver_signal():
    module = _load_release_criteria_module()
    commits = [
        _commit(module, "fix: patch bug"),
        _commit(module, "feat: add new capability"),
    ]
    result = module.decide_release(commits)
    assert result["should_release"] is True
    assert result["bump"] == "minor"


def test_decide_release_detects_breaking_change_from_bang():
    module = _load_release_criteria_module()
    commits = [_commit(module, "feat!: redesign API contract")]
    result = module.decide_release(commits)
    assert result["should_release"] is True
    assert result["bump"] == "major"


def test_decide_release_detects_breaking_change_from_body():
    module = _load_release_criteria_module()
    commits = [_commit(module, "feat: redesign API", "BREAKING CHANGE: old field removed")]
    result = module.decide_release(commits)
    assert result["should_release"] is True
    assert result["bump"] == "major"


def test_decide_release_skips_when_head_requests_skip():
    module = _load_release_criteria_module()
    commits = [
        _commit(module, "docs: update text [skip release]"),
        _commit(module, "feat: add capability"),
    ]
    result = module.decide_release(commits)
    assert result["should_release"] is False
    assert result["bump"] is None


def test_decide_release_head_override_forces_bump():
    module = _load_release_criteria_module()
    commits = [
        _commit(module, "docs: release note tweak [release patch]"),
        _commit(module, "feat: add capability"),
    ]
    result = module.decide_release(commits)
    assert result["should_release"] is True
    assert result["bump"] == "patch"


def test_decide_release_ignores_non_releasable_commit_types():
    module = _load_release_criteria_module()
    commits = [
        _commit(module, "docs: update readme"),
        _commit(module, "chore: refresh lockfile"),
        _commit(module, "test: add regression coverage"),
    ]
    result = module.decide_release(commits)
    assert result["should_release"] is False
    assert result["bump"] is None
