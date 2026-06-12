"""Tests for scripts/update_changelog.py helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "update_changelog.py"
    spec = importlib.util.spec_from_file_location("update_changelog_script", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_dirty_paths_parses_porcelain_output():
    module = _load_module()
    porcelain = " M CHANGELOG.md\n?? notes.txt\n"
    assert module.dirty_paths(porcelain) == ["CHANGELOG.md", "notes.txt"]


def test_dirty_paths_ignores_blank_lines():
    module = _load_module()
    assert module.dirty_paths("\n\n M README.md\n") == ["README.md"]