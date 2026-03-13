"""Tests that keep the product positioning aligned across public surfaces."""

from __future__ import annotations

from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_readme_positions_system_as_trust_calibrated_working_memory():
    readme = (_repo_root() / "README.md").read_text(encoding="utf-8")
    assert "Trust-calibrated working memory for coding agents." in readme
    assert "claims are the reusable unit" in readme
    assert "episodes are the raw evidence" in readme
    assert "trust_profile" in readme


def test_architecture_doc_positions_system_as_trust_layer():
    architecture = (_repo_root() / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8")
    assert "trust layer for coding-agent memory" in architecture
    assert "Claims are the reusable unit." in architecture
    assert "Episodes are the raw evidence" in architecture


def test_package_metadata_and_module_docstrings_match_positioning():
    pyproject = (_repo_root() / "pyproject.toml").read_text(encoding="utf-8")
    package_init = (_repo_root() / "src" / "consolidation_memory" / "__init__.py").read_text(
        encoding="utf-8"
    )
    cli_module = (_repo_root() / "src" / "consolidation_memory" / "cli.py").read_text(
        encoding="utf-8"
    )
    rest_module = (_repo_root() / "src" / "consolidation_memory" / "rest.py").read_text(
        encoding="utf-8"
    )

    assert "Trust-calibrated working memory for coding agents" in pyproject
    assert "trust-calibrated working memory for coding agents" in package_init
    assert 'description="Trust-calibrated working memory for coding agents"' in cli_module
    assert 'description="Trust-calibrated working memory for coding agents"' in rest_module
