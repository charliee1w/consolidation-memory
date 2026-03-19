"""Tests for scripts/check_release_docs.py guard logic."""

from __future__ import annotations

import importlib.util
from pathlib import Path


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
