"""Tests for scripts/changelog_builder.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "changelog_builder.py"
    spec = importlib.util.spec_from_file_location("changelog_builder_script", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_collect_release_subjects_filters_merge_and_release_commits():
    module = _load_module()
    subjects = [
        "feat: add utility trigger explanation",
        "Merge branch 'main'",
        "v0.14.0",
        "fix: harden scheduler state",
        "feat: add utility trigger explanation",
    ]
    notes = module.collect_release_subjects(subjects)
    assert notes == [
        "feat: add utility trigger explanation",
        "fix: harden scheduler state",
    ]


def test_render_changelog_entry_groups_by_category():
    module = _load_module()
    entry = module.render_changelog_entry(
        "0.16.0",
        [
            "fix: scheduler lease cleanup",
            "feat(status): expose utility breakdown",
            "docs: release automation guide",
        ],
        release_date="2026-06-12",
    )
    assert "## 0.16.0 - 2026-06-12" in entry
    assert "### Features" in entry
    assert "### Bug Fixes" in entry
    assert "### Documentation" in entry
    assert "- feat(status): expose utility breakdown" in entry
    assert "- fix: scheduler lease cleanup" in entry


def test_upsert_unreleased_section_inserts_and_replaces():
    module = _load_module()
    original = "# Changelog\n\n## 0.14.0 - 2026-03-19\n\n### Highlights\n\n- old\n"
    first = module.upsert_unreleased_section(original, ["feat: one", "fix: two"])
    assert "## Unreleased" in first
    assert "### Features" in first
    assert "### Bug Fixes" in first

    second = module.upsert_unreleased_section(first, ["feat: replaced"])
    assert second.count("## Unreleased") == 1
    assert "feat: replaced" in second
    assert "fix: two" not in second


def test_insert_version_entry_promotes_notes_and_skips_duplicates():
    module = _load_module()
    original = "# Changelog\n\n## Unreleased\n\n### Features\n\n- feat: pending\n\n"
    updated, inserted = module.insert_version_entry(original, "0.16.0", ["feat: pending"])
    assert inserted is True
    assert "## 0.16.0 -" in updated
    assert "feat: pending" in updated

    again, inserted_again = module.insert_version_entry(updated, "0.16.0", ["feat: pending"])
    assert inserted_again is False
    assert again == updated