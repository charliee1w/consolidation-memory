"""Regression tests for changelog subject selection, ranking, and truncation warnings.

Covers the defect where ``limit=20`` silently dropped the oldest user-visible
commits of a 27-commit range, so a 0.21.0 reader never learned about the MCP
dependency floor, the strict tool-input contract, or the structured result format.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

# Commits that were absent from CHANGELOG.md for 0.21.0 under the old limit of 20.
PREVIOUSLY_LOST_SUBJECTS = (
    "chore(deps): migrate to mcp 2.3 and MCPServer API",
    "feat(mcp): document tool inputs and reject unknown arguments",
    "fix(mcp): return tool results as structured objects",
)

# The real ``v0.20.3..HEAD`` range (27 commits, oldest first) that triggered the loss.
REAL_RANGE_SUBJECTS: tuple[str, ...] = (
    "chore(changelog): update Unreleased section [skip release]",
    "test: keep strategy reuse evidence fresh",
    "fix(mcp): return tool results as structured objects",
    "fix(tests): parse object tool results in surface contracts",
    "chore(deps): migrate to mcp 2.3 and MCPServer API",
    "feat(mcp): document tool inputs and reject unknown arguments",
    "feat(mcp): report tool execution errors as isError results",
    "feat(mcp): publish typed output contracts for all tools",
    "feat: add scope discovery across MCP, OpenAI and REST",
    "fix(ci): clear ruff and mypy gates for release",
    "ci: run desktop app tests in the optional surfaces job",
    "chore: drop stray my_script.js",
    "feat(scripts): generate the MCP tool reference",
    "docs: describe the MCP result contract and add the MCP guide",
    "docs: record the MCP surface wave in the roadmap",
    "ci: regenerate docs/TOOLS.md with the docs bot",
    "docs: brief CLAUDE.md on the typed MCP surface",
    "docs: document scope policies and ACL end to end",
    "fix(cli): correct the policy grant principal-type help",
    "docs: document the browser UI, TUI dashboard and desktop app",
    "docs: rebuild the README around the knowledge-layer story",
    "docs: align the tagline across metadata and runtime surfaces",
    "docs: fix post-rebuild inconsistencies in guides and meta",
    "test: guard tracked markdown links and anchors",
    "docs: extend the architecture doc with surfaces, contracts and modules",
    "docs: sync desktop and web copy with the positioning",
    "chore(changelog): refresh Unreleased section",
)

VERSION_HEADER_RE = re.compile(r"^## (?:Unreleased|\d+\.\d+\.\d+ - \d{4}-\d{2}-\d{2})$", re.MULTILINE)
HEADING_RE = re.compile(r"^(#{1,6}) (.+)$")
BULLET_RE = re.compile(r"^- .+$")


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "changelog_builder.py"
    spec = importlib.util.spec_from_file_location("changelog_builder_truncation_script", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _assert_valid_changelog_markdown(module, text: str) -> list[str]:
    """Assert a generated block is a clean Markdown document and return its bullets.

    The generator only ever emits a version header, ``### category`` headings and
    ``- subject`` bullets, so an unrecognised line means the output was corrupted -
    for example by a truncation warning leaking into the file. Only the generated
    block may be passed in: pre-existing sections may use legacy headings.
    """
    categories: list[str] = []
    bullets: list[str] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        heading = HEADING_RE.match(line)
        if heading:
            assert len(heading.group(1)) in (2, 3), f"unexpected heading depth: {line!r}"
            if len(heading.group(1)) == 3:
                categories.append(heading.group(2))
            continue
        bullet = BULLET_RE.match(line)
        assert bullet is not None, f"line is neither a heading nor a bullet: {line!r}"
        assert "\x1b" not in line, "ANSI escape leaked into the changelog"
        bullets.append(line[2:])
    assert VERSION_HEADER_RE.search(text), "missing release section header"
    assert categories == [name for name in module.CATEGORY_ORDER if name in categories], (
        f"category sections out of order: {categories}"
    )
    return bullets


def _unreleased_block(module, text: str) -> str:
    match = module.UNRELEASED_SECTION_RE.search(text)
    assert match is not None, "Unreleased section missing"
    return match.group(0)


def test_default_limit_does_not_truncate_a_realistic_range():
    module = _load_module()
    assert module.DEFAULT_SUBJECT_LIMIT >= 100, "the default limit must clear realistic PR ranges"
    selection = module.select_release_subjects(list(REAL_RANGE_SUBJECTS))
    assert len(REAL_RANGE_SUBJECTS) >= 26
    assert selection.truncated is False
    assert selection.dropped == []
    assert len(selection.subjects) == selection.total


def test_real_range_keeps_every_user_visible_entry_despite_docs_wave():
    module = _load_module()
    selection = module.select_release_subjects(list(REAL_RANGE_SUBJECTS))
    for subject in PREVIOUSLY_LOST_SUBJECTS:
        assert subject in selection.subjects, f"user-visible commit lost: {subject}"
    # Ten user-visible commits (feat/fix/perf plus the dependency floor) against a
    # wall of docs/chore/test/ci: a limit of ten must keep every one of them.
    assert len([s for s in selection.subjects if module.is_user_visible_subject(s)]) == 10


def test_noise_never_displaces_user_visible_entries_under_a_tight_limit():
    module = _load_module()
    subjects = [
        "feat(api)!: reject unknown tool arguments",
        "fix(mcp)!: drop the legacy result envelope",
        "perf: index claims in batches",
        "security: sign trust manifests",
        "style: f",
        "refactor: e",
        "ci: d",
        "test: c",
        "chore: b",
        "docs: a",
    ]
    generous = module.select_release_subjects(subjects, limit=4)
    assert generous.subjects == [
        "security: sign trust manifests",
        "perf: index claims in batches",
        "fix!(mcp): drop the legacy result envelope",
        "feat!(api): reject unknown tool arguments",
    ]
    assert generous.dropped == ["docs: a", "chore: b", "test: c", "ci: d", "refactor: e", "style: f"]
    assert generous.dropped_internal == generous.dropped
    assert generous.dropped_user_visible == []

    starved = module.select_release_subjects(subjects, limit=3)
    assert starved.dropped_user_visible == ["feat!(api): reject unknown tool arguments"]
    assert len(starved.dropped_internal) == 6


def test_dependency_scope_changes_are_ranked_user_visible():
    module = _load_module()
    for subject in (
        "chore(deps): migrate to mcp 2.3 and MCPServer API",
        "build(deps-dev): pin pytest 8.0",
        "chore(dependencies): drop the numpy floor",
    ):
        assert module.is_user_visible_subject(subject) is True, subject
    for subject in ("docs: a", "chore: b", "ci: regenerate docs/TOOLS.md", "style: f"):
        assert module.is_user_visible_subject(subject) is False, subject
    for subject in ("feat: a", "fix: b", "perf: c", "security: d", "revert: e", "docs(api)!: f", "raw note"):
        assert module.is_user_visible_subject(subject) is True, subject
    for subject in (
        "chore(deps): migrate to mcp 2.3 and MCPServer API",
        "build(deps-dev): pin pytest 8.0",
    ):
        assert module.categorize_commit_subject(subject) == "Dependencies", subject
    assert module.categorize_commit_subject("chore: drop stray my_script.js") == "Internal"


def test_internal_noise_is_dropped_before_user_visible_when_capped():
    module = _load_module()
    subjects = [
        *(f"docs: doc {index}" for index in range(40)),
        *(f"chore: chore {index}" for index in range(40)),
        "feat: user visible feature",
        "fix: user visible fix",
    ]
    selection = module.select_release_subjects(subjects, limit=5)
    assert selection.subjects == [
        "fix: user visible fix",
        "feat: user visible feature",
        "chore: chore 39",
        "chore: chore 38",
        "chore: chore 37",
    ]
    assert selection.truncated is True
    assert selection.dropped_user_visible == []
    assert len(selection.dropped_internal) == 77


def test_truncation_warns_with_dropped_count_and_reason():
    module = _load_module()
    subjects = [*(f"docs: doc {index}" for index in range(30)), "feat: kept feature"]
    with pytest.warns(module.ChangelogTruncationWarning) as caught:
        notes = module.collect_release_subjects(subjects, limit=1)
    assert notes == ["feat: kept feature"]
    message = str(caught[0].message)
    assert "Changelog truncated" in message
    assert "kept 1 of 31 releasable commits (limit 1)" in message
    assert "Dropped 30 entries: 30 internal" in message
    assert "0 user-visible" in message
    assert "No user-visible change was lost" in message
    assert f"--limit {module.select_release_subjects(subjects, limit=1).suggested_limit()}" in message


def test_truncation_warning_names_dropped_user_visible_commits():
    module = _load_module()
    subjects = ["feat: oldest feature", "docs: noise", "fix: oldest fix", "fix: newest fix"]
    with pytest.warns(module.ChangelogTruncationWarning) as caught:
        module.collect_release_subjects(subjects, limit=2)
    message = str(caught[0].message)
    assert "Dropped 2 entries: 1 internal" in message
    assert "1 user-visible" in message
    assert "User-visible changes were dropped (oldest first)" in message
    assert "- feat: oldest feature" in message
    assert "- docs: noise" in message


def test_no_warning_when_nothing_is_dropped():
    module = _load_module()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert module.collect_release_subjects(list(REAL_RANGE_SUBJECTS))


def test_rendered_markdown_stays_valid_and_excludes_the_warning():
    module = _load_module()
    subjects = [*(f"docs: doc {index}" for index in range(40)), "feat(mcp): typed tool results"]
    with pytest.warns(module.ChangelogTruncationWarning):
        entry = module.render_changelog_entry(
            "0.21.0",
            subjects,
            release_date="2026-07-20",
            limit=1,
        )
    bullets = _assert_valid_changelog_markdown(module, entry)
    assert bullets == ["feat(mcp): typed tool results"]
    assert "WARNING" not in entry
    assert module.TRUNCATION_WARNING_PREFIX not in entry
    assert "Changelog truncated" not in entry
    assert "docs: doc 0" not in entry
    assert entry.endswith("\n")


def test_selection_report_names_counts_and_stays_out_of_the_markdown():
    module = _load_module()
    subjects = [*(f"docs: doc {index}" for index in range(30)), "feat: kept feature", "Merge branch 'main'"]
    selection = module.select_release_subjects(subjects, limit=2)
    lines: list[str] = []
    module.emit_selection_report(selection, stream=_ListStream(lines))
    report = "".join(lines)
    assert "Changelog truncated: kept 2 of 31 releasable commits (limit 2)." in report
    assert "Dropped 29 entries: 29 internal" in report
    assert "skipped 1 merge/release/skip-marked commit(s) and 0 duplicate subject(s)" in report
    assert "::warning::" not in report
    with pytest.warns(module.ChangelogTruncationWarning):
        rendered = module.render_changelog_entry("0.21.0", subjects, limit=2)
    for chunk in lines:
        for line in chunk.splitlines():
            assert line.strip() not in rendered.splitlines(), f"report line leaked into markdown: {line!r}"


def test_selection_report_emits_github_annotation_in_ci(monkeypatch):
    module = _load_module()
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    selection = module.select_release_subjects(
        ["feat: oldest feature", "fix: oldest fix", "fix: newest fix"],
        limit=2,
    )
    lines: list[str] = []
    module.emit_selection_report(selection, stream=_ListStream(lines))
    report = "".join(lines)
    assert "::warning::Changelog truncated to 2/3 commits (limit 2)" in report
    assert "1 user-visible" in report


def test_upsert_unreleased_drops_noise_but_keeps_every_user_visible_entry():
    module = _load_module()
    original = "# Changelog\n\n## 0.20.3 - 2026-07-01\n\n### Highlights\n\n- old\n"
    with pytest.warns(module.ChangelogTruncationWarning):
        updated = module.upsert_unreleased_section(
            original,
            list(REAL_RANGE_SUBJECTS),
            limit=10,
        )
    assert updated.count("## Unreleased") == 1
    block = _unreleased_block(module, updated)
    bullets = _assert_valid_changelog_markdown(module, block)
    for subject in PREVIOUSLY_LOST_SUBJECTS:
        assert subject in bullets, f"user-visible commit lost from Unreleased: {subject}"
    assert not [bullet for bullet in bullets if bullet.startswith(("docs:", "chore:", "test:", "ci:"))]
    assert "WARNING" not in updated
    assert "## 0.20.3 - 2026-07-01" in updated


def test_upsert_unreleased_section_is_clean_for_the_full_real_range():
    module = _load_module()
    original = "# Changelog\n\n## 0.20.3 - 2026-07-01\n\n### Highlights\n\n- old\n"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        updated = module.upsert_unreleased_section(original, list(REAL_RANGE_SUBJECTS))
    bullets = _assert_valid_changelog_markdown(module, _unreleased_block(module, updated))
    for subject in PREVIOUSLY_LOST_SUBJECTS:
        assert subject in bullets
    assert "docs: sync desktop and web copy with the positioning" in bullets
    assert updated.startswith("# Changelog\n")


def test_limit_above_the_default_is_not_recapped_on_a_second_pass():
    module = _load_module()
    oversized = module.DEFAULT_SUBJECT_LIMIT + 50
    subjects = [f"feat: feature {index}" for index in range(oversized)]
    original = "# Changelog\n\n## 0.20.3 - 2026-07-01\n\n### Highlights\n\n- old\n"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        updated = module.upsert_unreleased_section(original, subjects, limit=oversized)
    assert updated.count(f"- feat: feature {oversized - 1}") == 1
    selection = module.select_release_subjects(subjects, limit=oversized)
    assert selection.truncated is False
    assert len(selection.subjects) == oversized


def test_release_script_limit_is_wired_through(monkeypatch):
    release_path = Path(__file__).resolve().parents[1] / "scripts" / "release.py"
    spec = importlib.util.spec_from_file_location("release_script_limit_check", release_path)
    assert spec is not None and spec.loader is not None
    release_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = release_module
    spec.loader.exec_module(release_module)

    assert release_module.DEFAULT_SUBJECT_LIMIT == 200
    stdout = "".join(f"{subject}\n" for subject in reversed(REAL_RANGE_SUBJECTS))
    monkeypatch.setattr(
        release_module,
        "run",
        lambda cmd, *, check=True, capture=False: subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout=stdout, stderr=""
        ),
    )
    notes = release_module.collect_release_notes("v0.20.3")
    for subject in PREVIOUSLY_LOST_SUBJECTS:
        assert subject in notes, f"release notes lost user-visible commit: {subject}"

    source = release_path.read_text(encoding="utf-8")
    assert '"--limit"' in source
    assert "collect_release_notes(previous_tag, limit=args.limit)" in source
    assert "add_changelog_entry(new_version, release_notes, limit=args.limit)" in source


def test_update_changelog_exposes_limit_flag():
    source = (Path(__file__).resolve().parents[1] / "scripts" / "update_changelog.py").read_text(
        encoding="utf-8"
    )
    assert '"--limit"' in source
    assert "DEFAULT_SUBJECT_LIMIT" in source
    assert "emit_selection_report" in source


class _ListStream:
    """Minimal writable stream so the report can be asserted without touching stderr."""

    def __init__(self, sink: list[str]) -> None:
        self._sink = sink

    def write(self, text: str) -> int:
        self._sink.append(text)
        return len(text)

    def flush(self) -> None:
        return None
