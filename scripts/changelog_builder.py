"""Shared changelog generation for release and CI automation."""

from __future__ import annotations

import argparse
import os
import re
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import IO

# Maximum number of changelog bullets kept per release.
#
# The cap must not bite on a realistic release range. Releases run every few days
# and 200 bullets is only a few kilobytes of Markdown, so the cap leaves orders of
# magnitude of headroom over any realistic range. Its job is to bound a
# pathological squash/merge (a thousand-commit vendored import), not to filter
# normal work. Recompute the widest observed range with
# `git rev-list --count <previous-tag>..<tag>` across `git tag | sort -V`.
DEFAULT_SUBJECT_LIMIT = 200

# Commits that only matter to maintainers. They are the first entries dropped when
# the limit bites, so a docs/chore/test wave can never displace a user-visible change.
NOISE_COMMIT_TYPES = frozenset({"build", "chore", "ci", "docs", "refactor", "style", "test"})

# Scopes that change what an installed package requires. A `chore(deps):` commit is
# typed as noise but is user-visible: a raised dependency floor breaks downstream pins
# even though the commit subject reads like maintenance.
DEPENDENCY_SCOPES = frozenset({"deps", "deps-dev", "dependencies"})

TRUNCATION_WARNING_PREFIX = "[changelog] WARNING:"
SELECTION_NOTE_PREFIX = "[changelog] NOTE:"

CHANGELOG_HEADER = "# Changelog\n"
UNRELEASED_HEADER = "## Unreleased"
VERSION_HEADER_RE = re.compile(r"^##\s+(\d+\.\d+\.\d+)\s+-\s+(\d{4}-\d{2}-\d{2})\s*$", re.MULTILINE)
UNRELEASED_SECTION_RE = re.compile(
    r"(?ms)^## Unreleased\s*\n.*?(?=^## \d+\.\d+\.\d+ |\Z)",
)
RELEASE_SUBJECT_RE = re.compile(r"^v\d+\.\d+\.\d+\b", re.IGNORECASE)
MERGE_SUBJECT_RE = re.compile(
    r"^(?:Merge (?:branch|pull request|remote-tracking branch)|Merged in\b)",
    re.IGNORECASE,
)
CONVENTIONAL_SUBJECT_RE = re.compile(
    r"^(?P<type>[a-z]+)(?:\((?P<scope>[^)]+)\))?(?P<bang>!)?:\s+(?P<summary>.+)$",
    re.IGNORECASE,
)
SKIP_SUBJECT_RE = re.compile(r"\[(?:skip\s+release|release\s+skip|changelog\s+skip)\]", re.IGNORECASE)

CATEGORY_ORDER = (
    "Features",
    "Bug Fixes",
    "Performance",
    "Security",
    "Dependencies",
    "Refactoring",
    "Documentation",
    "Internal",
    "Other",
)

_TYPE_TO_CATEGORY = {
    "feat": "Features",
    "fix": "Bug Fixes",
    "perf": "Performance",
    "security": "Security",
    "refactor": "Refactoring",
    "revert": "Bug Fixes",
    "docs": "Documentation",
    "test": "Internal",
    "ci": "Internal",
    "chore": "Internal",
    "build": "Internal",
    "style": "Internal",
}


def normalize_commit_subject(subject: str) -> str:
    """Return a display-friendly changelog bullet subject."""
    cleaned = subject.strip()
    if not cleaned:
        return ""
    match = CONVENTIONAL_SUBJECT_RE.match(cleaned)
    if not match:
        return cleaned
    commit_type = match.group("type").lower()
    bang = "!" if match.group("bang") else ""
    summary = match.group("summary").strip()
    scope_match = re.match(rf"^{re.escape(commit_type)}(?:\(([^)]+)\))?", cleaned, re.IGNORECASE)
    scope = ""
    if scope_match and scope_match.group(1):
        scope = f"({scope_match.group(1)})"
    return f"{commit_type}{bang}{scope}: {summary}".strip()


def should_ignore_commit_subject(subject: str) -> bool:
    cleaned = subject.strip()
    if not cleaned:
        return True
    return bool(
        RELEASE_SUBJECT_RE.match(cleaned)
        or MERGE_SUBJECT_RE.match(cleaned)
        or SKIP_SUBJECT_RE.search(cleaned)
        or cleaned.lower().startswith("chore(release):")
    )


def categorize_commit_subject(subject: str) -> str:
    cleaned = subject.strip()
    match = CONVENTIONAL_SUBJECT_RE.match(cleaned)
    if not match:
        return "Other"
    commit_type = match.group("type").lower()
    if match.group("bang"):
        return "Features"
    category = _TYPE_TO_CATEGORY.get(commit_type, "Other")
    if category == "Internal" and (match.group("scope") or "").lower() in DEPENDENCY_SCOPES:
        return "Dependencies"
    return category


def is_user_visible_subject(subject: str) -> bool:
    """Return True for bullets a reader needs, so internal noise can never displace them.

    Breaking subjects (``!`` suffix) and the known user-facing types are kept, as is
    any dependency-scope change under a noise type. An unclassifiable subject counts
    as user-visible: over-keeping a maintainer line is cheaper than silently dropping
    a change nobody can recognise.
    """
    match = CONVENTIONAL_SUBJECT_RE.match(subject.strip())
    if not match:
        return True
    if match.group("bang"):
        return True
    if match.group("type").lower() not in NOISE_COMMIT_TYPES:
        return True
    return (match.group("scope") or "").lower() in DEPENDENCY_SCOPES


class ChangelogTruncationWarning(UserWarning):
    """Raised when the subject limit drops changelog entries."""


@dataclass
class SubjectSelection:
    """Result of filtering commit subjects, with the audit trail for the caller."""

    subjects: list[str]
    total: int
    limit: int
    ignored: int = 0
    duplicates: int = 0
    dropped: list[str] = field(default_factory=list)

    @property
    def truncated(self) -> bool:
        return bool(self.dropped)

    @property
    def dropped_user_visible(self) -> list[str]:
        return [subject for subject in self.dropped if is_user_visible_subject(subject)]

    @property
    def dropped_internal(self) -> list[str]:
        return [subject for subject in self.dropped if not is_user_visible_subject(subject)]

    def suggested_limit(self) -> int:
        return max(self.limit * 2, self.total, DEFAULT_SUBJECT_LIMIT)

    def warning_message(self) -> str:
        """Human-readable, actionable report of everything the selection left out."""
        user_visible = self.dropped_user_visible
        internal = self.dropped_internal
        lines = [
            (
                f"Changelog truncated: kept {len(self.subjects)} of {self.total} releasable "
                f"commits (limit {self.limit})."
            ),
            (
                f"Dropped {len(self.dropped)} entries: {len(internal)} internal "
                f"(docs/chore/test/ci/refactor/style/build) and {len(user_visible)} user-visible "
                f"(feat/fix/perf/security/breaking)."
            ),
        ]
        if user_visible:
            lines.append(
                "User-visible changes were dropped (oldest first). Raise the limit before releasing."
            )
        else:
            lines.append("No user-visible change was lost; only maintainer-facing entries were dropped.")
        lines.append(
            f"Re-run with a higher limit (for example --limit {self.suggested_limit()}) to keep every entry."
        )
        lines.append("Dropped entries:")
        lines.extend(f"  - {subject}" for subject in self.dropped)
        return "\n".join(lines)

    def annotation_message(self) -> str:
        """Single-line summary for GitHub Actions annotations."""
        user_visible = len(self.dropped_user_visible)
        return (
            f"Changelog truncated to {len(self.subjects)}/{self.total} commits (limit {self.limit}); "
            f"dropped {user_visible} user-visible and {len(self.dropped) - user_visible} internal entries. "
            f"Raise the limit to at least {self.suggested_limit()}."
        )


def positive_limit(value: str) -> int:
    """argparse type for ``--limit``: at least one bullet, never zero.

    ``select_release_subjects`` used to clamp with ``max(1, limit)``, so
    ``--limit 0`` and ``--limit -1`` silently became 1: the run produced a
    one-bullet changelog and the truncation report named a limit the caller never
    asked for. Rejecting at the parser is louder and honest about intent.
    """
    try:
        parsed = int(value)
    except ValueError as exc:  # pragma: no cover - argparse formats the message
        raise argparse.ArgumentTypeError(f"limit must be an integer, got {value!r}") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError(
            f"limit must be at least 1, got {parsed}; "
            "use --limit to cap, not to remove entries"
        )
    return parsed


def select_release_subjects(
    subjects: list[str],
    *,
    limit: int = DEFAULT_SUBJECT_LIMIT,
) -> SubjectSelection:
    """Filter, dedupe, rank, and cap commit subjects for changelog bullets.

    Ranking guarantees that internal-only commits are dropped before user-visible
    ones when the limit bites. Returns the audit trail instead of warning, so
    callers decide how loudly to report it.

    ``limit`` below 1 is rejected rather than clamped: the CLI validates it with
    :func:`positive_limit`, and a library caller passing 0 gets told instead of
    quietly receiving a single bullet.
    """
    if limit < 1:
        raise ValueError(f"limit must be at least 1, got {limit}")
    budget = limit
    collected: list[str] = []
    seen: set[str] = set()
    ignored = 0
    duplicates = 0
    for subject in reversed(subjects):
        if should_ignore_commit_subject(subject):
            ignored += 1
            continue
        normalized = normalize_commit_subject(subject)
        if not normalized:
            ignored += 1
            continue
        if normalized in seen:
            duplicates += 1
            continue
        seen.add(normalized)
        collected.append(normalized)
    total = len(collected)
    if total == 0:
        return SubjectSelection(["Maintenance release."], 0, budget, ignored, duplicates, [])
    if total <= budget:
        return SubjectSelection(list(collected), total, budget, ignored, duplicates, [])
    user_visible = [subject for subject in collected if is_user_visible_subject(subject)]
    internal = [subject for subject in collected if not is_user_visible_subject(subject)]
    kept_visible = user_visible[:budget]
    kept_internal = internal[: budget - len(kept_visible)]
    kept = set(kept_visible) | set(kept_internal)
    return SubjectSelection(
        subjects=[subject for subject in collected if subject in kept],
        total=total,
        limit=budget,
        ignored=ignored,
        duplicates=duplicates,
        dropped=[subject for subject in collected if subject not in kept],
    )


def collect_release_subjects(
    subjects: list[str],
    *,
    limit: int = DEFAULT_SUBJECT_LIMIT,
    emit_warning: bool = True,
) -> list[str]:
    """Keep newest-first commit subjects for changelog bullets, warning when capped."""
    selection = select_release_subjects(subjects, limit=limit)
    if emit_warning and selection.truncated:
        warnings.warn(selection.warning_message(), ChangelogTruncationWarning, stacklevel=2)
    return selection.subjects


def emit_selection_report(selection: SubjectSelection, *, stream: IO[str] | None = None) -> None:
    """Report filtering and truncation to the release author.

    Written to stderr (never into the rendered Markdown) so the automation log
    carries the same audit trail as the code path that produced the notes.
    """
    out = stream if stream is not None else sys.stderr
    if selection.ignored or selection.duplicates:
        out.write(
            f"{SELECTION_NOTE_PREFIX} kept {len(selection.subjects)} of {selection.total} releasable commits; "
            f"skipped {selection.ignored} merge/release/skip-marked commit(s) and "
            f"{selection.duplicates} duplicate subject(s).\n"
        )
    if not selection.truncated:
        return
    for line in selection.warning_message().splitlines():
        out.write(f"{TRUNCATION_WARNING_PREFIX} {line}\n")
    if os.environ.get("GITHUB_ACTIONS"):
        out.write(f"::warning::{selection.annotation_message()}\n")


def group_subjects_by_category(subjects: list[str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {category: [] for category in CATEGORY_ORDER}
    for subject in subjects:
        category = categorize_commit_subject(subject)
        grouped.setdefault(category, [])
        if subject not in grouped[category]:
            grouped[category].append(subject)
    return {key: value for key, value in grouped.items() if value}


def render_categorized_body(grouped: dict[str, list[str]]) -> str:
    if not grouped:
        return "### Highlights\n\n- Maintenance release.\n"
    sections: list[str] = []
    for category in CATEGORY_ORDER:
        bullets = grouped.get(category)
        if not bullets:
            continue
        lines = "\n".join(f"- {bullet}" for bullet in bullets)
        sections.append(f"### {category}\n\n{lines}")
    return "\n\n".join(sections) + "\n"


def render_changelog_entry(
    version: str,
    subjects: list[str],
    *,
    release_date: str | None = None,
    limit: int = DEFAULT_SUBJECT_LIMIT,
    emit_warning: bool = True,
) -> str:
    """Render a full versioned changelog section."""
    # UTC, not the committer's local calendar: the same release range must render
    # the same date on every machine, and CI already runs in UTC.
    when = release_date or datetime.now(tz=timezone.utc).date().isoformat()
    notes = collect_release_subjects(subjects, limit=limit, emit_warning=emit_warning)
    grouped = group_subjects_by_category(notes)
    body = render_categorized_body(grouped)
    return f"## {version} - {when}\n\n{body}"


def render_unreleased_section(
    subjects: list[str],
    *,
    limit: int = DEFAULT_SUBJECT_LIMIT,
    emit_warning: bool = True,
) -> str:
    notes = collect_release_subjects(subjects, limit=limit, emit_warning=emit_warning)
    grouped = group_subjects_by_category(notes)
    body = render_categorized_body(grouped)
    return f"{UNRELEASED_HEADER}\n\n{body}"


def extract_unreleased_subjects(changelog_text: str) -> list[str]:
    match = UNRELEASED_SECTION_RE.search(changelog_text)
    if not match:
        return []
    section = match.group(0)
    return [line[2:].strip() for line in section.splitlines() if line.startswith("- ")]


def ensure_changelog_header(changelog_text: str) -> str:
    if CHANGELOG_HEADER not in changelog_text:
        raise RuntimeError("Could not find '# Changelog' header in CHANGELOG.md")
    return changelog_text


def version_entry_exists(changelog_text: str, version: str) -> bool:
    return bool(re.search(rf"^##\s+{re.escape(version)}\b", changelog_text, flags=re.MULTILINE))


def upsert_unreleased_section(
    changelog_text: str,
    subjects: list[str],
    *,
    limit: int = DEFAULT_SUBJECT_LIMIT,
    emit_warning: bool = True,
) -> str:
    """Insert or replace the Unreleased section from commit subjects."""
    ensure_changelog_header(changelog_text)
    unreleased = render_unreleased_section(subjects, limit=limit, emit_warning=emit_warning)
    if UNRELEASED_SECTION_RE.search(changelog_text):
        return UNRELEASED_SECTION_RE.sub(unreleased + "\n", changelog_text, count=1)
    return changelog_text.replace(CHANGELOG_HEADER, f"{CHANGELOG_HEADER}\n{unreleased}\n", 1)


def insert_version_entry(
    changelog_text: str,
    version: str,
    subjects: list[str],
    *,
    limit: int = DEFAULT_SUBJECT_LIMIT,
    emit_warning: bool = True,
) -> tuple[str, bool]:
    """Insert a versioned changelog entry below the header."""
    ensure_changelog_header(changelog_text)
    if version_entry_exists(changelog_text, version):
        return changelog_text, False
    entry = render_changelog_entry(
        version,
        subjects,
        limit=limit,
        emit_warning=emit_warning,
    )
    updated = changelog_text.replace(CHANGELOG_HEADER, f"{CHANGELOG_HEADER}\n{entry}\n", 1)
    return updated, True


def remove_unreleased_section(changelog_text: str) -> str:
    if not UNRELEASED_SECTION_RE.search(changelog_text):
        return changelog_text
    return UNRELEASED_SECTION_RE.sub("", changelog_text, count=1)