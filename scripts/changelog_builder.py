"""Shared changelog generation for release and CI automation."""

from __future__ import annotations

import re
from datetime import date

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
    r"^(?P<type>[a-z]+)(?:\([^)]+\))?(?P<bang>!)?:\s+(?P<summary>.+)$",
    re.IGNORECASE,
)
SKIP_SUBJECT_RE = re.compile(r"\[(?:skip\s+release|release\s+skip|changelog\s+skip)\]", re.IGNORECASE)

CATEGORY_ORDER = (
    "Features",
    "Bug Fixes",
    "Performance",
    "Security",
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
    if RELEASE_SUBJECT_RE.match(cleaned):
        return True
    if MERGE_SUBJECT_RE.match(cleaned):
        return True
    if SKIP_SUBJECT_RE.search(cleaned):
        return True
    if cleaned.lower().startswith("chore(release):"):
        return True
    return False


def categorize_commit_subject(subject: str) -> str:
    cleaned = subject.strip()
    match = CONVENTIONAL_SUBJECT_RE.match(cleaned)
    if not match:
        return "Other"
    commit_type = match.group("type").lower()
    if match.group("bang"):
        return "Features"
    return _TYPE_TO_CATEGORY.get(commit_type, "Other")


def collect_release_subjects(
    subjects: list[str],
    *,
    limit: int = 20,
) -> list[str]:
    """Filter, dedupe, and keep newest-first commit subjects for changelog bullets."""
    collected: list[str] = []
    seen: set[str] = set()
    for subject in reversed(subjects):
        if should_ignore_commit_subject(subject):
            continue
        normalized = normalize_commit_subject(subject)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        collected.append(normalized)
    if not collected:
        return ["Maintenance release."]
    return collected[: max(1, limit)]


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
) -> str:
    """Render a full versioned changelog section."""
    when = release_date or date.today().isoformat()
    grouped = group_subjects_by_category(collect_release_subjects(subjects))
    body = render_categorized_body(grouped)
    return f"## {version} - {when}\n\n{body}"


def render_unreleased_section(subjects: list[str]) -> str:
    grouped = group_subjects_by_category(collect_release_subjects(subjects))
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


def upsert_unreleased_section(changelog_text: str, subjects: list[str]) -> str:
    """Insert or replace the Unreleased section from commit subjects."""
    ensure_changelog_header(changelog_text)
    unreleased = render_unreleased_section(subjects)
    if UNRELEASED_SECTION_RE.search(changelog_text):
        return UNRELEASED_SECTION_RE.sub(unreleased + "\n", changelog_text, count=1)
    return changelog_text.replace(CHANGELOG_HEADER, f"{CHANGELOG_HEADER}\n{unreleased}\n", 1)


def insert_version_entry(changelog_text: str, version: str, subjects: list[str]) -> tuple[str, bool]:
    """Insert a versioned changelog entry below the header."""
    ensure_changelog_header(changelog_text)
    if version_entry_exists(changelog_text, version):
        return changelog_text, False
    entry = render_changelog_entry(version, subjects)
    updated = changelog_text.replace(CHANGELOG_HEADER, f"{CHANGELOG_HEADER}\n{entry}\n", 1)
    return updated, True


def remove_unreleased_section(changelog_text: str) -> str:
    if not UNRELEASED_SECTION_RE.search(changelog_text):
        return changelog_text
    return UNRELEASED_SECTION_RE.sub("", changelog_text, count=1)