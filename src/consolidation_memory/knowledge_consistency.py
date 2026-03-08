"""Consistency auditing between markdown topics and structured DB records."""

from __future__ import annotations

from typing import Any

from consolidation_memory.config import get_config
from consolidation_memory.consolidation.prompting import _parse_frontmatter
from consolidation_memory.database import get_all_knowledge_topics, get_records_by_topic
from consolidation_memory.markdown_records import parse_markdown_records


def _normalize_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def build_knowledge_consistency_report(*, max_issues: int = 20) -> dict[str, Any]:
    """Compare markdown topic files against active DB records.

    A topic is considered consistent when:
    - the markdown file exists inside ``KNOWLEDGE_DIR``,
    - markdown title/summary match topic metadata, and
    - markdown-derived record count matches active DB record count for the topic.
    """
    cfg = get_config()
    threshold = float(cfg.KNOWLEDGE_CONSISTENCY_THRESHOLD)
    topics = get_all_knowledge_topics()
    knowledge_root = cfg.KNOWLEDGE_DIR.resolve()

    checked_topics = 0
    consistent_topics = 0
    issues: list[dict[str, Any]] = []

    def _append_issue(payload: dict[str, Any]) -> None:
        if len(issues) < max_issues:
            issues.append(payload)

    for topic in topics:
        topic_id = str(topic.get("id") or "")
        filename = str(topic.get("filename") or "")
        file_path = (cfg.KNOWLEDGE_DIR / filename).resolve()
        checked_topics += 1

        if not file_path.is_relative_to(knowledge_root):
            _append_issue(
                {
                    "topic_id": topic_id,
                    "filename": filename,
                    "issue": "path_outside_knowledge_dir",
                }
            )
            continue
        if not file_path.exists():
            _append_issue(
                {
                    "topic_id": topic_id,
                    "filename": filename,
                    "issue": "markdown_missing",
                }
            )
            continue

        try:
            content = file_path.read_text(encoding="utf-8")
        except OSError:
            _append_issue(
                {
                    "topic_id": topic_id,
                    "filename": filename,
                    "issue": "markdown_unreadable",
                }
            )
            continue

        parsed = _parse_frontmatter(content)
        meta = parsed.get("meta", {})
        body = parsed.get("body", "")

        markdown_title = _normalize_text(meta.get("title"))
        markdown_summary = _normalize_text(meta.get("summary"))
        topic_title = _normalize_text(topic.get("title"))
        topic_summary = _normalize_text(topic.get("summary"))

        markdown_records = parse_markdown_records(str(body))
        db_records = get_records_by_topic(topic_id, include_expired=False)

        mismatch_reasons: list[str] = []
        if markdown_title and markdown_title != topic_title:
            mismatch_reasons.append("title_mismatch")
        if markdown_summary and markdown_summary != topic_summary:
            mismatch_reasons.append("summary_mismatch")
        if len(markdown_records) != len(db_records):
            mismatch_reasons.append("record_count_mismatch")

        if mismatch_reasons:
            _append_issue(
                {
                    "topic_id": topic_id,
                    "filename": filename,
                    "issue": ",".join(mismatch_reasons),
                    "markdown_record_count": len(markdown_records),
                    "db_record_count": len(db_records),
                }
            )
            continue

        consistent_topics += 1

    consistency_ratio = (
        (consistent_topics / checked_topics) if checked_topics > 0 else 1.0
    )

    return {
        "threshold": threshold,
        "checked_topics": checked_topics,
        "consistent_topics": consistent_topics,
        "consistency_ratio": round(consistency_ratio, 4),
        "meets_threshold": consistency_ratio >= threshold,
        "issue_count": checked_topics - consistent_topics,
        "issues": issues,
    }
