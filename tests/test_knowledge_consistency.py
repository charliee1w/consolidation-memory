"""Tests for markdown/DB knowledge consistency auditing."""

from __future__ import annotations

from consolidation_memory.config import get_config
from consolidation_memory.consolidation.prompting import _render_markdown_from_records
from consolidation_memory.database import (
    ensure_schema,
    get_knowledge_topic,
    insert_knowledge_records,
    upsert_knowledge_topic,
)
from consolidation_memory.knowledge_consistency import build_knowledge_consistency_report


def test_consistency_report_passes_for_aligned_topic(tmp_data_dir):
    ensure_schema()
    cfg = get_config()
    records = [
        {"type": "fact", "subject": "Python", "info": "3.13"},
        {"type": "preference", "key": "theme", "value": "light"},
    ]
    topic_id = upsert_knowledge_topic(
        filename="aligned.md",
        title="Aligned",
        summary="Aligned summary",
        source_episodes=[],
        fact_count=len(records),
    )
    insert_knowledge_records(
        topic_id,
        [
            {
                "record_type": rec["type"],
                "content": rec,
                "embedding_text": f"{rec['type']} {idx}",
            }
            for idx, rec in enumerate(records)
        ],
    )
    markdown = _render_markdown_from_records(
        "Aligned",
        "Aligned summary",
        [],
        0.8,
        records,
    )
    (cfg.KNOWLEDGE_DIR / "aligned.md").write_text(markdown, encoding="utf-8")

    report = build_knowledge_consistency_report()
    assert report["checked_topics"] == 1
    assert report["consistent_topics"] == 1
    assert report["issue_count"] == 0
    assert report["meets_threshold"] is True


def test_consistency_report_detects_markdown_record_drift(tmp_data_dir):
    ensure_schema()
    cfg = get_config()
    db_records = [
        {"type": "fact", "subject": "A", "info": "1"},
        {"type": "fact", "subject": "B", "info": "2"},
    ]
    topic_id = upsert_knowledge_topic(
        filename="drift.md",
        title="Drift",
        summary="Drift summary",
        source_episodes=[],
        fact_count=len(db_records),
    )
    insert_knowledge_records(
        topic_id,
        [
            {
                "record_type": rec["type"],
                "content": rec,
                "embedding_text": f"{rec['subject']}:{rec['info']}",
            }
            for rec in db_records
        ],
    )

    markdown = _render_markdown_from_records(
        "Drift",
        "Drift summary",
        [],
        0.8,
        [{"type": "fact", "subject": "A", "info": "1"}],
    )
    (cfg.KNOWLEDGE_DIR / "drift.md").write_text(markdown, encoding="utf-8")

    report = build_knowledge_consistency_report()
    assert report["checked_topics"] == 1
    assert report["consistent_topics"] == 0
    assert report["issue_count"] == 1
    assert report["meets_threshold"] is False
    assert report["issues"][0]["issue"] == "record_count_mismatch"


def test_consistency_report_uses_storage_filename_for_scoped_topic(tmp_data_dir):
    ensure_schema()
    cfg = get_config()
    scope = {
        "namespace_slug": "default",
        "project_slug": "default",
        "app_client_name": "codex-desktop",
        "app_client_type": "mcp",
    }
    records = [
        {"type": "fact", "subject": "Scoped topic", "info": "stored via canonical path"},
    ]
    topic_id = upsert_knowledge_topic(
        filename="scoped.md",
        title="Scoped",
        summary="Scoped summary",
        source_episodes=[],
        fact_count=len(records),
        scope=scope,
    )
    insert_knowledge_records(
        topic_id,
        [
            {
                "record_type": rec["type"],
                "content": rec,
                "embedding_text": rec["subject"],
            }
            for rec in records
        ],
        scope=scope,
    )
    topic_row = get_knowledge_topic("scoped.md", scope=scope)
    assert topic_row is not None
    storage_filename = str(topic_row["storage_filename"])
    assert storage_filename != "scoped.md"

    markdown = _render_markdown_from_records(
        "Scoped",
        "Scoped summary",
        [],
        0.8,
        records,
    )
    (cfg.KNOWLEDGE_DIR / storage_filename).write_text(markdown, encoding="utf-8")
    assert not (cfg.KNOWLEDGE_DIR / "scoped.md").exists()

    report = build_knowledge_consistency_report()
    assert report["checked_topics"] == 1
    assert report["consistent_topics"] == 1
    assert report["issue_count"] == 0
    assert report["meets_threshold"] is True
