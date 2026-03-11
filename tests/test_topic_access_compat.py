"""Regression tests for legacy topic-access APIs used by recall surfaces."""

from consolidation_memory.database import (
    ensure_schema,
    get_knowledge_topic,
    increment_topic_access,
    upsert_knowledge_topic,
)


def test_increment_topic_access_updates_by_filename(tmp_data_dir):
    ensure_schema()
    upsert_knowledge_topic(
        "python-runtime.md",
        "Python Runtime",
        "Tracks runtime details",
        source_episodes=[],
    )

    before = get_knowledge_topic("python-runtime.md")
    assert before is not None
    assert before["access_count"] == 0

    increment_topic_access(["python-runtime.md"])

    after = get_knowledge_topic("python-runtime.md")
    assert after is not None
    assert after["access_count"] == 1


def test_increment_topic_access_updates_by_storage_filename(tmp_data_dir):
    ensure_schema()
    upsert_knowledge_topic(
        "storage-key.md",
        "Storage Key",
        "Ensures storage filename lookups remain compatible",
        source_episodes=[],
    )

    topic = get_knowledge_topic("storage-key.md")
    assert topic is not None
    storage_filename = topic["storage_filename"]

    increment_topic_access([storage_filename])

    updated = get_knowledge_topic("storage-key.md")
    assert updated is not None
    assert updated["access_count"] == 1


def test_increment_topic_access_deduplicates_and_ignores_empty_tokens(tmp_data_dir):
    ensure_schema()
    upsert_knowledge_topic(
        "dedupe.md",
        "Dedupe",
        "No double count for duplicate inputs",
        source_episodes=[],
    )

    increment_topic_access(["dedupe.md", "", "dedupe.md", "   "])

    topic = get_knowledge_topic("dedupe.md")
    assert topic is not None
    assert topic["access_count"] == 1


def test_increment_topic_access_respects_scope_filters(tmp_data_dir):
    ensure_schema()
    scope_a = {
        "namespace_slug": "default",
        "project_slug": "default",
        "app_client_name": "client-a",
        "app_client_type": "python_sdk",
    }
    scope_b = {
        "namespace_slug": "default",
        "project_slug": "default",
        "app_client_name": "client-b",
        "app_client_type": "python_sdk",
    }
    upsert_knowledge_topic("shared.md", "Shared A", "scope a", source_episodes=[], scope=scope_a)
    upsert_knowledge_topic("shared.md", "Shared B", "scope b", source_episodes=[], scope=scope_b)

    increment_topic_access(["shared.md"], scope=scope_a)

    topic_a = get_knowledge_topic("shared.md", scope=scope_a)
    topic_b = get_knowledge_topic("shared.md", scope=scope_b)
    assert topic_a is not None
    assert topic_b is not None
    assert topic_a["access_count"] == 1
    assert topic_b["access_count"] == 0
