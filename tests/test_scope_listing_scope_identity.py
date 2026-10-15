"""Scope identity, liveness and metadata rules behind memory_scope_list.

Discovery must agree with the rest of the stack: a discovered scope is exactly
one scope as ``_apply_exact_scope_filters`` sees it, counts cover live rows
only, and the display-only columns are a deterministic function of the group
instead of an arbitrary row.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from consolidation_memory import server
from consolidation_memory.database import (
    close_all_connections,
    ensure_schema,
    get_connection,
    insert_episode,
    insert_knowledge_records,
    upsert_knowledge_topic,
)
from consolidation_memory.db.scope import list_scope_usage

_T1 = "2026-01-01T00:00:00+00:00"
_T2 = "2026-01-02T00:00:00+00:00"
_T3 = "2026-01-03T00:00:00+00:00"
_REPO_SRC = Path(__file__).resolve().parents[1] / "src"

# Columns whose values every other tool treats as exact-scope identity.
_CANONICAL_SEPARATORS: list[tuple[str, str]] = [
    ("namespace_sharing_mode", "shared"),
    ("app_client_provider", "acme-cloud"),
    ("app_client_external_key", "tenant-2"),
    ("agent_name", "planner"),
    ("session_kind", "thread"),
]

_DDL_PATTERN = re.compile(
    r"^\s*(CREATE|ALTER|DROP|INSERT|UPDATE|DELETE|REPLACE|PRAGMA)\b",
    re.IGNORECASE,
)


@pytest.fixture(autouse=True)
def _schema() -> None:
    """Fresh tmp data dirs start without tables; create them per test."""
    ensure_schema()


def _scope_row(**overrides: Any) -> dict[str, Any]:
    """Build a flat scope row the way the store path does."""
    row: dict[str, Any] = {
        "namespace_slug": "acme",
        "namespace_sharing_mode": "private",
        "app_client_name": "legacy_client",
        "app_client_type": "python_sdk",
        "app_client_provider": None,
        "app_client_external_key": None,
        "agent_name": None,
        "agent_external_key": None,
        "session_external_key": None,
        "session_kind": None,
        "project_slug": "alpha",
        "project_display_name": "Alpha",
        "project_root_uri": None,
        "project_repo_remote": None,
        "project_default_branch": "main",
    }
    row.update(overrides)
    return row


def _entry(payload: dict[str, Any], index: int = 0) -> dict[str, Any]:
    return payload["scopes"][index]


# ── Identity: distinct scopes must not merge ────────────────────────────────


@pytest.mark.parametrize(("column", "value"), _CANONICAL_SEPARATORS)
def test_rows_differing_in_one_canonical_key_stay_separate_scopes(
    column: str, value: str
) -> None:
    insert_episode(content="base row", scope=_scope_row(), created_at=_T1)
    insert_episode(content="variant row", scope=_scope_row(**{column: value}), created_at=_T2)

    payload = list_scope_usage(limit=10)

    assert payload["total"] == 2
    assert [entry["counts"]["episodes"] for entry in payload["scopes"]] == [1, 1]


def test_rows_differing_in_external_key_and_session_kind_stay_separate() -> None:
    """Regression: three rows over two scopes were reported as one scope."""
    insert_episode(
        content="first",
        scope=_scope_row(app_client_external_key="tenant-1", session_kind="conversation"),
        created_at=_T1,
    )
    insert_episode(
        content="second",
        scope=_scope_row(app_client_external_key="tenant-1", session_kind="conversation"),
        created_at=_T2,
    )
    insert_episode(
        content="third",
        scope=_scope_row(app_client_external_key="tenant-2", session_kind="thread"),
        created_at=_T3,
    )

    payload = list_scope_usage(limit=10)

    assert payload["total"] == 2
    discovered = {
        (entry["scope"]["app_client"]["external_key"], entry["scope"]["session"]["session_kind"]): entry
        for entry in payload["scopes"]
    }
    assert set(discovered) == {("tenant-1", "conversation"), ("tenant-2", "thread")}
    assert discovered[("tenant-1", "conversation")]["counts"]["episodes"] == 2
    assert discovered[("tenant-1", "conversation")]["last_used_at"] == _T2
    assert discovered[("tenant-2", "thread")]["counts"]["episodes"] == 1
    assert discovered[("tenant-2", "thread")]["last_used_at"] == _T3


def test_identity_metadata_is_reported_on_the_envelope() -> None:
    insert_episode(
        content="identified row",
        scope=_scope_row(
            namespace_sharing_mode="team",
            app_client_provider="acme-cloud",
            app_client_external_key="tenant-9",
            agent_name="planner",
            agent_external_key="agent-7",
            session_external_key="session-3",
            session_kind="workflow",
        ),
        created_at=_T1,
    )

    scope = _entry(list_scope_usage(limit=10))["scope"]

    assert scope["namespace"] == {"slug": "acme", "sharing_mode": "team"}
    assert scope["app_client"] == {
        "name": "legacy_client",
        "app_type": "python_sdk",
        "provider": "acme-cloud",
        "external_key": "tenant-9",
    }
    assert scope["agent"] == {"name": "planner", "external_key": "agent-7"}
    assert scope["session"] == {"external_key": "session-3", "session_kind": "workflow"}


# ── Liveness: soft-deleted rows are not stored data ─────────────────────────


def test_soft_deleted_episodes_are_not_counted_or_recent() -> None:
    insert_episode(content="live one", scope=_scope_row(), created_at=_T1)
    insert_episode(content="live two", scope=_scope_row(), created_at=_T2)
    insert_episode(content="forgotten", scope=_scope_row(), created_at=_T3, deleted=1)

    payload = list_scope_usage(limit=10)

    assert payload["total"] == 1
    assert _entry(payload)["counts"]["episodes"] == 2
    assert _entry(payload)["last_used_at"] == _T2


def test_scope_with_only_deleted_episodes_is_not_discovered() -> None:
    insert_episode(content="forgotten", scope=_scope_row(), created_at=_T1, deleted=1)

    payload = list_scope_usage(limit=10)

    assert payload == {"scopes": [], "total": 0, "offset": 0, "message": None}


def test_soft_deleted_records_are_not_counted() -> None:
    topic_id = upsert_knowledge_topic(
        filename="alpha.md",
        title="Alpha",
        summary="alpha summary",
        source_episodes=[],
        scope=_scope_row(),
        created_at=_T1,
    )
    record_ids = insert_knowledge_records(
        topic_id,
        [
            {"record_type": "fact", "content": {"text": "one"}, "embedding_text": "one"},
            {"record_type": "fact", "content": {"text": "two"}, "embedding_text": "two"},
        ],
        scope=_scope_row(),
    )
    with get_connection() as conn:
        conn.execute(
            "UPDATE knowledge_records SET deleted = 1 WHERE id = ?", (record_ids[0],)
        )

    entry = _entry(list_scope_usage(limit=10))

    assert entry["counts"] == {"episodes": 0, "records": 1, "topics": 1}


def test_topics_are_counted_as_stored_rows() -> None:
    """knowledge_topics carries no tombstone flag, so every row counts."""
    upsert_knowledge_topic(
        filename="alpha.md",
        title="Alpha",
        summary="alpha summary",
        source_episodes=[],
        scope=_scope_row(),
        created_at=_T1,
    )

    entry = _entry(list_scope_usage(limit=10))

    assert entry["counts"] == {"episodes": 0, "records": 0, "topics": 1}


# ── Display metadata: deterministic, not row-order dependent ────────────────


def _seed_display_conflict(*, blank_first: bool, blank_is_newer: bool) -> None:
    """Two rows of one canonical scope: one records display metadata, the other NULLs it."""
    named = _scope_row(
        project_root_uri="/srv/alpha",
        project_repo_remote="git@example.com:acme/alpha.git",
    )
    rows = [("named", named, _T2), ("blank", _scope_row(), _T1)]
    if blank_is_newer:
        rows = [("named", named, _T1), ("blank", _scope_row(), _T2)]
    if blank_first:
        rows.reverse()
    for content, scope_row, created_at in rows:
        insert_episode(content=content, scope=scope_row, created_at=created_at)
    with get_connection() as conn:
        conn.execute("UPDATE episodes SET project_display_name = NULL WHERE content = 'blank'")


@pytest.mark.parametrize("blank_first", [True, False])
@pytest.mark.parametrize("blank_is_newer", [True, False])
def test_display_metadata_is_identical_for_both_row_orders(
    blank_first: bool, blank_is_newer: bool
) -> None:
    """Bare columns in SELECT are row-order dependent; MAX() is not."""
    _seed_display_conflict(blank_first=blank_first, blank_is_newer=blank_is_newer)

    project = _entry(list_scope_usage(limit=10))["scope"]["project"]

    assert project == {
        "slug": "alpha",
        "display_name": "Alpha",
        "root_uri": "/srv/alpha",
        "repo_remote": "git@example.com:acme/alpha.git",
        "default_branch": "main",
    }


def test_display_metadata_picks_a_documented_value_when_rows_disagree() -> None:
    insert_episode(
        content="older name",
        scope=_scope_row(project_display_name="Alpha One", project_repo_remote="git@example.com:1"),
        created_at=_T1,
    )
    insert_episode(
        content="newer name",
        scope=_scope_row(project_display_name="Alpha Two", project_repo_remote="git@example.com:2"),
        created_at=_T2,
    )

    project = _entry(list_scope_usage(limit=10))["scope"]["project"]

    assert project["display_name"] == "Alpha Two"
    assert project["repo_remote"] == "git@example.com:2"


# ── Published contract ──────────────────────────────────────────────────────


def _published_schema(tool_name: str) -> dict[str, Any]:
    tools = asyncio.run(server.mcp.list_tools())
    tool = next(t for t in tools if t.name == tool_name)
    assert tool.output_schema is not None, tool_name
    return tool.output_schema


def test_payload_validates_against_published_schema_without_namespace_display_name() -> None:
    from consolidation_memory.tool_contracts import ScopeListNamespace

    insert_episode(
        content="contract row",
        scope=_scope_row(agent_name="planner", session_external_key="session-1"),
        created_at=_T1,
    )
    result = asyncio.run(server.mcp.call_tool("memory_scope_list", {"limit": 10}))

    assert result.is_error is False
    payload = result.structured_content
    jsonschema.validate(payload, _published_schema("memory_scope_list"))
    assert "display_name" not in payload["scopes"][0]["scope"]["namespace"]
    assert "display_name" not in ScopeListNamespace.model_fields


# ── Startup owns the schema, reads only read ────────────────────────────────


def test_read_path_issues_no_ddl_once_the_schema_exists() -> None:
    insert_episode(content="existing row", scope=_scope_row(), created_at=_T1)

    statements: list[str] = []
    with get_connection() as conn:
        conn.set_trace_callback(statements.append)
        try:
            list_scope_usage(limit=10)
        finally:
            conn.set_trace_callback(None)

    assert [sql for sql in statements if _DDL_PATTERN.match(sql)] == []


def test_cold_database_gets_a_schema_from_the_discovery_path(tmp_data_dir: Path) -> None:
    """Clientless entry points may run before any client is built."""
    from consolidation_memory.config import reset_config

    reset_config(
        _base_data_dir=Path(tmp_data_dir) / "cold" / "data",
        active_project="default",
        EMBEDDING_DIMENSION=384,
        EMBEDDING_BACKEND="fastembed",
    )
    close_all_connections()
    try:
        with get_connection() as conn:
            missing = conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'episodes'"
            ).fetchone()
        assert missing is None

        assert list_scope_usage(limit=10) == {
            "scopes": [],
            "total": 0,
            "offset": 0,
            "message": None,
        }
    finally:
        close_all_connections()
