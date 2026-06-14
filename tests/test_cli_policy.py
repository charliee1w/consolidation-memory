"""CLI tests for persisted policy administration."""

from __future__ import annotations

import json

import pytest

from consolidation_memory.database import ensure_schema, list_policy_admin_rows


def test_policy_list_reports_empty_state(capsys):
    from consolidation_memory.cli import cmd_policy_list

    ensure_schema()
    cmd_policy_list()
    output = capsys.readouterr().out
    assert "No persisted access policies." in output


def test_policy_grant_creates_binding(capsys):
    from consolidation_memory.cli import cmd_policy_grant, cmd_policy_list

    ensure_schema()
    cmd_policy_grant(
        namespace="team-a",
        project="repo-a",
        principal_type="app_client",
        principal_key="python_sdk:legacy_client",
        write_mode="deny",
        read_visibility="namespace",
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "granted"
    assert payload["write_mode"] == "deny"

    rows = list_policy_admin_rows()
    assert any(
        row.get("namespace_slug") == "team-a"
        and row.get("project_slug") == "repo-a"
        and row.get("write_mode") == "deny"
        for row in rows
    )

    capsys.readouterr()
    cmd_policy_list()
    listed = capsys.readouterr().out
    assert "team-a" in listed
    assert "deny" in listed


def test_policy_grant_requires_mode_or_visibility():
    from consolidation_memory.cli import cmd_policy_grant

    with pytest.raises(SystemExit):
        cmd_policy_grant(
            namespace=None,
            project=None,
            principal_type="app_client",
            principal_key="python_sdk:legacy_client",
            write_mode=None,
            read_visibility=None,
        )