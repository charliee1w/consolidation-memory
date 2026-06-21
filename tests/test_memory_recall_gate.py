"""Tests for the memory-recall PreToolUse gate hook."""

from __future__ import annotations

import importlib.util
import io
import json
import os
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

HOOK_PATH = Path(__file__).resolve().parents[1] / ".grok" / "hooks" / "require_memory_recall.py"


def _load_hook_module():
    spec = importlib.util.spec_from_file_location("require_memory_recall", HOOK_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def gate(tmp_path, monkeypatch):
    monkeypatch.setenv("GROK_HOOK_STATE_DIR", str(tmp_path))
    monkeypatch.delenv("GROK_MEMORY_RECALL_GATE", raising=False)
    return _load_hook_module()


def _recall_payload(event: str, session_id: str = "sess-1") -> dict:
    return {
        "hookEventName": event,
        "sessionId": session_id,
        "toolName": "CallMcpTool",
        "toolInput": {
            "server": "consolidation_memory",
            "toolName": "memory_recall",
            "arguments": {"query": "pytest", "include_knowledge": True},
        },
    }


def _shell_payload(event: str, session_id: str = "sess-1") -> dict:
    return {
        "hookEventName": event,
        "sessionId": session_id,
        "toolName": "Shell",
        "toolInput": {"command": "pytest -q"},
    }


def _run_hook(stdin: str, *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    """Exercise the hook entrypoint in-process (avoids flaky Windows subprocess init)."""
    module = _load_hook_module()
    merged_env = {**os.environ, **(env or {})}
    saved: dict[str, str | None] = {}
    for key, value in merged_env.items():
        saved[key] = os.environ.get(key)
        os.environ[key] = value

    stdout_stream = io.StringIO()
    returncode = 0
    old_stdin = sys.stdin
    try:
        sys.stdin = io.StringIO(stdin)
        with redirect_stdout(stdout_stream):
            try:
                module.main()
            except SystemExit as exc:
                returncode = int(exc.code) if exc.code is not None else 0
    finally:
        sys.stdin = old_stdin
        for key, previous in saved.items():
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous

    return subprocess.CompletedProcess(
        args=[sys.executable, str(HOOK_PATH)],
        returncode=returncode,
        stdout=stdout_stream.getvalue(),
        stderr="",
    )


def test_new_turn_blocks_non_recall_tools(gate):
    gate.handle_hook_event({"hookEventName": "user_prompt_submit", "sessionId": "sess-1"}, now=100.0)
    assert gate.handle_hook_event(_shell_payload("pre_tool_use"), now=106.0) == "deny"


def test_recall_attempt_allows_recall_and_marks_attempt(gate):
    gate.handle_hook_event({"hookEventName": "user_prompt_submit", "sessionId": "sess-1"})
    assert gate.handle_hook_event(_recall_payload("pre_tool_use")) == "allow"
    assert gate._attempt_file("sess-1").exists()


def test_post_recall_unblocks_other_tools(gate):
    gate.handle_hook_event({"hookEventName": "user_prompt_submit", "sessionId": "sess-1"})
    gate.handle_hook_event(_recall_payload("pre_tool_use"))
    gate.handle_hook_event(_recall_payload("post_tool_use"))
    assert gate.handle_hook_event(_shell_payload("pre_tool_use")) == "allow"


def test_user_prompt_submit_resets_gate(gate):
    gate.handle_hook_event(_recall_payload("pre_tool_use"))
    gate.handle_hook_event(_recall_payload("post_tool_use"))
    gate.handle_hook_event({"hookEventName": "user_prompt_submit", "sessionId": "sess-1"}, now=200.0)
    assert gate.handle_hook_event(_shell_payload("pre_tool_use"), now=206.0) == "deny"


def test_in_flight_recall_allows_parallel_tools(gate):
    gate.handle_hook_event({"hookEventName": "user_prompt_submit", "sessionId": "sess-1"})
    gate.handle_hook_event(_recall_payload("pre_tool_use"))
    assert gate.handle_hook_event(_shell_payload("pre_tool_use")) == "allow"
    assert gate._attempt_file("sess-1").exists()
    assert not gate._done_file("sess-1").exists()


def test_attempt_grace_marks_done_after_timeout_without_post(gate):
    gate.handle_hook_event({"hookEventName": "user_prompt_submit", "sessionId": "sess-1"})
    gate.handle_hook_event(_recall_payload("pre_tool_use"))
    started_at = float(gate._attempt_file("sess-1").read_text(encoding="utf-8"))
    assert gate.handle_hook_event(_shell_payload("pre_tool_use"), now=started_at + 1.0) == "allow"
    assert (
        gate.handle_hook_event(_shell_payload("pre_tool_use"), now=started_at + 13.0)
        == "allow"
    )
    assert gate._done_file("sess-1").exists()


def test_turn_parallel_grace_allows_early_siblings(gate):
    gate.handle_hook_event({"hookEventName": "user_prompt_submit", "sessionId": "sess-par"}, now=100.0)
    assert gate.handle_hook_event(_shell_payload("pre_tool_use", "sess-par"), now=102.0) == "allow"


def test_turn_failopen_unblocks_without_recall(gate):
    gate.handle_hook_event({"hookEventName": "user_prompt_submit", "sessionId": "sess-open"}, now=100.0)
    assert gate.handle_hook_event(_shell_payload("pre_tool_use", "sess-open"), now=108.0) == "allow"
    assert gate._done_file("sess-open").exists()


def test_raw_input_payload_unblocks_on_post(gate):
    gate.handle_hook_event({"hookEventName": "user_prompt_submit", "sessionId": "sess-raw"})
    gate.handle_hook_event(
        {
            "hookEventName": "pre_tool_use",
            "sessionId": "sess-raw",
            "title": "CallMcpTool",
            "rawInput": {
                "server": "consolidation_memory",
                "toolName": "memory_recall",
                "arguments": {"query": "hook", "include_knowledge": True},
            },
        }
    )
    gate.handle_hook_event(
        {
            "hookEventName": "post_tool_use",
            "conversationId": "sess-raw",
            "toolName": "CallMcpTool",
            "rawInput": {
                "server": "consolidation_memory",
                "toolName": "memory_recall",
                "arguments": {"query": "hook", "include_knowledge": True},
            },
        }
    )
    assert gate.handle_hook_event(_shell_payload("pre_tool_use", "sess-raw")) == "allow"


def test_use_tool_post_payload_unblocks(gate):
    gate.handle_hook_event({"hookEventName": "user_prompt_submit", "sessionId": "sess-use"})
    gate.handle_hook_event(
        {
            "hookEventName": "pre_tool_use",
            "sessionId": "sess-use",
            "toolName": "CallMcpTool",
            "rawInput": {
                "variant": "UseTool",
                "tool_name": "consolidation_memory__memory_recall",
                "tool_input": {"query": "hook", "include_knowledge": True},
            },
        }
    )
    gate.handle_hook_event(
        {
            "hookEventName": "post_tool_use",
            "sessionId": "sess-use",
            "toolName": "CallMcpTool",
            "rawInput": {
                "variant": "UseTool",
                "tool_name": "consolidation_memory__memory_recall",
                "tool_input": {"query": "hook", "include_knowledge": True},
            },
        }
    )
    assert gate.handle_hook_event(_shell_payload("pre_tool_use", "sess-use")) == "allow"


def test_direct_memory_recall_tool_name(gate):
    gate.handle_hook_event({"hookEventName": "user_prompt_submit", "sessionId": "sess-direct"})
    payload = {
        "hookEventName": "pre_tool_use",
        "sessionId": "sess-direct",
        "toolName": "memory_recall",
        "toolInput": {"query": "direct", "include_knowledge": True},
    }
    assert gate.handle_hook_event(payload) == "allow"
    gate.handle_hook_event({**payload, "hookEventName": "post_tool_use"})
    assert gate.handle_hook_event(_shell_payload("pre_tool_use", "sess-direct")) == "allow"


def test_non_recall_mcp_post_does_not_open_gate(gate):
    gate.handle_hook_event({"hookEventName": "user_prompt_submit", "sessionId": "sess-mcp"}, now=100.0)
    gate.handle_hook_event(
        {
            "hookEventName": "post_tool_use",
            "sessionId": "sess-mcp",
            "toolName": "CallMcpTool",
            "toolInput": {
                "server": "consolidation_memory",
                "toolName": "memory_store",
                "arguments": {"content": "x", "content_type": "fact", "tags": ["t"]},
            },
        }
    )
    assert gate.handle_hook_event(_shell_payload("pre_tool_use", "sess-mcp"), now=106.0) == "deny"


def test_parallel_batch_after_recall_pre(gate):
    gate.handle_hook_event({"hookEventName": "user_prompt_submit", "sessionId": "sess-par"})
    gate.handle_hook_event(_recall_payload("pre_tool_use", "sess-par"))
    assert gate.handle_hook_event(_shell_payload("pre_tool_use", "sess-par")) == "allow"
    assert gate.handle_hook_event(
        {
            "hookEventName": "pre_tool_use",
            "sessionId": "sess-par",
            "toolName": "Grep",
            "toolInput": {"pattern": "hook", "path": "."},
        }
    ) == "allow"


def test_recall_failure_unblocks_immediately_by_default(gate):
    gate.handle_hook_event({"hookEventName": "user_prompt_submit", "sessionId": "sess-fail"})
    gate.handle_hook_event(_recall_payload("pre_tool_use", "sess-fail"))
    gate.handle_hook_event(_recall_payload("post_tool_use_failure", "sess-fail"))
    assert gate._done_file("sess-fail").exists()
    assert gate.handle_hook_event(_shell_payload("pre_tool_use", "sess-fail")) == "allow"


def test_recall_failure_respects_configured_grace(gate, monkeypatch):
    monkeypatch.setenv("GROK_MEMORY_RECALL_FAILURE_GRACE_SECONDS", "3")
    gate = _load_hook_module()
    gate.handle_hook_event({"hookEventName": "user_prompt_submit", "sessionId": "sess-fail"})
    gate.handle_hook_event(_recall_payload("pre_tool_use", "sess-fail"))
    started_at = float(gate._attempt_file("sess-fail").read_text(encoding="utf-8"))
    gate.handle_hook_event(_recall_payload("post_tool_use_failure", "sess-fail"), now=started_at + 1.0)
    assert not gate._done_file("sess-fail").exists()
    gate.handle_hook_event(_recall_payload("post_tool_use_failure", "sess-fail"), now=started_at + 4.0)
    assert gate._done_file("sess-fail").exists()


def test_grok_session_id_env_used_when_payload_missing_session(gate, monkeypatch):
    monkeypatch.setenv("GROK_SESSION_ID", "env-session")
    gate.handle_hook_event({"hookEventName": "user_prompt_submit", "sessionId": "env-session"})
    gate.handle_hook_event(_recall_payload("pre_tool_use", ""))
    gate.handle_hook_event(
        {
            "hookEventName": "post_tool_use",
            "toolName": "CallMcpTool",
            "toolInput": {
                "server": "consolidation_memory",
                "toolName": "memory_recall",
                "arguments": {"query": "env"},
            },
        }
    )
    assert gate.handle_hook_event(_shell_payload("pre_tool_use", "env-session")) == "allow"


def test_gate_disabled_env_allows_everything(gate, monkeypatch):
    monkeypatch.setenv("GROK_MEMORY_RECALL_GATE", "off")
    gate = _load_hook_module()
    assert gate.handle_hook_event(_shell_payload("pre_tool_use")) == "allow"


def test_main_fail_open_on_invalid_json(tmp_path, monkeypatch):
    monkeypatch.setenv("GROK_HOOK_STATE_DIR", str(tmp_path))
    result = _run_hook("{not-json", env={"GROK_HOOK_STATE_DIR": str(tmp_path)})
    assert result.returncode == 0
    assert json.loads(result.stdout) == {"decision": "allow"}


def test_main_fail_open_on_empty_stdin(tmp_path, monkeypatch):
    monkeypatch.setenv("GROK_HOOK_STATE_DIR", str(tmp_path))
    result = _run_hook("", env={"GROK_HOOK_STATE_DIR": str(tmp_path)})
    assert result.returncode == 0


def test_main_deny_emits_json(tmp_path, monkeypatch):
    monkeypatch.setenv("GROK_HOOK_STATE_DIR", str(tmp_path))
    deny_payload = {
        "hookEventName": "pre_tool_use",
        "sessionId": "deny-sess",
        "toolName": "Shell",
        "toolInput": {"command": "ls"},
    }
    result = _run_hook(
        json.dumps(deny_payload),
        env={"GROK_HOOK_STATE_DIR": str(tmp_path)},
    )
    assert result.returncode == 2
    body = json.loads(result.stdout)
    assert body["decision"] == "deny"
    assert "memory_recall" in body["reason"]


def test_between_parallel_grace_and_failopen_denies(gate):
    gate.handle_hook_event({"hookEventName": "user_prompt_submit", "sessionId": "sess-gap"}, now=100.0)
    assert gate.handle_hook_event(_shell_payload("pre_tool_use", "sess-gap"), now=106.0) == "deny"


def test_corrupt_attempt_timestamp_allows_siblings(gate):
    gate.handle_hook_event({"hookEventName": "user_prompt_submit", "sessionId": "sess-bad"})
    gate.handle_hook_event(_recall_payload("pre_tool_use", "sess-bad"))
    gate._attempt_file("sess-bad").write_text("not-a-timestamp", encoding="utf-8")
    assert gate.handle_hook_event(_shell_payload("pre_tool_use", "sess-bad")) == "allow"


def test_consolidation_memory_hyphen_server_alias(gate):
    gate.handle_hook_event({"hookEventName": "user_prompt_submit", "sessionId": "sess-hyphen"})
    payload = {
        "hookEventName": "pre_tool_use",
        "sessionId": "sess-hyphen",
        "toolName": "call_mcp_tool",
        "toolInput": {
            "server": "consolidation-memory",
            "toolName": "memory_recall",
            "arguments": {"query": "alias", "include_knowledge": True},
        },
    }
    assert gate.handle_hook_event(payload) == "allow"
    gate.handle_hook_event({**payload, "hookEventName": "post_tool_use"})
    assert gate.handle_hook_event(_shell_payload("pre_tool_use", "sess-hyphen")) == "allow"


def test_session_start_clears_done_state(gate):
    gate.handle_hook_event(_recall_payload("pre_tool_use"))
    gate.handle_hook_event(_recall_payload("post_tool_use"))
    assert gate._done_file("sess-1").exists()
    gate.handle_hook_event({"hookEventName": "session_start", "sessionId": "sess-1"})
    assert not gate._done_file("sess-1").exists()


def test_main_passive_events_exit_zero(tmp_path, monkeypatch):
    monkeypatch.setenv("GROK_HOOK_STATE_DIR", str(tmp_path))
    for event in ("user_prompt_submit", "post_tool_use", "session_start"):
        result = _run_hook(
            json.dumps({"hookEventName": event, "sessionId": "passive-sess"}),
            env={"GROK_HOOK_STATE_DIR": str(tmp_path)},
        )
        assert result.returncode == 0


def test_full_turn_simulation_subprocess(tmp_path, monkeypatch):
    monkeypatch.setenv("GROK_HOOK_STATE_DIR", str(tmp_path))
    env = {"GROK_HOOK_STATE_DIR": str(tmp_path)}
    sid = "sim-sess"

    steps = [
        ({"hookEventName": "user_prompt_submit", "sessionId": sid}, 0, None),
        (
            {
                "hookEventName": "pre_tool_use",
                "sessionId": sid,
                "toolName": "CallMcpTool",
                "toolInput": {
                    "server": "consolidation_memory",
                    "toolName": "memory_recall",
                    "arguments": {"query": "x", "include_knowledge": True},
                },
            },
            0,
            "allow",
        ),
        (
            {
                "hookEventName": "pre_tool_use",
                "sessionId": sid,
                "toolName": "Grep",
                "toolInput": {"pattern": "hook"},
            },
            0,
            "allow",
        ),
        (
            {
                "hookEventName": "post_tool_use",
                "sessionId": sid,
                "toolName": "CallMcpTool",
                "rawInput": {
                    "variant": "UseTool",
                    "tool_name": "consolidation_memory__memory_recall",
                    "tool_input": {"query": "x"},
                },
            },
            0,
            None,
        ),
        (
            {
                "hookEventName": "pre_tool_use",
                "sessionId": sid,
                "toolName": "Read",
                "toolInput": {"path": "README.md"},
            },
            0,
            "allow",
        ),
    ]
    for payload, expect_rc, expect_decision in steps:
        result = _run_hook(json.dumps(payload), env=env)
        assert result.returncode == expect_rc
        if expect_decision is not None:
            assert json.loads(result.stdout)["decision"] == expect_decision