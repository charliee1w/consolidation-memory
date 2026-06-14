#!/usr/bin/env python3
"""Require consolidation_memory memory_recall before other agent tools.

Tracks per-session recall state. Blocks PreToolUse until memory_recall starts
for the current user turn, but fail-opens on parallel batches, slow recall,
hook errors, and client timeouts so agents never freeze.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

RECALL_TOOL = "memory_recall"
ALLOWED_PRE_RECALL = {RECALL_TOOL}
MCP_INVOKE_TOOLS = {"CallMcpTool", "call_mcp_tool", "mcp"}
CONSOLIDATION_SERVER_ALIASES = {"consolidation_memory", "consolidation-memory"}
DEFAULT_RECALL_GRACE_SECONDS = 12.0
DEFAULT_RECALL_FAILURE_GRACE_SECONDS = 0.0
DEFAULT_TURN_PARALLEL_GRACE_SECONDS = 3.0
DEFAULT_TURN_FAILOPEN_SECONDS = 8.0

_EVENT_ALIASES = {
    "session_start": "session_start",
    "sessionstart": "session_start",
    "user_prompt_submit": "user_prompt_submit",
    "beforesubmitprompt": "user_prompt_submit",
    "pre_tool_use": "pre_tool_use",
    "pretooluse": "pre_tool_use",
    "beforeshellexecution": "pre_tool_use",
    "beforemcpexecution": "pre_tool_use",
    "beforereadfile": "pre_tool_use",
    "post_tool_use": "post_tool_use",
    "posttooluse": "post_tool_use",
    "aftershellexecution": "post_tool_use",
    "aftermcpexecution": "post_tool_use",
    "afterfileedit": "post_tool_use",
    "post_tool_use_failure": "post_tool_use_failure",
    "posttoolusefailure": "post_tool_use_failure",
}


def _gate_disabled() -> bool:
    return os.environ.get("GROK_MEMORY_RECALL_GATE", "1").strip().lower() in {
        "0",
        "off",
        "false",
        "no",
        "disabled",
    }


def _positive_float_env(name: str, default: float) -> float:
    raw = os.environ.get(name, "")
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _state_dir() -> Path:
    base = os.environ.get("GROK_HOOK_STATE_DIR") or os.environ.get("TEMP") or "/tmp"
    path = Path(base) / "grok-memory-gate"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_session_id(session_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in session_id)


def _done_file(session_id: str) -> Path:
    return _state_dir() / f"{_safe_session_id(session_id)}.recall_done"


def _attempt_file(session_id: str) -> Path:
    return _state_dir() / f"{_safe_session_id(session_id)}.recall_attempt"


def _turn_start_file(session_id: str) -> Path:
    return _state_dir() / f"{_safe_session_id(session_id)}.turn_start"


def _recall_grace_seconds() -> float:
    return _positive_float_env("GROK_MEMORY_RECALL_GRACE_SECONDS", DEFAULT_RECALL_GRACE_SECONDS)


def _recall_failure_grace_seconds() -> float:
    return _positive_float_env(
        "GROK_MEMORY_RECALL_FAILURE_GRACE_SECONDS",
        DEFAULT_RECALL_FAILURE_GRACE_SECONDS,
    )


def _turn_parallel_grace_seconds() -> float:
    return _positive_float_env(
        "GROK_MEMORY_RECALL_TURN_PARALLEL_GRACE_SECONDS",
        DEFAULT_TURN_PARALLEL_GRACE_SECONDS,
    )


def _turn_failopen_seconds() -> float:
    return _positive_float_env(
        "GROK_MEMORY_RECALL_TURN_FAILOPEN_SECONDS",
        DEFAULT_TURN_FAILOPEN_SECONDS,
    )


def _normalize_event(name: str) -> str:
    return _EVENT_ALIASES.get(name.replace("-", "").replace("_", "").lower(), name.lower())


def _first_str(payload: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = payload.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _session_id(payload: dict[str, Any]) -> str:
    return (
        _first_str(payload, "sessionId", "session_id", "conversationId", "conversation_id")
        or str(os.environ.get("GROK_SESSION_ID") or "").strip()
        or "default"
    )


def _outer_tool_name(payload: dict[str, Any]) -> str:
    return _first_str(payload, "toolName", "tool", "title", "name")


def _tool_input(payload: dict[str, Any]) -> dict[str, Any]:
    for key in ("toolInput", "rawInput", "input", "arguments"):
        parsed = _as_dict(payload.get(key))
        if parsed:
            return parsed
    if payload.get("query") is not None:
        return payload
    return {}


def _split_server_tool_name(value: str) -> tuple[str, str]:
    if "__" not in value:
        return "", value
    server, tool = value.split("__", 1)
    return server.strip(), tool.strip()


def _mcp_server_name(tool_input: dict[str, Any]) -> str:
    server = _first_str(
        tool_input,
        "server",
        "mcpServer",
        "mcp_server",
        "serverName",
        "server_name",
    )
    if server:
        return server
    combined = _first_str(tool_input, "tool_name", "toolName", "name", "title")
    if combined:
        server, _ = _split_server_tool_name(combined)
        return server
    return ""


def _mcp_inner_tool_name(tool_input: dict[str, Any]) -> str:
    direct = _first_str(tool_input, "toolName", "tool", "name", "title", "tool_name")
    if direct:
        _, tool = _split_server_tool_name(direct)
        return tool or direct
    return ""


def _is_consolidation_server(server: str) -> bool:
    if not server:
        return True
    normalized = server.replace("-", "_").lower()
    return normalized in {alias.replace("-", "_") for alias in CONSOLIDATION_SERVER_ALIASES}


def _mcp_tool_name(payload: dict[str, Any]) -> str | None:
    tool = _outer_tool_name(payload)
    tool_input = _tool_input(payload)

    if tool in MCP_INVOKE_TOOLS:
        server = _mcp_server_name(tool_input)
        mcp_tool = _mcp_inner_tool_name(tool_input)
        if server and not _is_consolidation_server(server):
            return None
        return mcp_tool or None

    if tool == RECALL_TOOL or tool.startswith("memory_"):
        return tool

    if _first_str(tool_input, "tool", "toolName", "name", "tool_name") == RECALL_TOOL:
        return RECALL_TOOL

    if payload.get("query") is not None and (
        not tool or tool in {"", "mcp", "CallMcpTool", "call_mcp_tool"}
    ):
        return RECALL_TOOL

    return None


def _is_allowed_pre_recall(payload: dict[str, Any]) -> bool:
    return _mcp_tool_name(payload) in ALLOWED_PRE_RECALL


def _write_timestamp(path: Path, *, now: float | None = None) -> None:
    path.write_text(str(time.time() if now is None else now), encoding="utf-8")


def _read_timestamp(path: Path) -> float | None:
    if not path.exists():
        return None
    try:
        return float(path.read_text(encoding="utf-8").strip())
    except ValueError:
        return None


def _mark_attempt(session_id: str, *, now: float | None = None) -> None:
    _write_timestamp(_attempt_file(session_id), now=now)


def _mark_done(session_id: str) -> None:
    _done_file(session_id).write_text("1", encoding="utf-8")
    _attempt_file(session_id).unlink(missing_ok=True)


def _mark_turn_start(session_id: str, *, now: float | None = None) -> None:
    _write_timestamp(_turn_start_file(session_id), now=now)


def _clear_gate(session_id: str) -> None:
    _done_file(session_id).unlink(missing_ok=True)
    _attempt_file(session_id).unlink(missing_ok=True)
    _turn_start_file(session_id).unlink(missing_ok=True)


def _turn_age_seconds(session_id: str, *, now: float | None = None) -> float | None:
    started_at = _read_timestamp(_turn_start_file(session_id))
    if started_at is None:
        return None
    current = time.time() if now is None else now
    return max(0.0, current - started_at)


def _recall_gate_open(session_id: str, *, now: float | None = None) -> bool:
    if _done_file(session_id).exists():
        return True

    attempt = _attempt_file(session_id)
    if attempt.exists():
        started_at = _read_timestamp(attempt)
        if started_at is None:
            return True
        current = time.time() if now is None else now
        if current - started_at >= _recall_grace_seconds():
            _mark_done(session_id)
        return True

    turn_age = _turn_age_seconds(session_id, now=now)
    if turn_age is not None:
        if turn_age < _turn_parallel_grace_seconds():
            return True
        if turn_age >= _turn_failopen_seconds():
            _mark_done(session_id)
            return True

    return False


def _allow() -> None:
    print(json.dumps({"decision": "allow"}))
    raise SystemExit(0)


def _deny() -> None:
    print(
        json.dumps(
            {
                "decision": "deny",
                "reason": (
                    "Call consolidation_memory memory_recall first with a query "
                    "matching the user's goal (include_knowledge=true). "
                    "Do this before any shell, file, search, or other tool. "
                    "If recall is slow, use a short query and smaller n_results; "
                    "the gate auto-unblocks after a recall attempt times out."
                ),
            }
        )
    )
    raise SystemExit(2)


def handle_hook_event(payload: dict[str, Any], *, now: float | None = None) -> str:
    """Return hook action: allow, deny, reset, or noop."""
    if _gate_disabled():
        return "allow"

    event = _normalize_event(
        str(payload.get("hookEventName") or os.environ.get("GROK_HOOK_EVENT") or "")
    )
    session_id = _session_id(payload)

    if event == "session_start":
        _clear_gate(session_id)
        return "reset"

    if event == "user_prompt_submit":
        _clear_gate(session_id)
        _mark_turn_start(session_id, now=now)
        return "reset"

    if event == "post_tool_use":
        if _is_allowed_pre_recall(payload):
            _mark_done(session_id)
        return "noop"

    if event == "post_tool_use_failure":
        if not _is_allowed_pre_recall(payload):
            return "noop"
        if not _attempt_file(session_id).exists():
            return "noop"
        started_at = _read_timestamp(_attempt_file(session_id))
        if started_at is None:
            _mark_done(session_id)
            return "noop"
        current = time.time() if now is None else now
        if current - started_at >= _recall_failure_grace_seconds():
            _mark_done(session_id)
        return "noop"

    if event == "pre_tool_use":
        if _is_allowed_pre_recall(payload):
            _mark_attempt(session_id, now=now)
            return "allow"
        if _recall_gate_open(session_id, now=now):
            return "allow"
        return "deny"

    return "noop"


def main() -> None:
    if _gate_disabled():
        _allow()
    try:
        raw = sys.stdin.read()
        payload = json.loads(raw) if raw.strip() else {}
        action = handle_hook_event(payload)
    except Exception:
        _allow()
    if action == "allow":
        _allow()
    if action == "deny":
        _deny()
    raise SystemExit(0)


if __name__ == "__main__":
    main()