"""Wire-level MCP tests: the JSON-RPC frames that actually leave the process.

``tests/test_mcp_structured_output.py`` covers the encoding guarantee in-process
via ``server.mcp.call_tool(...)``, which never crosses the stdio transport. The
guarantee it claims (byte-identical non-ASCII, ``isError`` on a bad argument) is
produced jointly by this package and the unpinned ``mcp`` library, so the frame
itself is the contract worth testing.

These tests spawn the real server over a real pipe. The same assertions run in
``scripts/smoke_mcp_stdio.py``, which CI wires as a gate; the helpers it uses are
imported here so the gate's own logic cannot rot into always-pass.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import subprocess
import sys
import threading
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

SMOKE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "smoke_mcp_stdio.py"

# Bounds for the spawned server: generous enough for a cold embedding model,
# small enough that a hang fails the test instead of stalling the suite.
WIRE_INIT_TIMEOUT = 60.0
WIRE_TOOL_TIMEOUT = 90.0
WIRE_TOTAL_TIMEOUT = 240.0


def _load_smoke() -> ModuleType:
    spec = importlib.util.spec_from_file_location("smoke_mcp_stdio", SMOKE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def smoke() -> ModuleType:
    return _load_smoke()


@pytest.fixture(scope="module")
def wire(smoke: ModuleType):
    """One spawned server, one initialized session, shared by the wire tests.

    Module-scoped on purpose: the spawn is the expensive part, and these tests
    assert on a contract that cannot change mid-session.
    """
    import os
    import tempfile

    with tempfile.TemporaryDirectory(prefix="cm-wire-test-") as tmp:
        data_dir = str(Path(tmp) / "data")
        env = os.environ.copy()
        env.update(
            {
                "PYTHONUNBUFFERED": "1",
                "CONSOLIDATION_MEMORY_DATA_DIR": data_dir,
                "CONSOLIDATION_MEMORY_PROJECT": "wire_test",
                "CONSOLIDATION_MEMORY_EMBEDDING_BACKEND": "fastembed",
                "CONSOLIDATION_MEMORY_LLM_BACKEND": "disabled",
                "CONSOLIDATION_MEMORY_WARMUP_ON_START": "0",
                "CONSOLIDATION_MEMORY_PRELOAD_NUMERIC_BACKENDS_ON_START": "1",
                "CONSOLIDATION_MEMORY_PRELOAD_SCIPY_ON_START": "0",
                "CONSOLIDATION_MEMORY_STATUS_LIGHTWEIGHT": "1",
                "CONSOLIDATION_MEMORY_MCP_AUTO_CONSOLIDATE": "0",
                "CONSOLIDATION_MEMORY_STDIO_SINGLETON": "0",
                "CONSOLIDATION_MEMORY_IDLE_TIMEOUT_SECONDS": "0",
                "CONSOLIDATION_MEMORY_CLIENT_INIT_TIMEOUT_SECONDS": "30",
                "CONSOLIDATION_MEMORY_DEFERRED_KNOWLEDGE_RETRY_SECONDS": "0",
            }
        )
        repo_src = Path(__file__).resolve().parents[1] / "src"
        if repo_src.is_dir():
            env["PYTHONPATH"] = str(repo_src) + os.pathsep + env.get("PYTHONPATH", "")

        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "consolidation_memory",
                "--project",
                "wire_test",
                "serve",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="strict",
            env=env,
            bufsize=1,
        )
        stderr: list[str] = []
        threading.Thread(
            target=smoke._drain_stderr, args=(proc, stderr), daemon=True
        ).start()
        assert proc.stdout is not None
        reader = smoke._LineReader(proc.stdout)
        reader.start()
        deadline = smoke._Deadline(WIRE_TOTAL_TIMEOUT)
        state = {"next_id": 0}

        def request(method: str, params: dict[str, Any], timeout: float) -> dict:
            state["next_id"] += 1
            request_id = state["next_id"]
            smoke._send(
                proc,
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": method,
                    "params": params,
                },
            )
            msg, _ = smoke._read_until(
                reader,
                request_id,
                timeout=timeout,
                deadline=deadline,
                returncode=proc.poll,
            )
            return smoke._result(msg, request_id)

        def call(tool: str, arguments: dict[str, Any], timeout: float) -> tuple[dict, str]:
            state["next_id"] += 1
            request_id = state["next_id"]
            smoke._send(
                proc,
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": "tools/call",
                    "params": {"name": tool, "arguments": arguments},
                },
            )
            return smoke._read_until(
                reader,
                request_id,
                timeout=timeout,
                deadline=deadline,
                returncode=proc.poll,
            )

        def tools() -> list[dict[str, Any]]:
            listed = request("tools/list", {}, WIRE_INIT_TIMEOUT).get("tools")
            return [tool for tool in (listed or []) if isinstance(tool, dict)]

        try:
            smoke._send(
                proc,
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": smoke.REQUESTED_PROTOCOL_VERSION,
                        "capabilities": {},
                        "clientInfo": {"name": "wire_test", "version": "0"},
                    },
                },
            )
            init_msg, _ = smoke._read_until(
                reader,
                1,
                timeout=WIRE_INIT_TIMEOUT,
                deadline=deadline,
                returncode=proc.poll,
            )
            smoke._send(proc, {"jsonrpc": "2.0", "method": "notifications/initialized"})
            yield {
                "init": smoke._result(init_msg, 1),
                "call": call,
                "tools": tools,
                "stderr": stderr,
            }
        finally:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                proc.kill()


# ── the wire contract ────────────────────────────────────────────────────────


def test_initialize_negotiates_a_supported_protocol_version(wire) -> None:
    """The handshake must name a version, and the mcp library must agree to it."""
    negotiated = wire["init"].get("protocolVersion")
    assert isinstance(negotiated, str) and negotiated, wire["init"]

    smoke = _load_smoke()
    expected = smoke._expected_protocol_version(smoke.REQUESTED_PROTOCOL_VERSION)
    if expected is not None:
        assert negotiated == expected
    assert (wire["init"].get("serverInfo") or {}).get("name")


def test_tools_list_publishes_output_schemas(wire) -> None:
    """Every advertised tool ships a closed ``anyOf[success, error]`` schema."""
    smoke = _load_smoke()
    tools = wire["tools"]()
    assert tools, "tools/list returned no tools"
    detail = smoke._check_output_schemas(tools)
    assert f"{len(tools)} tools" in detail
    assert smoke._check_input_schemas(tools)
    assert f"{len(tools)} tools:" in smoke._check_tool_count(tools)


def test_non_ascii_round_trip_over_the_wire(wire) -> None:
    """Store non-ASCII text, read it back, and pin both frames byte-for-byte."""
    smoke = _load_smoke()
    detail = smoke._check_non_ascii_round_trip(
        lambda: wire["call"](
            "memory_store",
            {"content": smoke.ROUND_TRIP_CONTENT, "content_type": "exchange"},
            WIRE_TOOL_TIMEOUT,
        ),
        lambda: wire["call"](
            "memory_search", {"query": smoke._SEARCH_QUERY, "limit": 5}, WIRE_TOOL_TIMEOUT
        ),
    )
    assert "raw UTF-8 on the wire" in detail
    assert f"{len(smoke.ROUND_TRIP_CONTENT.encode('utf-8'))}B" in detail


def test_unknown_argument_is_rejected_on_the_wire(wire) -> None:
    """A bad argument is ``isError: true`` on the wire, not a dropped key."""
    smoke = _load_smoke()
    result = wire["call"](
        "memory_status", {"lightweight": True, "junk_argument": 1}, WIRE_TOOL_TIMEOUT
    )
    assert "isError=true" in smoke._check_unknown_argument(result)


# ── the gate's own logic, so it cannot rot into always-pass ─────────────────


def _stored_ok() -> tuple[dict, str]:
    """A successful ``memory_store`` response, as the server would send it."""
    return ({"result": {"structuredContent": {"status": "stored"}, "isError": False}}, "{}")



def test_round_trip_check_rejects_an_escaped_frame(smoke: ModuleType) -> None:
    """If the transport escaped structuredContent, the check must fail."""

    def frames(structured: dict[str, Any], *, ascii: bool) -> tuple[dict, str]:
        result = {
            "content": [{"text": json.dumps(structured)}],
            "structuredContent": structured,
        }
        raw = json.dumps({"result": result}, ensure_ascii=ascii)
        return {"result": result}, raw

    stored = _stored_ok
    payload = {"episodes": [{"content": smoke.ROUND_TRIP_CONTENT}]}

    assert smoke._check_non_ascii_round_trip(stored, lambda: frames(payload, ascii=False))
    with pytest.raises(RuntimeError, match="raw UTF-8"):
        smoke._check_non_ascii_round_trip(stored, lambda: frames(payload, ascii=True))


def test_round_trip_check_rejects_a_re_encoded_payload(smoke: ModuleType) -> None:
    """A payload whose literal ``\\uXXXX`` text was decoded must fail."""
    stored = _stored_ok
    corrupt = {"episodes": [{"content": smoke.ROUND_TRIP_CONTENT.replace(r"\u0393", "Γ")}]}
    result = {"content": [{"text": json.dumps(corrupt)}], "structuredContent": corrupt}
    raw = json.dumps({"result": result}, ensure_ascii=False)
    with pytest.raises(RuntimeError, match="byte-identically"):
        smoke._check_non_ascii_round_trip(stored, lambda: ({"result": result}, raw))


def test_round_trip_check_requires_both_frames_to_agree(smoke: ModuleType) -> None:
    """structuredContent and the text frame must decode to the same object."""
    stored = _stored_ok
    result = {
        "content": [{"text": json.dumps({"episodes": [{"content": "other"}]})}],
        "structuredContent": {"episodes": [{"content": smoke.ROUND_TRIP_CONTENT}]},
    }
    raw = json.dumps({"result": result}, ensure_ascii=False)
    with pytest.raises(RuntimeError, match="disagree"):
        smoke._check_non_ascii_round_trip(stored, lambda: ({"result": result}, raw))


def test_round_trip_check_rejects_an_empty_result(smoke: ModuleType) -> None:
    """No episodes means the round trip is unproven, not passed."""
    stored = _stored_ok
    empty = {"episodes": [], "total_matches": 0}
    result = {"content": [{"text": json.dumps(empty)}], "structuredContent": empty}
    raw = json.dumps({"result": result}, ensure_ascii=False)
    with pytest.raises(RuntimeError, match="unproven"):
        smoke._check_non_ascii_round_trip(stored, lambda: ({"result": result}, raw))


def test_unknown_argument_check_requires_is_error(smoke: ModuleType) -> None:
    ok = {
        "content": [
            {"text": "Error executing tool: junk_argument Extra inputs are not permitted"}
        ],
        "isError": True,
    }
    assert "junk_argument" in smoke._check_unknown_argument(({"result": ok}, "{}"))
    with pytest.raises(RuntimeError, match="not rejected"):
        smoke._check_unknown_argument(({"result": {"content": [{"text": "x"}]}}, "{}"))
    with pytest.raises(RuntimeError, match="does not name"):
        smoke._check_unknown_argument(({"result": {"content": [{"text": "boom"}], "isError": True}}, "{}"))


def test_read_until_times_out_instead_of_blocking(smoke: ModuleType) -> None:
    """A server that accepts a request and never answers must fail, not hang."""
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert proc.stdout is not None
        reader = smoke._LineReader(proc.stdout)
        reader.start()
        with pytest.raises(TimeoutError):
            smoke._read_until(
                reader,
                1,
                timeout=30.0,
                deadline=smoke._Deadline(0.5),
                returncode=proc.poll,
            )
    finally:
        proc.kill()
        proc.wait(timeout=5)


def test_read_until_surfaces_eof(smoke: ModuleType) -> None:
    """A server that dies mid-session reports EOF rather than hanging."""
    proc = subprocess.Popen(
        [sys.executable, "-c", "pass"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert proc.stdout is not None
        reader = smoke._LineReader(proc.stdout)
        reader.start()
        with pytest.raises(RuntimeError, match="EOF"):
            smoke._read_until(
                reader,
                1,
                timeout=30.0,
                deadline=smoke._Deadline(30.0),
                returncode=proc.poll,
            )
    finally:
        proc.kill()
        proc.wait(timeout=5)


def test_expected_protocol_version_is_derived_from_the_library(smoke: ModuleType) -> None:
    """A known handshake version echoes; an unknown one falls back to latest."""
    supported = smoke._expected_protocol_version(smoke.REQUESTED_PROTOCOL_VERSION)
    assert supported == smoke.REQUESTED_PROTOCOL_VERSION

    from mcp_types.version import LATEST_HANDSHAKE_VERSION

    assert smoke._expected_protocol_version("1999-01-01") == LATEST_HANDSHAKE_VERSION


def test_documented_tool_names_are_derived_from_the_file(smoke: ModuleType) -> None:
    """docs/TOOLS.md supplies the documented count; no literal lives in code."""
    names = smoke._documented_tool_names()
    assert "memory_status" in names
    assert names == list(dict.fromkeys(names)), "table of contents lists a duplicate"
    assert names == _registered_names(), (
        "docs/TOOLS.md table of contents is stale; run scripts/generate_tool_reference.py"
    )


def _registered_names() -> list[str]:
    from consolidation_memory import server

    return [tool.name for tool in asyncio.run(server.mcp.list_tools())]
