#!/usr/bin/env python3
"""MCP stdio wire gate: handshake, published schemas, non-ASCII round trip.

Spawns the real server over the real stdio JSON-RPC transport and asserts what
in-process tests cannot: the frames that actually travel over the pipe. The
guarantees below are produced jointly by this package and the (unpinned) ``mcp``
library, so only a spawned process pins them.

Every check runs against one initialized session but is self-contained, so a
single failure is reported without hiding the rest. Each wait is bounded by a
deadline against a reader thread, never by a blocking ``readline()``: a hung
server fails the gate instead of hanging the job.

Hermetic by construction: stdlib only, no API keys (LLM disabled), a throwaway
data directory, and no network beyond the embedding model the CI job already
downloads for its pytest leg.

Usage:
  python scripts/smoke_mcp_stdio.py
  python scripts/smoke_mcp_stdio.py --init-timeout 30 --recall-timeout 45
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path

# Sent in `initialize`; the server's answer is derived from this and from the
# installed `mcp` library (see `_expected_protocol_version`), never hardcoded.
REQUESTED_PROTOCOL_VERSION = "2024-11-05"

# Real non-ASCII plus literal backslash-u sequences: a byte-identical round trip
# over this string fails if the payload is decoded, re-encoded or escaped
# anywhere between SQLite and the wire.
ROUND_TRIP_CONTENT = (
    "wire round trip marker: mcp-nonascii-wire\n"
    "non-ascii: Γειά σου κόσμε — مرحبا 世界 你好, ẞ\n"
    'literal escapes stay six characters each: \\u0393 and \\u043f, quote " and backslash \\'
)
_LITERAL_ESCAPES = re.compile(r"\\u[0-9a-fA-F]{4}")
_EXPECTED_ESCAPES = [r"\u0393", r"\u043f"]
_NON_ASCII_SAMPLE = "Γειά σου κόσμε — مرحبا 世界 你好, ẞ"
# Substring of the stored content; `memory_search` is a case-insensitive
# keyword search, so the ASCII marker is enough to find the episode back.
_SEARCH_QUERY = "mcp-nonascii-wire"


class _Deadline:
    """Monotonic wall-clock budget shared by every wait in one run."""

    def __init__(self, total: float) -> None:
        self._expiry = time.monotonic() + total

    def remaining(self) -> float:
        return max(0.0, self._expiry - time.monotonic())

    def expired(self) -> bool:
        return self.remaining() <= 0.0


class _LineReader(threading.Thread):
    """Feed every stdout line to a queue so waits can time out instead of block."""

    def __init__(self, stream: object) -> None:
        super().__init__(daemon=True)
        self._stream = stream
        self._lines: queue.Queue[str | None] = queue.Queue()

    def run(self) -> None:
        try:
            # The stream raises when the child process is killed; the reader
            # thread just stops reading and the sentinel unblocks the consumer.
            with contextlib.suppress(Exception):
                for line in self._stream:  # type: ignore[attr-defined]
                    self._lines.put(line)
        finally:
            self._lines.put(None)

    def next_line(self, timeout: float) -> str | None:
        try:
            return self._lines.get(timeout=timeout)
        except queue.Empty:
            return None

    @property
    def exhausted(self) -> bool:
        """True once stdout closed and every line it produced has been consumed."""
        return self._lines.empty() and not self.is_alive()


def _expected_protocol_version(requested: str) -> str | None:
    """What the installed ``mcp`` server agrees to, or None if it cannot say.

    ``runner._negotiate_initialize`` echoes the requested version when it is a
    handshake version and otherwise answers with its latest. Reading the ladder
    from the library keeps the expectation correct across ``mcp`` upgrades.
    """
    try:
        from mcp_types.version import (
            HANDSHAKE_PROTOCOL_VERSIONS,
            LATEST_HANDSHAKE_VERSION,
        )
    except ImportError:
        return None
    return requested if requested in HANDSHAKE_PROTOCOL_VERSIONS else LATEST_HANDSHAKE_VERSION


def _send(proc: subprocess.Popen[str], obj: dict[str, object]) -> None:
    assert proc.stdin is not None
    proc.stdin.write(json.dumps(obj) + "\n")
    proc.stdin.flush()


def _read_until(
    reader: _LineReader,
    request_id: int,
    *,
    timeout: float,
    deadline: _Deadline,
    returncode: Callable[[], int | None],
) -> tuple[dict[str, object], str]:
    """Wait for the JSON-RPC response to ``request_id``; return it with its raw line.

    The raw line is part of the contract under test: only the raw frame shows
    whether the transport escaped non-ASCII on the way out.
    """
    while True:
        remaining = min(timeout, deadline.remaining())
        if remaining <= 0:
            raise TimeoutError(
                f"no MCP response for id={request_id}: step budget {timeout:g}s "
                f"or total budget exhausted"
            )
        line = reader.next_line(remaining)
        if line is None:
            if reader.exhausted:
                raise RuntimeError(f"MCP stdout EOF (rc={returncode()})")
            raise TimeoutError(f"no MCP response for id={request_id} within {remaining:g}s")
        stripped = line.strip()
        if not stripped:
            continue
        try:
            msg = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if msg.get("id") != request_id:
            continue  # ignore notifications / other ids
        if "error" in msg:
            raise RuntimeError(f"MCP error for id={request_id}: {msg['error']}")
        return msg, stripped


def _result(msg: dict[str, object], request_id: int) -> dict[str, object]:
    result = msg.get("result")
    if not isinstance(result, dict):
        raise RuntimeError(  # noqa: TRY004 - payload shape assertion, not an argument validator
            f"id={request_id}: result is not an object: {json.dumps(msg)[:400]}"
        )
    return result


def _structured_region(raw_line: str) -> str:
    """The ``structuredContent`` slice of a raw JSON-RPC frame."""
    marker = '"structuredContent":'
    index = raw_line.find(marker)
    if index < 0:
        raise RuntimeError(f"frame carries no structuredContent: {raw_line[:400]}")
    return raw_line[index:]


def _documented_tool_names() -> list[str]:
    """Tool names listed in the generated docs/TOOLS.md table of contents.

    Derived from the file, never a literal count, so it tracks the published
    surface (``tests/test_tool_reference_sync.py`` keeps the file generated).
    """
    path = Path(__file__).resolve().parents[1] / "docs" / "TOOLS.md"
    if not path.is_file():
        raise RuntimeError(f"{path} is missing; cannot cross-check the documented tool count")
    names: list[str] = []
    in_toc = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            in_toc = line.strip() == "## Tools"
            continue
        if not in_toc:
            continue
        entry = re.fullmatch(r"- \[([^\]]+)\]\(#\1\)", line.strip())
        if entry is None:
            if names:  # first non-entry line ends the table of contents
                break
            continue
        names.append(entry.group(1))
    if not names:
        raise RuntimeError(f"{path} lists no tools in its table of contents")
    return names


def _registered_tool_names() -> list[str]:
    """Tool names the code registers, read from the same list clients see."""
    from consolidation_memory import server

    return [tool.name for tool in asyncio.run(server.mcp.list_tools())]


def _check_output_schemas(tools: list[dict[str, object]]) -> str:
    """Every tool publishes ``anyOf[success, error]``, and the success arm is real."""
    for tool in tools:
        name = str(tool.get("name"))
        schema = tool.get("outputSchema")
        if not isinstance(schema, dict) or not schema:
            raise RuntimeError(f"{name}: no outputSchema published")
        arms = schema.get("anyOf")
        if not isinstance(arms, list) or len(arms) != 2:
            raise RuntimeError(f"{name}: outputSchema is not an anyOf pair: {json.dumps(schema)[:200]}")
        success, error = arms
        if not isinstance(success, dict) or success.get("type") != "object":
            raise RuntimeError(f"{name}: success arm is not an object")
        if not (success.get("properties") or {}):
            raise RuntimeError(f"{name}: success arm declares no properties")
        if not isinstance(error, dict) or error.get("type") != "object":
            raise RuntimeError(f"{name}: error arm is not an object")
        if "error" not in (error.get("properties") or {}):
            raise RuntimeError(f"{name}: error arm has no 'error' key")
    return f"{len(tools)} tools publish anyOf[success, error]"


def _check_input_schemas(tools: list[dict[str, object]]) -> str:
    """Published input schemas must carry descriptions and forbid extra keys."""
    for tool in tools:
        name = str(tool.get("name"))
        schema = tool.get("inputSchema") or {}
        if schema.get("additionalProperties") is not False:
            raise RuntimeError(f"{name}: inputSchema allows additional properties")
        for prop_name, prop_spec in (schema.get("properties") or {}).items():
            if not prop_spec.get("description"):
                raise RuntimeError(f"{name}.{prop_name}: missing description")
    return f"{len(tools)} input schemas documented and closed"


def _check_tool_count(tools: list[dict[str, object]]) -> str:
    """The wire surface, the registered surface and the docs must agree exactly."""
    wire = sorted(str(tool.get("name")) for tool in tools)
    registered = sorted(_registered_tool_names())
    documented = sorted(_documented_tool_names())
    if wire != registered:
        only_wire = sorted(set(wire) - set(registered))
        only_code = sorted(set(registered) - set(wire))
        raise RuntimeError(f"tools/list != registered: wire-only={only_wire} code-only={only_code}")
    if wire != documented:
        only_wire = sorted(set(wire) - set(documented))
        only_docs = sorted(set(documented) - set(wire))
        raise RuntimeError(f"tools/list != docs/TOOLS.md: wire-only={only_wire} docs-only={only_docs}")
    return (
        f"{len(wire)} tools: tools/list == registered == docs/TOOLS.md "
        f"({len(tools)} on the wire)"
    )


def _check_non_ascii_round_trip(
    store: Callable[[], tuple[dict[str, object], str]],
    search: Callable[[], tuple[dict[str, object], str]],
) -> str:
    """Store non-ASCII text, read it back, and pin both frames byte-for-byte.

    Three independent guarantees, so a regression names itself:

    1. ``structuredContent`` and the ``content`` text frame decode to the same
       bytes as the stored string, over the real transport;
    2. nothing double-decodes the literal ``\\\\u0393`` text into a real character;
    3. the raw frame carries ``structuredContent`` as raw UTF-8 rather than
       ``\\\\uXXXX`` escapes.

    Both channels are encoded by the ``mcp`` library through pydantic
    (``pydantic_core.to_json(..., indent=2)`` builds the text frame), which emits
    non-ASCII as raw UTF-8, so neither channel introduces ``\\\\uXXXX`` escaping and
    the literal ``\\\\u0393`` text stays the six characters it was stored as. The text
    frame sits one JSON encoding layer deeper, because the payload is a string
    nested inside the outer frame; that layer escapes backslashes and newlines,
    not characters, so both channels decode to the same bytes.
    """
    store_result = _result(store()[0], "store")
    if store_result.get("isError"):
        raise RuntimeError(f"memory_store failed: {json.dumps(store_result)[:400]}")
    stored = store_result.get("structuredContent") or {}
    if stored.get("status") not in {"stored", "duplicate_detected"}:
        raise RuntimeError(f"memory_store did not store the episode: {json.dumps(stored)[:400]}")

    search_msg, raw_line = search()
    search_result = _result(search_msg, "search")
    if search_result.get("isError"):
        raise RuntimeError(f"memory_search failed: {json.dumps(search_result)[:400]}")

    frames = search_result.get("content")
    if not isinstance(frames, list) or not frames:
        raise RuntimeError("memory_search returned no content frames")
    text = frames[0].get("text")
    if not isinstance(text, str):
        raise RuntimeError(  # noqa: TRY004 - payload shape assertion, not an argument validator
            "memory_search content frame carries no text"
        )
    structured = search_result.get("structuredContent")
    if not isinstance(structured, dict):
        raise RuntimeError(  # noqa: TRY004 - payload shape assertion, not an argument validator
            "memory_search returned no structuredContent"
        )

    from_text = json.loads(text)
    from_structured = json.loads(json.dumps(structured))
    if from_text != from_structured:
        raise RuntimeError("content text frame and structuredContent disagree")

    episodes = from_structured.get("episodes") or []
    if not episodes:
        raise RuntimeError("memory_search returned no episodes; the round trip is unproven")
    matches = [row for row in episodes if row.get("content") == ROUND_TRIP_CONTENT]
    if not matches:
        raise RuntimeError(
            f"stored content not returned byte-identically; got "
            f"{json.dumps([row.get('content') for row in episodes], ensure_ascii=False)[:400]}"
        )
    for row in matches:
        content = str(row["content"])
        if content.encode("utf-8") != ROUND_TRIP_CONTENT.encode("utf-8"):
            raise RuntimeError("returned content differs at the byte level")
        if _LITERAL_ESCAPES.findall(content) != _EXPECTED_ESCAPES:
            raise RuntimeError(
                f"literal escape sequences were re-encoded or decoded: "
                f"{_LITERAL_ESCAPES.findall(content)}"
            )
    if _NON_ASCII_SAMPLE not in str(matches[0]["content"]):
        raise RuntimeError("non-ASCII sample missing from the returned content")

    region = _structured_region(raw_line)
    if _NON_ASCII_SAMPLE not in region:
        raise RuntimeError("structuredContent left the process as \\uXXXX escapes, not raw UTF-8")

    return (
        f"{len(ROUND_TRIP_CONTENT.encode('utf-8'))}B non-ASCII identical in both frames; "
        f"raw UTF-8 on the wire"
    )


def _drain_stderr(proc: subprocess.Popen[str], sink: list[str]) -> None:
    assert proc.stderr is not None
    for line in proc.stderr:
        sink.append(line.rstrip())


def run_smoke(
    *,
    status_timeout: float,
    recall_timeout: float,
    init_timeout: float,
    total_timeout: float,
) -> int:
    checks: list[tuple[str, bool, str]] = []

    def record(name: str, detail: str) -> None:
        checks.append((name, True, detail))
        print(f"  PASS  {name}: {detail}", flush=True)

    def attempt(name: str, check: Callable[[], str]) -> None:
        try:
            record(name, check())
        except Exception as exc:
            checks.append((name, False, str(exc)))
            print(f"  FAIL  {name}: {exc}", flush=True)

    with tempfile.TemporaryDirectory(prefix="cm-mcp-smoke-") as tmp:
        data_dir = str(Path(tmp) / "data")
        env = os.environ.copy()
        env.update(
            {
                "PYTHONUNBUFFERED": "1",
                "CONSOLIDATION_MEMORY_DATA_DIR": data_dir,
                "CONSOLIDATION_MEMORY_PROJECT": "mcp_smoke",
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
                "CONSOLIDATION_MEMORY_RECALL_TIMEOUT_SECONDS": str(int(recall_timeout)),
                "CONSOLIDATION_MEMORY_TIMEOUT_MEMORY_STATUS": str(int(status_timeout)),
                "CONSOLIDATION_MEMORY_DEFERRED_KNOWLEDGE_RETRY_SECONDS": "0",
            }
        )
        # The tool-count cross-check imports the package in this process; point
        # it at the throwaway data dir so the gate touches no real state.
        os.environ["CONSOLIDATION_MEMORY_DATA_DIR"] = data_dir
        os.environ.setdefault("CONSOLIDATION_MEMORY_EMBEDDING_BACKEND", "fastembed")
        # Prefer local src tree when present.
        repo_src = Path(__file__).resolve().parents[1] / "src"
        if repo_src.is_dir():
            env["PYTHONPATH"] = str(repo_src) + os.pathsep + env.get("PYTHONPATH", "")

        cmd = [
            sys.executable,
            "-m",
            "consolidation_memory",
            "--project",
            "mcp_smoke",
            "serve",
        ]
        print(f"spawning: {' '.join(cmd)}", flush=True)
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="strict",
            env=env,
            bufsize=1,
        )
        stderr_lines: list[str] = []
        threading.Thread(target=_drain_stderr, args=(proc, stderr_lines), daemon=True).start()
        assert proc.stdout is not None
        reader = _LineReader(proc.stdout)
        reader.start()
        deadline = _Deadline(total_timeout)
        t0 = time.monotonic()
        request_id = 0

        def call(tool: str, arguments: dict[str, object], timeout: float) -> tuple[dict, str]:
            nonlocal request_id
            request_id += 1
            current = request_id
            _send(
                proc,
                {
                    "jsonrpc": "2.0",
                    "id": current,
                    "method": "tools/call",
                    "params": {"name": tool, "arguments": arguments},
                },
            )
            return _read_until(
                reader, current, timeout=timeout, deadline=deadline, returncode=proc.poll
            )

        try:
            _send(
                proc,
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": REQUESTED_PROTOCOL_VERSION,
                        "capabilities": {},
                        "clientInfo": {"name": "smoke_mcp_stdio", "version": "0"},
                    },
                },
            )
            init_msg, _ = _read_until(
                reader, 1, timeout=init_timeout, deadline=deadline, returncode=proc.poll
            )
            init_result = _result(init_msg, 1)
            _send(proc, {"jsonrpc": "2.0", "method": "notifications/initialized"})
            print(f"initialize ok in {time.monotonic() - t0:.2f}s", flush=True)

            def check_protocol_version() -> str:
                negotiated = init_result.get("protocolVersion")
                if not isinstance(negotiated, str) or not negotiated:
                    raise RuntimeError(f"initialize result carries no protocolVersion: {init_result}")
                expected = _expected_protocol_version(REQUESTED_PROTOCOL_VERSION)
                if expected is not None and negotiated != expected:
                    raise RuntimeError(
                        f"negotiated protocolVersion={negotiated!r} but the installed mcp "
                        f"server agrees to {expected!r}"
                    )
                server_info = init_result.get("serverInfo") or {}
                if not server_info.get("name"):
                    raise RuntimeError(f"initialize result carries no serverInfo.name: {init_result}")
                return (
                    f"requested {REQUESTED_PROTOCOL_VERSION} -> negotiated {negotiated} "
                    f"(server {server_info.get('name')})"
                )

            attempt("initialize_protocol_version", check_protocol_version)

            tools: list[dict[str, object]] = []

            def fetch_tools() -> list[dict[str, object]]:
                if tools:
                    return tools
                _send(proc, {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
                msg, _ = _read_until(
                    reader, 2, timeout=init_timeout, deadline=deadline, returncode=proc.poll
                )
                fetched = ((_result(msg, 2).get("tools")) or [])
                if not isinstance(fetched, list) or not fetched:
                    raise RuntimeError("tools/list returned no tools")
                tools.extend(tool for tool in fetched if isinstance(tool, dict))
                return tools

            attempt("published_output_schemas", lambda: _check_output_schemas(fetch_tools()))
            attempt("published_input_schemas", lambda: _check_input_schemas(fetch_tools()))
            attempt("tool_count_cross_check", lambda: _check_tool_count(fetch_tools()))

            attempt(
                "memory_status",
                lambda: _check_no_timeout(call("memory_status", {"lightweight": True}, status_timeout)),
            )
            attempt(
                "memory_recall",
                lambda: _check_no_timeout(
                    call(
                        "memory_recall",
                        {"query": "mcp smoke", "n_results": 3, "include_knowledge": True},
                        recall_timeout,
                    )
                ),
            )
            attempt(
                "non_ascii_round_trip",
                lambda: _check_non_ascii_round_trip(
                    lambda: call(
                        "memory_store",
                        {"content": ROUND_TRIP_CONTENT, "content_type": "exchange"},
                        status_timeout,
                    ),
                    lambda: call(
                        "memory_search", {"query": _SEARCH_QUERY, "limit": 5}, status_timeout
                    ),
                ),
            )
            # Spec: an unknown tool argument is an input validation error
            # (isError: true), not a silently dropped key and not a
            # protocol-level failure.
            attempt(
                "unknown_argument_rejected",
                lambda: _check_unknown_argument(
                    call("memory_status", {"lightweight": True, "junk_argument": 1}, status_timeout)
                ),
            )
        finally:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                proc.kill()

        failed = [name for name, ok, _ in checks if not ok]
        total = time.monotonic() - t0
        print(flush=True)
        print(f"--- smoke_mcp_stdio summary ({total:.2f}s) ---", flush=True)
        for name, ok, detail in checks:
            print(f"  {'PASS' if ok else 'FAIL'}  {name}", flush=True)
            if not ok:
                print(f"        {detail}", flush=True)
        if failed:
            print(flush=True)
            print("--- stderr tail ---", flush=True)
            for line in stderr_lines[-40:]:
                print(line, flush=True)
            print(
                f"smoke_mcp_stdio FAIL: {len(failed)}/{len(checks)} checks failed: {failed}",
                file=sys.stderr,
                flush=True,
            )
            return 1
        print(
            f"smoke_mcp_stdio PASS: {len(checks)}/{len(checks)} checks in {total:.2f}s",
            flush=True,
        )
        return 0


def _check_no_timeout(response: tuple[dict, str]) -> str:
    """A tool call must not come back carrying a timeout payload."""
    result = _result(response[0], "call")
    text = json.dumps(result, ensure_ascii=False)
    lowered = text.lower()
    if "timed out" in lowered or "timeout" in lowered:
        raise RuntimeError(f"tool returned a timeout payload: {text[:400]}")
    if result.get("isError"):
        raise RuntimeError(f"tool returned isError: {text[:400]}")
    return "no timeout payload, isError absent"


def _check_unknown_argument(response: tuple[dict, str]) -> str:
    """Input validation errors arrive as isError=true text, with no payload.

    Rejection happens inside the SDK before the handler runs, so there is no
    ``structuredContent`` to check here — the wire contract is isError plus
    actionable text naming the rejected key.
    """
    result = _result(response[0], "call")
    if not result.get("isError"):
        raise RuntimeError(f"extra tool argument not rejected: {json.dumps(result)[:400]}")
    frames = result.get("content") or []
    text = " ".join(str(frame.get("text") or "") for frame in frames if isinstance(frame, dict))
    if "junk_argument" not in text:
        raise RuntimeError(f"rejection does not name the rejected key: {text[:400]}")
    if "not permitted" not in text.lower():
        raise RuntimeError(f"rejection is not an input validation error: {text[:400]}")
    return "isError=true naming junk_argument"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-timeout", type=float, default=30.0)
    parser.add_argument("--status-timeout", type=float, default=30.0)
    parser.add_argument("--recall-timeout", type=float, default=45.0)
    parser.add_argument(
        "--total-timeout",
        type=float,
        default=240.0,
        help="wall-clock budget for the whole run; a hung server fails instead of hanging",
    )
    args = parser.parse_args()
    return run_smoke(
        status_timeout=args.status_timeout,
        recall_timeout=args.recall_timeout,
        init_timeout=args.init_timeout,
        total_timeout=args.total_timeout,
    )


if __name__ == "__main__":
    raise SystemExit(main())
