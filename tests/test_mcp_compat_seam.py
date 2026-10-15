"""The MCP outputSchema contract must survive private-SDK drift.

Every tool has to publish ``anyOf[success contract, error payload]`` and reject
unknown arguments. Both ride on private, unversioned ``mcp`` SDK internals
(``ArgModelBase``, ``MCPServer._tool_manager._tools``, the cached
``Tool.output_schema``), so this module pins the three guarantees the
consolidation-memory contract depends on:

1. the full documented tool set publishes the two-arm schema at startup;
2. publication self-heals, so a tool registered after startup is fixed on the
   next listing instead of serving a success-only schema forever;
3. the compat seam in ``mcp_compat`` fails loudly and names the installed mcp
   version, the supported range, and every attribute it tried — instead of the
   previous silent contract loss or a bare ``ImportError`` traceback.
"""

from __future__ import annotations

import asyncio
from typing import Any

import jsonschema
import pytest
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import BaseModel

from consolidation_memory import mcp_compat, schemas, server, tool_dispatch

# The documented full MCP profile. Kept explicit (not derived from the server)
# so a tool silently dropped from the registry fails here.
EXPECTED_TOOL_NAMES = {
    "memory_ask",
    "memory_browse",
    "memory_claim_browse",
    "memory_claim_search",
    "memory_compact",
    "memory_consolidate",
    "memory_consolidation_log",
    "memory_contradictions",
    "memory_correct",
    "memory_decay_report",
    "memory_detect_drift",
    "memory_export",
    "memory_forget",
    "memory_hygiene_apply",
    "memory_hygiene_scan",
    "memory_outcome_browse",
    "memory_outcome_record",
    "memory_policy_grant",
    "memory_policy_list",
    "memory_protect",
    "memory_read_topic",
    "memory_recall",
    "memory_remember",
    "memory_scope_list",
    "memory_search",
    "memory_status",
    "memory_store",
    "memory_store_batch",
    "memory_timeline",
}

_LATE_TOOL_NAME = "memory_late_registered_probe"
_LATE_DESCRIPTION = "A tool registered after the import-time publication ran."
_UNPUBLISHED_TOOL_NAME = "memory_unpublished_probe"


class _LateOutput(BaseModel):
    ok: bool


def _published() -> dict[str, Any]:
    return {tool.name: tool for tool in asyncio.run(server.mcp.list_tools())}


def _register_late_tool() -> None:
    """Register a tool after startup, the way a plugin or dynamic tool would.

    Also appended to ``schemas.openai_tools``: a tool registered on MCP but not
    published to the OpenAI surface has no argument contract to enforce, and the
    listing now verifies that, so the probe has to be a legitimate tool.
    """
    schemas.openai_tools.append(
        {
            "type": "function",
            "function": {
                "name": _LATE_TOOL_NAME,
                "description": _LATE_DESCRIPTION,
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {},
                    "required": [],
                },
            },
        }
    )

    @server.mcp.tool(name=_LATE_TOOL_NAME, description=_LATE_DESCRIPTION)
    async def _late() -> _LateOutput:
        """A tool registered after the one-shot import-time publication ran."""
        return _LateOutput(ok=True)

    # The allowed-argument map is memoized over openai_tools; a tool published
    # after the first call has to invalidate it or the argument check keeps
    # seeing the set from import time.
    tool_dispatch._published_argument_names.cache_clear()


def _remove_late_tool() -> None:
    del mcp_compat.registered_tools(server.mcp)[_LATE_TOOL_NAME]
    schemas.openai_tools[:] = [
        tool
        for tool in schemas.openai_tools
        if str(tool["function"]["name"]) != _LATE_TOOL_NAME
    ]
    tool_dispatch._published_argument_names.cache_clear()


def test_every_documented_tool_publishes_the_two_arm_output_schema() -> None:
    published = _published()
    assert set(published) == EXPECTED_TOOL_NAMES, sorted(set(published) ^ EXPECTED_TOOL_NAMES)
    for name, tool in sorted(published.items()):
        schema = tool.output_schema
        assert schema is not None, f"{name}: no outputSchema published"
        assert schema.get("type") == "object", f"{name}: root type must stay object"
        arms = schema.get("anyOf")
        assert isinstance(arms, list) and len(arms) == 2, f"{name}: {arms!r}"
        assert arms[0].get("type") == "object" and arms[0].get("properties"), name
        assert arms[1].get("title") == "ToolErrorPayload", name
        assert arms[1].get("required") == ["error"], name


def test_error_payload_matches_the_published_error_arm() -> None:
    """The error arm has to accept exactly what a failure actually returns."""
    schema = _published()["memory_status"].output_schema
    assert schema is not None
    jsonschema.validate({"error": "memory_status timed out after 30s"}, schema)


def test_self_check_passes_on_the_live_server() -> None:
    server._verify_published_output_schemas()


def test_self_check_raises_when_a_tool_loses_its_error_arm() -> None:
    tool = mcp_compat.registered_tools(server.mcp)["memory_status"]
    original = mcp_compat.output_schema_dict(tool)
    try:
        mcp_compat.set_output_schema(tool, {"type": "object", "properties": {"ok": {}}})
        with pytest.raises(mcp_compat.MCPCompatError) as excinfo:
            server._verify_published_output_schemas()
        message = str(excinfo.value)
        assert "memory_status" in message
        assert mcp_compat.MCP_COMPAT_REQUIREMENT in message
    finally:
        mcp_compat.set_output_schema(tool, original)
    server._verify_published_output_schemas()


def test_self_check_catches_a_double_wrapped_error_arm() -> None:
    tool = mcp_compat.registered_tools(server.mcp)["memory_compact"]
    original = mcp_compat.output_schema_dict(tool)
    try:
        assert original is not None
        mcp_compat.set_output_schema(tool, {"type": "object", "anyOf": [original, original]})
        with pytest.raises(mcp_compat.MCPCompatError, match="ToolErrorPayload"):
            server._verify_published_output_schemas()
    finally:
        mcp_compat.set_output_schema(tool, original)
    server._verify_published_output_schemas()


def test_self_check_raises_when_a_tool_has_no_output_schema() -> None:
    tool = mcp_compat.registered_tools(server.mcp)["memory_remember"]
    original = mcp_compat.output_schema_dict(tool)
    try:
        tool.fn_metadata.output_schema = None
        with pytest.raises(mcp_compat.MCPCompatError, match="memory_remember"):
            server._verify_published_output_schemas()
    finally:
        tool.fn_metadata.output_schema = original
    server._verify_published_output_schemas()


def test_late_registered_tool_gets_its_output_schema_on_the_next_listing() -> None:
    _register_late_tool()
    try:
        # Publication ran at import, before this tool existed: the SDK has a
        # success-only schema for it right now.
        tool = mcp_compat.registered_tools(server.mcp)[_LATE_TOOL_NAME]
        assert "anyOf" not in (mcp_compat.output_schema_dict(tool) or {}), "precondition"

        published = _published()[_LATE_TOOL_NAME]
        assert published.output_schema is not None
        arms = published.output_schema.get("anyOf")
        assert isinstance(arms, list) and len(arms) == 2, published.output_schema
        assert arms[1].get("title") == "ToolErrorPayload"
    finally:
        _remove_late_tool()


def test_a_late_tool_that_breaks_the_contract_is_refused_on_listing() -> None:
    """Self-healing must not become self-exempting.

    The listing used to run publication only, so a tool registered at runtime
    was repaired and then served unverified: no argument-set check, no
    description check. Register one that has no entry in the OpenAI surface -
    nothing enforces a contract for it - and the listing must fail instead.
    """
    @server.mcp.tool(name=_UNPUBLISHED_TOOL_NAME, description="Never published.")
    async def _unpublished() -> _LateOutput:
        """Registered on MCP but absent from schemas.openai_tools."""
        return _LateOutput(ok=True)

    try:
        with pytest.raises(mcp_compat.MCPCompatError, match=_UNPUBLISHED_TOOL_NAME):
            _published()
    finally:
        del mcp_compat.registered_tools(server.mcp)[_UNPUBLISHED_TOOL_NAME]


def test_publication_is_idempotent_and_preserves_the_published_object() -> None:
    """Repeat calls must be no-ops, not rebuilds (and never re-wrap an arm)."""
    before = {
        name: mcp_compat.output_schema_dict(tool)
        for name, tool in mcp_compat.registered_tools(server.mcp).items()
    }
    assert server._publish_output_schemas() == []
    after = {
        name: mcp_compat.output_schema_dict(tool)
        for name, tool in mcp_compat.registered_tools(server.mcp).items()
    }
    assert before.keys() == after.keys()
    for name, schema in before.items():
        assert after[name] is schema, f"{name}: published schema was rebuilt"
    for name, tool in _published().items():
        arms = tool.output_schema["anyOf"]
        assert arms[1]["title"] == "ToolErrorPayload", f"{name}: error arm was re-wrapped"


def test_publication_repairs_a_regressed_schema_on_the_next_listing() -> None:
    """Even a schema that loses its anyOf is rebuilt, not trusted."""
    tool = mcp_compat.registered_tools(server.mcp)["memory_status"]
    original = mcp_compat.output_schema_dict(tool)
    try:
        mcp_compat.set_output_schema(tool, {"type": "object", "properties": {"ok": {}}})
        _published()
        arms = _published()["memory_status"].output_schema["anyOf"]
        assert arms[1]["title"] == "ToolErrorPayload"
    finally:
        mcp_compat.set_output_schema(tool, original)
    _published()


def test_cached_property_eviction_is_what_keeps_the_listing_fresh() -> None:
    """Private surface 5: ``Tool.output_schema`` is a ``cached_property``.

    Assigning ``fn_metadata.output_schema`` on its own leaves the SDK serving the
    pre-assignment dict, so a publication path that skipped the eviction would
    leave every tool success-only forever with no error anywhere. This pins the
    staleness the seam has to defeat.
    """
    tool = mcp_compat.registered_tools(server.mcp)["memory_search"]
    original = tool.output_schema
    assert original is not None
    try:
        probe = {"type": "object", "anyOf": [], "marker": "fresh-probe"}
        tool.fn_metadata.output_schema = probe
        assert tool.output_schema is original, "precondition: the SDK cache is stale"
        assert mcp_compat.output_schema_dict(tool) is probe
        assert tool.output_schema is probe
    finally:
        mcp_compat.set_output_schema(tool, original)
        assert tool.output_schema is original
    _published()


class _RenamedToolManager:
    """Stands in for an mcp minor bump that renamed the tool mapping."""

    def __init__(self) -> None:
        self.tools: dict[str, Any] = {}


class _RenamedServer:
    def __init__(self) -> None:
        self._tool_manager = _RenamedToolManager()


def test_registered_tools_reports_every_attribute_it_tried() -> None:
    with pytest.raises(mcp_compat.MCPCompatError) as excinfo:
        mcp_compat.registered_tools(_RenamedServer())  # type: ignore[arg-type]
    message = str(excinfo.value)
    assert "'_tool_manager._tools'" in message
    assert "'_tool_manager._tool_map'" in message
    assert mcp_compat.installed_mcp_version() in message
    assert mcp_compat.MCP_COMPAT_REQUIREMENT in message
    assert "mcp_compat.py" in message


def test_registered_tools_reports_a_missing_tool_manager() -> None:
    with pytest.raises(mcp_compat.MCPCompatError, match="_tool_manager"):
        mcp_compat.registered_tools(object())  # type: ignore[arg-type]


def test_registered_tools_falls_back_to_the_probe_order() -> None:
    """A rename that keeps a mapping attribute is picked up, not lost."""

    class _MappedManager:
        def __init__(self) -> None:
            self._tool_map = {"memory_probe": object()}

    class _Server:
        _tool_manager = _MappedManager()

    assert set(mcp_compat.registered_tools(_Server())) == {"memory_probe"}  # type: ignore[arg-type]


def test_tool_without_fn_metadata_is_reported() -> None:
    with pytest.raises(mcp_compat.MCPCompatError, match="fn_metadata"):
        mcp_compat.output_schema_dict(object())


def test_arg_model_base_still_forbids_unknown_arguments() -> None:
    """The load-bearing half of the private surface: extra="forbid".

    Losing it would make pydantic silently drop misspelled arguments and stop
    publishing additionalProperties: false, so unknown-argument rejection and
    the input-schema guard both depend on it holding through the seam.
    """
    assert mcp_compat.arg_model_extra_mode() == "forbid"
    tool = _published()["memory_status"]
    assert (tool.input_schema or {}).get("additionalProperties") is False
    with pytest.raises(ToolError, match="Extra inputs are not permitted"):
        asyncio.run(server.mcp.call_tool("memory_status", {"lightweight": True, "junk": 1}))


def test_heal_hook_is_installed_once() -> None:
    wrapper = server.mcp.list_tools
    mcp_compat.install_list_tools_heal(server.mcp, server._publish_output_schemas)
    assert server.mcp.list_tools is wrapper
