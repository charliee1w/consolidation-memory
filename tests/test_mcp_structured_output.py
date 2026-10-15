"""Regression tests for MCP tool result encoding (issue #5).

Tool handlers must hand MCPServer a JSON-serializable object. Returning JSON text
makes the SDK wrap the payload as ``{"result": "<json text>"}`` in
``structured_content``, and the default serializer can emit ``\\uXXXX`` escapes
that the client passes to the model verbatim.

The same contract has to show up in the published schemas: ``input_schema`` must
describe the handler parameters, and ``output_schema`` must describe the payload
the handler actually returns.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from typing import Any

import jsonschema
import pytest
from mcp.types import Tool as MCPTool

from consolidation_memory import server

# A complete memory_status payload: every required contract key with exact
# JSON types, carrying non-ASCII text in free-form fields. Passing this through
# FastMCP proves the contract accepts real payloads and keeps Unicode intact.
_STATUS_PAYLOAD: dict[str, Any] = {
    "episodic_buffer": {"total": 2, "pending_consolidation": 1, "consolidated": 1, "pruned": 0},
    "knowledge_base": {
        "total_topics": 0,
        "total_facts": 0,
        "total_records": 0,
        "records_by_type": {
            "facts": 0,
            "solutions": 0,
            "preferences": 0,
            "procedures": 0,
            "strategies": 0,
        },
    },
    "last_consolidation": None,
    "embedding_backend": "fastembed",
    "embedding_model": "bge-Γειά σου κόσμε",
    "faiss_index_size": 2,
    "faiss_tombstones": 0,
    "db_size_mb": 0.25,
    "version": "0.0.0-test",
    "health": {"status": "healthy", "issues": [], "backend_reachable": True},
    "consolidation_metrics": [],
    "consolidation_quality": None,
    "fast_path_hits": 0,
    "llm_fallbacks": 0,
    "recent_activity": [],
    "utility_scheduler": None,
    "knowledge_consistency": None,
    "scaling": None,
    "trust_profile": {"note": "Γειά σου κόσμε — память"},
}

# ASCII, non-ASCII and literal \uXXXX sequences in one stored string.
_BYTE_IDENTITY_CONTENT = (
    "ascii: the quick brown fox jumps over the lazy dog\n"
    "non-ascii: Γειά σου κόσμε — مرحبا 世界 你好\n"
    'literal escape sequence: \\u0393 and \\u043f, quote " and backslash \\'
)


def _published_tools() -> list[MCPTool]:
    return asyncio.run(server.mcp.list_tools())


def _handler_for(tool: MCPTool) -> inspect.Signature:
    return inspect.signature(getattr(server, tool.name))


def _call_status(
    monkeypatch: Any,
    *,
    payload: dict[str, Any] | None = None,
    error: BaseException | None = None,
) -> Any:
    """Run ``memory_status`` through MCPServer with a canned dispatch result."""

    if error is not None:

        async def fake_call_tool_payload(*args: Any, **kwargs: Any) -> dict[str, object]:
            raise error

    else:

        async def fake_call_tool_payload(*args: Any, **kwargs: Any) -> dict[str, object]:
            return dict(payload or {})

    monkeypatch.setattr(server, "_call_tool_payload", fake_call_tool_payload)
    monkeypatch.setattr(server, "tool_requires_client", lambda name: False)
    return asyncio.run(server.mcp.call_tool("memory_status", {"lightweight": True}))


def test_structured_content_is_the_payload_object(monkeypatch: Any) -> None:
    result = _call_status(monkeypatch, payload=_STATUS_PAYLOAD)
    assert result.structured_content == _STATUS_PAYLOAD


def test_text_content_is_single_encoded_json_with_unicode(monkeypatch: Any) -> None:
    result = _call_status(monkeypatch, payload=_STATUS_PAYLOAD)
    text = result.content[0].text
    assert "Γειά" in text, "non-ASCII text must be returned as-is"
    assert "\\u0393" not in text, "Greek characters must not be \\uXXXX-escaped"
    assert "\\u043f" not in text, "Cyrillic characters must not be \\uXXXX-escaped"
    assert json.loads(text) == _STATUS_PAYLOAD


def test_stored_content_round_trips_byte_identically() -> None:
    """Stored text must come back unchanged: nothing re-encoded, nothing decoded."""
    from consolidation_memory.database import ensure_schema, insert_episode

    ensure_schema()
    insert_episode(content=_BYTE_IDENTITY_CONTENT)

    result = asyncio.run(
        server.mcp.call_tool("memory_search", {"query": "quick brown fox", "limit": 5})
    )
    structured = result.structured_content

    assert structured.get("episodes"), "search must return the stored episode"
    from_structured = structured["episodes"][0]["content"]
    from_text = json.loads(result.content[0].text)["episodes"][0]["content"]

    for returned in (from_structured, from_text):
        assert returned == _BYTE_IDENTITY_CONTENT
        assert returned.encode("utf-8") == _BYTE_IDENTITY_CONTENT.encode("utf-8")


def test_error_result_is_is_error_with_structured_payload(monkeypatch: Any) -> None:
    """Execution errors follow the spec: isError=true, actionable text, payload kept."""
    result = _call_status(monkeypatch, error=TimeoutError("boom"))
    assert result.is_error is True
    assert "timed out" in result.content[0].text
    structured = result.structured_content
    assert isinstance(structured, dict)
    assert isinstance(structured.get("error"), str)
    assert "timed out" in structured["error"]


def test_handler_returns_object_not_json_text(monkeypatch: Any) -> None:
    async def fake_call_tool_payload(*args: Any, **kwargs: Any) -> dict[str, object]:
        return dict(_STATUS_PAYLOAD)

    monkeypatch.setattr(server, "_call_tool_payload", fake_call_tool_payload)
    result = asyncio.run(server.memory_status(lightweight=True))
    assert isinstance(result, dict)
    assert result == _STATUS_PAYLOAD


def test_input_schemas_are_valid_json_schema() -> None:
    for tool in _published_tools():
        assert tool.input_schema is not None, tool.name
        jsonschema.Draft202012Validator.check_schema(tool.input_schema)
        assert tool.input_schema.get("type") == "object", tool.name


def test_input_schemas_match_handler_signatures() -> None:
    for tool in _published_tools():
        signature = _handler_for(tool)
        expected = set(signature.parameters)
        properties = set((tool.input_schema or {}).get("properties") or {})
        assert properties == expected, (
            f"{tool.name}: input_schema properties {sorted(properties)} "
            f"!= handler parameters {sorted(expected)}"
        )
        required = set((tool.input_schema or {}).get("required") or [])
        expected_required = {
            name
            for name, parameter in signature.parameters.items()
            if parameter.default is inspect.Parameter.empty
        }
        assert required == expected_required, (
            f"{tool.name}: required {sorted(required)} != parameters without defaults "
            f"{sorted(expected_required)}"
        )


def _openai_input_descriptions() -> dict[str, dict[str, str]]:
    from consolidation_memory.schemas import openai_tools

    out: dict[str, dict[str, str]] = {}
    for tool in openai_tools:
        fn = tool.get("function", tool)
        props = fn.get("parameters", {}).get("properties", {})
        out[fn["name"]] = {
            name: spec["description"]
            for name, spec in props.items()
            if spec.get("description")
        }
    return out


def test_every_input_property_has_a_description() -> None:
    missing = []
    for tool in _published_tools():
        for name, spec in ((tool.input_schema or {}).get("properties") or {}).items():
            if not str(spec.get("description") or "").strip():
                missing.append(f"{tool.name}.{name}")
    assert not missing, f"input properties without description: {missing}"


def test_input_descriptions_match_the_openai_surface() -> None:
    """Both surfaces publish the same wording for every shared parameter."""
    expected = _openai_input_descriptions()
    for tool in _published_tools():
        props = (tool.input_schema or {}).get("properties") or {}
        assert set(props) == set(expected[tool.name]), tool.name
        for name, spec in props.items():
            assert spec.get("description") == expected[tool.name][name], f"{tool.name}.{name}"


def test_input_schemas_forbid_additional_properties() -> None:
    for tool in _published_tools():
        assert (tool.input_schema or {}).get("additionalProperties") is False, tool.name


def test_unknown_argument_is_rejected() -> None:
    from mcp.server.mcpserver.exceptions import ToolError

    with pytest.raises(ToolError, match="Extra inputs are not permitted"):
        asyncio.run(server.mcp.call_tool("memory_status", {"lightweight": True, "junk": 1}))


def test_output_schemas_are_valid_json_schema() -> None:
    """Published schemas are anyOf[success contract, error payload] objects."""
    for tool in _published_tools():
        schema = tool.output_schema
        assert schema is not None, tool.name
        jsonschema.Draft202012Validator.check_schema(schema)
        arms = schema.get("anyOf")
        assert isinstance(arms, list) and len(arms) == 2, tool.name
        assert arms[0].get("type") == "object", tool.name
        assert arms[0].get("properties"), tool.name
        assert arms[1].get("title") == "ToolErrorPayload", tool.name
        assert arms[1].get("required") == ["error"], tool.name


def _iter_object_schemas(schema: dict[str, Any]) -> list[dict[str, Any]]:
    """Collect every object schema node reachable through properties/items/$defs."""
    found: list[dict[str, Any]] = []

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object" and isinstance(node.get("properties"), dict):
                found.append(node)
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for item in node:
                visit(item)

    visit(schema)
    return found


def test_output_schema_properties_all_carry_descriptions() -> None:
    """Every published property, including nested models, documents itself."""
    missing: list[str] = []
    for tool in _published_tools():
        schema = tool.output_schema or {}
        success_arm = (schema.get("anyOf") or [{}])[0]
        for node in _iter_object_schemas(success_arm):
            for prop_name, prop_spec in (node.get("properties") or {}).items():
                if not str(prop_spec.get("description") or "").strip():
                    missing.append(f"{tool.name}.{prop_name}")
    assert not missing, f"output properties without description: {missing}"


def test_output_schema_describes_the_payload(monkeypatch: Any) -> None:
    result = _call_status(monkeypatch, payload=_STATUS_PAYLOAD)

    tool = next(tool for tool in _published_tools() if tool.name == "memory_status")
    assert tool.output_schema is not None
    success_arm = tool.output_schema["anyOf"][0]
    properties = set(success_arm.get("properties") or {})
    assert "result" not in properties, "payload must not be wrapped in a result key"
    jsonschema.validate(result.structured_content, tool.output_schema)


def test_output_schema_rejects_non_object_payload(monkeypatch: Any) -> None:
    """A str payload is not what the schema promises."""
    tool = next(tool for tool in _published_tools() if tool.name == "memory_status")
    assert tool.output_schema is not None
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(json.dumps(dict(_STATUS_PAYLOAD)), tool.output_schema)
