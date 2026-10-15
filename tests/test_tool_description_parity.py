"""Both surfaces must publish the same tool description.

The description is the only thing an agent host reads to decide whether and how
to call a tool. It reaches a host through two different places: the MCP
``tools/list`` entry, and the OpenAI-compatible ``openai_tools`` schema. When
those drift, the richer text is published to one audience and silently withheld
from the other — which is how ``memory_scope_list`` ended up describing itself
as a plain scope listing on MCP while three documents explained that it is a
deployment-wide audit tool that is not filtered by ``read_visibility``.

The contract is single-sourced from ``schemas.openai_tools`` and enforced by
``server._verify_tool_descriptions()`` at import and in the lifespan. These tests
pin the two halves of that: parity today, and a loud failure when it breaks.
"""

from __future__ import annotations

import asyncio
import re

import pytest

from consolidation_memory import mcp_compat, schemas, server


def _normalize(text: str) -> str:
    return " ".join(text.split())


def _mcp_descriptions() -> dict[str, str]:
    return {tool.name: tool.description or "" for tool in asyncio.run(server.mcp.list_tools())}


def _openai_descriptions() -> dict[str, str]:
    return {
        str(tool["function"]["name"]): str(tool["function"]["description"])
        for tool in schemas.openai_tools
    }


def test_every_tool_publishes_one_description_on_both_surfaces() -> None:
    mcp = _mcp_descriptions()
    openai = _openai_descriptions()

    assert set(mcp) == set(openai), (
        "the MCP and OpenAI surfaces must expose the same tool set"
    )
    assert len(openai) == 29

    mismatched = {
        name: (_normalize(mcp[name]), _normalize(openai[name]))
        for name in openai
        if _normalize(mcp[name]) != _normalize(openai[name])
    }
    assert not mismatched, f"tool descriptions diverged between surfaces: {mismatched}"


@pytest.mark.parametrize(
    "name",
    ["memory_scope_list", "memory_policy_list"],
)
def test_unfiltered_listings_say_so_in_the_published_description(name: str) -> None:
    """The discovery tools must carry their visibility semantics where hosts read them.

    Both listings are intentionally outside the ``read_visibility`` filter. That
    is a deliberate design decision with a security consequence, and a host that
    cannot see it in the tool description will route a multi-tenant deployment
    through them without knowing. Asserted on both surfaces because the two
    publish from different places.
    """
    for description in (_mcp_descriptions()[name], _openai_descriptions()[name]):
        assert "not filtered" in description.lower() or "NOT filtered" in description
        assert "read_visibility" in description


def test_verifier_rejects_a_docstring_that_contradicts_the_published_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drift has to fail at import, not reach a client.

    A silent fallback is the failure mode this guard exists to prevent: the
    server would keep serving a stale summary for one tool while everything
    else stayed green.
    """

    class _Stale:
        """Publishes an unrelated summary."""

    monkeypatch.setitem(server.__dict__, "memory_search", _Stale)

    with pytest.raises(mcp_compat.MCPCompatError) as excinfo:
        server._verify_tool_descriptions()

    message = str(excinfo.value)
    assert "memory_search" in message
    assert "docstring" in message and "published" in message


def test_verifier_rejects_a_tool_with_no_docstring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _bare() -> None:  # pragma: no cover - never called
        return None

    monkeypatch.setitem(server.__dict__, "memory_search", _bare)

    with pytest.raises(mcp_compat.MCPCompatError) as excinfo:
        server._verify_tool_descriptions()

    assert "no docstring" in str(excinfo.value)


def test_verifier_rejects_a_published_tool_with_no_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(server.__dict__, "memory_search", raising=False)

    with pytest.raises(mcp_compat.MCPCompatError) as excinfo:
        server._verify_tool_descriptions()

    assert "no handler is" in str(excinfo.value)


def _published_output_property_names() -> dict[str, set[str]]:
    """Output-contract property names per tool, including nested `$defs` models."""
    import asyncio

    from consolidation_memory import server

    def _collect(node: object, found: set[str]) -> None:
        if isinstance(node, dict):
            properties = node.get("properties")
            if isinstance(properties, dict):
                found.update(str(key) for key in properties)
            for value in node.values():
                _collect(value, found)
        elif isinstance(node, list):
            for item in node:
                _collect(item, found)

    per_tool: dict[str, set[str]] = {}
    for tool in asyncio.run(server.mcp.list_tools()):
        found: set[str] = set()
        _collect(tool.output_schema or {}, found)
        per_tool[str(tool.name)] = found
    return per_tool


def test_published_descriptions_only_name_keys_the_contract_declares() -> None:
    """A description must not send a caller to a key that does not exist.

    The description is the only thing an agent reads to decide how to call a
    tool. A snake_case token in prose reads as a parameter or output field; if the
    published schema declares no such key, the call is rejected on every surface
    (`additionalProperties: false`) or the field is simply absent from the
    result. This caught `memory_remember` advertising `content_type`, which it
    does not publish - a key its own description told agents to prefer the other
    tool for.

    Tool names are legitimate prose references, so are domain field names another
    tool in the same surface does declare: the two discovery tools say "not
    filtered by read_visibility", naming the ACL field memory_policy_grant accepts,
    not an argument of their own.
    """
    tool_names = {str(tool["function"]["name"]) for tool in schemas.openai_tools}
    surface_arguments = {
        str(key)
        for tool in schemas.openai_tools
        for key in (tool["function"]["parameters"].get("properties") or {})
    }
    output_names = _published_output_property_names()

    offenders: list[str] = []
    for tool in schemas.openai_tools:
        name = str(tool["function"]["name"])
        published = {
            str(key)
            for key in (tool["function"]["parameters"].get("properties") or {})
        }
        published |= output_names.get(name, set())
        published |= surface_arguments
        description = str(tool["function"]["description"])
        for claim in re.findall(r"\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\b", description):
            if claim in tool_names or claim in published:
                continue
            offenders.append(f"{name} mentions {claim!r}, which it does not declare")

    assert not offenders, "; ".join(offenders)
