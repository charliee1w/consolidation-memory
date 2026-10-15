"""Discovery, reuse and pagination of memory_scope_list."""

from __future__ import annotations

import asyncio

import pytest

from consolidation_memory import server


@pytest.fixture(autouse=True)
def _schema():
    """Fresh tmp data dirs start without tables; create them per test."""
    from consolidation_memory.database import ensure_schema

    ensure_schema()


def _list_scopes(**arguments):
    return asyncio.run(server.mcp.call_tool("memory_scope_list", arguments))


def _seed_scopes() -> set[str]:
    """Store one episode per scope and return the seeded project slugs."""
    asyncio.run(server.memory_store(content="alpha repo memory", scope={"project": {"slug": "alpha"}}))
    asyncio.run(server.memory_store(content="beta repo memory", scope={"project": {"slug": "beta"}}))
    asyncio.run(server.memory_store(content="gamma repo memory", scope={"project": {"slug": "gamma"}}))
    return {"alpha", "beta", "gamma"}


def test_empty_corpus_lists_no_scopes() -> None:
    result = _list_scopes(limit=10)
    assert result.is_error is False
    assert result.structured_content == {"scopes": [], "total": 0, "offset": 0, "message": None}


def test_discovers_seeded_scopes_with_counts_and_envelopes() -> None:
    slugs = _seed_scopes()
    result = _list_scopes(limit=10)
    assert result.is_error is False
    payload = result.structured_content

    assert payload["total"] == 3
    assert payload["offset"] == 0
    assert payload["message"] is None
    found = {entry["scope"]["project"]["slug"] for entry in payload["scopes"]}
    assert found == slugs

    for entry in payload["scopes"]:
        assert entry["counts"] == {"episodes": 1, "records": 0, "topics": 0}
        assert entry["last_used_at"]
        scope = entry["scope"]
        assert scope["namespace"]["slug"] == "default"
        assert scope["app_client"]["name"] == "legacy_client"
        assert scope["agent"] is None
        assert scope["session"] is None


def test_listed_scope_is_reusable_as_scope_argument() -> None:
    _seed_scopes()
    listed = _list_scopes(limit=10).structured_content["scopes"][0]["scope"]
    stored = asyncio.run(server.memory_store(content="reused scope write", scope=listed))
    assert stored["status"] == "stored"

    after = _list_scopes(limit=10).structured_content
    match = next(
        entry
        for entry in after["scopes"]
        if entry["scope"]["project"]["slug"] == listed["project"]["slug"]
    )
    assert match["counts"]["episodes"] == 2
    assert after["total"] == 3


def test_listed_envelope_validates_against_the_published_scope_schema() -> None:
    """A discovered envelope must satisfy the schema the tools publish.

    The guide promises an entry can be passed back verbatim. Discovery emits
    ``null`` for every identity key the rows do not carry, so the published
    scope schema has to accept that: it previously declared strings only, and a
    host validating arguments against it (OpenAI strict mode) rejected the
    envelope that MCP and REST both accept.
    """
    import jsonschema

    from consolidation_memory.schemas import SCOPE_INPUT_SCHEMA

    _seed_scopes()
    listed = _list_scopes(limit=10).structured_content["scopes"][0]["scope"]

    jsonschema.validate(instance=listed, schema=SCOPE_INPUT_SCHEMA)


def test_a_null_identity_key_is_rejected_still() -> None:
    """Nullable is not permissive: an unknown key and a bad type still fail."""
    import jsonschema
    from jsonschema.exceptions import ValidationError

    from consolidation_memory.schemas import SCOPE_INPUT_SCHEMA

    for envelope in (
        {"namespace": {"nope": "x"}},
        {"app_client": {"provider": 7}},
        {"namespace": {"sharing_mode": "public"}},
    ):
        with pytest.raises(ValidationError):
            jsonschema.validate(instance=envelope, schema=SCOPE_INPUT_SCHEMA)


def test_pages_iterate_without_overlap_or_gaps() -> None:
    _seed_scopes()
    full = _list_scopes(limit=10).structured_content
    assert full["total"] == 3

    page1 = _list_scopes(limit=1, offset=0).structured_content
    assert page1["scopes"] == full["scopes"][0:1]
    assert page1["offset"] == 0
    assert page1["message"] and "1 of 3" in page1["message"]

    page2 = _list_scopes(limit=1, offset=1).structured_content
    assert page2["scopes"] == full["scopes"][1:2]
    assert page2["offset"] == 1

    page3 = _list_scopes(limit=1, offset=2).structured_content
    assert page3["scopes"] == full["scopes"][2:3]
    assert page3["message"] and "offset 2" in page3["message"]

    past = _list_scopes(limit=1, offset=99).structured_content
    assert past["scopes"] == []
    assert past["total"] == 3
    assert past["offset"] == 99
    assert past["message"]


def test_out_of_range_window_is_an_error() -> None:
    _seed_scopes()
    result = _list_scopes(limit=0)
    assert result.is_error is True
    assert "between 1 and 1000" in result.content[0].text

    negative = _list_scopes(limit=10, offset=-1)
    assert negative.is_error is True
    assert "between 0 and" in negative.content[0].text

try:
    from fastapi.testclient import TestClient  # noqa: F401

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


@pytest.mark.skipif(not HAS_FASTAPI, reason="fastapi not installed")
def test_scope_list_matches_across_surfaces() -> None:
    """OpenAI dispatch, the MCP handler and REST return the same payload."""
    from consolidation_memory.server import memory_scope_list
    from tests.surface_contract_helpers import invoke_surfaces_with_execute_tool_call

    expected = {"scopes": [], "total": 0, "offset": 0, "message": None}
    dispatch_out, mcp_out, rest_out, mock_execute = invoke_surfaces_with_execute_tool_call(
        tool_name="memory_scope_list",
        tool_args={"limit": 10, "offset": 0},
        expected_result=expected,
        mcp_coro_factory=lambda: memory_scope_list(limit=10, offset=0),
        rest_path="/memory/scopes?limit=10&offset=0",
        rest_method="GET",
    )

    assert dispatch_out == expected
    assert mcp_out == expected
    assert rest_out == expected
    assert mock_execute.recorded_tool_calls == [
        ("memory_scope_list", {"limit": 10, "offset": 0}),
        ("memory_scope_list", {"limit": 10, "offset": 0}),
        ("memory_scope_list", {"limit": 10, "offset": 0}),
    ]
