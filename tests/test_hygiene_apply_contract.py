"""The applied hygiene payload must satisfy the contract it publishes.

``memory_hygiene_apply`` used to omit ``episode_ids`` from the non-dry-run
payload while the published ``outputSchema`` required it. The MCP SDK
validates ``structuredContent`` against that schema *after* the handler
returned, so a caller saw a protocol error for an episode set that was already
soft-deleted, with no ``forgotten``/``not_found`` counters to reconcile.

These tests drive the mutating branch through ``mcp.call_tool`` — the entry
point that performs output validation, unlike the raw handler — and pin the
contract to the producer result type so the two cannot drift.
"""

from __future__ import annotations

import asyncio
import dataclasses
from typing import Any
from unittest.mock import patch

import jsonschema

from consolidation_memory import server
from consolidation_memory.tool_contracts import HygieneApplyOutput
from consolidation_memory.types import HygieneApplyResult
from tests.helpers import make_normalized_vec as _vec

TOOL = "memory_hygiene_apply"
_SCHEMA_CACHE: dict[str, dict[str, Any]] = {}


def _published_schema(tool_name: str) -> dict[str, Any]:
    if tool_name not in _SCHEMA_CACHE:
        tools = asyncio.run(server.mcp.list_tools())
        tool = next(t for t in tools if t.name == tool_name)
        assert tool.output_schema is not None, tool_name
        _SCHEMA_CACHE[tool_name] = tool.output_schema
    return _SCHEMA_CACHE[tool_name]


def _status_contract() -> dict[str, Any]:
    """Published ``status`` property of the success arm."""
    success = _published_schema(TOOL)["anyOf"][0]
    return success["properties"]["status"]


def _call_via_mcp(case: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Call the tool the way a client does and validate what comes back.

    ``mcp.call_tool`` is the only path that validates the payload against the
    published schema, so it is the only path that can catch this class of bug.
    """
    result = asyncio.run(server.mcp.call_tool(TOOL, arguments))
    assert getattr(result, "is_error", False) is not True, (
        f"{case}: expected success, got isError: {result.content[0].text}"
    )
    payload = result.structured_content
    assert isinstance(payload, dict), f"{case}: payload is not an object"
    try:
        jsonschema.validate(payload, _published_schema(TOOL))
    except jsonschema.ValidationError as exc:
        raise AssertionError(f"{case}: payload violates output schema: {exc.message}") from exc
    # The SDK validates against the contract model itself; assert the wire
    # payload kept every declared key instead of dropping it to fit.
    assert set(payload) == set(HygieneApplyOutput.model_fields), sorted(payload)
    return payload


def _seed_episode(content: str) -> str:
    """Store one episode the hygiene scan will flag, without a real backend."""
    from consolidation_memory.client import MemoryClient
    from consolidation_memory.database import ensure_schema

    ensure_schema()
    client = MemoryClient(auto_consolidate=False)
    try:
        return client.store(content, content_type="exchange").id
    finally:
        client.close()


def _deleted_flags(episode_ids: list[str]) -> dict[str, int]:
    from consolidation_memory.database import get_connection

    with get_connection() as conn:
        return {
            episode_id: conn.execute(
                "SELECT deleted FROM episodes WHERE id = ?", (episode_id,)
            ).fetchone()["deleted"]
            for episode_id in episode_ids
        }


class TestHygieneApplyMutatingPath:
    def test_applied_payload_validates_through_mcp_call_tool(self, tmp_data_dir) -> None:
        """Recommended cleanup: payload validates and reports what it forgot."""
        with patch("consolidation_memory.backends.encode_documents") as mock_embed:
            mock_embed.side_effect = lambda texts: _vec(seed=hash(texts[0]) % 997).reshape(1, -1)
            ids = [
                _seed_episode("user asked a short follow up question"),
                _seed_episode("ok thanks that works now"),
            ]

            payload = _call_via_mcp("applied_recommended", {"use_recommended": True})

        assert payload["status"] == "applied"
        assert payload["forgotten"] == 2
        assert payload["not_found"] == 0
        assert payload["episode_targets"] == 2
        # The reported ids are the ones actually forgotten.
        assert sorted(payload["episode_ids"]) == sorted(ids)
        assert _deleted_flags(ids) == dict.fromkeys(ids, 1)
        assert payload["expire_orphans"] is False
        assert payload["orphan_repair"] is None

    def test_default_call_without_arguments_succeeds(self, tmp_data_dir) -> None:
        """The default path is a no-op, not a validation error after a cleanup."""
        with patch("consolidation_memory.backends.encode_documents") as mock_embed:
            mock_embed.side_effect = lambda texts: _vec(seed=hash(texts[0]) % 997).reshape(1, -1)
            episode_id = _seed_episode("user asked a short follow up question")

            payload = _call_via_mcp("default_no_arguments", {})

        assert payload["status"] == "applied"
        assert payload["episode_targets"] == 0
        assert payload["episode_ids"] == []
        assert payload["forgotten"] == 0
        assert payload["not_found"] == 0
        # Nothing selected, so nothing deleted.
        assert _deleted_flags([episode_id]) == {episode_id: 0}

    def test_explicit_ids_report_forgotten_and_not_found(self, tmp_data_dir) -> None:
        """Counters the broken path could not report at all."""
        with patch("consolidation_memory.backends.encode_documents") as mock_embed:
            mock_embed.side_effect = lambda texts: _vec(seed=hash(texts[0]) % 997).reshape(1, -1)
            episode_id = _seed_episode("user asked a short follow up question")

            payload = _call_via_mcp(
                "explicit_ids",
                {"episode_ids": [episode_id, "ep-does-not-exist"]},
            )

        assert payload["status"] == "applied"
        assert payload["episode_targets"] == 2
        assert payload["episode_ids"] == sorted([episode_id, "ep-does-not-exist"])
        assert payload["forgotten"] == 1
        assert payload["not_found"] == 1
        assert _deleted_flags([episode_id]) == {episode_id: 1}

    def test_expire_orphans_reports_repair_on_the_applied_path(self, tmp_data_dir) -> None:
        """``expire_orphans`` fills ``orphan_repair`` in both modes."""
        with patch("consolidation_memory.backends.encode_documents") as mock_embed:
            mock_embed.side_effect = lambda texts: _vec(seed=hash(texts[0]) % 997).reshape(1, -1)
            _seed_episode("user asked a short follow up question")

            dry = _call_via_mcp("dry_run_expire_orphans", {"expire_orphans": True, "dry_run": True})
            applied = _call_via_mcp("applied_expire_orphans", {"expire_orphans": True})

        assert dry["status"] == "dry_run"
        assert dry["orphan_repair"] is not None
        assert applied["status"] == "applied"
        assert applied["orphan_repair"] is not None
        # Structurally identical apart from status and the mutation counters.
        assert set(dry) == set(applied)
        assert dry["forgotten"] == 0 and applied["forgotten"] == 0
        assert dry["not_found"] == 0 and applied["not_found"] == 0

    def test_status_value_matches_published_contract(self, tmp_data_dir) -> None:
        """Emitted ``status`` values are exactly the ones the schema advertises."""
        contract = _status_contract()
        assert contract["enum"] == ["dry_run", "applied"]
        description = contract["description"]
        for value in contract["enum"]:
            assert value in description, f"{value} missing from contract description: {description}"
        # The old producer said "ok", which the description never promised.
        assert "ok" not in description.replace("dry_run", "").replace("applied", "")

        with patch("consolidation_memory.backends.encode_documents") as mock_embed:
            mock_embed.side_effect = lambda texts: _vec(seed=hash(texts[0]) % 997).reshape(1, -1)
            _seed_episode("user asked a short follow up question")
            dry = _call_via_mcp("dry_run_status", {"dry_run": True})
            applied = _call_via_mcp("applied_status", {})

        assert dry["status"] in contract["enum"]
        assert applied["status"] in contract["enum"]


class TestHygieneApplyContractMirrorsResult:
    def test_contract_fields_match_producer_result_type(self) -> None:
        """The published contract is the producer payload, field for field."""
        result_fields = {item.name for item in dataclasses.fields(HygieneApplyResult)}
        assert set(HygieneApplyOutput.model_fields) == result_fields
        assert result_fields == set(
            HygieneApplyResult(status="applied").as_payload()
        ), "as_payload() must send every field the contract requires"

    def test_every_contract_field_is_required(self) -> None:
        """Required fields are what the producer always sends — no defaults."""
        optional = [
            name
            for name, spec in HygieneApplyOutput.model_fields.items()
            if not spec.is_required()
        ]
        assert optional == [], f"optional contract fields: {optional}"

    def test_payload_keys_are_stable_across_modes(self) -> None:
        """Both branches build the same dataclass, so keys cannot differ."""
        dry = HygieneApplyResult(status="dry_run", episode_ids=["a"])
        applied = HygieneApplyResult(status="applied", episode_ids=["a"], forgotten=1)
        assert set(dry.as_payload()) == set(applied.as_payload())
        assert dry.as_payload()["forgotten"] == 0
        assert applied.as_payload()["forgotten"] == 1
