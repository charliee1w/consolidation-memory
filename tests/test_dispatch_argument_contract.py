"""Contract enforcement and MCP/dispatch parity for tool arguments.

Every published ``inputSchema`` declares ``additionalProperties: false``, so an
argument the contract does not name must be rejected on every surface — never
silently dropped. MCP rejects it in the SDK, before the handler body runs
(pydantic ``extra="forbid"`` on ``ArgModelBase``);
``tool_dispatch.reject_unknown_arguments`` is the same check for the dispatch
seam shared by the OpenAI, REST and desktop surfaces, reading the allowed names
straight from ``schemas.openai_tools`` so enforcement cannot drift from the
published contract. REST request models forbid extras too, so the rejection
survives the HTTP hop instead of being dropped by pydantic.

Cross-surface tests live here (not in ``test_schemas.py``) because they need the
MCP server registry as well as the dispatch entry point.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from consolidation_memory import mcp_compat, server
from consolidation_memory.schemas import openai_tools
from consolidation_memory.tool_dispatch import (
    ToolContractError,
    accepted_argument_names,
    dispatch_tool_call,
    execute_tool_call,
    reject_unknown_arguments,
)
from consolidation_memory.types import RecallResult, StatusResult

try:
    from fastapi.testclient import TestClient

    HAS_FASTAPI = True
except ImportError:  # pragma: no cover - exercised only without the rest extra
    HAS_FASTAPI = False

try:
    from mcp.server.mcpserver.exceptions import ToolError
except ImportError:  # pragma: no cover - depends on the installed mcp layout
    from mcp.server.fastmcp.exceptions import ToolError

PUBLISHED_TOOL_NAMES = [tool["function"]["name"] for tool in openai_tools]

# A no-argument tool (the audit repro), a tool with many optional parameters, a
# tool with a nested/dict parameter, and a tool whose name collides with a
# parameter that only other tools declare.
PARITY_SAMPLES: list[tuple[str, dict[str, Any], list[str]]] = [
    ("memory_policy_list", {}, ["dry_run", "junk"]),
    (
        "memory_recall",
        {"query": "deploy", "n_results": 5, "scope": {"project": {"slug": "repo-a"}}},
        ["junk"],
    ),
    (
        "memory_status",
        {"lightweight": True, "scope": {"namespace": {"slug": "team-a"}}},
        ["dry_run", "junk"],
    ),
    ("memory_hygiene_apply", {"dry_run": True}, ["junk"]),
]

# The same question asked over HTTP: tool name, REST route, a body the published
# contract accepts, and keys that contract does not name. ``dry_run``/``last_n``
# are published by *other* tools, and ``surprise``/``include_knowledge`` by the
# full-memory tools the simple surface deliberately does not expose, so these
# cover "wrong key for this tool" as well as "no such key anywhere".
REST_PARITY_SAMPLES: list[tuple[str, str, dict[str, Any], list[str]]] = [
    ("memory_status", "/memory/status", {"lightweight": True}, ["junk", "dry_run"]),
    (
        "memory_hygiene_apply",
        "/memory/hygiene/apply",
        {"dry_run": True},
        ["junk", "episode_id"],
    ),
    (
        "memory_consolidation_log",
        "/memory/consolidation-log",
        {"last_n": 2},
        ["junk", "dry_run"],
    ),
    (
        "memory_decay_report",
        "/memory/decay-report",
        {"global_scope": False},
        ["junk", "last_n"],
    ),
    (
        "memory_remember",
        "/memory/remember",
        {"content": "rest parity probe", "kind": "fix"},
        ["junk", "surprise"],
    ),
    (
        "memory_ask",
        "/memory/ask",
        {"query": "rest parity probe", "n_results": 3},
        ["junk", "include_knowledge"],
    ),
]


def _published_parameters(name: str) -> dict[str, Any]:
    tool = next(t for t in openai_tools if t["function"]["name"] == name)
    return tool["function"]["parameters"]


def _mcp_input_schema(name: str) -> dict[str, Any]:
    published = {tool.name: tool for tool in asyncio.run(server.mcp.list_tools())}
    return dict(published[name].input_schema or {})


def _mcp_reject_message(name: str, arguments: dict[str, Any]) -> str:
    with pytest.raises(ToolError) as excinfo:
        asyncio.run(server.mcp.call_tool(name, arguments))
    return str(excinfo.value)


def _mcp_rejection_text(name: str, arguments: dict[str, Any]) -> str:
    """Rejection text from MCP, whichever of the two rejection shapes arrives.

    A key the handler signature omits never reaches the body: the SDK rejects it
    and raises ``ToolError``. Handlers that translate and dispatch themselves
    (``memory_ask``, ``memory_remember``) instead meet the shared contract's
    error inside their own ``except Exception`` and report it as ``isError``.
    Both are rejections, so both are readable here.
    """
    try:
        result = asyncio.run(server.mcp.call_tool(name, arguments))
    except ToolError as exc:
        return str(exc)
    if result.is_error:
        return str(result.content[0].text)
    raise AssertionError(f"{name} accepted undeclared arguments: {sorted(arguments)}")


def _fake_registry(**parameters: dict[str, Any]) -> dict[str, Any]:
    """Registry stub shaped like the SDK's per-tool ``parameters`` schema."""
    return {name: SimpleNamespace(parameters=schema) for name, schema in parameters.items()}


class TestEnforcedContractMatchesPublishedSchemas:
    def test_every_published_tool_is_enforced(self):
        """A new published tool is enforced the day it is published, not later."""
        for name in PUBLISHED_TOOL_NAMES:
            assert accepted_argument_names(name) is not None, (
                f"{name} is published but carries no allowed-argument set"
            )

    def test_the_enforced_set_is_per_tool_not_a_union(self):
        """One set per tool, not one table shared by all of them.

        The derivation is per tool, so a key another tool publishes is rejected
        here. This is what makes "derived once from the published schemas" a
        statement about behaviour rather than about the source of the data: the
        two facts are the same data, and only the behaviour is falsifiable.
        """
        for name in PUBLISHED_TOOL_NAMES:
            own = accepted_argument_names(name) or frozenset()
            others: set[str] = set()
            for other in PUBLISHED_TOOL_NAMES:
                if other == name:
                    continue
                others |= set(_published_parameters(other).get("properties") or {})
            foreign = others - own
            if not foreign:
                continue

            foreign_key = min(foreign)
            with pytest.raises(ToolContractError) as excinfo:
                execute_tool_call(name, {foreign_key: 1})
            assert foreign_key in str(excinfo.value)

    def test_published_schemas_forbid_extra_properties(self):
        """The published side of the same contract is `additionalProperties: false`."""
        for name in PUBLISHED_TOOL_NAMES:
            parameters = _published_parameters(name)
            assert parameters["additionalProperties"] is False, name

    def test_mcp_publishes_the_same_argument_set_it_enforces(self):
        """MCP and dispatch enforce one contract, not two that look alike."""
        for name in PUBLISHED_TOOL_NAMES:
            input_schema = _mcp_input_schema(name)
            assert input_schema["additionalProperties"] is False, name
            assert frozenset(input_schema["properties"]) == accepted_argument_names(name), name

    @pytest.mark.parametrize("name", PUBLISHED_TOOL_NAMES)
    def test_undeclared_argument_is_rejected_for_every_tool(self, name: str):
        with pytest.raises(ToolContractError) as excinfo:
            execute_tool_call(name, {"undeclared_probe": 1})

        message = str(excinfo.value)
        assert "undeclared_probe" in message
        assert name in message
        assert "Extra inputs are not permitted" in message
        assert "inputSchema" in message

    def test_unknown_tools_keep_their_own_error(self):
        with pytest.raises(ValueError, match="Unknown tool: nonexistent_tool"):
            execute_tool_call("nonexistent_tool", {"junk": 1})


class TestSdkAndDispatchShareOneAllowedSet:
    """The two implementations of "which arguments are allowed" cannot drift.

    The SDK validates a request against the per-tool arg model derived from the
    handler signature *before* any body runs, so a rejected key never reaches
    ``reject_unknown_arguments``: the SDK check is the MCP fast path and cannot
    delegate. ``server._verify_published_argument_contract`` is the delegation
    that is possible — at startup it looks every registered tool up in the shared
    contract and compares the two, and refuses to serve on a divergence.
    """

    def test_every_registered_tool_is_registered_for_both_surfaces(self):
        registered = set(mcp_compat.registered_tools(server.mcp))
        assert registered == set(PUBLISHED_TOOL_NAMES)

    def test_startup_guard_passes_for_the_shipped_tools(self):
        server._verify_published_argument_contract()

    def test_sdk_enforced_argument_set_is_the_shared_argument_set(self):
        for name, tool in mcp_compat.registered_tools(server.mcp).items():
            parameters = getattr(tool, "parameters", None) or {}
            assert parameters.get("additionalProperties") is False, name
            enforced = frozenset(parameters.get("properties") or {})
            shared = accepted_argument_names(name)
            assert shared is not None, name
            assert enforced == shared, name
            # And the shared helper is the one that answers, both ways.
            reject_unknown_arguments(name, {key: None for key in enforced})
            with pytest.raises(ToolContractError, match="undeclared_contract_probe"):
                reject_unknown_arguments(name, {"undeclared_contract_probe": None})

    @pytest.mark.parametrize(
        ("registry", "expected"),
        [
            pytest.param(
                _fake_registry(
                    memory_hygiene_apply={
                        "additionalProperties": False,
                        "properties": {"dry_run": {}},
                    }
                ),
                "MCP accepts ['dry_run'] but the published inputSchema declares",
                id="signature-narrower-than-schema",
            ),
            pytest.param(
                _fake_registry(memory_compact={"properties": {}}),
                "does not publish additionalProperties: false",
                id="sdk-strictness-lost",
            ),
            pytest.param(
                _fake_registry(
                    memory_compact={
                        "additionalProperties": False,
                        "properties": {},
                    }
                ),
                "",
                id="agrees",
            ),
            pytest.param(
                _fake_registry(
                    memory_ghost_tool={
                        "additionalProperties": False,
                        "properties": {"anything": {}},
                    }
                ),
                "absent from schemas.openai_tools",
                id="registered-but-unpublished",
            ),
        ],
    )
    def test_startup_guard_names_the_drift(self, registry, expected):
        with patch.object(mcp_compat, "registered_tools", return_value=registry):
            if not expected:
                server._verify_published_argument_contract()
                return
            with pytest.raises(mcp_compat.MCPCompatError, match="drifted") as excinfo:
                server._verify_published_argument_contract()
        assert expected in str(excinfo.value)



class TestMcpDispatchParity:
    @pytest.mark.parametrize(
        ("name", "declared", "unknown"),
        PARITY_SAMPLES,
        ids=[sample[0] for sample in PARITY_SAMPLES],
    )
    def test_both_surfaces_reject_the_same_unknown_keys(
        self,
        name: str,
        declared: dict[str, Any],
        unknown: list[str],
    ):
        mcp_arguments = {**declared, **{key: 1 for key in unknown}}
        mcp_message = _mcp_reject_message(name, mcp_arguments)

        with pytest.raises(ToolContractError) as excinfo:
            execute_tool_call(name, mcp_arguments, client=MagicMock())
        dispatch_message = str(excinfo.value)

        for key in unknown:
            assert key in mcp_message, f"MCP must name {key} for {name}"
            assert key in dispatch_message, f"dispatch must name {key} for {name}"
        for key in declared:
            assert key not in dispatch_message, f"{name}: {key} is declared, not unknown"
        assert "Extra inputs are not permitted" in mcp_message
        assert "Extra inputs are not permitted" in dispatch_message

    def test_declared_arguments_reach_the_dispatch_seam_from_mcp(self):
        """Parity is not only about rejection: valid arguments keep working."""
        declared = {"dry_run": True}
        from consolidation_memory.types import HygieneApplyResult

        canned = HygieneApplyResult(status="dry_run").as_payload()

        with patch(
            "consolidation_memory.server.execute_tool_call",
            return_value=canned,
        ) as mock_execute:
            result = asyncio.run(server.mcp.call_tool("memory_hygiene_apply", declared))

        assert result.is_error is not True
        assert mock_execute.call_args.args[1]["dry_run"] is True

    def test_nested_scope_argument_is_a_value_not_an_unknown_key(self):
        """Dict-shaped parameters reach the client untouched on both surfaces."""
        scope = {"namespace": {"slug": "team-a"}, "policy": {"write_mode": "deny"}}

        with patch(
            "consolidation_memory.server.execute_tool_call",
            return_value={"status": "forgotten", "id": "ep-1", "message": None},
        ) as mock_execute:
            forget_args = {"episode_id": "ep-1", "scope": scope}
            asyncio.run(server.mcp.call_tool("memory_forget", forget_args))
        assert mock_execute.call_args.args[1]["scope"] == scope

        client = MagicMock()
        client.status.return_value = StatusResult(
            version="0.1.0",
            embedding_backend="fastembed",
            fast_path_hits=0,
            llm_fallbacks=0,
        )
        dispatch_tool_call(client, "memory_status", {"scope": scope})
        assert client.status.call_args.kwargs["scope"] == scope


class TestErrorSurface:
    def test_mcp_dispatch_path_reports_a_contract_violation_as_is_error(self):
        """The same rejection MCP produces for bad input, via the dispatch seam."""
        result = asyncio.run(server._call_tool_result("memory_policy_list", {"junk": 1}))

        assert result.is_error is True
        text = result.content[0].text
        assert "junk" in text
        assert "Extra inputs are not permitted" in text

    def test_contract_error_is_a_value_error(self):
        """REST maps ValueError to HTTP 422, so the REST path rejects too."""
        assert issubclass(ToolContractError, ValueError)

    def test_dispatch_raises_instead_of_returning_a_soft_error_payload(self):
        with pytest.raises(ToolContractError):
            dispatch_tool_call(MagicMock(), "memory_policy_list", {"junk": 1, "dry_run": True})

    def test_other_failures_keep_the_soft_error_payload_convention(self):
        """Only contract violations raise; runtime failures still return ``error``."""
        result = dispatch_tool_call(MagicMock(), "nonexistent_tool", {})
        assert result == {"error": "Unknown tool: nonexistent_tool"}


class TestRecallDeadlineIsNotAnArgument:
    """The deadline travels as a keyword, so no exemption list exists.

    As an injected argument key it had to be exempted from the unknown-argument
    check, and that exemption was global: ``memory_store`` accepted
    ``_recall_deadline_monotonic`` and silently dropped it, which is exactly what
    SECURITY.md says cannot happen.
    """

    def test_recall_deadline_keyword_reaches_the_client(self):
        client = MagicMock()
        client.query_recall.return_value = RecallResult(episodes=[], knowledge=[])

        execute_tool_call(
            "memory_recall",
            {"query": "deploy"},
            client=client,
            recall_deadline_monotonic=123.5,
        )

        assert client.query_recall.call_args.kwargs["recall_deadline_monotonic"] == 123.5

    @pytest.mark.parametrize("name", ["memory_store", "memory_recall", "memory_status"])
    def test_the_old_argument_key_is_now_rejected_for_every_tool(self, name: str):
        arguments = {"query": "deploy"} if name != "memory_store" else {"content": "x"}
        arguments["_recall_deadline_monotonic"] = 123.5

        with pytest.raises(ToolContractError, match="_recall_deadline_monotonic"):
            dispatch_tool_call(MagicMock(), name, arguments)

    def test_undeclared_internal_looking_key_is_still_rejected(self):
        with pytest.raises(ToolContractError, match="_junk"):
            execute_tool_call("memory_recall", {"query": "deploy", "_junk": 1})

    def test_simple_surface_re_translation_stays_inside_the_contract(self):
        client = MagicMock()
        client.query_recall.return_value = RecallResult(episodes=[], knowledge=[])

        result = dispatch_tool_call(
            client,
            "memory_ask",
            {"query": "deploy"},
            recall_deadline_monotonic=7.5,
        )

        assert result["query"] == "deploy"
        assert client.query_recall.call_args.kwargs["recall_deadline_monotonic"] == 7.5


@pytest.mark.skipif(not HAS_FASTAPI, reason="fastapi not installed")
class TestRestSurface:
    @patch("consolidation_memory.backends.encode_documents")
    def test_rest_recall_path_survives_the_internal_deadline_key(self, mock_embed):
        """REST injects the deadline key; enforcement must not reject the hop."""
        from tests.helpers import make_normalized_vec

        mock_embed.return_value = make_normalized_vec(seed=11).reshape(1, -1)

        from consolidation_memory.rest import create_app

        with TestClient(create_app()) as client:
            response = client.post("/memory/ask", json={"query": "rest contract test"})

        assert response.status_code == 200, response.text
        assert response.json()["query"] == "rest contract test"

    def test_rest_rejects_a_badly_typed_argument_with_an_error_status(self):
        """A rejected argument is an error status, never a soft ``error`` body."""
        from consolidation_memory.rest import create_app

        with TestClient(create_app()) as client:
            response = client.post("/memory/hygiene/apply", json={"dry_run": "not-a-bool"})

        assert response.status_code == 422
        assert "error" not in response.json()

    def test_rest_rejects_an_unknown_body_key_by_name_and_stores_nothing(self):
        """The request model is the REST half of the contract, not a silent filter.

        Pydantic's default ``extra="ignore"`` used to drop the key here, before
        dispatch could see it, so REST accepted junk MCP rejects. ``surprise`` is
        the exact key: it belongs to ``memory_store``, not to the published
        ``memory_remember`` schema.
        """
        from consolidation_memory.rest import create_app

        with TestClient(create_app()) as client:
            response = client.post(
                "/memory/remember",
                json={"content": "rest unknown key", "kind": "fact", "surprise": 0.9},
            )
            assert response.status_code == 422, response.text
            episodes = client.get("/memory/decay-report").json()["prunable_episodes"]
            totals = client.get("/memory/status").json()["episodic_buffer"]["total"]

        detail = response.json()["detail"]
        rejected = [entry for entry in detail if entry.get("type") == "extra_forbidden"]
        assert [entry["loc"][-1] for entry in rejected] == ["surprise"]
        assert all(entry["msg"] == "Extra inputs are not permitted" for entry in rejected)
        # No soft error payload that reads like a successful call, and nothing
        # reached the store.
        assert "error" not in response.json()
        assert totals == 0
        assert episodes == 0

    @pytest.mark.parametrize(
        ("name", "path", "valid", "unknown"),
        REST_PARITY_SAMPLES,
        ids=[sample[0] for sample in REST_PARITY_SAMPLES],
    )
    def test_rest_rejects_exactly_the_keys_mcp_and_dispatch_reject(
        self,
        name: str,
        path: str,
        valid: dict[str, Any],
        unknown: list[str],
    ):
        from consolidation_memory.rest import create_app

        offending = {key: 1 for key in unknown}
        body = {**valid, **offending}

        with TestClient(create_app()) as client:
            response = client.post(path, json=body)

        assert response.status_code == 422, response.text
        rejected = {
            entry["loc"][-1]
            for entry in response.json()["detail"]
            if entry.get("type") == "extra_forbidden"
        }
        assert rejected == set(unknown), f"REST must name exactly the undeclared keys for {name}"
        # The keys the contract does declare are not collateral damage.
        assert not rejected & set(valid)

        for key in unknown:
            assert key in _mcp_rejection_text(name, body), f"MCP must name {key} for {name}"
        with pytest.raises(ToolContractError) as excinfo:
            execute_tool_call(name, body, client=MagicMock())
        dispatch_message = str(excinfo.value)
        for key in unknown:
            assert key in dispatch_message, f"dispatch must name {key} for {name}"

    @pytest.mark.parametrize(
        ("name", "path", "valid", "unknown"),
        REST_PARITY_SAMPLES,
        ids=[sample[0] for sample in REST_PARITY_SAMPLES],
    )
    def test_rest_accepts_exactly_the_keys_the_contract_declares(
        self,
        name: str,
        path: str,
        valid: dict[str, Any],
        unknown: list[str],
    ):
        from consolidation_memory.rest import create_app

        del unknown
        with TestClient(create_app()) as client:
            response = client.post(path, json=valid)

        assert response.status_code == 200, response.text
        assert "error" not in response.json()
        # Same answer from the shared contract, without a client: the transport
        # is not what decides, the published schema is.
        reject_unknown_arguments(name, dict(valid))

