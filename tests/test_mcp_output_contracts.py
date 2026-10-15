"""Real handler payloads must validate against the published output schemas.

Each tool runs through its actual dispatch path against the throwaway data
directory (conftest ``tmp_data_dir``) and the raw payload is validated against
the published ``anyOf[success, error]`` schema. This is the MCP spec contract:
servers MUST return structured results that conform to outputSchema, so a
payload/schema drift has to fail here before it fails a client.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

import jsonschema

from consolidation_memory import server

_REPO_SRC = Path(__file__).resolve().parents[1] / "src"
_SCHEMA_CACHE: dict[str, dict[str, Any]] = {}


def _published_schema(tool_name: str) -> dict[str, Any]:
    if tool_name not in _SCHEMA_CACHE:
        tools = asyncio.run(server.mcp.list_tools())
        tool = next(t for t in tools if t.name == tool_name)
        assert tool.output_schema is not None, tool_name
        _SCHEMA_CACHE[tool_name] = tool.output_schema
    return _SCHEMA_CACHE[tool_name]


def _validate(tool_name: str, case: str, result: Any) -> dict[str, Any]:
    payload = result.structured_content if hasattr(result, "structured_content") else result
    assert isinstance(payload, dict), f"{tool_name}/{case}: payload is not an object"
    try:
        jsonschema.validate(payload, _published_schema(tool_name))
    except jsonschema.ValidationError as exc:
        raise AssertionError(
            f"{tool_name}/{case}: payload violates output schema: {exc.message}"
        ) from exc
    return payload


def _call(tool_name: str, case: str, *, expect_success: bool = True, **kwargs: Any) -> Any:
    result = asyncio.run(getattr(server, tool_name)(**kwargs))
    if expect_success:
        assert not hasattr(result, "is_error"), f"{tool_name}/{case}: expected success, got isError"
    _validate(tool_name, case, result)
    return result


def _seed_episode() -> str | None:
    from consolidation_memory.database import ensure_schema, insert_episode

    ensure_schema()
    insert_episode(content="The deploy pipeline fails when the migration checksum mismatches.")
    stored = _call(
        "memory_store",
        "seed",
        content="unique falcon encoding issue in parser",
        tags=["bug"],
    )
    payload = stored.structured_content if hasattr(stored, "structured_content") else stored
    return payload.get("id")


def test_success_payloads_validate_against_published_schemas(
    monkeypatch: Any, tmp_path: Path
) -> None:
    monkeypatch.setenv("PYTHONPATH", str(_REPO_SRC) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    monkeypatch.setenv("CONSOLIDATION_MEMORY_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("CONSOLIDATION_MEMORY_PROJECT", "default")
    monkeypatch.setenv("CONSOLIDATION_MEMORY_LLM_BACKEND", "disabled")

    store_id = _seed_episode()

    _call("memory_store", "success", content="Redis cache misses under load spike", tags=["perf"])
    _call("memory_remember", "success", content="Prefer podman over docker locally")
    _call(
        "memory_store_batch",
        "success",
        episodes=[{"content": "batch one"}, {"content": "batch two", "content_type": "fact"}],
    )
    _call("memory_search", "success", query="deploy", limit=5)
    _call("memory_search", "empty", query="zzzz-no-match")
    _call("memory_recall", "success", query="deploy failure", n_results=5)
    _call("memory_ask", "success", query="what broke the deploy?")
    _call("memory_status", "full")
    _call("memory_status", "lightweight", lightweight=True)
    _call("memory_export", "success")
    _call("memory_claim_browse", "empty")
    _call("memory_claim_search", "empty", query="deploy")
    _call(
        "memory_outcome_record",
        "success",
        action_summary="Fixed the flaky migration test",
        outcome_type="success",
        source_episode_ids=[store_id] if store_id else None,
        confidence=0.9,
    )
    _call("memory_outcome_browse", "success")
    _call("memory_browse", "empty")
    _call("memory_read_topic", "not_found", filename="no-such-topic.md")
    _call("memory_correct", "not_found", topic_filename="no-such.md", correction="x")
    _call("memory_timeline", "empty", topic="deploy")
    _call("memory_contradictions", "empty")
    _call("memory_decay_report", "success")
    _call("memory_consolidation_log", "success")
    _call("memory_hygiene_scan", "success")
    _call("memory_hygiene_apply", "dry_run", dry_run=True)
    _call("memory_policy_list", "success")
    _call("memory_scope_list", "success", limit=10)
    _call(
        "memory_policy_grant",
        "success",
        principal_type="agent",
        principal_key="ci-bot",
        write_mode="allow",
    )
    _call("memory_compact", "success")
    from consolidation_memory.config import override_config

    with override_config(LLM_BACKEND="disabled"):
        _call("memory_consolidate", "disabled_llm")
    if store_id:
        _call("memory_protect", "success", episode_id=store_id)
        _call("memory_forget", "success", episode_id=store_id)
    _call("memory_forget", "not_found", episode_id="ep-missing")
    _call("memory_protect", "not_found", episode_id="ep-missing")

    repo = str(Path(__file__).resolve().parents[1])
    _call("memory_detect_drift", "success", repo_path=repo)


def test_error_payloads_validate_against_published_schemas() -> None:
    async def timeout_error(*args: Any, **kwargs: Any) -> dict[str, object]:
        raise TimeoutError("client timeout")

    with (
        patch.object(server, "_call_tool_payload", timeout_error),
        patch.object(server, "tool_requires_client", lambda name: False),
    ):
        _call("memory_status", "timeout_error", expect_success=False, lightweight=True)
        result = _call("memory_consolidate", "timeout_error", expect_success=False)
        assert result.is_error is True

    with (
        patch.object(server, "_run_blocking", timeout_error),
        patch.object(server, "tool_requires_client", lambda name: False),
    ):
        recall_result = _call("memory_recall", "double_timeout", expect_success=False, query="anything")
        assert recall_result.is_error is True
        _call("memory_ask", "timeout", expect_success=False, query="anything")

    with patch.object(server, "run_detect_drift_subprocess", timeout_error):
        degraded = _call("memory_detect_drift", "degraded_timeout", expect_success=False, base_ref="main")
        assert degraded.is_error is True

    # The degraded payload satisfies both arms of the published anyOf, so
    # validating it proves nothing. It must carry the error arm's required key,
    # or a client that checks `error` reads a timeout as a successful empty scan.
    assert degraded.structured_content.get("error"), (
        "degraded drift payload has no `error` key; it validates against the "
        "success arm and reads as an empty scan"
    )
    arms = _published_schema("memory_detect_drift")["anyOf"]
    error_arm = next(arm for arm in arms if "error" in (arm.get("properties") or {}))
    assert set(error_arm["required"]) <= set(degraded.structured_content)

    async def drift_crash(*args: Any, **kwargs: Any) -> dict[str, object]:
        raise RuntimeError("drift worker exploded")

    with patch.object(server, "run_detect_drift_subprocess", drift_crash):
        _call("memory_detect_drift", "error", expect_success=False)

    non_repo = _call(
        "memory_detect_drift",
        "non_repo_error",
        expect_success=False,
        repo_path="/nonexistent-repo",
    )
    assert non_repo.is_error is True


def test_recall_keyword_fallback_payload_validates() -> None:
    from consolidation_memory.database import ensure_schema, insert_episode
    from consolidation_memory.tool_adapter import build_recall_timeout_fallback_result

    ensure_schema()
    insert_episode(content="The deploy pipeline fails when the migration checksum mismatches.")
    search = _call("memory_search", "fallback_source", query="deploy")
    fallback = build_recall_timeout_fallback_result(
        search, recall_timeout_seconds=1.0, include_knowledge=True
    )
    _validate("memory_recall", "fallback_success", fallback)
    assert "entity_resolution" not in fallback, "fallback omits the optional resolver block"


def test_error_results_carry_is_error_and_actionable_text(monkeypatch: Any) -> None:
    async def timeout_error(*args: Any, **kwargs: Any) -> dict[str, object]:
        raise TimeoutError("client timeout")

    monkeypatch.setattr(server, "_call_tool_payload", timeout_error)
    monkeypatch.setattr(server, "tool_requires_client", lambda name: False)
    result = asyncio.run(server.mcp.call_tool("memory_status", {"lightweight": True}))
    assert result.is_error is True
    assert "timed out" in result.content[0].text
    payload = _validate("memory_status", "timeout_via_mcp", result)
    assert "timed out" in payload["error"]
