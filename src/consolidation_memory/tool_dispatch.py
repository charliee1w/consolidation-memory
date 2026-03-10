"""Canonical sync tool dispatch for all adapter surfaces."""

from __future__ import annotations

import dataclasses
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from consolidation_memory.client import MemoryClient

_MAX_CONTENT_LENGTH = 50_000
_MAX_BATCH_SIZE = 100
_UNSAFE_FILENAME_RE = re.compile(r"[/\\]|\.\.")


def tool_requires_client(name: str) -> bool:
    return name != "memory_detect_drift"


def _require_client(client: MemoryClient | None, name: str) -> MemoryClient:
    if client is None:
        raise RuntimeError(f"Tool {name} requires a MemoryClient instance")
    return client


def _run_detect_drift(*, base_ref: str | None = None, repo_path: str | None = None) -> dict[str, Any]:
    from consolidation_memory.drift import detect_code_drift

    return dict(detect_code_drift(base_ref=base_ref, repo_path=repo_path))


def _validate_content(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("content must be a string")
    if len(value) > _MAX_CONTENT_LENGTH:
        raise ValueError(
            f"Content too long ({len(value)} chars). Maximum is {_MAX_CONTENT_LENGTH} characters."
        )
    return value


def _validate_batch_episodes(episodes: object) -> list[dict[str, Any]]:
    if not isinstance(episodes, list):
        raise ValueError("episodes must be a list")
    if len(episodes) > _MAX_BATCH_SIZE:
        raise ValueError(f"Batch size {len(episodes)} exceeds maximum of {_MAX_BATCH_SIZE}")
    validated: list[dict[str, Any]] = []
    for index, item in enumerate(episodes):
        if not isinstance(item, dict):
            raise ValueError(f"Episode {index} must be an object")
        if "content" not in item:
            raise ValueError(f"Episode {index} is missing required field 'content'")
        content = _validate_content(item["content"])
        validated.append({**item, "content": content})
    return validated


def execute_tool_call(
    name: str,
    arguments: dict[str, Any],
    *,
    client: MemoryClient | None = None,
) -> dict[str, Any]:
    """Execute a canonical tool call and return a JSON-serializable dict."""
    if name == "memory_store":
        client = _require_client(client, name)
        content = _validate_content(arguments["content"])
        scope = arguments.get("scope")
        if scope is not None and hasattr(client, "store_with_scope"):
            store_result = client.store_with_scope(
                content=content,
                content_type=arguments.get("content_type", "exchange"),
                tags=arguments.get("tags"),
                surprise=arguments.get("surprise", 0.5),
                scope=scope,
            )
        else:
            store_result = client.store(
                content=content,
                content_type=arguments.get("content_type", "exchange"),
                tags=arguments.get("tags"),
                surprise=arguments.get("surprise", 0.5),
            )
        return dataclasses.asdict(store_result)

    if name == "memory_store_batch":
        client = _require_client(client, name)
        episodes = _validate_batch_episodes(arguments["episodes"])
        scope = arguments.get("scope")
        if scope is not None and hasattr(client, "store_batch_with_scope"):
            batch_result = client.store_batch_with_scope(episodes=episodes, scope=scope)
        else:
            batch_result = client.store_batch(episodes=episodes)
        return dataclasses.asdict(batch_result)

    if name == "memory_recall":
        client = _require_client(client, name)
        recall_result = client.query_recall(
            query=arguments["query"],
            n_results=max(1, min(arguments.get("n_results", 10), 50)),
            include_knowledge=arguments.get("include_knowledge", True),
            content_types=arguments.get("content_types"),
            tags=arguments.get("tags"),
            after=arguments.get("after"),
            before=arguments.get("before"),
            include_expired=arguments.get("include_expired", False),
            as_of=arguments.get("as_of"),
            scope=arguments.get("scope"),
        )
        return dataclasses.asdict(recall_result)

    if name == "memory_search":
        client = _require_client(client, name)
        search_result = client.query_search(
            query=arguments.get("query"),
            content_types=arguments.get("content_types"),
            tags=arguments.get("tags"),
            after=arguments.get("after"),
            before=arguments.get("before"),
            limit=max(1, min(arguments.get("limit", 20), 50)),
            scope=arguments.get("scope"),
        )
        return dataclasses.asdict(search_result)

    if name == "memory_claim_browse":
        client = _require_client(client, name)
        claim_browse_result = client.query_browse_claims(
            claim_type=arguments.get("claim_type"),
            as_of=arguments.get("as_of"),
            limit=max(1, min(arguments.get("limit", 50), 200)),
            scope=arguments.get("scope"),
        )
        return dataclasses.asdict(claim_browse_result)

    if name == "memory_claim_search":
        client = _require_client(client, name)
        claim_search_result = client.query_search_claims(
            query=arguments["query"],
            claim_type=arguments.get("claim_type"),
            as_of=arguments.get("as_of"),
            limit=max(1, min(arguments.get("limit", 50), 200)),
            scope=arguments.get("scope"),
        )
        return dataclasses.asdict(claim_search_result)

    if name == "memory_detect_drift":
        if client is not None and hasattr(client, "query_detect_drift"):
            return dict(
                client.query_detect_drift(
                    base_ref=arguments.get("base_ref"),
                    repo_path=arguments.get("repo_path"),
                )
            )
        return _run_detect_drift(
            base_ref=arguments.get("base_ref"),
            repo_path=arguments.get("repo_path"),
        )

    if name == "memory_status":
        client = _require_client(client, name)
        return dataclasses.asdict(client.status())

    if name == "memory_forget":
        client = _require_client(client, name)
        return dataclasses.asdict(client.forget(episode_id=arguments["episode_id"]))

    if name == "memory_export":
        client = _require_client(client, name)
        return dataclasses.asdict(client.export())

    if name == "memory_correct":
        client = _require_client(client, name)
        return dataclasses.asdict(
            client.correct(
                topic_filename=arguments["topic_filename"],
                correction=arguments["correction"],
            )
        )

    if name == "memory_compact":
        client = _require_client(client, name)
        return dataclasses.asdict(client.compact())

    if name == "memory_consolidate":
        client = _require_client(client, name)
        consolidation_result = client.consolidate()
        if isinstance(consolidation_result, dict):
            return dict(consolidation_result)
        return dataclasses.asdict(consolidation_result)

    if name == "memory_protect":
        client = _require_client(client, name)
        return dataclasses.asdict(
            client.protect(
                episode_id=arguments.get("episode_id"),
                tag=arguments.get("tag"),
            )
        )

    if name == "memory_timeline":
        client = _require_client(client, name)
        return dataclasses.asdict(client.timeline(topic=arguments["topic"]))

    if name == "memory_contradictions":
        client = _require_client(client, name)
        return dataclasses.asdict(client.contradictions(topic=arguments.get("topic")))

    if name == "memory_browse":
        client = _require_client(client, name)
        return dataclasses.asdict(client.browse())

    if name == "memory_read_topic":
        client = _require_client(client, name)
        filename = arguments["filename"]
        if not isinstance(filename, str):
            raise ValueError("filename must be a string")
        if _UNSAFE_FILENAME_RE.search(filename):
            raise ValueError("Invalid filename: must not contain '/', '\\', or '..'")
        return dataclasses.asdict(client.read_topic(filename=filename))

    if name == "memory_decay_report":
        client = _require_client(client, name)
        return dataclasses.asdict(client.decay_report())

    if name == "memory_consolidation_log":
        client = _require_client(client, name)
        return dataclasses.asdict(
            client.consolidation_log(last_n=max(1, min(arguments.get("last_n", 5), 20)))
        )

    raise ValueError(f"Unknown tool: {name}")


def dispatch_tool_call(
    client: MemoryClient,
    name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """OpenAI-style wrapper that returns error payloads instead of raising."""
    try:
        return execute_tool_call(name, arguments, client=client)
    except Exception as exc:
        return {"error": str(exc)}
