"""Canonical sync tool dispatch for all adapter surfaces."""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from consolidation_memory.client import MemoryClient

from consolidation_memory.types import (
    ContentType,
    OUTCOME_TYPES,
    ScopeEnvelope,
    coerce_scope_envelope,
)

_MAX_CONTENT_LENGTH = 50_000
_MAX_BATCH_SIZE = 100
_MAX_QUERY_LENGTH = 10_000
_MAX_TOPIC_LENGTH = 500
_MAX_FILENAME_LENGTH = 255
_MAX_PATH_LENGTH = 4096
_MAX_TAGS = 100
_MAX_TAG_LENGTH = 100
_MAX_OUTCOME_SOURCE_IDS = 500
_UNSAFE_FILENAME_RE = re.compile(r"[/\\]|\.\.")
_WINDOWS_ABS_PATH_RE = re.compile(r"^[a-zA-Z]:[/\\]")
_URI_PREFIX_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")
_VALID_CONTENT_TYPES = frozenset(content_type.value for content_type in ContentType)
_VALID_OUTCOME_TYPES = frozenset(OUTCOME_TYPES)


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


def _validate_optional_text(
    field_name: str,
    value: object,
    *,
    max_length: int,
    allow_empty: bool = True,
) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    if not allow_empty and not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    if len(value) > max_length:
        raise ValueError(
            f"{field_name} too long ({len(value)} chars). Maximum is {max_length} characters."
        )
    return value


def _validate_required_text(field_name: str, value: object, *, max_length: int) -> str:
    validated = _validate_optional_text(
        field_name,
        value,
        max_length=max_length,
        allow_empty=False,
    )
    if validated is None:
        raise ValueError(f"{field_name} is required")
    return validated


def _validate_content_type(value: object, *, field_name: str = "content_type") -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    if value not in _VALID_CONTENT_TYPES:
        allowed = ", ".join(sorted(_VALID_CONTENT_TYPES))
        raise ValueError(f"{field_name} must be one of: {allowed}")
    return value


def _validate_content_type_list(field_name: str, value: object) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list of strings")
    validated: list[str] = []
    for index, item in enumerate(value):
        validated.append(
            _validate_content_type(
                item,
                field_name=f"{field_name}[{index}]",
            )
        )
    return validated


def _validate_tags(value: object) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError("tags must be a list of strings")
    if len(value) > _MAX_TAGS:
        raise ValueError(f"tags exceeds maximum of {_MAX_TAGS} entries")
    validated: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise ValueError(f"tags[{index}] must be a string")
        if len(item) > _MAX_TAG_LENGTH:
            raise ValueError(
                f"tags[{index}] too long ({len(item)} chars). Maximum is {_MAX_TAG_LENGTH} characters."
            )
        validated.append(item)
    return validated


def _validate_surprise(value: object, *, field_name: str = "surprise") -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a number between 0.0 and 1.0")
    surprise = float(value)
    if not 0.0 <= surprise <= 1.0:
        raise ValueError(f"{field_name} must be between 0.0 and 1.0")
    return surprise


def _validate_string_list(
    field_name: str,
    value: object,
    *,
    max_items: int,
    max_length: int,
) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list of strings")
    if len(value) > max_items:
        raise ValueError(f"{field_name} exceeds maximum of {max_items} entries")
    validated: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise ValueError(f"{field_name}[{index}] must be a string")
        token = item.strip()
        if not token:
            continue
        if len(token) > max_length:
            raise ValueError(
                f"{field_name}[{index}] too long ({len(token)} chars). Maximum is {max_length} characters."
            )
        validated.append(token)
    return validated


def _validate_outcome_type(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("outcome_type must be a string")
    token = value.strip().lower()
    if token not in _VALID_OUTCOME_TYPES:
        allowed = ", ".join(sorted(_VALID_OUTCOME_TYPES))
        raise ValueError(f"outcome_type must be one of: {allowed}")
    return token


def _validate_code_anchors(value: object) -> list[dict[str, str]] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError("code_anchors must be a list of objects")
    anchors: list[dict[str, str]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError(f"code_anchors[{index}] must be an object")
        anchor_type = item.get("anchor_type", item.get("type"))
        anchor_value = item.get("anchor_value", item.get("value"))
        if not isinstance(anchor_type, str) or not anchor_type.strip():
            raise ValueError(f"code_anchors[{index}].anchor_type must be a non-empty string")
        if not isinstance(anchor_value, str) or not anchor_value.strip():
            raise ValueError(f"code_anchors[{index}].anchor_value must be a non-empty string")
        anchors.append(
            {
                "anchor_type": anchor_type.strip(),
                "anchor_value": anchor_value.strip(),
            }
        )
    return anchors


def _validate_details(value: object, *, field_name: str) -> dict[str, object] | str | None:
    if value is None:
        return None
    if isinstance(value, str):
        if len(value) > _MAX_CONTENT_LENGTH:
            raise ValueError(
                f"{field_name} too long ({len(value)} chars). Maximum is {_MAX_CONTENT_LENGTH} characters."
            )
        return value
    if isinstance(value, Mapping):
        return dict(value)
    raise ValueError(f"{field_name} must be an object or string")


def _validate_bounded_int(
    field_name: str,
    value: object,
    *,
    minimum: int,
    maximum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer between {minimum} and {maximum}")
    if value < minimum or value > maximum:
        raise ValueError(f"{field_name} must be between {minimum} and {maximum}")
    return value


def _validate_bool(field_name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean")
    return value


def _validate_scope(value: object) -> ScopeEnvelope | dict[str, object] | None:
    if value is None:
        return None
    if isinstance(value, ScopeEnvelope):
        coerce_scope_envelope(value)
        return value
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("scope string must not be empty")

        project_scope: dict[str, str] = {"slug": cleaned}
        if (
            _WINDOWS_ABS_PATH_RE.match(cleaned)
            or cleaned.startswith(("/", "\\\\", "./", "../"))
            or "/" in cleaned
            or "\\" in cleaned
            or _URI_PREFIX_RE.match(cleaned)
        ):
            project_scope = {"root_uri": cleaned}

        normalized_from_string: dict[str, object] = {"project": project_scope}
        coerce_scope_envelope(normalized_from_string)
        return normalized_from_string
    if not isinstance(value, Mapping):
        raise ValueError("scope must be an object or string")

    normalized: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError("scope keys must be strings")
        normalized[key] = item

    coerce_scope_envelope(normalized)
    return normalized


def _validate_filename(value: object, *, field_name: str = "filename") -> str:
    filename = _validate_required_text(field_name, value, max_length=_MAX_FILENAME_LENGTH)
    if _UNSAFE_FILENAME_RE.search(filename):
        raise ValueError(f"Invalid {field_name}: must not contain '/', '\\\\', or '..'")
    return filename


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
        content_type = item.get("content_type", "exchange")
        surprise = item.get("surprise", 0.5)
        validated.append(
            {
                **item,
                "content": content,
                "content_type": _validate_content_type(
                    content_type,
                    field_name=f"episodes[{index}].content_type",
                ),
                "tags": _validate_tags(item.get("tags")),
                "surprise": _validate_surprise(
                    surprise,
                    field_name=f"episodes[{index}].surprise",
                ),
            }
        )
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
        content_type = _validate_content_type(arguments.get("content_type", "exchange"))
        tags = _validate_tags(arguments.get("tags"))
        surprise = _validate_surprise(arguments.get("surprise", 0.5))
        scope = _validate_scope(arguments.get("scope"))
        if scope is not None and hasattr(client, "store_with_scope"):
            store_result = client.store_with_scope(
                content=content,
                content_type=content_type,
                tags=tags,
                surprise=surprise,
                scope=scope,
            )
        else:
            store_result = client.store(
                content=content,
                content_type=content_type,
                tags=tags,
                surprise=surprise,
            )
        return dataclasses.asdict(store_result)

    if name == "memory_store_batch":
        client = _require_client(client, name)
        episodes = _validate_batch_episodes(arguments["episodes"])
        scope = _validate_scope(arguments.get("scope"))
        if scope is not None and hasattr(client, "store_batch_with_scope"):
            batch_result = client.store_batch_with_scope(episodes=episodes, scope=scope)
        else:
            batch_result = client.store_batch(episodes=episodes)
        return dataclasses.asdict(batch_result)

    if name == "memory_recall":
        client = _require_client(client, name)
        query = _validate_required_text("query", arguments["query"], max_length=_MAX_QUERY_LENGTH)
        n_results = _validate_bounded_int(
            "n_results",
            arguments.get("n_results", 10),
            minimum=1,
            maximum=50,
        )
        recall_result = client.query_recall(
            query=query,
            n_results=n_results,
            include_knowledge=_validate_bool(
                "include_knowledge",
                arguments.get("include_knowledge", True),
            ),
            content_types=_validate_content_type_list(
                "content_types",
                arguments.get("content_types"),
            ),
            tags=_validate_tags(arguments.get("tags")),
            after=_validate_optional_text("after", arguments.get("after"), max_length=64),
            before=_validate_optional_text("before", arguments.get("before"), max_length=64),
            include_expired=_validate_bool(
                "include_expired",
                arguments.get("include_expired", False),
            ),
            as_of=_validate_optional_text("as_of", arguments.get("as_of"), max_length=64),
            scope=_validate_scope(arguments.get("scope")),
        )
        return dataclasses.asdict(recall_result)

    if name == "memory_search":
        client = _require_client(client, name)
        limit = _validate_bounded_int("limit", arguments.get("limit", 20), minimum=1, maximum=50)
        search_result = client.query_search(
            query=_validate_optional_text(
                "query",
                arguments.get("query"),
                max_length=_MAX_QUERY_LENGTH,
                allow_empty=False,
            ),
            content_types=_validate_content_type_list(
                "content_types",
                arguments.get("content_types"),
            ),
            tags=_validate_tags(arguments.get("tags")),
            after=_validate_optional_text("after", arguments.get("after"), max_length=64),
            before=_validate_optional_text("before", arguments.get("before"), max_length=64),
            limit=limit,
            scope=_validate_scope(arguments.get("scope")),
        )
        return dataclasses.asdict(search_result)

    if name == "memory_claim_browse":
        client = _require_client(client, name)
        claim_browse_result = client.query_browse_claims(
            claim_type=_validate_optional_text(
                "claim_type",
                arguments.get("claim_type"),
                max_length=64,
                allow_empty=False,
            ),
            as_of=_validate_optional_text("as_of", arguments.get("as_of"), max_length=64),
            limit=_validate_bounded_int(
                "limit",
                arguments.get("limit", 50),
                minimum=1,
                maximum=200,
            ),
            scope=_validate_scope(arguments.get("scope")),
        )
        return dataclasses.asdict(claim_browse_result)

    if name == "memory_claim_search":
        client = _require_client(client, name)
        claim_search_result = client.query_search_claims(
            query=_validate_required_text("query", arguments["query"], max_length=_MAX_QUERY_LENGTH),
            claim_type=_validate_optional_text(
                "claim_type",
                arguments.get("claim_type"),
                max_length=64,
                allow_empty=False,
            ),
            as_of=_validate_optional_text("as_of", arguments.get("as_of"), max_length=64),
            limit=_validate_bounded_int(
                "limit",
                arguments.get("limit", 50),
                minimum=1,
                maximum=200,
            ),
            scope=_validate_scope(arguments.get("scope")),
        )
        return dataclasses.asdict(claim_search_result)

    if name == "memory_outcome_record":
        client = _require_client(client, name)
        source_claim_ids = _validate_string_list(
            "source_claim_ids",
            arguments.get("source_claim_ids"),
            max_items=_MAX_OUTCOME_SOURCE_IDS,
            max_length=_MAX_FILENAME_LENGTH,
        )
        source_record_ids = _validate_string_list(
            "source_record_ids",
            arguments.get("source_record_ids"),
            max_items=_MAX_OUTCOME_SOURCE_IDS,
            max_length=_MAX_FILENAME_LENGTH,
        )
        source_episode_ids = _validate_string_list(
            "source_episode_ids",
            arguments.get("source_episode_ids"),
            max_items=_MAX_OUTCOME_SOURCE_IDS,
            max_length=_MAX_FILENAME_LENGTH,
        )
        if not source_claim_ids and not source_record_ids and not source_episode_ids:
            raise ValueError(
                "At least one source_claim_ids, source_record_ids, or source_episode_ids entry is required"
            )
        outcome_result = client.record_outcome(
            action_summary=_validate_required_text(
                "action_summary",
                arguments["action_summary"],
                max_length=_MAX_QUERY_LENGTH,
            ),
            outcome_type=_validate_outcome_type(arguments.get("outcome_type")),
            source_claim_ids=source_claim_ids,
            source_record_ids=source_record_ids,
            source_episode_ids=source_episode_ids,
            code_anchors=_validate_code_anchors(arguments.get("code_anchors")),
            issue_ids=_validate_string_list(
                "issue_ids",
                arguments.get("issue_ids"),
                max_items=_MAX_OUTCOME_SOURCE_IDS,
                max_length=_MAX_FILENAME_LENGTH,
            ),
            pr_ids=_validate_string_list(
                "pr_ids",
                arguments.get("pr_ids"),
                max_items=_MAX_OUTCOME_SOURCE_IDS,
                max_length=_MAX_FILENAME_LENGTH,
            ),
            action_key=_validate_optional_text(
                "action_key",
                arguments.get("action_key"),
                max_length=_MAX_FILENAME_LENGTH,
                allow_empty=False,
            ),
            summary=_validate_optional_text(
                "summary",
                arguments.get("summary"),
                max_length=_MAX_CONTENT_LENGTH,
                allow_empty=False,
            ),
            details=_validate_details(arguments.get("details"), field_name="details"),
            confidence=_validate_surprise(
                arguments.get("confidence", 0.8),
                field_name="confidence",
            ),
            provenance=_validate_details(arguments.get("provenance"), field_name="provenance"),
            observed_at=_validate_optional_text(
                "observed_at",
                arguments.get("observed_at"),
                max_length=64,
                allow_empty=False,
            ),
            scope=_validate_scope(arguments.get("scope")),
        )
        return dataclasses.asdict(outcome_result)

    if name == "memory_outcome_browse":
        client = _require_client(client, name)
        outcome_browse_result = client.query_browse_outcomes(
            outcome_type=(
                _validate_outcome_type(arguments.get("outcome_type"))
                if arguments.get("outcome_type") is not None
                else None
            ),
            action_key=_validate_optional_text(
                "action_key",
                arguments.get("action_key"),
                max_length=_MAX_FILENAME_LENGTH,
                allow_empty=False,
            ),
            source_claim_id=_validate_optional_text(
                "source_claim_id",
                arguments.get("source_claim_id"),
                max_length=_MAX_FILENAME_LENGTH,
                allow_empty=False,
            ),
            source_record_id=_validate_optional_text(
                "source_record_id",
                arguments.get("source_record_id"),
                max_length=_MAX_FILENAME_LENGTH,
                allow_empty=False,
            ),
            source_episode_id=_validate_optional_text(
                "source_episode_id",
                arguments.get("source_episode_id"),
                max_length=_MAX_FILENAME_LENGTH,
                allow_empty=False,
            ),
            as_of=_validate_optional_text("as_of", arguments.get("as_of"), max_length=64),
            limit=_validate_bounded_int(
                "limit",
                arguments.get("limit", 50),
                minimum=1,
                maximum=200,
            ),
            scope=_validate_scope(arguments.get("scope")),
        )
        return dataclasses.asdict(outcome_browse_result)

    if name == "memory_detect_drift":
        base_ref = _validate_optional_text(
            "base_ref",
            arguments.get("base_ref"),
            max_length=_MAX_FILENAME_LENGTH,
            allow_empty=False,
        )
        repo_path = _validate_optional_text(
            "repo_path",
            arguments.get("repo_path"),
            max_length=_MAX_PATH_LENGTH,
            allow_empty=False,
        )
        if client is not None and hasattr(client, "query_detect_drift"):
            return dict(
                client.query_detect_drift(
                    base_ref=base_ref,
                    repo_path=repo_path,
                )
            )
        return _run_detect_drift(
            base_ref=base_ref,
            repo_path=repo_path,
        )

    if name == "memory_status":
        client = _require_client(client, name)
        return dataclasses.asdict(client.status())

    if name == "memory_forget":
        client = _require_client(client, name)
        return dataclasses.asdict(
            client.forget(
                episode_id=_validate_required_text(
                    "episode_id",
                    arguments["episode_id"],
                    max_length=_MAX_FILENAME_LENGTH,
                ),
                scope=_validate_scope(arguments.get("scope")),
            )
        )

    if name == "memory_export":
        client = _require_client(client, name)
        return dataclasses.asdict(client.export(scope=_validate_scope(arguments.get("scope"))))

    if name == "memory_correct":
        client = _require_client(client, name)
        return dataclasses.asdict(
            client.correct(
                topic_filename=_validate_filename(arguments["topic_filename"], field_name="topic_filename"),
                correction=_validate_required_text(
                    "correction",
                    arguments["correction"],
                    max_length=_MAX_CONTENT_LENGTH,
                ),
                scope=_validate_scope(arguments.get("scope")),
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
                episode_id=_validate_optional_text(
                    "episode_id",
                    arguments.get("episode_id"),
                    max_length=_MAX_FILENAME_LENGTH,
                    allow_empty=False,
                ),
                tag=_validate_optional_text(
                    "tag",
                    arguments.get("tag"),
                    max_length=_MAX_TAG_LENGTH,
                    allow_empty=False,
                ),
                scope=_validate_scope(arguments.get("scope")),
            )
        )

    if name == "memory_timeline":
        client = _require_client(client, name)
        return dataclasses.asdict(
            client.timeline(
                topic=_validate_required_text("topic", arguments["topic"], max_length=_MAX_TOPIC_LENGTH),
                scope=_validate_scope(arguments.get("scope")),
            )
        )

    if name == "memory_contradictions":
        client = _require_client(client, name)
        return dataclasses.asdict(
            client.contradictions(
                topic=_validate_optional_text(
                    "topic",
                    arguments.get("topic"),
                    max_length=_MAX_FILENAME_LENGTH,
                    allow_empty=False,
                )
            )
        )

    if name == "memory_browse":
        client = _require_client(client, name)
        return dataclasses.asdict(client.browse(scope=_validate_scope(arguments.get("scope"))))

    if name == "memory_read_topic":
        client = _require_client(client, name)
        filename = _validate_filename(arguments["filename"])
        return dataclasses.asdict(
            client.read_topic(
                filename=filename,
                scope=_validate_scope(arguments.get("scope")),
            )
        )

    if name == "memory_decay_report":
        client = _require_client(client, name)
        return dataclasses.asdict(client.decay_report())

    if name == "memory_consolidation_log":
        client = _require_client(client, name)
        return dataclasses.asdict(
            client.consolidation_log(
                last_n=_validate_bounded_int(
                    "last_n",
                    arguments.get("last_n", 5),
                    minimum=1,
                    maximum=20,
                )
            )
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
