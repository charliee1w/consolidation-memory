"""Simplified remember/ask API shared by MCP, REST, and the browser UI."""

from __future__ import annotations

from typing import Any

_SIMPLE_KIND_TO_CONTENT_TYPE: dict[str, str] = {
    "note": "exchange",
    "fact": "fact",
    "fix": "solution",
    "preference": "preference",
}

_SIMPLE_KINDS = frozenset(_SIMPLE_KIND_TO_CONTENT_TYPE)


def map_simple_kind(kind: str) -> str:
    """Map user-facing memory kinds to episode content_type values."""
    normalized = kind.strip().lower()
    if normalized not in _SIMPLE_KIND_TO_CONTENT_TYPE:
        allowed = ", ".join(sorted(_SIMPLE_KINDS))
        raise ValueError(f"kind must be one of: {allowed}")
    return _SIMPLE_KIND_TO_CONTENT_TYPE[normalized]


def build_remember_store_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    """Translate memory_remember arguments into memory_store arguments."""
    content = arguments["content"]
    kind = arguments.get("kind", "note")
    if not isinstance(kind, str):
        raise ValueError("kind must be a string")
    store_args: dict[str, Any] = {
        "content": content,
        "content_type": map_simple_kind(kind),
    }
    if "tags" in arguments and arguments["tags"] is not None:
        store_args["tags"] = arguments["tags"]
    if "surprise" in arguments and arguments["surprise"] is not None:
        store_args["surprise"] = arguments["surprise"]
    if "scope" in arguments and arguments["scope"] is not None:
        store_args["scope"] = arguments["scope"]
    return store_args


def build_ask_recall_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    """Translate memory_ask arguments into memory_recall arguments."""
    recall_args: dict[str, Any] = {
        "query": arguments["query"],
        "n_results": arguments.get("n_results", 8),
        "include_knowledge": True,
    }
    if "scope" in arguments and arguments["scope"] is not None:
        recall_args["scope"] = arguments["scope"]
    deadline = arguments.get("_recall_deadline_monotonic")
    if deadline is not None:
        recall_args["_recall_deadline_monotonic"] = deadline
    return recall_args


def simplify_recall_result(payload: dict[str, object]) -> dict[str, object]:
    """Trim a memory_recall envelope for simple consumers."""
    episodes = payload.get("episodes") or []
    knowledge = payload.get("knowledge") or []
    records = payload.get("records") or []
    claims = payload.get("claims") or []

    def _episode_row(item: object) -> dict[str, object]:
        if not isinstance(item, dict):
            return {}
        content = str(item.get("content") or "")
        preview = content if len(content) <= 280 else content[:277] + "..."
        return {
            "id": item.get("id"),
            "kind": item.get("content_type"),
            "preview": preview,
            "tags": item.get("tags") or [],
            "score": item.get("score"),
            "created_at": item.get("created_at"),
        }

    def _knowledge_row(item: object) -> dict[str, object]:
        if not isinstance(item, dict):
            return {}
        return {
            "title": item.get("title"),
            "summary": item.get("summary"),
            "confidence": item.get("confidence"),
            "filename": item.get("filename"),
        }

    def _record_row(item: object) -> dict[str, object]:
        if not isinstance(item, dict):
            return {}
        return {
            "type": item.get("record_type"),
            "topic": item.get("topic_title"),
            "text": item.get("embedding_text"),
            "confidence": item.get("confidence"),
        }

    def _claim_row(item: object) -> dict[str, object]:
        if not isinstance(item, dict):
            return {}
        reliability = item.get("reliability")
        band = None
        if isinstance(reliability, dict):
            band = reliability.get("band")
        return {
            "id": item.get("id"),
            "type": item.get("claim_type"),
            "text": item.get("canonical_text"),
            "status": item.get("status"),
            "trust": band,
            "score": item.get("relevance"),
        }

    return {
        "episodes": [_episode_row(item) for item in episodes if isinstance(item, dict)],
        "knowledge": [_knowledge_row(item) for item in knowledge if isinstance(item, dict)],
        "records": [_record_row(item) for item in records if isinstance(item, dict)],
        "claims": [_claim_row(item) for item in claims if isinstance(item, dict)],
        "total_episodes": payload.get("total_episodes"),
        "message": payload.get("message"),
        "warnings": payload.get("warnings") or [],
    }