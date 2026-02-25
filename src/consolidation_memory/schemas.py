"""OpenAI function calling schemas for consolidation-memory tools.

Provides tool definitions in the OpenAI function calling format and a
dispatch helper for routing tool calls to MemoryClient methods.

Usage with any OpenAI-compatible client::

    from openai import OpenAI
    from consolidation_memory import MemoryClient
    from consolidation_memory.schemas import openai_tools, dispatch_tool_call

    client = OpenAI()
    mem = MemoryClient()

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=openai_tools,
    )

    for call in response.choices[0].message.tool_calls:
        result = dispatch_tool_call(mem, call.function.name, json.loads(call.function.arguments))
        # Feed result back as tool response message
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from consolidation_memory.client import MemoryClient


# ── Tool Schemas ─────────────────────────────────────────────────────────────

MEMORY_STORE_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "memory_store",
        "description": (
            "Store a memory episode in the episodic buffer. "
            "Always store memories when you learn something new about the user, "
            "solve a problem, discover a preference, or encounter something surprising. "
            "Write content as a self-contained note that future-you can understand without context. "
            "Include both the problem AND solution for solution-type memories. "
            "Do NOT store trivial exchanges like greetings."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The text content to store. Include relevant context.",
                },
                "content_type": {
                    "type": "string",
                    "enum": ["exchange", "fact", "solution", "preference"],
                    "description": (
                        "Category of the memory. 'exchange' for conversation, "
                        "'fact' for learned info, 'solution' for problem+fix, "
                        "'preference' for user preference."
                    ),
                    "default": "exchange",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional topic tags for organization (e.g., ['python', 'debugging']).",
                },
                "surprise": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "How novel this is, 0.0 (routine) to 1.0 (very surprising).",
                    "default": 0.5,
                },
            },
            "required": ["content"],
        },
    },
}

MEMORY_STORE_BATCH_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "memory_store_batch",
        "description": (
            "Store multiple memory episodes in a single operation. "
            "More efficient than calling memory_store repeatedly."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "episodes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "The text content to store.",
                            },
                            "content_type": {
                                "type": "string",
                                "enum": ["exchange", "fact", "solution", "preference"],
                                "default": "exchange",
                            },
                            "tags": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "surprise": {
                                "type": "number",
                                "minimum": 0.0,
                                "maximum": 1.0,
                                "default": 0.5,
                            },
                        },
                        "required": ["content"],
                    },
                    "description": "List of episode objects to store.",
                },
            },
            "required": ["episodes"],
        },
    },
}

MEMORY_RECALL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "memory_recall",
        "description": (
            "Retrieve relevant memories by semantic similarity. "
            "Call this at the start of every new conversation and when context "
            "about the user's setup or preferences would improve your response. "
            "This is your persistent memory."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language description of what to recall.",
                },
                "n_results": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "description": "Maximum number of episode results.",
                    "default": 10,
                },
                "include_knowledge": {
                    "type": "boolean",
                    "description": "Whether to include consolidated knowledge documents.",
                    "default": True,
                },
                "content_types": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["exchange", "fact", "solution", "preference"],
                    },
                    "description": "Filter to specific content types (e.g. ['solution', 'fact']).",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter to episodes with at least one matching tag.",
                },
                "after": {
                    "type": "string",
                    "description": "Only episodes created after this ISO date (e.g. '2025-01-01').",
                },
                "before": {
                    "type": "string",
                    "description": "Only episodes created before this ISO date.",
                },
            },
            "required": ["query"],
        },
    },
}

MEMORY_STATUS_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "memory_status",
        "description": "Show memory system statistics including episode counts, knowledge base size, and backend info.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}

MEMORY_FORGET_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "memory_forget",
        "description": (
            "Mark an episode for removal from the memory system. "
            "Use to forget specific memories that are incorrect, outdated, "
            "or that the user wants removed."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "episode_id": {
                    "type": "string",
                    "description": "The UUID of the episode to forget.",
                },
            },
            "required": ["episode_id"],
        },
    },
}

MEMORY_EXPORT_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "memory_export",
        "description": (
            "Export all episodes and knowledge to a JSON snapshot. "
            "Creates a timestamped backup file and returns the file path."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}

MEMORY_CORRECT_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "memory_correct",
        "description": (
            "Correct a knowledge document with new information. "
            "Use when you discover that a knowledge document contains outdated "
            "or incorrect information and needs to be updated."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "topic_filename": {
                    "type": "string",
                    "description": "The filename of the knowledge topic (e.g., 'vr_setup.md').",
                },
                "correction": {
                    "type": "string",
                    "description": "Description of what needs to be corrected and the correct information.",
                },
            },
            "required": ["topic_filename", "correction"],
        },
    },
}

MEMORY_SEARCH_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "memory_search",
        "description": (
            "Keyword/metadata search over episodes. Works without embedding backend. "
            "Unlike memory_recall (semantic similarity), this does plain text matching. "
            "Use when the embedding backend is down, or for exact substring searches."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Text substring to search for in episode content (case-insensitive).",
                },
                "content_types": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["exchange", "fact", "solution", "preference"],
                    },
                    "description": "Filter to specific content types.",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter to episodes with at least one matching tag.",
                },
                "after": {
                    "type": "string",
                    "description": "Only episodes created after this ISO date (e.g. '2025-01-01').",
                },
                "before": {
                    "type": "string",
                    "description": "Only episodes created before this ISO date.",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "description": "Maximum results to return.",
                    "default": 20,
                },
            },
            "required": [],
        },
    },
}

# Convenience list of all tool schemas
openai_tools: list[dict[str, Any]] = [
    MEMORY_STORE_SCHEMA,
    MEMORY_STORE_BATCH_SCHEMA,
    MEMORY_RECALL_SCHEMA,
    MEMORY_SEARCH_SCHEMA,
    MEMORY_STATUS_SCHEMA,
    MEMORY_FORGET_SCHEMA,
    MEMORY_EXPORT_SCHEMA,
    MEMORY_CORRECT_SCHEMA,
]


# ── Dispatch ─────────────────────────────────────────────────────────────────

def dispatch_tool_call(
    client: MemoryClient,
    name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """Execute a tool call against a MemoryClient and return the result as a dict.

    Intended for use with OpenAI function calling responses::

        for call in response.choices[0].message.tool_calls:
            result = dispatch_tool_call(
                client,
                call.function.name,
                json.loads(call.function.arguments),
            )

    Args:
        client: MemoryClient instance.
        name: Tool function name (e.g. 'memory_store').
        arguments: Parsed arguments dict from the function call.

    Returns:
        Dict representation of the result.

    Raises:
        ValueError: If the tool name is unknown.
    """
    if name == "memory_store":
        result = client.store(
            content=arguments["content"],
            content_type=arguments.get("content_type", "exchange"),
            tags=arguments.get("tags"),
            surprise=arguments.get("surprise", 0.5),
        )
        return dataclasses.asdict(result)

    elif name == "memory_store_batch":
        result = client.store_batch(episodes=arguments["episodes"])
        return dataclasses.asdict(result)

    elif name == "memory_recall":
        result = client.recall(
            query=arguments["query"],
            n_results=arguments.get("n_results", 10),
            include_knowledge=arguments.get("include_knowledge", True),
            content_types=arguments.get("content_types"),
            tags=arguments.get("tags"),
            after=arguments.get("after"),
            before=arguments.get("before"),
        )
        return dataclasses.asdict(result)

    elif name == "memory_search":
        result = client.search(
            query=arguments.get("query"),
            content_types=arguments.get("content_types"),
            tags=arguments.get("tags"),
            after=arguments.get("after"),
            before=arguments.get("before"),
            limit=arguments.get("limit", 20),
        )
        return dataclasses.asdict(result)

    elif name == "memory_status":
        result = client.status()
        return dataclasses.asdict(result)

    elif name == "memory_forget":
        result = client.forget(episode_id=arguments["episode_id"])
        return dataclasses.asdict(result)

    elif name == "memory_export":
        result = client.export()
        return dataclasses.asdict(result)

    elif name == "memory_correct":
        result = client.correct(
            topic_filename=arguments["topic_filename"],
            correction=arguments["correction"],
        )
        return dataclasses.asdict(result)

    else:
        raise ValueError(f"Unknown tool: {name}")
