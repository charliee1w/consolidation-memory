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
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from consolidation_memory.client import MemoryClient


# ── Tool Schemas ─────────────────────────────────────────────────────────────

SCOPE_ENVELOPE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "description": (
        "Optional canonical scope envelope for universal shared memory. "
        "If omitted, legacy single-project defaults are used."
    ),
    "properties": {
        "namespace": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "slug": {"type": "string"},
                "display_name": {"type": "string"},
                "sharing_mode": {
                    "type": "string",
                    "enum": ["private", "shared", "team", "managed"],
                },
            },
        },
        "app_client": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "app_type": {
                    "type": "string",
                    "enum": [
                        "mcp",
                        "python_sdk",
                        "rest",
                        "openai_agents",
                        "langgraph",
                        "adk",
                        "letta",
                        "cli",
                        "other",
                    ],
                },
                "name": {"type": "string"},
                "provider": {"type": "string"},
                "external_key": {"type": "string"},
            },
        },
        "agent": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "name": {"type": "string"},
                "external_key": {"type": "string"},
                "model_provider": {"type": "string"},
                "model_name": {"type": "string"},
            },
        },
        "session": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "external_key": {"type": "string"},
                "session_kind": {
                    "type": "string",
                    "enum": ["conversation", "thread", "workflow", "job"],
                },
            },
        },
        "project": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "slug": {"type": "string"},
                "display_name": {"type": "string"},
                "root_uri": {"type": "string"},
                "repo_remote": {"type": "string"},
                "default_branch": {"type": "string"},
            },
        },
        "policy": {
            "type": "object",
            "properties": {
                "read_visibility": {
                    "type": "string",
                    "enum": ["private", "namespace", "project"],
                },
                "write_mode": {
                    "type": "string",
                    "enum": ["allow", "deny"],
                },
            },
        },
    },
}

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
                "scope": SCOPE_ENVELOPE_SCHEMA,
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
                "scope": SCOPE_ENVELOPE_SCHEMA,
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
            "Returns episodes, knowledge documents, and individual knowledge records "
            "(facts, solutions, preferences, procedures). "
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
                "include_expired": {
                    "type": "boolean",
                    "description": "Include temporally expired knowledge records. Default False.",
                    "default": False,
                },
                "as_of": {
                    "type": "string",
                    "description": (
                        "ISO datetime for temporal belief queries. Returns knowledge "
                        "state at that point in time, including records that have since "
                        "been superseded (e.g. '2025-06-15T00:00:00+00:00')."
                    ),
                },
                "scope": SCOPE_ENVELOPE_SCHEMA,
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
                "scope": SCOPE_ENVELOPE_SCHEMA,
            },
            "required": [],
        },
    },
}

MEMORY_CLAIM_BROWSE_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "memory_claim_browse",
        "description": (
            "Browse claims from the claim graph. "
            "Supports optional type filtering and temporal snapshot queries via as_of."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "claim_type": {
                    "type": "string",
                    "description": "Optional claim type filter (e.g. 'fact', 'solution').",
                },
                "as_of": {
                    "type": "string",
                    "description": (
                        "Optional ISO datetime for temporal claim queries. "
                        "When set, returns claims valid at that point in time."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 200,
                    "description": "Maximum claims to return.",
                    "default": 50,
                },
                "scope": SCOPE_ENVELOPE_SCHEMA,
            },
            "required": [],
        },
    },
}

MEMORY_CLAIM_SEARCH_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "memory_claim_search",
        "description": (
            "Search claims using deterministic phrase and keyword matching. "
            "Supports optional claim type filtering and temporal as_of queries."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search text to match against claim canonical text and payload.",
                },
                "claim_type": {
                    "type": "string",
                    "description": "Optional claim type filter (e.g. 'fact', 'solution').",
                },
                "as_of": {
                    "type": "string",
                    "description": (
                        "Optional ISO datetime for temporal claim queries. "
                        "When set, searches claims valid at that point in time."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 200,
                    "description": "Maximum matched claims to return.",
                    "default": 50,
                },
                "scope": SCOPE_ENVELOPE_SCHEMA,
            },
            "required": ["query"],
        },
    },
}

MEMORY_DETECT_DRIFT_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "memory_detect_drift",
        "description": (
            "Detect code drift by checking changed files and challenge impacted claims. "
            "Use after substantial file edits."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "base_ref": {
                    "type": "string",
                    "description": "Optional git base ref for comparison (e.g. 'origin/main').",
                },
                "repo_path": {
                    "type": "string",
                    "description": "Optional repository path (defaults to current working directory).",
                },
            },
            "required": [],
        },
    },
}

MEMORY_COMPACT_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "memory_compact",
        "description": (
            "Compact the FAISS index by removing tombstoned vectors. "
            "Call when memory_status shows high tombstone count or ratio."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}

MEMORY_CONSOLIDATE_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "memory_consolidate",
        "description": (
            "Manually trigger a consolidation run. "
            "Clusters unconsolidated episodes, synthesizes knowledge, "
            "prunes old episodes, and compacts FAISS. Can take several minutes."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}

MEMORY_PROTECT_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "memory_protect",
        "description": (
            "Mark episodes as immune to pruning. "
            "Protect specific episodes or all episodes with a given tag from "
            "being pruned during consolidation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "episode_id": {
                    "type": "string",
                    "description": "Protect a specific episode by its UUID.",
                },
                "tag": {
                    "type": "string",
                    "description": "Protect all episodes with this tag.",
                },
            },
            "required": [],
        },
    },
}

MEMORY_TIMELINE_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "memory_timeline",
        "description": (
            "Show how understanding of a topic has changed over time. "
            "Returns all knowledge records matching the topic sorted chronologically, "
            "including expired/superseded records."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": (
                        "Natural language topic to query "
                        "(e.g., 'frontend framework preference')."
                    ),
                },
            },
            "required": ["topic"],
        },
    },
}

MEMORY_CONTRADICTIONS_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "memory_contradictions",
        "description": (
            "List detected contradictions from the audit log. "
            "Shows cases where knowledge records contradicted each other during "
            "consolidation, including both the old and new content and how it was resolved."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "Optional topic filename or title to filter results.",
                },
            },
            "required": [],
        },
    },
}

MEMORY_BROWSE_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "memory_browse",
        "description": (
            "Browse all knowledge topics with summaries and metadata. "
            "Returns titles, summaries, record counts by type, confidence scores, "
            "and file paths."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}

MEMORY_READ_TOPIC_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "memory_read_topic",
        "description": (
            "Read the full markdown content of a knowledge topic. "
            "Use memory_browse first to see available topics."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "The filename of the knowledge topic (e.g., 'python_setup.md').",
                },
            },
            "required": ["filename"],
        },
    },
}

MEMORY_DECAY_REPORT_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "memory_decay_report",
        "description": (
            "Show what would be forgotten if pruning ran right now. "
            "Reports prunable episodes, low-confidence records, and protected episode counts. "
            "Does NOT actually delete anything."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}

MEMORY_CONSOLIDATION_LOG_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "memory_consolidation_log",
        "description": (
            "Show recent consolidation activity as a human-readable changelog. "
            "Returns summaries of recent runs: topics created/updated, "
            "contradictions detected, episodes pruned."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "last_n": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                    "description": "Number of recent runs to show (1-20, default 5).",
                    "default": 5,
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
    MEMORY_CLAIM_BROWSE_SCHEMA,
    MEMORY_CLAIM_SEARCH_SCHEMA,
    MEMORY_DETECT_DRIFT_SCHEMA,
    MEMORY_STATUS_SCHEMA,
    MEMORY_FORGET_SCHEMA,
    MEMORY_EXPORT_SCHEMA,
    MEMORY_CORRECT_SCHEMA,
    MEMORY_COMPACT_SCHEMA,
    MEMORY_CONSOLIDATE_SCHEMA,
    MEMORY_PROTECT_SCHEMA,
    MEMORY_TIMELINE_SCHEMA,
    MEMORY_CONTRADICTIONS_SCHEMA,
    MEMORY_BROWSE_SCHEMA,
    MEMORY_READ_TOPIC_SCHEMA,
    MEMORY_DECAY_REPORT_SCHEMA,
    MEMORY_CONSOLIDATION_LOG_SCHEMA,
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
        Dict representation of the result.  On any error, returns
        ``{"error": "<message>"}`` instead of raising.
    """
    _MAX_CONTENT_LENGTH = 50_000
    _MAX_BATCH_SIZE = 100
    _UNSAFE_FILENAME_RE = re.compile(r"[/\\]|\.\.")

    try:
        if name == "memory_store":
            content = arguments["content"]
            if len(content) > _MAX_CONTENT_LENGTH:
                return {
                    "error": (
                        f"Content too long ({len(content)} chars). "
                        f"Maximum is {_MAX_CONTENT_LENGTH} characters."
                    )
                }
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

        elif name == "memory_store_batch":
            episodes = arguments["episodes"]
            if len(episodes) > _MAX_BATCH_SIZE:
                return {
                    "error": f"Batch size {len(episodes)} exceeds maximum of {_MAX_BATCH_SIZE}"
                }
            scope = arguments.get("scope")
            if scope is not None and hasattr(client, "store_batch_with_scope"):
                batch_result = client.store_batch_with_scope(episodes=episodes, scope=scope)
            else:
                batch_result = client.store_batch(episodes=episodes)
            return dataclasses.asdict(batch_result)

        elif name == "memory_recall":
            n_results = max(1, min(arguments.get("n_results", 10), 50))
            recall_result = client.query_recall(
                query=arguments["query"],
                n_results=n_results,
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

        elif name == "memory_search":
            search_result = client.query_search(
                query=arguments.get("query"),
                content_types=arguments.get("content_types"),
                tags=arguments.get("tags"),
                after=arguments.get("after"),
                before=arguments.get("before"),
                limit=min(arguments.get("limit", 20), 50),
                scope=arguments.get("scope"),
            )
            return dataclasses.asdict(search_result)

        elif name == "memory_claim_browse":
            browse_claims_result = client.query_browse_claims(
                claim_type=arguments.get("claim_type"),
                as_of=arguments.get("as_of"),
                limit=min(arguments.get("limit", 50), 200),
                scope=arguments.get("scope"),
            )
            return dataclasses.asdict(browse_claims_result)

        elif name == "memory_claim_search":
            search_claims_result = client.query_search_claims(
                query=arguments["query"],
                claim_type=arguments.get("claim_type"),
                as_of=arguments.get("as_of"),
                limit=min(arguments.get("limit", 50), 200),
                scope=arguments.get("scope"),
            )
            return dataclasses.asdict(search_claims_result)

        elif name == "memory_detect_drift":
            drift_result = client.query_detect_drift(
                base_ref=arguments.get("base_ref"),
                repo_path=arguments.get("repo_path"),
            )
            return dict(drift_result)

        elif name == "memory_status":
            status_result = client.status()
            return dataclasses.asdict(status_result)

        elif name == "memory_forget":
            forget_result = client.forget(episode_id=arguments["episode_id"])
            return dataclasses.asdict(forget_result)

        elif name == "memory_export":
            export_result = client.export()
            return dataclasses.asdict(export_result)

        elif name == "memory_correct":
            correct_result = client.correct(
                topic_filename=arguments["topic_filename"],
                correction=arguments["correction"],
            )
            return dataclasses.asdict(correct_result)

        elif name == "memory_compact":
            compact_result = client.compact()
            return dataclasses.asdict(compact_result)

        elif name == "memory_consolidate":
            consolidate_result = client.consolidate()
            if isinstance(consolidate_result, dict):
                return dict(consolidate_result)
            return dataclasses.asdict(consolidate_result)

        elif name == "memory_protect":
            protect_result = client.protect(
                episode_id=arguments.get("episode_id"),
                tag=arguments.get("tag"),
            )
            return dataclasses.asdict(protect_result)

        elif name == "memory_timeline":
            timeline_result = client.timeline(topic=arguments["topic"])
            return dataclasses.asdict(timeline_result)

        elif name == "memory_contradictions":
            contradictions_result = client.contradictions(
                topic=arguments.get("topic"),
            )
            return dataclasses.asdict(contradictions_result)

        elif name == "memory_browse":
            browse_result = client.browse()
            return dataclasses.asdict(browse_result)

        elif name == "memory_read_topic":
            filename = arguments["filename"]
            if _UNSAFE_FILENAME_RE.search(filename):
                return {"error": "Invalid filename: must not contain '/', '\\', or '..'"}
            read_topic_result = client.read_topic(filename=filename)
            return dataclasses.asdict(read_topic_result)

        elif name == "memory_decay_report":
            decay_report_result = client.decay_report()
            return dataclasses.asdict(decay_report_result)

        elif name == "memory_consolidation_log":
            log_result = client.consolidation_log(
                last_n=max(1, min(arguments.get("last_n", 5), 20)),
            )
            return dataclasses.asdict(log_result)

        else:
            return {"error": f"Unknown tool: {name}"}

    except Exception as e:
        return {"error": str(e)}
