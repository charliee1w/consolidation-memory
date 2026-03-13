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

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from consolidation_memory.client import MemoryClient

from consolidation_memory.tool_dispatch import dispatch_tool_call as _dispatch_tool_call


# ── Tool Schemas ─────────────────────────────────────────────────────────────

_MAX_CONTENT_LENGTH = 50_000
_MAX_QUERY_LENGTH = 10_000
_MAX_TOPIC_LENGTH = 500
_MAX_FILENAME_LENGTH = 255
_MAX_PATH_LENGTH = 4096

SCOPE_ENVELOPE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "description": (
        "Optional canonical scope envelope for universal shared memory. "
        "If omitted, legacy single-project defaults are used."
    ),
    "additionalProperties": False,
    "properties": {
        "namespace": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "id": {"type": "string", "maxLength": _MAX_FILENAME_LENGTH},
                "slug": {"type": "string", "maxLength": _MAX_FILENAME_LENGTH},
                "display_name": {"type": "string", "maxLength": _MAX_TOPIC_LENGTH},
                "sharing_mode": {
                    "type": "string",
                    "enum": ["private", "shared", "team", "managed"],
                },
            },
        },
        "app_client": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "id": {"type": "string", "maxLength": _MAX_FILENAME_LENGTH},
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
                "name": {"type": "string", "maxLength": _MAX_FILENAME_LENGTH},
                "provider": {"type": "string", "maxLength": _MAX_FILENAME_LENGTH},
                "external_key": {"type": "string", "maxLength": _MAX_FILENAME_LENGTH},
            },
        },
        "agent": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "id": {"type": "string", "maxLength": _MAX_FILENAME_LENGTH},
                "name": {"type": "string", "maxLength": _MAX_FILENAME_LENGTH},
                "external_key": {"type": "string", "maxLength": _MAX_FILENAME_LENGTH},
                "model_provider": {"type": "string", "maxLength": _MAX_FILENAME_LENGTH},
                "model_name": {"type": "string", "maxLength": _MAX_FILENAME_LENGTH},
            },
        },
        "session": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "id": {"type": "string", "maxLength": _MAX_FILENAME_LENGTH},
                "external_key": {"type": "string", "maxLength": _MAX_FILENAME_LENGTH},
                "session_kind": {
                    "type": "string",
                    "enum": ["conversation", "thread", "workflow", "job"],
                },
            },
        },
        "project": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "id": {"type": "string", "maxLength": _MAX_FILENAME_LENGTH},
                "slug": {"type": "string", "maxLength": _MAX_FILENAME_LENGTH},
                "display_name": {"type": "string", "maxLength": _MAX_TOPIC_LENGTH},
                "root_uri": {"type": "string", "maxLength": _MAX_PATH_LENGTH},
                "repo_remote": {"type": "string", "maxLength": _MAX_PATH_LENGTH},
                "default_branch": {"type": "string", "maxLength": _MAX_FILENAME_LENGTH},
            },
        },
        "policy": {
            "type": "object",
            "additionalProperties": False,
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

SCOPE_INPUT_SCHEMA: dict[str, Any] = {
    "description": (
        "Optional scope input. Use a canonical scope object, or pass a string shorthand "
        "that auto-maps to project scope "
        "(path-like values -> project.root_uri, otherwise -> project.slug)."
    ),
    "oneOf": [
        SCOPE_ENVELOPE_SCHEMA,
        {
            "type": "string",
            "maxLength": _MAX_PATH_LENGTH,
        },
    ],
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
            "additionalProperties": False,
            "properties": {
                "content": {
                    "type": "string",
                    "maxLength": _MAX_CONTENT_LENGTH,
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
                "scope": SCOPE_INPUT_SCHEMA,
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
            "additionalProperties": False,
            "properties": {
                "episodes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "content": {
                                "type": "string",
                                "maxLength": _MAX_CONTENT_LENGTH,
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
                "scope": SCOPE_INPUT_SCHEMA,
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
            "additionalProperties": False,
            "properties": {
                "query": {
                    "type": "string",
                    "maxLength": _MAX_QUERY_LENGTH,
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
                    "maxLength": 64,
                    "description": "Only episodes created after this ISO date (e.g. '2025-01-01').",
                },
                "before": {
                    "type": "string",
                    "maxLength": 64,
                    "description": "Only episodes created before this ISO date.",
                },
                "include_expired": {
                    "type": "boolean",
                    "description": "Include temporally expired knowledge records. Default False.",
                    "default": False,
                },
                "as_of": {
                    "type": "string",
                    "maxLength": 64,
                    "description": (
                        "ISO datetime for temporal belief queries. Returns knowledge "
                        "state at that point in time, including records that have since "
                        "been superseded (e.g. '2025-06-15T00:00:00+00:00')."
                    ),
                },
                "scope": SCOPE_INPUT_SCHEMA,
            },
            "required": ["query"],
        },
    },
}

MEMORY_STATUS_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "memory_status",
        "description": (
            "Show memory system statistics, including trust posture, claim coverage, "
            "provenance coverage, drift-watch pressure, episode counts, and backend info."
        ),
        "parameters": {
            "type": "object",
            "additionalProperties": False,
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
            "additionalProperties": False,
            "properties": {
                "episode_id": {
                    "type": "string",
                    "maxLength": _MAX_FILENAME_LENGTH,
                    "description": "The UUID of the episode to forget.",
                },
                "scope": SCOPE_INPUT_SCHEMA,
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
            "additionalProperties": False,
            "properties": {
                "scope": SCOPE_INPUT_SCHEMA,
            },
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
            "additionalProperties": False,
            "properties": {
                "topic_filename": {
                    "type": "string",
                    "maxLength": _MAX_FILENAME_LENGTH,
                    "description": "The filename of the knowledge topic (e.g., 'vr_setup.md').",
                },
                "correction": {
                    "type": "string",
                    "maxLength": _MAX_CONTENT_LENGTH,
                    "description": "Description of what needs to be corrected and the correct information.",
                },
                "scope": SCOPE_INPUT_SCHEMA,
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
            "additionalProperties": False,
            "properties": {
                "query": {
                    "type": "string",
                    "maxLength": _MAX_QUERY_LENGTH,
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
                    "maxLength": 64,
                    "description": "Only episodes created after this ISO date (e.g. '2025-01-01').",
                },
                "before": {
                    "type": "string",
                    "maxLength": 64,
                    "description": "Only episodes created before this ISO date.",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "description": "Maximum results to return.",
                    "default": 20,
                },
                "scope": SCOPE_INPUT_SCHEMA,
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
            "additionalProperties": False,
            "properties": {
                "claim_type": {
                    "type": "string",
                    "maxLength": 64,
                    "description": "Optional claim type filter (e.g. 'fact', 'solution').",
                },
                "as_of": {
                    "type": "string",
                    "maxLength": 64,
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
                "scope": SCOPE_INPUT_SCHEMA,
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
            "additionalProperties": False,
            "properties": {
                "query": {
                    "type": "string",
                    "maxLength": _MAX_QUERY_LENGTH,
                    "description": "Search text to match against claim canonical text and payload.",
                },
                "claim_type": {
                    "type": "string",
                    "maxLength": 64,
                    "description": "Optional claim type filter (e.g. 'fact', 'solution').",
                },
                "as_of": {
                    "type": "string",
                    "maxLength": 64,
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
                "scope": SCOPE_INPUT_SCHEMA,
            },
            "required": ["query"],
        },
    },
}

MEMORY_OUTCOME_RECORD_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "memory_outcome_record",
        "description": (
            "Record whether a strategy/action worked. "
            "Outcome observations are durable evidence for trust scoring and can be linked "
            "to claim/record/episode provenance, code anchors, and issue/PR identifiers."
        ),
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "action_summary": {
                    "type": "string",
                    "maxLength": _MAX_QUERY_LENGTH,
                    "description": "Concise description of the attempted strategy or action.",
                },
                "outcome_type": {
                    "type": "string",
                    "enum": ["success", "failure", "partial_success", "reverted", "superseded"],
                    "description": "Outcome classification for this observation.",
                },
                "source_claim_ids": {
                    "type": "array",
                    "items": {"type": "string", "maxLength": _MAX_FILENAME_LENGTH},
                    "description": "Claim IDs this outcome is linked to.",
                },
                "source_record_ids": {
                    "type": "array",
                    "items": {"type": "string", "maxLength": _MAX_FILENAME_LENGTH},
                    "description": "Knowledge record IDs this outcome is linked to.",
                },
                "source_episode_ids": {
                    "type": "array",
                    "items": {"type": "string", "maxLength": _MAX_FILENAME_LENGTH},
                    "description": "Episode IDs this outcome is linked to.",
                },
                "code_anchors": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "anchor_type": {"type": "string", "maxLength": 64},
                            "anchor_value": {"type": "string", "maxLength": _MAX_PATH_LENGTH},
                        },
                        "required": ["anchor_type", "anchor_value"],
                    },
                    "description": "Optional code anchors associated with the outcome.",
                },
                "issue_ids": {
                    "type": "array",
                    "items": {"type": "string", "maxLength": _MAX_FILENAME_LENGTH},
                    "description": "Optional issue identifiers associated with the outcome.",
                },
                "pr_ids": {
                    "type": "array",
                    "items": {"type": "string", "maxLength": _MAX_FILENAME_LENGTH},
                    "description": "Optional pull request identifiers associated with the outcome.",
                },
                "action_key": {
                    "type": "string",
                    "maxLength": _MAX_FILENAME_LENGTH,
                    "description": "Optional stable strategy key. Auto-derived when omitted.",
                },
                "summary": {
                    "type": "string",
                    "maxLength": _MAX_CONTENT_LENGTH,
                    "description": "Optional short human summary of the observed outcome.",
                },
                "details": {
                    "oneOf": [
                        {"type": "object"},
                        {"type": "string", "maxLength": _MAX_CONTENT_LENGTH},
                    ],
                    "description": "Optional structured details payload for this outcome.",
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.8,
                    "description": "Confidence in this observation.",
                },
                "provenance": {
                    "oneOf": [
                        {"type": "object"},
                        {"type": "string", "maxLength": _MAX_CONTENT_LENGTH},
                    ],
                    "description": "Optional provenance metadata (agent/tool/run identifiers, etc.).",
                },
                "observed_at": {
                    "type": "string",
                    "maxLength": 64,
                    "description": "Optional ISO datetime when the outcome was observed.",
                },
                "scope": SCOPE_INPUT_SCHEMA,
            },
            "required": ["action_summary", "outcome_type"],
        },
    },
}

MEMORY_OUTCOME_BROWSE_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "memory_outcome_browse",
        "description": (
            "Browse recorded outcome observations over time with optional source and temporal filters."
        ),
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "outcome_type": {
                    "type": "string",
                    "enum": ["success", "failure", "partial_success", "reverted", "superseded"],
                    "description": "Optional outcome type filter.",
                },
                "action_key": {
                    "type": "string",
                    "maxLength": _MAX_FILENAME_LENGTH,
                    "description": "Optional action/strategy key filter.",
                },
                "source_claim_id": {
                    "type": "string",
                    "maxLength": _MAX_FILENAME_LENGTH,
                    "description": "Optional claim source filter.",
                },
                "source_record_id": {
                    "type": "string",
                    "maxLength": _MAX_FILENAME_LENGTH,
                    "description": "Optional record source filter.",
                },
                "source_episode_id": {
                    "type": "string",
                    "maxLength": _MAX_FILENAME_LENGTH,
                    "description": "Optional episode source filter.",
                },
                "as_of": {
                    "type": "string",
                    "maxLength": 64,
                    "description": "Optional ISO datetime upper bound for observed outcomes.",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 200,
                    "default": 50,
                    "description": "Maximum outcomes to return.",
                },
                "scope": SCOPE_INPUT_SCHEMA,
            },
            "required": [],
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
            "additionalProperties": False,
            "properties": {
                "base_ref": {
                    "type": "string",
                    "maxLength": _MAX_FILENAME_LENGTH,
                    "description": "Optional git base ref for comparison (e.g. 'origin/main').",
                },
                "repo_path": {
                    "type": "string",
                    "maxLength": _MAX_PATH_LENGTH,
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
            "additionalProperties": False,
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
            "additionalProperties": False,
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
            "additionalProperties": False,
            "properties": {
                "episode_id": {
                    "type": "string",
                    "maxLength": _MAX_FILENAME_LENGTH,
                    "description": "Protect a specific episode by its UUID.",
                },
                "tag": {
                    "type": "string",
                    "maxLength": 100,
                    "description": "Protect all episodes with this tag.",
                },
                "scope": SCOPE_INPUT_SCHEMA,
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
            "additionalProperties": False,
            "properties": {
                "topic": {
                    "type": "string",
                    "maxLength": _MAX_TOPIC_LENGTH,
                    "description": (
                        "Natural language topic to query "
                        "(e.g., 'frontend framework preference')."
                    ),
                },
                "scope": SCOPE_INPUT_SCHEMA,
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
            "additionalProperties": False,
            "properties": {
                "topic": {
                    "type": "string",
                    "maxLength": _MAX_FILENAME_LENGTH,
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
            "additionalProperties": False,
            "properties": {
                "scope": SCOPE_INPUT_SCHEMA,
            },
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
            "additionalProperties": False,
            "properties": {
                "filename": {
                    "type": "string",
                    "maxLength": _MAX_FILENAME_LENGTH,
                    "description": "The filename of the knowledge topic (e.g., 'python_setup.md').",
                },
                "scope": SCOPE_INPUT_SCHEMA,
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
            "additionalProperties": False,
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
            "additionalProperties": False,
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
    MEMORY_OUTCOME_RECORD_SCHEMA,
    MEMORY_OUTCOME_BROWSE_SCHEMA,
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
    """Execute a tool call against a MemoryClient and return the result as a dict."""
    return _dispatch_tool_call(client, name, arguments)
