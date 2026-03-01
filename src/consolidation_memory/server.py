"""Consolidation Memory MCP Server.

Thin wrapper over MemoryClient — exposes memory tools to Claude Desktop
via stdio transport.  All business logic lives in client.py.
"""

import asyncio
import dataclasses
import json
import logging
import sys
from contextlib import asynccontextmanager

from mcp.server.fastmcp import FastMCP

# Configure logging to stderr (stdout is the MCP JSON-RPC channel)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("consolidation_memory")

# ── Global client initialized in lifespan ─────────────────────────────────
_client = None


@asynccontextmanager
async def lifespan(server: FastMCP):
    """Initialize MemoryClient on startup, shut down on exit."""
    global _client

    from consolidation_memory import __version__
    from consolidation_memory.client import MemoryClient

    logger.info("Starting consolidation_memory MCP server v%s...", __version__)
    from consolidation_memory.config import get_active_project
    logger.info("Active project: %s", get_active_project())

    _client = MemoryClient()

    yield

    _client.close()
    _client = None
    logger.info("Shutting down consolidation_memory MCP server.")


mcp = FastMCP(
    "consolidation_memory",
    lifespan=lifespan,
)


# ── Tools ────────────────────────────────────────────────────────────────────

@mcp.tool()
async def memory_store(
    content: str,
    content_type: str = "exchange",
    tags: list[str] | None = None,
    surprise: float = 0.5,
) -> str:
    """Store a memory episode in the episodic buffer.

    IMPORTANT: Always store memories when you learn something new about the user,
    solve a problem, discover a preference, or encounter something surprising.
    Write content as a self-contained note that future-you can understand without context.
    Include both the problem AND solution for solution-type memories.
    Do NOT store trivial exchanges like greetings.

    Args:
        content: The text content to store. Include relevant context.
        content_type: One of 'exchange' (conversation), 'fact' (learned info),
                      'solution' (problem+fix), 'preference' (user preference).
        tags: Optional topic tags for organization (e.g., ['vr', 'steamvr']).
        surprise: How novel this is, 0.0 (routine) to 1.0 (very surprising).
    """
    if _client is None:
        return json.dumps({"error": "Memory system not initialized"})
    if len(content) > 50_000:
        return json.dumps({"error": "Content exceeds maximum length of 50KB"})
    result = await asyncio.to_thread(_client.store, content, content_type, tags, surprise)
    return json.dumps(dataclasses.asdict(result), default=str)


@mcp.tool()
async def memory_recall(
    query: str,
    n_results: int = 10,
    include_knowledge: bool = True,
    content_types: list[str] | None = None,
    tags: list[str] | None = None,
    after: str | None = None,
    before: str | None = None,
    include_expired: bool = False,
) -> str:
    """Retrieve relevant memories by semantic similarity.

    CRITICAL: You MUST call this at the START of EVERY new conversation, using a
    query that matches the user's opening message topic. Also call when the user
    references past interactions or when context about their setup/preferences
    would improve your response. This is your persistent memory — use it.

    Args:
        query: Natural language description of what to recall.
        n_results: Maximum number of episode results (1-50). Default 10.
        include_knowledge: Whether to include consolidated knowledge. Default True.
        content_types: Filter to specific types (e.g. ['solution', 'fact']).
        tags: Filter to episodes with at least one matching tag.
        after: Only episodes created after this ISO date (e.g. '2025-01-01').
        before: Only episodes created before this ISO date.
        include_expired: Include temporally expired knowledge records. Default False.
    """
    if _client is None:
        return json.dumps({"error": "Memory system not initialized"})
    n_results = min(n_results, 50)
    result = await asyncio.to_thread(
        lambda: _client.recall(
            query, n_results, include_knowledge,
            content_types=content_types, tags=tags, after=after, before=before,
            include_expired=include_expired,
        )
    )
    return json.dumps(dataclasses.asdict(result), default=str)


@mcp.tool()
async def memory_store_batch(
    episodes: list[dict],
) -> str:
    """Store multiple memory episodes in a single operation.

    More efficient than calling memory_store repeatedly. Single embedding call
    and batch FAISS insertion.

    Args:
        episodes: List of episode objects, each with:
            - content (str, required): The text content to store.
            - content_type (str): One of 'exchange', 'fact', 'solution', 'preference'.
            - tags (list[str]): Optional topic tags.
            - surprise (float): Novelty score 0.0-1.0.
    """
    if _client is None:
        return json.dumps({"error": "Memory system not initialized"})
    result = await asyncio.to_thread(_client.store_batch, episodes)
    return json.dumps(dataclasses.asdict(result), default=str)


@mcp.tool()
async def memory_search(
    query: str | None = None,
    content_types: list[str] | None = None,
    tags: list[str] | None = None,
    after: str | None = None,
    before: str | None = None,
    limit: int = 20,
) -> str:
    """Keyword/metadata search over episodes. Works without embedding backend.

    Unlike memory_recall (semantic similarity), this does plain text matching
    in SQLite. Use when the embedding backend is down, or for exact substring
    searches. At least one filter parameter should be provided.

    Args:
        query: Text substring to search for in episode content (case-insensitive).
        content_types: Filter to specific types (e.g. ['solution', 'fact']).
        tags: Filter to episodes with at least one matching tag.
        after: Only episodes created after this ISO date (e.g. '2025-01-01').
        before: Only episodes created before this ISO date.
        limit: Maximum results (default 20, max 50).
    """
    if _client is None:
        return json.dumps({"error": "Memory system not initialized"})
    result = await asyncio.to_thread(
        lambda: _client.search(
            query=query,
            content_types=content_types,
            tags=tags,
            after=after,
            before=before,
            limit=min(limit, 50),
        )
    )
    return json.dumps(dataclasses.asdict(result), default=str)


@mcp.tool()
async def memory_status() -> str:
    """Show memory system statistics.

    Call this to check the health and state of the memory system.
    """
    if _client is None:
        return json.dumps({"error": "Memory system not initialized"})
    result = await asyncio.to_thread(_client.status)
    return json.dumps(dataclasses.asdict(result), default=str)


@mcp.tool()
async def memory_forget(episode_id: str) -> str:
    """Mark an episode for removal from the memory system.

    Call this to forget specific memories that are incorrect,
    outdated, or that the user wants removed.

    Args:
        episode_id: The UUID of the episode to forget.
    """
    if _client is None:
        return json.dumps({"error": "Memory system not initialized"})
    result = await asyncio.to_thread(_client.forget, episode_id)
    return json.dumps(dataclasses.asdict(result), default=str)


@mcp.tool()
async def memory_export() -> str:
    """Export all episodes and knowledge to a JSON snapshot.

    Creates a timestamped JSON file in the backups directory containing
    all episodes (non-deleted) and knowledge topics with their content.
    Returns the file path.
    """
    if _client is None:
        return json.dumps({"error": "Memory system not initialized"})
    result = await asyncio.to_thread(_client.export)
    return json.dumps(dataclasses.asdict(result), default=str)


@mcp.tool()
async def memory_correct(topic_filename: str, correction: str) -> str:
    """Correct a knowledge document with new information.

    Use this when you discover that a knowledge document contains outdated
    or incorrect information and needs to be updated.

    Args:
        topic_filename: The filename of the knowledge topic (e.g., 'vr_setup.md').
        correction: Description of what needs to be corrected and the correct information.
    """
    if _client is None:
        return json.dumps({"error": "Memory system not initialized"})
    result = await asyncio.to_thread(_client.correct, topic_filename, correction)
    return json.dumps(dataclasses.asdict(result), default=str)


@mcp.tool()
async def memory_compact() -> str:
    """Compact the FAISS index by removing tombstoned vectors.

    Call when memory_status shows high tombstone count or ratio.
    Tombstones accumulate from forget and prune operations.
    Compaction rebuilds the index without dead vectors.
    """
    if _client is None:
        return json.dumps({"error": "Memory system not initialized"})
    result = await asyncio.to_thread(_client.compact)
    return json.dumps(dataclasses.asdict(result), default=str)


@mcp.tool()
async def memory_consolidate() -> str:
    """Manually trigger a consolidation run.

    Clusters unconsolidated episodes by semantic similarity, synthesizes
    knowledge documents via LLM, prunes old episodes, and compacts FAISS.
    Returns a run report. Will refuse if a consolidation is already in progress.

    NOTE: This can take several minutes depending on episode count and LLM speed.
    """
    if _client is None:
        return json.dumps({"error": "Memory system not initialized"})
    result = await asyncio.to_thread(_client.consolidate)
    if result.get("status") == "already_running":
        return json.dumps({"status": "already_running", "message": "A consolidation run is already in progress"})
    return json.dumps({"status": "completed", "report": result}, default=str)


@mcp.tool()
async def memory_decay_report() -> str:
    """Show what would be forgotten if pruning ran right now.

    Reports prunable episodes (consolidated and older than threshold),
    low-confidence records, and protected episode counts.
    Does NOT actually delete anything — just reports.
    """
    if _client is None:
        return json.dumps({"error": "Memory system not initialized"})
    result = await asyncio.to_thread(_client.decay_report)
    return json.dumps(dataclasses.asdict(result), default=str)


@mcp.tool()
async def memory_protect(
    episode_id: str | None = None,
    tag: str | None = None,
) -> str:
    """Mark episodes as immune to pruning.

    Protect specific episodes or all episodes with a given tag from
    being pruned during consolidation. Use this for important memories
    that should never be forgotten.

    Args:
        episode_id: Protect a specific episode by its UUID.
        tag: Protect all episodes with this tag.
    """
    if _client is None:
        return json.dumps({"error": "Memory system not initialized"})
    result = await asyncio.to_thread(_client.protect, episode_id, tag)
    return json.dumps(dataclasses.asdict(result), default=str)


@mcp.tool()
async def memory_timeline(topic: str) -> str:
    """Show how understanding of a topic has changed over time.

    Returns all knowledge records matching the topic sorted chronologically,
    including expired/superseded records. Shows what was believed, when it
    changed, and what replaced it. Useful for questions like "how has my
    understanding of X evolved?"

    Args:
        topic: Natural language topic to query (e.g., 'frontend framework preference').
    """
    if _client is None:
        return json.dumps({"error": "Memory system not initialized"})
    result = await asyncio.to_thread(_client.timeline, topic)
    return json.dumps(dataclasses.asdict(result), default=str)


@mcp.tool()
async def memory_contradictions(topic: str | None = None) -> str:
    """List detected contradictions from the audit log.

    Shows cases where knowledge records contradicted each other during
    consolidation, including both the old and new content and how it
    was resolved. Use this to review belief changes over time.

    Args:
        topic: Optional topic filename or title to filter results.
    """
    if _client is None:
        return json.dumps({"error": "Memory system not initialized"})
    result = await asyncio.to_thread(_client.contradictions, topic)
    return json.dumps(dataclasses.asdict(result), default=str)


@mcp.tool()
async def memory_browse() -> str:
    """Browse all knowledge topics with summaries and metadata.

    Returns a list of all knowledge topics including titles, summaries,
    record counts by type, confidence scores, and file paths. Use this
    to see what the memory system has learned and consolidated.
    """
    if _client is None:
        return json.dumps({"error": "Memory system not initialized"})
    result = await asyncio.to_thread(_client.browse)
    return json.dumps(dataclasses.asdict(result), default=str)


@mcp.tool()
async def memory_read_topic(filename: str) -> str:
    """Read the full markdown content of a knowledge topic.

    Use memory_browse first to see available topics, then read specific
    ones to see the full details including all extracted facts, solutions,
    preferences, and procedures.

    Args:
        filename: The filename of the knowledge topic (e.g., 'python_setup.md').
    """
    if _client is None:
        return json.dumps({"error": "Memory system not initialized"})
    result = await asyncio.to_thread(_client.read_topic, filename)
    return json.dumps(dataclasses.asdict(result), default=str)


# ── Entry point ──────────────────────────────────────────────────────────────

def run_server():
    """Run the MCP server on stdio transport."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    run_server()
