"""Consolidation Memory MCP Server.

Thin wrapper over MemoryClient — exposes memory tools to Claude Desktop
via stdio transport.  All business logic lives in client.py.
"""

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
    result = _client.store(content, content_type, tags, surprise)
    return json.dumps(dataclasses.asdict(result), default=str)


@mcp.tool()
async def memory_recall(
    query: str,
    n_results: int = 10,
    include_knowledge: bool = True,
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
    """
    result = _client.recall(query, n_results, include_knowledge)
    return json.dumps(dataclasses.asdict(result), default=str)


@mcp.tool()
async def memory_status() -> str:
    """Show memory system statistics.

    Call this to check the health and state of the memory system.
    """
    result = _client.status()
    return json.dumps(dataclasses.asdict(result), default=str)


@mcp.tool()
async def memory_forget(episode_id: str) -> str:
    """Mark an episode for removal from the memory system.

    Call this to forget specific memories that are incorrect,
    outdated, or that the user wants removed.

    Args:
        episode_id: The UUID of the episode to forget.
    """
    result = _client.forget(episode_id)
    return json.dumps(dataclasses.asdict(result), default=str)


@mcp.tool()
async def memory_export() -> str:
    """Export all episodes and knowledge to a JSON snapshot.

    Creates a timestamped JSON file in the backups directory containing
    all episodes (non-deleted) and knowledge topics with their content.
    Returns the file path.
    """
    result = _client.export()
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
    result = _client.correct(topic_filename, correction)
    return json.dumps(dataclasses.asdict(result), default=str)


# ── Entry point ──────────────────────────────────────────────────────────────

def run_server():
    """Run the MCP server on stdio transport."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    run_server()
