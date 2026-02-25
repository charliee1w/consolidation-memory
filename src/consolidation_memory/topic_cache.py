"""Shared topic embedding cache for recall and consolidation.

Caches embedded topic title+summary vectors. Invalidated when topics change
(after consolidation or correction). Thread-safe.

Race condition prevention: invalidate() bumps a version counter. If the
version changes between the start and end of a cache-miss fetch, the
stale result is discarded rather than written to the cache.
"""

import logging
import threading

import numpy as np

from consolidation_memory.database import get_all_knowledge_topics
from consolidation_memory.backends import encode_documents

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_version: int = 0  # Bumped on every invalidate(); guards against stale writes
_cache: dict = {
    "version": -1,  # Version at time of last successful populate
    "texts": [],
    "vecs": None,
    "topics": [],
}


def invalidate() -> None:
    """Force re-embedding on next get_topic_vecs() call."""
    global _version
    with _lock:
        _version += 1


def get_topic_vecs() -> tuple[list[dict], np.ndarray | None]:
    """Return (topics, embedding_matrix) with caching.

    Cache is valid as long as its version matches the current version.
    Returns ([], None) if no topics exist.
    """
    with _lock:
        if _cache["version"] == _version and _cache["vecs"] is not None:
            return _cache["topics"], _cache["vecs"]
        # Snapshot the version before we release the lock to fetch
        fetch_version = _version

    # Fetch and embed outside the lock (can be slow)
    topics = get_all_knowledge_topics()
    if not topics:
        return [], None

    summary_texts = [f"{t['title']}. {t['summary']}" for t in topics]
    try:
        vecs = encode_documents(summary_texts)
    except Exception as e:
        logger.warning("Failed to embed topic summaries: %s", e, exc_info=True)
        return topics, None

    with _lock:
        # Only write if no invalidation happened while we were fetching
        if _version == fetch_version:
            _cache["version"] = fetch_version
            _cache["texts"] = summary_texts
            _cache["vecs"] = vecs
            _cache["topics"] = topics
        else:
            logger.debug(
                "Topic cache populate discarded: version changed %d -> %d during fetch",
                fetch_version, _version,
            )

    return topics, vecs
