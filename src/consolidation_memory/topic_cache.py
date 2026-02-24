"""Shared topic embedding cache for recall and consolidation.

Caches embedded topic title+summary vectors. Invalidated when topics change
(after consolidation or correction). Thread-safe.
"""

import logging
import threading

import numpy as np

from consolidation_memory.database import get_all_knowledge_topics
from consolidation_memory.backends import encode_documents

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_cache: dict = {
    "topic_count": -1,
    "texts": [],
    "vecs": None,
    "topics": [],
}


def invalidate() -> None:
    """Force re-embedding on next get_topic_vecs() call."""
    with _lock:
        _cache["topic_count"] = -1


def get_topic_vecs() -> tuple[list[dict], np.ndarray | None]:
    """Return (topics, embedding_matrix) with caching.

    Cache is valid as long as topic count hasn't changed and cache is populated.
    Returns ([], None) if no topics exist.
    """
    with _lock:
        if _cache["topic_count"] >= 0 and _cache["vecs"] is not None:
            return _cache["topics"], _cache["vecs"]

    # Fetch and embed outside the lock (can be slow)
    topics = get_all_knowledge_topics()
    if not topics:
        return [], None

    summary_texts = [f"{t['title']}. {t['summary']}" for t in topics]
    try:
        vecs = encode_documents(summary_texts)
    except Exception as e:
        logger.warning("Failed to embed topic summaries: %s", e)
        return topics, None

    with _lock:
        _cache["topic_count"] = len(topics)
        _cache["texts"] = summary_texts
        _cache["vecs"] = vecs
        _cache["topics"] = topics

    return topics, vecs
