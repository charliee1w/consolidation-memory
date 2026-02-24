"""Priority-ranked retrieval combining vector similarity with metadata scoring.

Returns both episodic memories and relevant knowledge base entries.
Topic embeddings are cached via topic_cache (shared with consolidation).
"""

import json
import logging
import math
from datetime import datetime, timezone

import numpy as np

from consolidation_memory.config import (
    CONSOLIDATION_PRIORITY_WEIGHTS,
    KNOWLEDGE_DIR,
    RECALL_MAX_N,
)
from consolidation_memory.database import (
    get_episodes_batch,
    increment_access,
    increment_topic_access,
)
from consolidation_memory.backends import encode_query
from consolidation_memory.vector_store import VectorStore
from consolidation_memory import topic_cache

logger = logging.getLogger(__name__)


def invalidate_topic_cache() -> None:
    """Call after consolidation to force re-embedding on next recall."""
    topic_cache.invalidate()


# ── Scoring ───────────────────────────────────────────────────────────────────

def _recency_decay(created_at_iso: str, half_life_days: float = 30.0) -> float:
    try:
        created = datetime.fromisoformat(created_at_iso)
        age_days = (datetime.now(timezone.utc) - created).total_seconds() / 86400.0
        return math.exp(-age_days / half_life_days)
    except Exception:
        return 0.5


def _priority_score(similarity: float, episode: dict) -> float:
    w = CONSOLIDATION_PRIORITY_WEIGHTS
    surprise = episode.get("surprise_score", 0.5)
    recency = _recency_decay(episode.get("created_at", ""))
    access = episode.get("access_count", 0)

    access_factor = 1.0 + math.log1p(access) * w["access_frequency"]
    metadata_boost = (
        (surprise ** w["surprise"])
        * (recency ** w["recency"])
        * access_factor
    )
    return similarity * metadata_boost


# ── Main retrieval ────────────────────────────────────────────────────────────

def recall(
    query: str,
    n_results: int = 10,
    include_knowledge: bool = True,
    vector_store: VectorStore | None = None,
) -> dict:
    """Main retrieval function. Returns ranked episodes + knowledge excerpts."""
    n_results = min(n_results, RECALL_MAX_N)

    query_vec = encode_query(query)
    candidates = vector_store.search(query_vec, k=n_results * 3)

    # Batch-fetch all candidate episodes in one query instead of N individual SELECTs
    candidate_ids = [eid for eid, _ in candidates]
    episodes_by_id = get_episodes_batch(candidate_ids)

    scored = []
    for episode_id, similarity in candidates:
        ep = episodes_by_id.get(episode_id)
        if ep is None:
            continue
        score = _priority_score(similarity, ep)
        scored.append((ep, score, similarity))

    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:n_results]

    episodes = []
    for ep, score, sim in top:
        tags = json.loads(ep["tags"]) if isinstance(ep["tags"], str) else ep["tags"]
        episodes.append({
            "id": ep["id"],
            "content": ep["content"],
            "content_type": ep["content_type"],
            "tags": tags,
            "created_at": ep["created_at"],
            "score": round(score, 4),
            "similarity": round(sim, 4),
            "access_count": ep["access_count"],
        })

    accessed_ids = [ep["id"] for ep, _, _ in top]
    increment_access(accessed_ids)

    knowledge = []
    if include_knowledge:
        knowledge = _search_knowledge(query, query_vec)

    return {
        "episodes": episodes,
        "knowledge": knowledge,
    }


def _search_knowledge(query: str, query_vec: np.ndarray | None = None) -> list[dict]:
    topics, summary_vecs = topic_cache.get_topic_vecs()
    if not topics:
        return []

    try:
        if query_vec is None:
            query_vec = encode_query(query)
        if summary_vecs is not None:
            sims = (query_vec @ summary_vecs.T).flatten()
        else:
            sims = None
    except Exception as e:
        logger.warning("Semantic knowledge search failed, falling back to keyword: %s", e)
        sims = None

    query_lower = query.lower()
    query_words = set(query_lower.split())

    scored_topics = []
    for i, topic in enumerate(topics):
        sem_score = float(sims[i]) if sims is not None else 0.0

        title_lower = topic["title"].lower()
        summary_lower = topic["summary"].lower()
        kw_hits = sum(1 for w in query_words if w in title_lower or w in summary_lower)
        kw_score = kw_hits / len(query_words) if query_words else 0

        relevance = sem_score * 0.8 + kw_score * 0.2

        if relevance < 0.15:
            continue

        filepath = KNOWLEDGE_DIR / topic["filename"]
        content = ""
        if filepath.exists():
            content = filepath.read_text(encoding="utf-8")

        scored_topics.append({
            "topic": topic["filename"].replace(".md", ""),
            "filename": topic["filename"],
            "title": topic["title"],
            "summary": topic["summary"],
            "content": content,
            "confidence": topic["confidence"],
            "relevance": round(relevance, 3),
        })

    if scored_topics:
        increment_topic_access([t["filename"] for t in scored_topics])

    scored_topics.sort(key=lambda x: x["relevance"], reverse=True)
    return scored_topics[:5]
