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
    KNOWLEDGE_KEYWORD_WEIGHT,
    KNOWLEDGE_MAX_RESULTS,
    KNOWLEDGE_RELEVANCE_THRESHOLD,
    KNOWLEDGE_SEMANTIC_WEIGHT,
    RECALL_MAX_N,
    RECENCY_HALF_LIFE_DAYS,
)
from consolidation_memory.database import (
    get_episodes_batch,
    increment_access,
    increment_topic_access,
)
from consolidation_memory import backends
from consolidation_memory.vector_store import VectorStore
from consolidation_memory import topic_cache

logger = logging.getLogger(__name__)


def _parse_tags(tags_value: str | list) -> list:
    """Parse tags from DB storage format (JSON string or already-parsed list)."""
    if isinstance(tags_value, str):
        return json.loads(tags_value)
    return tags_value


def invalidate_topic_cache() -> None:
    """Call after consolidation to force re-embedding on next recall."""
    topic_cache.invalidate()


# ── Scoring ───────────────────────────────────────────────────────────────────

def _recency_decay(created_at_iso: str, half_life_days: float = RECENCY_HALF_LIFE_DAYS) -> float:
    try:
        created = datetime.fromisoformat(created_at_iso)
        # Handle naive datetimes (no timezone info) by assuming UTC
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
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
    *,
    content_types: list[str] | None = None,
    tags: list[str] | None = None,
    after: str | None = None,
    before: str | None = None,
) -> dict:
    """Main retrieval function. Returns ranked episodes + knowledge excerpts.

    Optional filters (all applied post-vector-search):
        content_types: Only return episodes matching these types.
        tags: Only return episodes that have at least one of these tags.
        after: Only return episodes created after this ISO date string.
        before: Only return episodes created before this ISO date string.
    """
    n_results = min(n_results, RECALL_MAX_N)

    # Fetch more candidates when filtering, since many will be discarded
    fetch_k = n_results * 5 if (content_types or tags or after or before) else n_results * 3

    query_vec = backends.encode_query(query)
    candidates = vector_store.search(query_vec, k=fetch_k)

    logger.debug(
        "recall: query_len=%d, n_results=%d, vector_candidates=%d, filters=%s",
        len(query), n_results, len(candidates),
        {"content_types": content_types, "tags": tags, "after": after, "before": before},
    )

    # Batch-fetch all candidate episodes in one query instead of N individual SELECTs
    candidate_ids = [eid for eid, _ in candidates]
    episodes_by_id = get_episodes_batch(candidate_ids)

    # Precompute filter sets
    _ct_set = set(content_types) if content_types else None
    _tag_set = set(tags) if tags else None

    scored = []
    for episode_id, similarity in candidates:
        ep = episodes_by_id.get(episode_id)
        if ep is None:
            continue

        # Apply filters
        if _ct_set and ep["content_type"] not in _ct_set:
            continue
        if _tag_set:
            ep_tags = _parse_tags(ep["tags"])
            if not _tag_set.intersection(ep_tags):
                continue
        if after and ep["created_at"] < after:
            continue
        if before and ep["created_at"] > before:
            continue

        score = _priority_score(similarity, ep)
        scored.append((ep, score, similarity))

    logger.debug(
        "recall: db_matches=%d, scored_after_priority=%d",
        len(episodes_by_id), len(scored),
    )

    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:n_results]

    episodes = []
    for ep, score, sim in top:
        tags = _parse_tags(ep["tags"])
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
    warnings = []
    if include_knowledge:
        knowledge, warnings = _search_knowledge(query, query_vec)

    return {
        "episodes": episodes,
        "knowledge": knowledge,
        "warnings": warnings,
    }


def _search_knowledge(
    query: str, query_vec: np.ndarray | None = None,
) -> tuple[list[dict], list[str]]:
    warnings: list[str] = []
    topics, summary_vecs = topic_cache.get_topic_vecs()
    if not topics:
        return [], warnings

    try:
        if query_vec is None:
            query_vec = backends.encode_query(query)
        if summary_vecs is not None:
            sims = (query_vec @ summary_vecs.T).flatten()
        else:
            sims = None
    except Exception as e:
        logger.warning(
            "Semantic knowledge search failed, falling back to keyword: %s", e,
            exc_info=True,
        )
        sims = None
        warnings.append("Knowledge search fell back to keyword-only (embedding failed)")

    query_lower = query.lower()
    query_words = set(query_lower.split())

    scored_topics = []
    for i, topic in enumerate(topics):
        sem_score = float(sims[i]) if sims is not None else 0.0

        title_lower = topic["title"].lower()
        summary_lower = topic["summary"].lower()
        kw_hits = sum(1 for w in query_words if w in title_lower or w in summary_lower)
        kw_score = kw_hits / len(query_words) if query_words else 0

        relevance = sem_score * KNOWLEDGE_SEMANTIC_WEIGHT + kw_score * KNOWLEDGE_KEYWORD_WEIGHT

        if relevance < KNOWLEDGE_RELEVANCE_THRESHOLD:
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

    logger.debug(
        "knowledge_search: %d topics checked, %d passed relevance threshold (>=%s)",
        len(topics), len(scored_topics), KNOWLEDGE_RELEVANCE_THRESHOLD,
    )

    if scored_topics:
        increment_topic_access([t["filename"] for t in scored_topics])

    scored_topics.sort(key=lambda x: x["relevance"], reverse=True)
    return scored_topics[:KNOWLEDGE_MAX_RESULTS], warnings
