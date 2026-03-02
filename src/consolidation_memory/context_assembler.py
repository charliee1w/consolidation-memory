"""Priority-ranked retrieval combining vector similarity with metadata scoring.

Returns both episodic memories and relevant knowledge base entries.
Topic embeddings are cached via topic_cache (shared with consolidation).
"""

import json
import logging
import math
from datetime import datetime, timezone

import numpy as np

from consolidation_memory.config import get_config
from consolidation_memory.database import (
    fts_available,
    fts_search,
    get_episodes_batch,
    get_tag_pairs_in_set,
    increment_access,
    increment_record_access,
    increment_topic_access,
)
from consolidation_memory import backends
from consolidation_memory.vector_store import VectorStore
from consolidation_memory import topic_cache
from consolidation_memory import record_cache

_TASK_INDICATORS: frozenset[str] = frozenset({
    "how", "workflow", "steps", "process", "deploy", "build",
    "test", "commit", "release", "setup", "configure", "run",
})

logger = logging.getLogger(__name__)


def _parse_tags(tags_value: str | list) -> list:
    """Parse tags from DB storage format (JSON string or already-parsed list)."""
    if isinstance(tags_value, str):
        try:
            parsed: list = json.loads(tags_value)
            return parsed
        except (json.JSONDecodeError, ValueError):
            return []
    return tags_value if tags_value is not None else []


def invalidate_topic_cache() -> None:
    """Call after consolidation to force re-embedding on next recall."""
    topic_cache.invalidate()


def invalidate_record_cache() -> None:
    """Call after consolidation to force re-embedding on next recall."""
    record_cache.invalidate()


# ── Scoring ───────────────────────────────────────────────────────────────────

def _recency_decay(created_at_iso: str, half_life_days: float | None = None) -> float:
    if half_life_days is None:
        half_life_days = get_config().RECENCY_HALF_LIFE_DAYS
    try:
        created = datetime.fromisoformat(created_at_iso)
        # Handle naive datetimes (no timezone info) by assuming UTC
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        age_days = (datetime.now(timezone.utc) - created).total_seconds() / 86400.0
        # Clamp to non-negative to prevent >1.0 scores from future-dated episodes
        age_days = max(0.0, age_days)
        return math.exp(-age_days * math.log(2) / half_life_days)
    except Exception:
        return 0.5


def _priority_score(similarity: float, episode: dict) -> float:
    w = get_config().CONSOLIDATION_PRIORITY_WEIGHTS
    surprise: float = episode.get("surprise_score", 0.5)
    recency = _recency_decay(episode.get("created_at", ""))
    access: int = episode.get("access_count", 0)

    access_factor: float = 1.0 + math.log1p(access) * w["access_frequency"]
    metadata_boost: float = (
        (surprise ** w["surprise"])
        * (recency ** w["recency"])
        * access_factor
    )
    return similarity * metadata_boost


def _apply_cooccurrence_boost(
    scored: list[tuple[dict, float, float, float]],
) -> list[tuple[dict, float, float, float]]:
    """Boost scores for episodes whose tags co-occur with those of other candidates.

    Queries the tag_cooccurrence table for pairs where both tags appear among the
    scored candidates. Episodes whose tags participate in these co-occurrence
    connections get a 10% score boost, clustering results around intent motifs
    (e.g., "diet" + "exercise" + "weight" form a fitness cluster).
    """
    # Collect all tags across candidates
    all_tags: set[str] = set()
    episode_tags: list[list[str]] = []
    for ep, _, _, _ in scored:
        ep_tags = _parse_tags(ep.get("tags", "[]"))
        episode_tags.append(ep_tags)
        all_tags.update(ep_tags)

    if len(all_tags) < 2:
        return scored

    # Find co-occurrence pairs where both tags are in the candidate set
    try:
        pairs = get_tag_pairs_in_set(list(all_tags), min_count=2)
    except Exception:
        return scored

    if not pairs:
        return scored

    # Build set of tags involved in co-occurrence connections
    connected_tags: set[str] = set()
    for tag_a, tag_b, _count in pairs:
        connected_tags.add(tag_a)
        connected_tags.add(tag_b)

    # Boost episodes that have at least one connected tag
    boosted = []
    for i, (ep, score, sim, bm25) in enumerate(scored):
        ep_tag_set = set(episode_tags[i])
        if ep_tag_set & connected_tags:
            score *= 1.10  # 10% boost
        boosted.append((ep, score, sim, bm25))

    return boosted


# ── Deduplication ─────────────────────────────────────────────────────────────

def _deduplicate_episodes(
    episodes: list[dict],
    records: list[dict],
) -> list[dict]:
    """Remove episodes that are already represented by a returned knowledge record.

    When a knowledge record's source_episodes overlap with returned episode IDs,
    the episode is redundant — the record has higher signal density (structured,
    consolidated). Prefer the record and drop the overlapping episode.
    """
    if not episodes or not records:
        return episodes

    # Collect all episode IDs that are covered by returned records
    covered_ids: set[str] = set()
    for rec in records:
        src_eps = rec.get("source_episodes", [])
        if src_eps:
            covered_ids.update(src_eps)

    if not covered_ids:
        return episodes

    original_count = len(episodes)
    filtered = [ep for ep in episodes if ep["id"] not in covered_ids]
    removed = original_count - len(filtered)

    if removed:
        logger.debug(
            "recall dedup: removed %d episodes covered by knowledge records "
            "(covered_ids=%d)",
            removed, len(covered_ids),
        )

    return filtered


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
    include_expired: bool = False,
) -> dict:
    """Main retrieval function. Returns ranked episodes + knowledge excerpts.

    Optional filters (all applied post-vector-search):
        content_types: Only return episodes matching these types.
        tags: Only return episodes that have at least one of these tags.
        after: Only return episodes created after this ISO date string.
        before: Only return episodes created before this ISO date string.
        include_expired: If True, include temporally expired knowledge records.
    """
    cfg = get_config()
    n_results = min(n_results, cfg.RECALL_MAX_N)

    # Fetch more candidates when filtering, since many will be discarded
    fetch_k = n_results * 5 if (content_types or tags or after or before) else n_results * 3

    query_vec = backends.encode_query(query)
    if vector_store is None:
        raise RuntimeError("vector_store is required for recall")
    candidates = vector_store.search(query_vec, k=fetch_k)

    # Build cosine similarity map from FAISS results
    cosine_map: dict[str, float] = {eid: sim for eid, sim in candidates}

    # FTS5 keyword search (hybrid)
    bm25_map: dict[str, float] = {}
    _hybrid = cfg.HYBRID_SEARCH_ENABLED and fts_available()
    if _hybrid:
        fts_results = fts_search(query, limit=cfg.HYBRID_FTS_CANDIDATES)
        bm25_map = {eid: score for eid, score in fts_results}

    # Merge candidate IDs from both sources
    all_candidate_ids = list(dict.fromkeys(
        [eid for eid, _ in candidates] + list(bm25_map.keys())
    ))

    logger.debug(
        "recall: query_len=%d, n_results=%d, vector_candidates=%d, "
        "fts_candidates=%d, merged=%d, filters=%s",
        len(query), n_results, len(candidates), len(bm25_map),
        len(all_candidate_ids),
        {"content_types": content_types, "tags": tags, "after": after, "before": before},
    )

    # Batch-fetch all candidate episodes in one query instead of N individual SELECTs
    episodes_by_id = get_episodes_batch(all_candidate_ids)

    # Precompute filter sets
    _ct_set = set(content_types) if content_types else None
    _tag_set = set(tags) if tags else None

    sem_w = cfg.HYBRID_SEMANTIC_WEIGHT if _hybrid else 1.0
    kw_w = cfg.HYBRID_KEYWORD_WEIGHT if _hybrid else 0.0

    scored = []
    for episode_id in all_candidate_ids:
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

        cosine_sim = cosine_map.get(episode_id, 0.0)
        bm25_norm = bm25_map.get(episode_id, 0.0)
        hybrid_sim = sem_w * cosine_sim + kw_w * bm25_norm

        score = _priority_score(hybrid_sim, ep)
        scored.append((ep, score, cosine_sim, bm25_norm))

    logger.debug(
        "recall: db_matches=%d, scored_after_priority=%d",
        len(episodes_by_id), len(scored),
    )

    # Tag co-occurrence boost: episodes whose tags co-occur with tags
    # from other high-scoring candidates get a 10% boost.
    if len(scored) >= 2:
        scored = _apply_cooccurrence_boost(scored)

    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:n_results]

    episodes = []
    for ep, score, sim, bm25 in top:
        ep_parsed_tags = _parse_tags(ep["tags"])
        entry: dict[str, object] = {
            "id": ep["id"],
            "content": ep["content"],
            "content_type": ep["content_type"],
            "tags": ep_parsed_tags,
            "created_at": ep["created_at"],
            "score": round(score, 4),
            "similarity": round(sim, 4),
            "access_count": ep["access_count"],
        }
        if _hybrid:
            entry["bm25_score"] = round(bm25, 4)
        episodes.append(entry)

    accessed_ids = [ep["id"] for ep, _, _, _ in top]
    increment_access(accessed_ids)

    knowledge: list[dict] = []
    records: list[dict] = []
    warnings = []
    if include_knowledge:
        knowledge, kw_warnings = _search_knowledge(query, query_vec)
        warnings.extend(kw_warnings)
        records, rec_warnings = _search_records(query, query_vec, include_expired=include_expired)
        warnings.extend(rec_warnings)

    # Deduplicate: remove episodes already represented by knowledge records
    if cfg.RECALL_DEDUP_ENABLED and records:
        episodes = _deduplicate_episodes(episodes, records)

    return {
        "episodes": episodes,
        "knowledge": knowledge,
        "records": records,
        "warnings": warnings,
    }


def _search_knowledge(
    query: str, query_vec: np.ndarray | None = None,
) -> tuple[list[dict], list[str]]:
    cfg = get_config()
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

        relevance = sem_score * cfg.KNOWLEDGE_SEMANTIC_WEIGHT + kw_score * cfg.KNOWLEDGE_KEYWORD_WEIGHT

        if relevance < cfg.KNOWLEDGE_RELEVANCE_THRESHOLD:
            continue

        filepath = cfg.KNOWLEDGE_DIR / topic["filename"]
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
        len(topics), len(scored_topics), cfg.KNOWLEDGE_RELEVANCE_THRESHOLD,
    )

    if scored_topics:
        increment_topic_access([t["filename"] for t in scored_topics])

    scored_topics.sort(key=lambda x: x["relevance"], reverse=True)
    return scored_topics[:cfg.KNOWLEDGE_MAX_RESULTS], warnings


def _search_records(
    query: str, query_vec: np.ndarray | None = None, *, include_expired: bool = False,
) -> tuple[list[dict], list[str]]:
    """Search individual knowledge records by semantic + keyword similarity."""
    cfg = get_config()
    warnings: list[str] = []
    records, record_vecs = record_cache.get_record_vecs(include_expired=include_expired)
    if not records:
        return [], warnings

    try:
        if query_vec is None:
            query_vec = backends.encode_query(query)
        if record_vecs is not None:
            sims = (query_vec @ record_vecs.T).flatten()
        else:
            sims = None
    except Exception as e:
        logger.warning(
            "Semantic record search failed, falling back to keyword: %s", e,
            exc_info=True,
        )
        sims = None
        warnings.append("Record search fell back to keyword-only (embedding failed)")

    query_lower = query.lower()
    query_words = set(query_lower.split())

    # Detect task-oriented queries that benefit from procedure records
    _is_task_query = bool(query_words & _TASK_INDICATORS)

    scored_records = []
    for i, rec in enumerate(records):
        sem_score = float(sims[i]) if sims is not None else 0.0

        embed_text = rec.get("embedding_text", "").lower()
        kw_hits = sum(1 for w in query_words if w in embed_text)
        kw_score = kw_hits / len(query_words) if query_words else 0

        relevance = sem_score * cfg.RECORDS_SEMANTIC_WEIGHT + kw_score * cfg.RECORDS_KEYWORD_WEIGHT

        # Boost procedure records for task-oriented queries
        if _is_task_query and rec.get("record_type") == "procedure":
            relevance *= 1.15

        if relevance < cfg.RECORDS_RELEVANCE_THRESHOLD:
            continue

        try:
            content = json.loads(rec["content"]) if isinstance(rec["content"], str) else rec["content"]
        except (json.JSONDecodeError, TypeError):
            content = {}

        # Parse source_episodes for deduplication downstream
        raw_src = rec.get("source_episodes", "[]")
        if isinstance(raw_src, str):
            try:
                src_eps: list[str] = json.loads(raw_src)
            except (json.JSONDecodeError, ValueError):
                src_eps = []
        else:
            src_eps = raw_src if raw_src is not None else []

        scored_records.append({
            "id": rec["id"],
            "record_type": rec["record_type"],
            "content": content,
            "embedding_text": rec.get("embedding_text", ""),
            "topic_title": rec.get("topic_title", ""),
            "topic_filename": rec.get("topic_filename", ""),
            "confidence": rec.get("confidence", 0.8),
            "relevance": round(relevance, 3),
            "source_episodes": src_eps,
        })

    logger.debug(
        "record_search: %d records checked, %d passed relevance threshold (>=%s)",
        len(records), len(scored_records), cfg.RECORDS_RELEVANCE_THRESHOLD,
    )

    if scored_records:
        increment_record_access([r["id"] for r in scored_records])

    scored_records.sort(key=lambda x: x["relevance"], reverse=True)
    return scored_records[:cfg.RECORDS_MAX_RESULTS], warnings
