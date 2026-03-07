"""Priority-ranked retrieval combining vector similarity with metadata scoring.

Returns both episodic memories and relevant knowledge base entries.
Topic embeddings are cached via topic_cache (shared with consolidation).
"""

import json
import logging
import math
from datetime import datetime, timedelta, timezone

import numpy as np

from consolidation_memory.config import get_config
from consolidation_memory.utils import parse_datetime, parse_json_list
from consolidation_memory.database import (
    fts_available,
    fts_search,
    get_active_claims,
    get_all_active_records,
    get_claims_as_of,
    get_claim_source_scope_rows,
    get_connection,
    get_episodes_batch,
    get_recently_contradicted_topic_ids,
    get_records_as_of,
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

# Uncertainty signaling thresholds
_LOW_CONFIDENCE_THRESHOLD = 0.6
_LOW_CONFIDENCE_WARNING = "Low confidence — based on limited or conflicting information"
_EVOLVING_TOPIC_WARNING = "Evolving — this topic has had recent contradictions"
_RECENTLY_CONTRADICTED_CLAIM_WARNING = (
    "Recently contradicted - verify against newer evidence"
)

logger = logging.getLogger(__name__)


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
        created = parse_datetime(created_at_iso)
        age_days = (datetime.now(timezone.utc) - created).total_seconds() / 86400.0
        # Clamp to non-negative to prevent >1.0 scores from future-dated episodes
        age_days = max(0.0, age_days)
        return math.exp(-age_days * math.log(2) / half_life_days)
    except (ValueError, TypeError, OverflowError):
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
        ep_tags = parse_json_list(ep.get("tags", "[]"))
        episode_tags.append(ep_tags)
        all_tags.update(ep_tags)

    if len(all_tags) < 2:
        return scored

    # Find co-occurrence pairs where both tags are in the candidate set
    try:
        pairs = get_tag_pairs_in_set(list(all_tags), min_count=2)
    except (OSError, RuntimeError) as exc:
        logger.warning("Tag co-occurrence lookup failed: %s", exc)
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


# ── Source traceability ───────────────────────────────────────────────────────

def _format_source_dates(dates: list[str]) -> str:
    """Format a list of ISO date strings into a human-readable summary.

    Returns e.g. "Based on 2 conversations (Jan 15, Feb 3, 2025)"
    """
    if not dates:
        return ""

    parsed: list[datetime] = []
    for d in dates:
        try:
            dt = parse_datetime(d)
            parsed.append(dt)
        except (ValueError, TypeError):
            continue

    if not parsed:
        return ""

    parsed.sort()

    # Format dates — use "Mon DD" for same year, "Mon DD, YYYY" otherwise
    now_year = datetime.now(timezone.utc).year
    formatted: list[str] = []
    seen: set[str] = set()
    for dt in parsed:
        if dt.year == now_year:
            label = dt.strftime("%b %d")
        else:
            label = dt.strftime("%b %d, %Y")
        if label not in seen:
            formatted.append(label)
            seen.add(label)

    n = len(parsed)
    date_list = ", ".join(formatted[:5])
    if len(formatted) > 5:
        date_list += f" (+{len(formatted) - 5} more)"

    if n == 1:
        return f"Based on 1 conversation ({date_list})"
    return f"Based on {n} conversations ({date_list})"


def _enrich_source_traceability(records: list[dict]) -> list[dict]:
    """Add source_summary and source_dates to records from their source_episodes."""
    # Collect all unique source episode IDs across all records
    all_src_ids: set[str] = set()
    for rec in records:
        src_eps = rec.get("source_episodes", [])
        if src_eps:
            all_src_ids.update(src_eps)

    # Batch-fetch source episodes for their created_at dates
    episodes_by_id = get_episodes_batch(list(all_src_ids)) if all_src_ids else {}

    for rec in records:
        src_eps = rec.get("source_episodes", [])
        if not src_eps:
            rec["source_summary"] = ""
            rec["source_dates"] = []
            continue

        dates = []
        for eid in src_eps:
            ep = episodes_by_id.get(eid)
            if ep:
                dates.append(ep["created_at"])

        rec["source_dates"] = dates
        rec["source_summary"] = _format_source_dates(dates)

    return records


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


def _matches_scope_filter(
    row: dict[str, object],
    scope: dict[str, str | None] | None,
) -> bool:
    if not scope:
        return True
    for key, expected in scope.items():
        if expected is None:
            continue
        actual = row.get(key)
        if actual is None or str(actual) != expected:
            return False
    return True


def _filter_claims_for_scope(
    claims: list[dict],
    scope: dict[str, str | None] | None,
) -> list[dict]:
    if not scope or not claims:
        return claims

    claim_ids = [str(claim["id"]) for claim in claims if claim.get("id")]
    if not claim_ids:
        return []

    source_rows = get_claim_source_scope_rows(claim_ids)
    allowed_ids: set[str] = set()
    for claim_id in claim_ids:
        rows = source_rows.get(claim_id, [])
        if not rows:
            continue
        if all(_matches_scope_filter(row, scope) for row in rows):
            allowed_ids.add(claim_id)

    return [claim for claim in claims if str(claim.get("id")) in allowed_ids]


# ── Uncertainty signaling ──────────────────────────────────────────────────

def _apply_uncertainty_signals(
    records: list[dict], warnings: list[str],
) -> None:
    """Attach uncertainty flags to low-confidence records and add aggregate warnings.

    Modifies records in-place, adding an "uncertainty" field when confidence
    is below the threshold. Also adds a warning to the warnings list if any
    low-confidence records are present.
    """
    low_conf_count = 0
    for rec in records:
        confidence = rec.get("confidence", 0.8)
        if confidence < _LOW_CONFIDENCE_THRESHOLD:
            rec["uncertainty"] = _LOW_CONFIDENCE_WARNING
            low_conf_count += 1

    if low_conf_count:
        warnings.append(
            f"{low_conf_count} record{'s' if low_conf_count != 1 else ''} "
            f"ha{'ve' if low_conf_count != 1 else 's'} low confidence (< {_LOW_CONFIDENCE_THRESHOLD})"
        )


def _apply_evolving_topic_signals(
    topics: list[dict], warnings: list[str],
) -> None:
    """Flag topics with recent contradictions as "evolving".

    Queries the contradiction log for topic IDs with contradictions in the
    last 30 days and marks matching returned topics.
    """
    if not topics:
        return

    try:
        cfg = get_config()
        contradicted_ids = get_recently_contradicted_topic_ids(
            days=cfg.EVOLVING_TOPIC_LOOKBACK_DAYS,
        )
    except (OSError, RuntimeError) as exc:
        logger.warning("Failed to check recent contradictions: %s", exc)
        return

    if not contradicted_ids:
        return

    # Try to use _topic_id from scored topics (set by _search_knowledge).
    # Fall back to DB lookup only if _topic_id is missing.
    topic_filename_to_id: dict[str, str] = {}
    needs_fallback = any(not t.get("_topic_id") for t in topics)
    if needs_fallback:
        from consolidation_memory.database import get_all_knowledge_topics
        try:
            all_topics = get_all_knowledge_topics()
            topic_filename_to_id = {t["filename"]: t["id"] for t in all_topics}
        except (OSError, RuntimeError) as exc:
            logger.warning("Failed to look up topic IDs for evolving signals: %s", exc)
            return

    evolving_count = 0
    for topic in topics:
        topic_id = topic.get("_topic_id") or topic_filename_to_id.get(topic.get("filename", ""))
        if topic_id and topic_id in contradicted_ids:
            topic["uncertainty"] = _EVOLVING_TOPIC_WARNING
            evolving_count += 1

    if evolving_count:
        warnings.append(
            f"{evolving_count} topic{'s' if evolving_count != 1 else ''} "
            f"{'are' if evolving_count != 1 else 'is'} evolving (recent contradictions detected)"
        )


def _recently_contradicted_claim_ids(
    claim_ids: list[str],
    *,
    as_of: str | None = None,
) -> set[str]:
    """Return claim IDs with recent contradiction events."""
    if not claim_ids:
        return set()

    cfg = get_config()
    reference_dt = parse_datetime(as_of) if as_of else datetime.now(timezone.utc)
    cutoff_dt = reference_dt - timedelta(days=cfg.EVOLVING_TOPIC_LOOKBACK_DAYS)

    placeholders = ",".join("?" for _ in claim_ids)
    params = [cutoff_dt.isoformat(), reference_dt.isoformat(), *claim_ids]
    with get_connection() as conn:
        rows = conn.execute(
            f"""SELECT DISTINCT claim_id
                FROM claim_events
                WHERE event_type = 'contradiction'
                  AND created_at >= ?
                  AND created_at <= ?
                  AND claim_id IN ({placeholders})""",
            params,
        ).fetchall()
    return {row["claim_id"] for row in rows}


def _parse_claim_payload(payload_raw: object) -> dict[str, object]:
    """Parse a claim payload from DB storage into a dict."""
    if isinstance(payload_raw, dict):
        return dict(payload_raw)
    if isinstance(payload_raw, str):
        try:
            parsed = json.loads(payload_raw)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError, ValueError):
            return {}
    return {}


def _search_claims(
    query: str,
    query_vec: np.ndarray | None = None,
    *,
    as_of: str | None = None,
    scope: dict[str, str | None] | None = None,
) -> tuple[list[dict], list[str]]:
    """Search claims by semantic and keyword relevance with uncertainty labels."""
    cfg = get_config()
    warnings: list[str] = []

    fetch_limit = max(cfg.RECORDS_MAX_RESULTS * 10, cfg.RECALL_MAX_N * 5)
    if as_of:
        claims = get_claims_as_of(as_of, limit=fetch_limit)
    else:
        claims = get_active_claims(limit=fetch_limit)

    if not claims:
        return [], warnings

    claim_payloads: list[dict[str, object]] = []
    claim_texts: list[str] = []
    for claim in claims:
        payload = _parse_claim_payload(claim.get("payload"))
        claim_payloads.append(payload)
        payload_text = " ".join(f"{k} {v}" for k, v in sorted(payload.items()))
        claim_texts.append(f"{claim.get('canonical_text', '')} {payload_text}".strip())

    try:
        if query_vec is None:
            query_vec = backends.encode_query(query)
        claim_vecs = backends.encode_documents(claim_texts)
        sims = (query_vec @ claim_vecs.T).flatten()
    except (ConnectionError, RuntimeError, ValueError) as e:
        logger.warning(
            "Semantic claim search failed, falling back to keyword: %s", e,
            exc_info=True,
        )
        sims = None
        warnings.append("Claim search fell back to keyword-only (embedding failed)")

    query_words = set(query.lower().split())
    scored_claims: list[dict] = []
    for i, claim in enumerate(claims):
        sem_score = float(sims[i]) if sims is not None else 0.0

        text_lower = claim_texts[i].lower()
        kw_hits = sum(1 for w in query_words if w in text_lower)
        kw_score = kw_hits / len(query_words) if query_words else 0.0

        relevance = sem_score * cfg.RECORDS_SEMANTIC_WEIGHT + kw_score * cfg.RECORDS_KEYWORD_WEIGHT

        confidence = float(claim.get("confidence", 0.8) or 0.8)
        relevance *= 0.5 + 0.5 * confidence
        if relevance < cfg.RECORDS_RELEVANCE_THRESHOLD:
            continue

        scored_claims.append({
            "id": claim["id"],
            "claim_type": claim.get("claim_type", ""),
            "canonical_text": claim.get("canonical_text", ""),
            "payload": claim_payloads[i],
            "status": claim.get("status", "active"),
            "confidence": confidence,
            "valid_from": claim.get("valid_from"),
            "valid_until": claim.get("valid_until"),
            "relevance": round(relevance, 3),
        })

    logger.debug(
        "claim_search: %d claims checked, %d passed relevance threshold (>=%s)",
        len(claims), len(scored_claims), cfg.RECORDS_RELEVANCE_THRESHOLD,
    )

    scored_claims.sort(key=lambda x: x["relevance"], reverse=True)
    top_claims = scored_claims[:cfg.RECORDS_MAX_RESULTS]

    scoped_claims = _filter_claims_for_scope(top_claims, scope)
    contradicted_ids = _recently_contradicted_claim_ids(
        [claim["id"] for claim in scoped_claims],
        as_of=as_of,
    )
    low_conf_count = 0
    contradicted_count = 0
    for claim in scoped_claims:
        signals: list[str] = []
        if claim["confidence"] < _LOW_CONFIDENCE_THRESHOLD:
            signals.append(_LOW_CONFIDENCE_WARNING)
            low_conf_count += 1
        if claim["id"] in contradicted_ids:
            signals.append(_RECENTLY_CONTRADICTED_CLAIM_WARNING)
            contradicted_count += 1
        if signals:
            claim["uncertainty"] = " | ".join(signals)

    if low_conf_count:
        warnings.append(
            f"{low_conf_count} claim{'s' if low_conf_count != 1 else ''} "
            f"ha{'ve' if low_conf_count != 1 else 's'} low confidence (< {_LOW_CONFIDENCE_THRESHOLD})"
        )
    if contradicted_count:
        warnings.append(
            f"{contradicted_count} claim{'s' if contradicted_count != 1 else ''} "
            f"{'were' if contradicted_count != 1 else 'was'} recently contradicted "
            f"(last {cfg.EVOLVING_TOPIC_LOOKBACK_DAYS} days)"
        )

    return scoped_claims, warnings


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
    as_of: str | None = None,
    scope: dict[str, str | None] | None = None,
) -> dict:
    """Main retrieval function. Returns ranked episodes + knowledge + claims.

    Optional filters (all applied post-vector-search):
        content_types: Only return episodes matching these types.
        tags: Only return episodes that have at least one of these tags.
        after: Only return episodes created after this ISO date string.
        before: Only return episodes created before this ISO date string.
        include_expired: If True, include temporally expired knowledge records.
        as_of: ISO datetime for temporal belief queries. When set, returns
            knowledge state as it was at that point in time, including
            records that have since been superseded and excluding records
            that did not yet exist. Implicitly caps episode results to
            episodes created on or before this time.
        scope: Canonical scope filter dict for namespace/project/client isolation.
    """
    cfg = get_config()
    n_results = min(n_results, cfg.RECALL_MAX_N)

    # Parse date filters to proper datetime objects once, so all downstream
    # comparisons are temporal rather than lexicographic string ordering.
    after_dt = parse_datetime(after) if after else None
    before_dt = parse_datetime(before) if before else None
    as_of_dt = parse_datetime(as_of) if as_of else None

    # Temporal belief query: as_of caps episode results to that point in time.
    # Use whichever is earlier: the explicit `before` or `as_of`.
    if as_of_dt:
        if before_dt is None or as_of_dt < before_dt:
            before_dt = as_of_dt

    # Fetch more candidates when filtering, since many will be discarded
    fetch_k = n_results * 5 if (content_types or tags or after or before) else n_results * 3
    if scope:
        fetch_k = max(fetch_k, n_results * 8)

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
        "fts_candidates=%d, merged=%d, filters=%s, as_of=%s",
        len(query), n_results, len(candidates), len(bm25_map),
        len(all_candidate_ids),
        {"content_types": content_types, "tags": tags, "after": after, "before": before},
        as_of,
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

        if scope and not _matches_scope_filter(ep, scope):
            continue

        # Apply filters
        if _ct_set and ep["content_type"] not in _ct_set:
            continue
        if _tag_set:
            ep_tags = parse_json_list(ep["tags"])
            if not _tag_set.intersection(ep_tags):
                continue
        if after_dt or before_dt:
            try:
                ep_dt = parse_datetime(ep["created_at"])
            except (ValueError, TypeError):
                continue
            if after_dt and ep_dt < after_dt:
                continue
            if before_dt and ep_dt > before_dt:
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
        ep_parsed_tags = parse_json_list(ep["tags"])
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
    claims: list[dict] = []
    warnings = []
    if include_knowledge:
        knowledge, kw_warnings = _search_knowledge(query, query_vec, as_of=as_of, scope=scope)
        warnings.extend(kw_warnings)
        records, rec_warnings = _search_records(
            query, query_vec, include_expired=include_expired, as_of=as_of, scope=scope,
        )
        warnings.extend(rec_warnings)
        claims, claim_warnings = _search_claims(query, query_vec, as_of=as_of, scope=scope)
        warnings.extend(claim_warnings)

        # Source traceability: enrich records and topics with source dates
        if records:
            records = _enrich_source_traceability(records)
        if knowledge:
            knowledge = _enrich_source_traceability(knowledge)

    # Deduplicate: remove episodes already represented by knowledge records
    if cfg.RECALL_DEDUP_ENABLED and records:
        episodes = _deduplicate_episodes(episodes, records)

    return {
        "episodes": episodes,
        "knowledge": knowledge,
        "records": records,
        "claims": claims,
        "warnings": warnings,
    }


def _search_knowledge(
    query: str,
    query_vec: np.ndarray | None = None,
    *,
    as_of: str | None = None,
    scope: dict[str, str | None] | None = None,
) -> tuple[list[dict], list[str]]:
    cfg = get_config()
    warnings: list[str] = []
    topics, summary_vecs = topic_cache.get_topic_vecs()
    if not topics:
        return [], warnings

    if scope:
        scope_indices = [i for i, topic in enumerate(topics) if _matches_scope_filter(topic, scope)]
        topics = [topics[i] for i in scope_indices]
        if summary_vecs is not None:
            summary_vecs = summary_vecs[scope_indices] if scope_indices else None
        if not topics:
            return [], warnings

    # Temporal belief query: filter topics to those that existed at as_of
    if as_of:
        as_of_dt = parse_datetime(as_of)
        filtered_indices = []
        for i, t in enumerate(topics):
            topic_created = t.get("created_at", "")
            if not topic_created:
                continue
            try:
                if parse_datetime(topic_created) <= as_of_dt:
                    filtered_indices.append(i)
            except (ValueError, TypeError):
                continue
        topics = [topics[i] for i in filtered_indices]
        if summary_vecs is not None and len(filtered_indices) < len(summary_vecs):
            summary_vecs = summary_vecs[filtered_indices] if filtered_indices else None
        if not topics:
            return [], warnings

    try:
        if query_vec is None:
            query_vec = backends.encode_query(query)
        if summary_vecs is not None:
            sims = (query_vec @ summary_vecs.T).flatten()
        else:
            sims = None
    except (ConnectionError, RuntimeError, ValueError) as e:
        logger.warning(
            "Semantic knowledge search failed, falling back to keyword: %s", e,
            exc_info=True,
        )
        sims = None
        warnings.append("Knowledge search fell back to keyword-only (embedding failed)")

    query_lower = query.lower()
    query_words = set(query_lower.split())

    scored_topics = []
    knowledge_resolved = cfg.KNOWLEDGE_DIR.resolve()
    for i, topic in enumerate(topics):
        sem_score = float(sims[i]) if sims is not None else 0.0

        title_lower = topic["title"].lower()
        summary_lower = topic["summary"].lower()
        kw_hits = sum(1 for w in query_words if w in title_lower or w in summary_lower)
        kw_score = kw_hits / len(query_words) if query_words else 0

        relevance = sem_score * cfg.KNOWLEDGE_SEMANTIC_WEIGHT + kw_score * cfg.KNOWLEDGE_KEYWORD_WEIGHT

        # Confidence-aware ranking: higher-confidence topics score higher
        topic_confidence = topic.get("confidence", 0.8)
        relevance *= 0.5 + 0.5 * topic_confidence

        # Access-weighted ranking: frequently-recalled topics rank higher
        topic_access = topic.get("access_count", 0)
        relevance *= 1.0 + math.log1p(topic_access) * cfg.CONSOLIDATION_PRIORITY_WEIGHTS["access_frequency"]

        if relevance < cfg.KNOWLEDGE_RELEVANCE_THRESHOLD:
            continue

        filepath = (cfg.KNOWLEDGE_DIR / topic["filename"]).resolve()
        content = ""
        if filepath.is_relative_to(knowledge_resolved) and filepath.exists():
            content = filepath.read_text(encoding="utf-8")

        # Parse source_episodes for traceability
        topic_src_eps: list[str] = parse_json_list(topic.get("source_episodes"))

        scored_topics.append({
            "topic": topic["filename"].replace(".md", ""),
            "filename": topic["filename"],
            "title": topic["title"],
            "summary": topic["summary"],
            "content": content,
            "confidence": topic["confidence"],
            "relevance": round(relevance, 3),
            "source_episodes": topic_src_eps,
            "_topic_id": topic.get("id"),
        })

    logger.debug(
        "knowledge_search: %d topics checked, %d passed relevance threshold (>=%s)",
        len(topics), len(scored_topics), cfg.KNOWLEDGE_RELEVANCE_THRESHOLD,
    )

    scored_topics.sort(key=lambda x: x["relevance"], reverse=True)
    top_topics = scored_topics[:cfg.KNOWLEDGE_MAX_RESULTS]

    if top_topics:
        increment_topic_access([t["filename"] for t in top_topics])

    # Uncertainty signaling: flag evolving topics with recent contradictions
    _apply_evolving_topic_signals(top_topics, warnings)

    # Strip internal _topic_id before returning to callers
    for t in top_topics:
        t.pop("_topic_id", None)

    return top_topics, warnings


def _search_records(
    query: str,
    query_vec: np.ndarray | None = None,
    *,
    include_expired: bool = False,
    as_of: str | None = None,
    scope: dict[str, str | None] | None = None,
) -> tuple[list[dict], list[str]]:
    """Search individual knowledge records by semantic + keyword similarity.

    When ``as_of`` is set, bypasses the record cache and queries the database
    directly for records that were valid at that point in time, then embeds
    them on the fly.
    """
    cfg = get_config()
    warnings: list[str] = []

    if as_of or scope:
        # Temporal query: fetch records valid at that point in time directly
        # from the database — the cache only holds current state.
        if as_of:
            if scope:
                records = get_records_as_of(as_of, scope=scope)
            else:
                records = get_records_as_of(as_of)
        else:
            records = get_all_active_records(include_expired=include_expired, scope=scope)
        if not records:
            return [], warnings
        texts = [r["embedding_text"] for r in records]
        try:
            record_vecs = backends.encode_documents(texts)
        except (ConnectionError, RuntimeError, ValueError) as e:
            logger.warning("Failed to embed temporal records: %s", e, exc_info=True)
            record_vecs = None
    else:
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
    except (ConnectionError, RuntimeError, ValueError) as e:
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

        # Confidence-aware ranking: higher-confidence records score higher
        # 0.5 confidence → 0.75x, 0.8 → 0.9x, 1.0 → 1.0x
        confidence = rec.get("confidence", 0.8)
        relevance *= 0.5 + 0.5 * confidence

        # Access-weighted ranking: frequently-recalled records rank higher
        rec_access = rec.get("access_count", 0)
        relevance *= 1.0 + math.log1p(rec_access) * cfg.CONSOLIDATION_PRIORITY_WEIGHTS["access_frequency"]

        if relevance < cfg.RECORDS_RELEVANCE_THRESHOLD:
            continue

        try:
            content = json.loads(rec["content"]) if isinstance(rec["content"], str) else rec["content"]
        except (json.JSONDecodeError, TypeError):
            content = {}

        # Parse source_episodes for deduplication downstream
        src_eps: list[str] = parse_json_list(rec.get("source_episodes"))

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

    scored_records.sort(key=lambda x: x["relevance"], reverse=True)
    top_records = scored_records[:cfg.RECORDS_MAX_RESULTS]

    if top_records:
        increment_record_access([r["id"] for r in top_records])

    # Uncertainty signaling: flag low-confidence records
    _apply_uncertainty_signals(top_records, warnings)

    return top_records, warnings
