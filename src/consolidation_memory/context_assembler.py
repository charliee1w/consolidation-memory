"""Priority-ranked retrieval combining vector similarity with metadata scoring.

Returns both episodic memories and relevant knowledge base entries.
Topic embeddings are cached via topic_cache (shared with consolidation).
"""

import json
import logging
import math
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import SupportsFloat

import numpy as np

from consolidation_memory.config import get_config
from consolidation_memory.utils import parse_datetime, parse_json_list
from consolidation_memory.database import (
    fts_available,
    fts_search,
    get_active_claims,
    get_claim_outcome_evidence,
    get_claims_as_of,
    get_connection,
    get_episodes_batch,
    get_recently_contradicted_topic_ids,
    get_records_as_of,
    get_tag_pairs_in_set,
    increment_access,
    increment_record_access,
    increment_topic_access,
    increment_topic_access_by_ids,
)
from consolidation_memory import backends
from consolidation_memory import claim_cache
from consolidation_memory.knowledge_paths import resolve_topic_path
from consolidation_memory.query_semantics import (
    claim_reliability_profile as _claim_reliability_profile,
    coerce_numeric_float as _coerce_numeric_float,
    filter_claims_for_scope as _filter_claims_for_scope,
    matches_scope_filter as _matches_scope_filter,
    strategy_reuse_profile as _strategy_reuse_profile,
)
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
_TOPIC_VERSION_TIMESTAMP_FORMAT = "%Y-%m-%dT%H-%M-%S-%f"


def invalidate_topic_cache() -> None:
    """Call after consolidation to force re-embedding on next recall."""
    topic_cache.invalidate()


def invalidate_record_cache() -> None:
    """Call after consolidation to force re-embedding on next recall."""
    record_cache.invalidate()


def invalidate_claim_cache() -> None:
    """Call after claim graph changes to force re-embedding on next recall."""
    claim_cache.invalidate()


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


def _get_claim_search_candidates(
    *,
    fetch_limit: int,
    as_of: str | None = None,
    scope: dict[str, str | None] | None = None,
) -> list[dict]:
    """Fetch claim candidates, paging until scoped windows are satisfied."""
    if not scope:
        if as_of:
            return get_claims_as_of(as_of, limit=fetch_limit)
        return get_active_claims(limit=fetch_limit)

    page_size = min(max(fetch_limit * 5, 50), 1000)
    collected: list[dict] = []
    offset = 0
    while len(collected) < fetch_limit:
        if as_of:
            rows = get_claims_as_of(as_of, limit=page_size, offset=offset)
        else:
            rows = get_active_claims(limit=page_size, offset=offset)
        if not rows:
            break
        collected.extend(_filter_claims_for_scope(rows, scope))
        offset += len(rows)
        if len(rows) < page_size:
            break
    return collected[:fetch_limit]


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
    claims = _get_claim_search_candidates(
        fetch_limit=fetch_limit,
        as_of=as_of,
        scope=scope,
    )

    if not claims:
        return [], warnings

    claim_payloads, claim_texts = claim_cache.build_claim_texts(claims)

    try:
        if query_vec is None:
            query_vec = backends.encode_query(query)
        claim_vecs: np.ndarray | None
        if as_of:
            # Temporal snapshots vary by as_of; embed ad-hoc to avoid unbounded cache keys.
            claim_vecs = backends.encode_documents(claim_texts)
        else:
            claim_vecs = claim_cache.get_claim_vecs(claims, claim_texts)
        if claim_vecs is not None:
            sims = (query_vec @ claim_vecs.T).flatten()
        else:
            sims = None
    except (ConnectionError, RuntimeError, ValueError) as e:
        logger.warning(
            "Semantic claim search failed, falling back to keyword: %s", e,
            exc_info=True,
        )
        sims = None
        warnings.append("Claim search fell back to keyword-only (embedding failed)")

    query_words = set(query.lower().split())
    claim_ids = [str(claim.get("id")) for claim in claims if claim.get("id")]
    claim_evidence = get_claim_outcome_evidence(
        claim_ids,
        as_of=as_of,
        scope=scope,
    )
    scored_claims: list[dict] = []
    for i, claim in enumerate(claims):
        sem_score = float(sims[i]) if sims is not None else 0.0

        text_lower = claim_texts[i].lower()
        kw_hits = sum(1 for w in query_words if w in text_lower)
        kw_score = kw_hits / len(query_words) if query_words else 0.0

        relevance = sem_score * cfg.RECORDS_SEMANTIC_WEIGHT + kw_score * cfg.RECORDS_KEYWORD_WEIGHT

        confidence = float(claim.get("confidence", 0.8) or 0.8)
        relevance *= 0.5 + 0.5 * confidence
        claim_id = str(claim.get("id", ""))
        evidence_payload = {
            **claim_evidence.get(claim_id, {}),
            "claim_status": claim.get("status", "active"),
        }
        reliability_profile = _claim_reliability_profile(
            evidence_payload,
            claim_status=str(claim.get("status", "active")),
            as_of=as_of,
            claim_updated_at=str(claim.get("updated_at") or ""),
        )
        relevance *= _coerce_numeric_float(
            reliability_profile.get("ranking_multiplier"),
            default=1.0,
        )

        strategy_profile: dict[str, object] | None = None
        if claim.get("claim_type") == "strategy":
            strategy_profile = _strategy_reuse_profile(evidence_payload)
            relevance *= _coerce_numeric_float(
                strategy_profile.get("reuse_multiplier"),
                default=1.0,
            )
        if relevance < cfg.RECORDS_RELEVANCE_THRESHOLD:
            continue

        row = {
            "id": claim["id"],
            "claim_type": claim.get("claim_type", ""),
            "canonical_text": claim.get("canonical_text", ""),
            "payload": claim_payloads[i],
            "status": claim.get("status", "active"),
            "confidence": confidence,
            "valid_from": claim.get("valid_from"),
            "valid_until": claim.get("valid_until"),
            "relevance": round(relevance, 3),
            "reliability": reliability_profile,
        }
        if strategy_profile is not None:
            row["strategy_evidence"] = strategy_profile
        scored_claims.append(row)

    logger.debug(
        "claim_search: %d claims checked, %d passed relevance threshold (>=%s)",
        len(claims), len(scored_claims), cfg.RECORDS_RELEVANCE_THRESHOLD,
    )

    scored_claims.sort(key=lambda x: x["relevance"], reverse=True)
    scoped_claims = scored_claims[:cfg.RECORDS_MAX_RESULTS]
    claim_ids = [str(claim.get("id", "")) for claim in scoped_claims]
    contradicted_ids = _recently_contradicted_claim_ids(claim_ids, as_of=as_of)
    low_conf_count = 0
    contradicted_count = 0
    for claim in scoped_claims:
        claim_id = str(claim.get("id", ""))
        raw_confidence = claim.get("confidence", 0.8)
        if isinstance(raw_confidence, (int, float)):
            confidence = float(raw_confidence)
        elif isinstance(raw_confidence, str):
            try:
                confidence = float(raw_confidence)
            except ValueError:
                confidence = 0.8
        else:
            confidence = 0.8
        signals: list[str] = []
        if confidence < _LOW_CONFIDENCE_THRESHOLD:
            signals.append(_LOW_CONFIDENCE_WARNING)
            low_conf_count += 1
        if claim_id in contradicted_ids:
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


def _parse_topic_version_timestamp(version_path: Path, stem: str) -> datetime | None:
    """Parse the timestamp embedded in a versioned topic filename."""
    prefix = f"{stem}."
    suffix = version_path.suffix
    name = version_path.name
    if not name.startswith(prefix) or suffix != ".md":
        return None
    token = name[len(prefix):-len(suffix)]
    try:
        return datetime.strptime(token, _TOPIC_VERSION_TIMESTAMP_FORMAT).replace(
            tzinfo=timezone.utc,
        )
    except ValueError:
        return None


def _topic_requires_historical_snapshot(topic: dict, as_of_dt: datetime) -> bool:
    """Return True when the current topic row is newer than the requested time."""
    updated_at = topic.get("updated_at")
    if not updated_at:
        return False
    try:
        return parse_datetime(str(updated_at)) > as_of_dt
    except (TypeError, ValueError):
        return False


def _select_versioned_topic_snapshot(filepath: Path, as_of_dt: datetime) -> Path | None:
    """Return the version file whose update superseded the requested point in time."""
    cfg = get_config()
    selected_path: Path | None = None
    selected_dt: datetime | None = None
    for version_path in cfg.KNOWLEDGE_VERSIONS_DIR.glob(f"{filepath.stem}.*.md"):
        version_dt = _parse_topic_version_timestamp(version_path, filepath.stem)
        if version_dt is None or version_dt <= as_of_dt:
            continue
        if selected_dt is None or version_dt < selected_dt:
            selected_dt = version_dt
            selected_path = version_path
    return selected_path


def _coerce_topic_confidence(raw_confidence: object, default: float) -> float:
    """Normalize topic confidence values into a bounded float."""
    confidence_value: float
    try:
        if isinstance(raw_confidence, str):
            confidence_value = float(raw_confidence.strip())
        elif isinstance(raw_confidence, SupportsFloat):
            confidence_value = float(raw_confidence)
        else:
            return default
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, confidence_value))


def _build_topic_snapshot_from_content(
    topic: dict,
    content: str,
    *,
    allow_current_summary_fallback: bool,
) -> dict:
    """Overlay parsed frontmatter metadata onto a topic row."""
    from consolidation_memory.consolidation.prompting import _parse_frontmatter

    parsed = _parse_frontmatter(content)
    meta = parsed.get("meta", {}) if isinstance(parsed, dict) else {}
    snapshot = dict(topic)
    snapshot["title"] = str(
        meta.get("title")
        or topic.get("title")
        or Path(str(topic.get("filename") or "topic.md")).stem,
    )
    if "summary" in meta:
        snapshot["summary"] = str(meta.get("summary") or "")
    elif allow_current_summary_fallback:
        snapshot["summary"] = str(topic.get("summary") or "")
    else:
        snapshot["summary"] = ""
    snapshot["confidence"] = _coerce_topic_confidence(
        meta.get("confidence", topic.get("confidence", 0.8)),
        _coerce_topic_confidence(topic.get("confidence", 0.8), 0.8),
    )
    snapshot["content"] = content
    return snapshot


def _load_topic_snapshot_as_of(topic: dict, as_of_dt: datetime) -> dict | None:
    """Load the best historical topic snapshot for an as_of query."""
    cfg = get_config()
    try:
        filepath = resolve_topic_path(cfg.KNOWLEDGE_DIR, topic, prefer_existing=True)
    except ValueError:
        return None

    snapshot_path = _select_versioned_topic_snapshot(filepath, as_of_dt)
    if snapshot_path is None or not snapshot_path.exists():
        return None

    try:
        content = snapshot_path.read_text(encoding="utf-8")
    except OSError:
        return None

    return _build_topic_snapshot_from_content(
        topic,
        content,
        allow_current_summary_fallback=False,
    )


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

    reembed_topic_texts: list[str] | None = None
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
        filtered_topics: list[dict] = []
        filtered_indices: list[int] = []
        missing_snapshot_count = 0
        for i, t in enumerate(topics):
            topic_created = t.get("created_at", "")
            if not topic_created:
                continue
            try:
                if parse_datetime(str(topic_created)) > as_of_dt:
                    continue
            except (ValueError, TypeError):
                continue

            if _topic_requires_historical_snapshot(t, as_of_dt):
                snapshot = _load_topic_snapshot_as_of(t, as_of_dt)
                if snapshot is None:
                    missing_snapshot_count += 1
                    continue
                filtered_topics.append(snapshot)
            else:
                filtered_topics.append(dict(t))
                filtered_indices.append(i)
        topics = filtered_topics
        if any("content" in topic for topic in topics):
            reembed_topic_texts = [f"{t['title']}. {t['summary']}" for t in topics]
            summary_vecs = None
        elif summary_vecs is not None and len(filtered_indices) < len(summary_vecs):
            summary_vecs = summary_vecs[filtered_indices] if filtered_indices else None
        if not topics:
            return [], warnings
        if missing_snapshot_count:
            warnings.append(
                f"{missing_snapshot_count} historical topic snapshot"
                f"{'s were' if missing_snapshot_count != 1 else ' was'} unavailable for as_of={as_of}"
            )

    try:
        if query_vec is None:
            query_vec = backends.encode_query(query)
        if reembed_topic_texts is not None:
            summary_vecs = backends.encode_documents(reembed_topic_texts) if reembed_topic_texts else None
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

        content = str(topic.get("content") or "")
        if not content:
            try:
                filepath = resolve_topic_path(cfg.KNOWLEDGE_DIR, topic, prefer_existing=True)
                if filepath.exists():
                    content = filepath.read_text(encoding="utf-8")
            except ValueError:
                # Keep recall robust when malformed filenames are encountered.
                content = ""

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
        topic_ids = [str(t["_topic_id"]) for t in top_topics if t.get("_topic_id")]
        if topic_ids:
            try:
                increment_topic_access_by_ids(topic_ids)
            except (OSError, RuntimeError, ValueError, sqlite3.Error) as exc:
                logger.warning("Topic access update by id failed: %s", exc)
                increment_topic_access([t["filename"] for t in top_topics])
        else:
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

    if as_of:
        # Temporal query: fetch records valid at that point in time directly
        # from the database — the cache only holds current state.
        if scope:
            records = get_records_as_of(as_of, scope=scope)
        else:
            records = get_records_as_of(as_of)
        if not records:
            return [], warnings
        texts = [r["embedding_text"] for r in records]
        try:
            record_vecs = backends.encode_documents(texts)
        except (ConnectionError, RuntimeError, ValueError) as e:
            logger.warning("Failed to embed temporal records: %s", e, exc_info=True)
            record_vecs = None
    elif scope:
        records, record_vecs = record_cache.get_record_vecs(
            include_expired=include_expired,
            scope=scope,
        )
        if not records:
            return [], warnings
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

