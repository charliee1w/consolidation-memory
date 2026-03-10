"""Main consolidation orchestration engine.

Includes: knowledge-file versioning, contradiction detection, topic merging,
cluster processing, the run_consolidation entry point, and index updating.
"""

import json
import logging
import shutil
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from consolidation_memory import claim_cache, record_cache, topic_cache
from consolidation_memory.claim_graph import claim_from_record
from consolidation_memory.config import get_config
from consolidation_memory.database import (
    complete_consolidation_run,
    ensure_schema,
    expire_claim,
    expire_record,
    get_connection,
    get_all_knowledge_topics,
    get_prunable_episodes,
    get_records_by_topic,
    get_unconsolidated_episodes,
    increment_consolidation_attempts,
    insert_claim_edge,
    insert_claim_event,
    insert_claim_sources,
    insert_consolidation_metrics,
    insert_contradiction,
    insert_knowledge_records,
    mark_consolidated,
    mark_pruned,
    reset_stale_consolidation_attempts,
    start_consolidation_run,
    upsert_claim,
    upsert_knowledge_topic,
)
from consolidation_memory.vector_store import VectorStore
from consolidation_memory.backends import encode_documents
from consolidation_memory.consolidation.clustering import (
    _compute_cluster_confidence,
    _find_similar_topic,
)
from consolidation_memory.consolidation.prompting import (
    _build_contradiction_prompt,
    _build_extraction_prompt,
    _build_merge_extraction_prompt,
    _call_llm,
    _embedding_text_for_record,
    _llm_extract_with_validation,
    _parse_frontmatter,
    _render_markdown_from_records,
    _slugify,
    _strip_code_fences,
)
from consolidation_memory.consolidation.scoring import _adjust_surprise_scores
from consolidation_memory.plugins import get_plugin_manager
from consolidation_memory.types import (
    ConsolidationReport,
    RUN_STATUS_COMPLETED,
    RUN_STATUS_FAILED,
)
from consolidation_memory.utils import parse_json_list

logger = logging.getLogger(__name__)


def _episode_scope_row(episode: dict[str, object]) -> dict[str, str | None]:
    """Extract canonical scope columns from an episode row."""
    return {
        "namespace_slug": str(episode.get("namespace_slug") or "default"),
        "namespace_sharing_mode": str(episode.get("namespace_sharing_mode") or "private"),
        "app_client_name": str(episode.get("app_client_name") or "legacy_client"),
        "app_client_type": str(episode.get("app_client_type") or "python_sdk"),
        "app_client_provider": str(episode.get("app_client_provider")) if episode.get("app_client_provider") else None,
        "app_client_external_key": (
            str(episode.get("app_client_external_key"))
            if episode.get("app_client_external_key")
            else None
        ),
        "agent_name": str(episode.get("agent_name")) if episode.get("agent_name") else None,
        "agent_external_key": (
            str(episode.get("agent_external_key")) if episode.get("agent_external_key") else None
        ),
        "session_external_key": (
            str(episode.get("session_external_key"))
            if episode.get("session_external_key")
            else None
        ),
        "session_kind": str(episode.get("session_kind")) if episode.get("session_kind") else None,
        "project_slug": str(episode.get("project_slug") or "default"),
        "project_display_name": (
            str(episode.get("project_display_name")) if episode.get("project_display_name") else None
        ),
        "project_root_uri": str(episode.get("project_root_uri")) if episode.get("project_root_uri") else None,
        "project_repo_remote": (
            str(episode.get("project_repo_remote")) if episode.get("project_repo_remote") else None
        ),
        "project_default_branch": (
            str(episode.get("project_default_branch")) if episode.get("project_default_branch") else None
        ),
    }


def _default_scope_row() -> dict[str, str | None]:
    return {
        "namespace_slug": "default",
        "namespace_sharing_mode": "private",
        "app_client_name": "legacy_client",
        "app_client_type": "python_sdk",
        "app_client_provider": None,
        "app_client_external_key": None,
        "agent_name": None,
        "agent_external_key": None,
        "session_external_key": None,
        "session_kind": None,
        "project_slug": "default",
        "project_display_name": "default",
        "project_root_uri": None,
        "project_repo_remote": None,
        "project_default_branch": None,
    }


def _scope_key(scope: dict[str, str | None]) -> tuple[str, ...]:
    """Deterministic key for scope isolation in consolidation clustering."""
    return (
        scope.get("namespace_slug") or "default",
        scope.get("project_slug") or "default",
        scope.get("app_client_name") or "legacy_client",
        scope.get("app_client_type") or "python_sdk",
        scope.get("app_client_provider") or "",
        scope.get("app_client_external_key") or "",
        scope.get("agent_external_key") or "",
        scope.get("agent_name") or "",
        scope.get("session_external_key") or "",
        scope.get("session_kind") or "",
    )


def _scope_filename_prefix(scope: dict[str, str | None]) -> str:
    """Build a stable scope prefix so topic filenames remain scope-safe."""
    parts: list[str] = [
        scope.get("namespace_slug") or "default",
        scope.get("project_slug") or "default",
    ]
    if scope.get("namespace_sharing_mode") == "private":
        parts.append(scope.get("app_client_name") or "legacy_client")
        parts.append(scope.get("app_client_type") or "python_sdk")
    cleaned = [_slugify(part) for part in parts if part]
    return "_".join(part for part in cleaned if part)


# ── Knowledge versioning ─────────────────────────────────────────────────────


def _version_knowledge_file(filepath: Path) -> None:
    if not filepath.exists():
        return

    get_config().KNOWLEDGE_VERSIONS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S-%f")
    stem = filepath.stem
    versioned_name = f"{stem}.{timestamp}.md"
    versioned_path = get_config().KNOWLEDGE_VERSIONS_DIR / versioned_name

    shutil.copy2(str(filepath), str(versioned_path))
    logger.info("Versioned %s -> %s", filepath.name, versioned_name)

    pattern = f"{stem}.*.md"
    existing_versions = sorted(
        get_config().KNOWLEDGE_VERSIONS_DIR.glob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for old in existing_versions[get_config().KNOWLEDGE_MAX_VERSIONS :]:
        old.unlink()
        logger.debug("Pruned old version: %s", old.name)


# ── Contradiction detection ───────────────────────────────────────────────────


def _detect_contradictions(
    new_records: list[dict],
    existing_records: list[dict],
) -> list[tuple[int, str]]:
    """Detect contradictions between new and existing knowledge records.

    Uses semantic similarity to find candidate pairs, then LLM verification
    for pairs above the similarity threshold.

    Args:
        new_records: List of new record dicts (with 'embedding_text' and content fields).
        existing_records: List of existing DB record dicts (with 'id', 'embedding_text', 'content').

    Returns:
        List of (new_record_index, existing_record_id) pairs that contradict.
    """
    if not new_records or not existing_records:
        return []

    # Embed new records
    new_texts = []
    for rec in new_records:
        text = rec.get("embedding_text", "")
        if not text:
            text = _embedding_text_for_record(rec)
        new_texts.append(text)

    existing_texts = []
    for rec in existing_records:
        text = rec.get("embedding_text", "")
        if not text:
            content = rec.get("content", {})
            if isinstance(content, str):
                try:
                    content = json.loads(content)
                except (json.JSONDecodeError, TypeError):
                    content = {}
            text = _embedding_text_for_record(content)
        existing_texts.append(text)

    try:
        new_vecs = encode_documents(new_texts)
        existing_vecs = encode_documents(existing_texts)
    except Exception as e:
        logger.warning("Failed to embed records for contradiction detection: %s", e)
        return []

    # Compute similarities
    sims = new_vecs @ existing_vecs.T  # shape: (len(new), len(existing))

    # Find candidate pairs above threshold
    candidate_pairs: list[tuple[int, int]] = []
    for new_idx in range(len(new_records)):
        for ex_idx in range(len(existing_records)):
            if float(sims[new_idx, ex_idx]) >= get_config().CONTRADICTION_SIMILARITY_THRESHOLD:
                candidate_pairs.append((new_idx, ex_idx))

    if not candidate_pairs:
        return []

    logger.info(
        "Contradiction detection: %d candidate pairs above threshold %.2f",
        len(candidate_pairs),
        get_config().CONTRADICTION_SIMILARITY_THRESHOLD,
    )

    if not get_config().CONTRADICTION_LLM_ENABLED:
        # Without LLM, treat all high-similarity pairs as contradictions
        return [(new_idx, existing_records[ex_idx]["id"]) for new_idx, ex_idx in candidate_pairs]

    # Build pair contents for LLM verification
    pair_contents: list[tuple[dict, dict]] = []
    for new_idx, ex_idx in candidate_pairs:
        ex_content = existing_records[ex_idx].get("content", {})
        if isinstance(ex_content, str):
            try:
                ex_content = json.loads(ex_content)
            except (json.JSONDecodeError, TypeError):
                ex_content = {}
        new_content = new_records[new_idx]
        # Strip embedding_text from content sent to LLM to keep prompt focused
        new_for_llm = {k: v for k, v in new_content.items() if k != "embedding_text"}
        pair_contents.append((ex_content, new_for_llm))

    prompt = _build_contradiction_prompt(pair_contents)

    try:
        raw = _call_llm(prompt, max_retries=2)
        verdicts = json.loads(_strip_code_fences(raw))
    except Exception as e:
        logger.warning("LLM contradiction verification failed: %s", e)
        return []

    if not isinstance(verdicts, list):
        logger.warning(
            "LLM returned non-list (%s) for contradiction verdicts. Skipping.",
            type(verdicts).__name__,
        )
        return []

    if len(verdicts) != len(candidate_pairs):
        logger.warning(
            "LLM returned %d verdicts for %d pairs; processing available verdicts.",
            len(verdicts),
            len(candidate_pairs),
        )

    contradictions = []
    for (new_idx, ex_idx), verdict in zip(candidate_pairs, verdicts):
        if isinstance(verdict, str) and "CONTRADICT" in verdict.upper():
            contradictions.append((new_idx, existing_records[ex_idx]["id"]))

    logger.info(
        "Contradiction detection: %d/%d candidate pairs confirmed as contradictions",
        len(contradictions),
        len(candidate_pairs),
    )
    return contradictions


# ── Silent drop detection ─────────────────────────────────────────────────────


def _detect_silent_drops(
    pre_merge_records: list[dict],
    merged_records: list[dict],
) -> list[tuple[int, float]]:
    """Detect pre-merge records silently dropped during LLM merge.

    Compares each pre-merge record against all merged records using cosine
    similarity. Returns indices of pre-merge records whose max similarity
    to any merged record falls below the configured threshold.

    Returns:
        List of (pre_merge_index, max_similarity) for dropped records.
    """
    if not pre_merge_records or not merged_records:
        return []

    cfg = get_config()

    pre_texts = [_embedding_text_for_record(rec) for rec in pre_merge_records]
    merged_texts = [_embedding_text_for_record(rec) for rec in merged_records]

    try:
        pre_vecs = encode_documents(pre_texts)
        merged_vecs = encode_documents(merged_texts)
    except Exception as e:
        logger.warning("Failed to embed records for silent drop detection: %s", e)
        return []

    sims = pre_vecs @ merged_vecs.T  # shape: (len(pre), len(merged))

    drops: list[tuple[int, float]] = []
    for idx in range(len(pre_merge_records)):
        max_sim = float(np.max(sims[idx]))
        if max_sim < cfg.MERGE_DROP_SIMILARITY_THRESHOLD:
            drops.append((idx, max_sim))

    return drops


# ── Topic merging ─────────────────────────────────────────────────────────────


def _normalize_record_field(value: object) -> str:
    return str(value or "").strip().lower()


def _record_dedup_key(record: dict) -> tuple:
    """Build a deterministic dedup key for a structured knowledge record."""
    rtype = _normalize_record_field(record.get("type", "fact"))
    if rtype == "fact":
        return (
            "fact",
            _normalize_record_field(record.get("subject")),
            _normalize_record_field(record.get("info")),
        )
    if rtype == "solution":
        return (
            "solution",
            _normalize_record_field(record.get("problem")),
            _normalize_record_field(record.get("fix")),
            _normalize_record_field(record.get("context")),
        )
    if rtype == "preference":
        return (
            "preference",
            _normalize_record_field(record.get("key")),
            _normalize_record_field(record.get("value")),
            _normalize_record_field(record.get("context")),
        )
    if rtype == "procedure":
        return (
            "procedure",
            _normalize_record_field(record.get("trigger")),
            _normalize_record_field(record.get("steps")),
            _normalize_record_field(record.get("context")),
        )
    # Unknown record shape: keep stable by sorted JSON key.
    return ("unknown", json.dumps(record, sort_keys=True, default=str))


def _record_specificity_score(record: dict) -> int:
    # Prefer records with denser content when deduplicating.
    return len(_embedding_text_for_record(record)) + len(json.dumps(record, default=str))


def _dedupe_records(records: list[dict]) -> list[dict]:
    """Deduplicate records by type-specific keys, keeping the most specific."""
    best_by_key: dict[tuple, dict] = {}
    order: list[tuple] = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        key = _record_dedup_key(rec)
        if key not in best_by_key:
            best_by_key[key] = rec
            order.append(key)
            continue
        if _record_specificity_score(rec) >= _record_specificity_score(best_by_key[key]):
            best_by_key[key] = rec
    return [best_by_key[k] for k in order]


def _merge_tags(existing_tags: list[str], new_tags: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for raw in [*existing_tags, *new_tags]:
        tag = str(raw).strip()
        if not tag:
            continue
        norm = tag.lower()
        if norm in seen:
            continue
        seen.add(norm)
        merged.append(tag)
    return merged


def _build_deterministic_merge_payload(
    existing_records: list[dict],
    new_records: list[dict],
    existing_title: str,
    existing_summary: str,
    existing_tags: list[str],
    extraction_data: dict,
) -> tuple[str, str, list[str], list[dict]]:
    """Fallback merge that never calls the LLM."""
    merged_records = _dedupe_records([*existing_records, *new_records])
    merged_title = extraction_data.get("title") or existing_title
    merged_summary = extraction_data.get("summary") or existing_summary
    merged_tags = _merge_tags(existing_tags, extraction_data.get("tags", []))
    return merged_title, merged_summary, merged_tags, merged_records


def _coerce_content_dict(content: object) -> dict:
    """Best-effort conversion of a stored record payload to a dict."""
    if isinstance(content, dict):
        return dict(content)
    if isinstance(content, str):
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError, ValueError):
            return {}
    return {}


def _materialize_claim_for_record(
    record: dict,
    *,
    topic_id: str,
    source_episode_ids: list[str] | None = None,
    source_record_id: str | None = None,
    confidence: float = 0.8,
    status: str = "active",
    valid_from: str | None = None,
    valid_until: str | None = None,
    event_type: str | None = None,
    event_details: dict | None = None,
) -> str | None:
    """Upsert claim + sources + optional event for one record.

    Any failure is logged and isolated to this record.
    """
    try:
        claim_obj = claim_from_record(record)
    except Exception as e:
        logger.warning("Claim canonicalization failed for record %s: %s", record, e)
        return None

    claim_id = claim_obj["id"]
    try:
        upsert_claim(
            claim_id=claim_id,
            claim_type=claim_obj["claim_type"],
            canonical_text=claim_obj["canonical_text"],
            payload=claim_obj["payload"],
            status=status,
            confidence=confidence,
            valid_from=valid_from,
            valid_until=valid_until,
        )
    except Exception as e:
        logger.warning("Failed to upsert claim %s: %s", claim_id, e)
        return None

    source_rows: list[dict] = []
    if source_episode_ids:
        for episode_id in source_episode_ids:
            source_rows.append(
                {
                    "source_episode_id": episode_id,
                    "source_topic_id": topic_id,
                    "source_record_id": source_record_id,
                }
            )
    else:
        source_rows.append(
            {
                "source_episode_id": None,
                "source_topic_id": topic_id,
                "source_record_id": source_record_id,
            }
        )

    try:
        insert_claim_sources(claim_id, source_rows)
    except Exception as e:
        logger.warning("Failed to insert claim sources for %s: %s", claim_id, e)

    if event_type:
        details: dict[str, object] = {"topic_id": topic_id}
        if source_record_id:
            details["source_record_id"] = source_record_id
        if event_details:
            details.update(event_details)
        try:
            insert_claim_event(claim_id, event_type=event_type, details=details)
        except Exception as e:
            logger.warning("Failed to insert claim event (%s) for %s: %s", event_type, claim_id, e)

    return claim_id


def _emit_claims_for_records(
    records: list[dict],
    *,
    topic_id: str,
    source_episode_ids: list[str],
    source_record_ids: list[str] | None,
    confidence: float,
    default_valid_from: str | None = None,
    event_type: str,
) -> list[str]:
    """Emit claims for a record list. Returns claim IDs emitted successfully."""
    claim_ids: list[str] = []
    for idx, rec in enumerate(records):
        source_record_id = None
        if source_record_ids and idx < len(source_record_ids):
            source_record_id = source_record_ids[idx]
        claim_id = _materialize_claim_for_record(
            rec,
            topic_id=topic_id,
            source_episode_ids=source_episode_ids,
            source_record_id=source_record_id,
            confidence=confidence,
            valid_from=default_valid_from,
            event_type=event_type,
        )
        if claim_id:
            claim_ids.append(claim_id)
    return claim_ids


def _load_existing_topic_merge_state(
    existing: dict,
    *,
    cluster_ep_ids: list[str],
) -> tuple[list[dict], list[dict], list[str], Path] | None:
    """Load DB records/frontmatter needed to merge into an existing topic."""
    existing_db_records = get_records_by_topic(existing["id"])
    existing_records: list[dict] = []
    for row in existing_db_records:
        try:
            content = json.loads(row["content"]) if isinstance(row["content"], str) else row["content"]
        except (json.JSONDecodeError, TypeError):
            content = {"type": "fact", "subject": "?", "info": row.get("content", "")}
        existing_records.append(content)

    existing_tags: list[str] = []
    filepath = get_config().KNOWLEDGE_DIR / existing["filename"]

    # Validate that the resolved path stays within KNOWLEDGE_DIR (path traversal guard)
    knowledge_dir_resolved = get_config().KNOWLEDGE_DIR.resolve()
    if not filepath.resolve().is_relative_to(knowledge_dir_resolved):
        logger.error(
            "Path traversal detected: %s resolves outside KNOWLEDGE_DIR", existing["filename"]
        )
        increment_consolidation_attempts(cluster_ep_ids)
        return None

    if filepath.exists():
        existing_content = filepath.read_text(encoding="utf-8")
        parsed_fm = _parse_frontmatter(existing_content)
        existing_tags = parsed_fm["meta"].get("tags", [])

    return existing_db_records, existing_records, existing_tags, filepath


def _resolve_merged_payload(
    *,
    existing: dict,
    extraction_data: dict,
    existing_db_records: list[dict],
    existing_records: list[dict],
    existing_tags: list[str],
    cluster_episodes: list[dict],
    cluster_ep_ids: list[str],
) -> tuple[tuple[str, str, list[str], list[dict]] | None, int]:
    """Resolve merged topic metadata/records with deterministic fallbacks."""
    new_records = extraction_data.get("records", [])
    merge_prompt = _build_merge_extraction_prompt(
        existing_records=existing_records,
        new_records=new_records,
        existing_title=existing["title"],
        existing_summary=existing["summary"],
        existing_tags=existing_tags,
    )

    merge_calls = 0
    try:
        merged_data, merge_calls = _llm_extract_with_validation(merge_prompt, cluster_episodes)
        merged_title = merged_data.get("title", existing["title"])
        merged_summary = merged_data.get("summary", existing["summary"])
        merged_tags = merged_data.get("tags", existing_tags)
        merged_records = merged_data.get("records", [])
    except Exception as e:
        logger.warning(
            "LLM merge failed for %s (%s); using deterministic merge fallback",
            existing["filename"],
            e,
        )
        merged_title, merged_summary, merged_tags, merged_records = _build_deterministic_merge_payload(
            existing_records=existing_records,
            new_records=new_records,
            existing_title=existing["title"],
            existing_summary=existing["summary"],
            existing_tags=existing_tags,
            extraction_data=extraction_data,
        )

    if not merged_records:
        logger.warning(
            "Merged result for %s had no records; retrying deterministic merge",
            existing["filename"],
        )
        merged_title, merged_summary, merged_tags, merged_records = _build_deterministic_merge_payload(
            existing_records=existing_records,
            new_records=new_records,
            existing_title=existing["title"],
            existing_summary=existing["summary"],
            existing_tags=existing_tags,
            extraction_data=extraction_data,
        )
        if not merged_records:
            logger.error("Deterministic merge also produced no records for %s", existing["filename"])
            increment_consolidation_attempts(cluster_ep_ids)
            return None, merge_calls

    # Guard: reject merge if LLM drastically reduced record count
    if len(existing_db_records) >= 4 and len(merged_records) < len(existing_db_records) * 0.5:
        logger.warning(
            "Merge for %s dropped too many records (%d -> %d); "
            "using deterministic merge fallback to prevent data loss.",
            existing["filename"],
            len(existing_db_records),
            len(merged_records),
        )
        merged_title, merged_summary, merged_tags, merged_records = _build_deterministic_merge_payload(
            existing_records=existing_records,
            new_records=new_records,
            existing_title=existing["title"],
            existing_summary=existing["summary"],
            existing_tags=existing_tags,
            extraction_data=extraction_data,
        )
        if len(existing_db_records) >= 4 and len(merged_records) < len(existing_db_records) * 0.5:
            logger.error(
                "Deterministic merge still dropped too many records for %s (%d -> %d)",
                existing["filename"],
                len(existing_db_records),
                len(merged_records),
            )
            increment_consolidation_attempts(cluster_ep_ids)
            return None, merge_calls

    return (merged_title, merged_summary, merged_tags, merged_records), merge_calls


def _merge_into_existing(
    existing: dict,
    extraction_data: dict,
    cluster_episodes: list[dict],
    cluster_ep_ids: list[str],
    confidence: float,
    cluster_scope: dict[str, str | None] | None = None,
) -> tuple[str, int]:
    """Merge new extracted records into an existing topic.

    Returns:
        Tuple of (status, api_calls) where status is 'updated' or 'failed'.

    Raises:
        Exception: Propagated from LLM or file I/O failures (caller handles).
    """
    if cluster_scope is None:
        cluster_scope = _episode_scope_row(cluster_episodes[0]) if cluster_episodes else _default_scope_row()

    state = _load_existing_topic_merge_state(existing, cluster_ep_ids=cluster_ep_ids)
    if state is None:
        return "failed", 0
    existing_db_records, existing_records, existing_tags, filepath = state

    merged_payload, merge_calls = _resolve_merged_payload(
        existing=existing,
        extraction_data=extraction_data,
        existing_db_records=existing_db_records,
        existing_records=existing_records,
        existing_tags=existing_tags,
        cluster_episodes=cluster_episodes,
        cluster_ep_ids=cluster_ep_ids,
    )
    if merged_payload is None:
        return "failed", merge_calls
    merged_title, merged_summary, merged_tags, merged_records = merged_payload
    new_records = extraction_data.get("records", [])

    # Detect silent drops: pre-merge records not preserved in merged output
    cfg = get_config()
    if cfg.MERGE_DROP_DETECTION_ENABLED:
        pre_merge = existing_records + new_records  # all claims before merge
        silent_drops = _detect_silent_drops(pre_merge, merged_records)
        for drop_idx, max_sim in silent_drops:
            dropped_rec = pre_merge[drop_idx]
            dropped_content = json.dumps(dropped_rec, default=str)
            insert_contradiction(
                topic_id=existing["id"],
                old_record_id=None,
                new_record_id=None,
                old_content=dropped_content,
                new_content="",
                resolution="silent_drop",
                reason=f"max_similarity={max_sim:.3f}",
            )
        if silent_drops:
            logger.warning(
                "Merge for %s: %d/%d pre-merge records potentially dropped (threshold=%.2f)",
                existing["filename"],
                len(silent_drops),
                len(pre_merge),
                cfg.MERGE_DROP_SIMILARITY_THRESHOLD,
            )

    # Detect contradictions between new extraction and existing records
    new_for_detection = []
    for rec in new_records:
        det = dict(rec)
        det["embedding_text"] = _embedding_text_for_record(rec)
        new_for_detection.append(det)

    contradictions = _detect_contradictions(new_for_detection, existing_db_records)
    contradicted_existing_ids = {ex_id for _, ex_id in contradictions}

    if contradicted_existing_ids:
        contradicted_keys: set[tuple] = set()
        for row in existing_db_records:
            if row["id"] not in contradicted_existing_ids:
                continue
            content = row.get("content", {})
            if isinstance(content, str):
                try:
                    content = json.loads(content)
                except (json.JSONDecodeError, TypeError):
                    content = {}
            if isinstance(content, dict):
                contradicted_keys.add(_record_dedup_key(content))
        if contradicted_keys:
            merged_records = [
                rec for rec in merged_records
                if _record_dedup_key(rec) not in contradicted_keys
            ]
            if not merged_records:
                logger.warning(
                    "All merged records for %s were contradicted old claims; "
                    "falling back to deduped new records only.",
                    existing["filename"],
                )
                merged_records = _dedupe_records(new_records)
                if not merged_records:
                    increment_consolidation_attempts(cluster_ep_ids)
                    return "failed", merge_calls

    now_ts = datetime.now(timezone.utc).isoformat()

    # Log contradictions to audit log before expiring
    existing_by_id = {r["id"]: r for r in existing_db_records}
    pm = get_plugin_manager()
    for new_idx, ex_id in contradictions:
        old_rec = existing_by_id.get(ex_id, {})
        old_content = old_rec.get("content", "")
        old_content_dict = _coerce_content_dict(old_content)
        if isinstance(old_content, dict):
            old_content = json.dumps(old_content)
        new_content = json.dumps(new_for_detection[new_idx])
        contradiction_id: str | None = None
        try:
            contradiction_id = insert_contradiction(
                topic_id=existing["id"],
                old_record_id=ex_id,
                new_record_id=None,
                old_content=old_content,
                new_content=new_content,
                resolution="expired_old",
            )
            pm.fire(
                "on_contradiction",
                topic_filename=existing["filename"],
                old_content=old_content,
                new_content=new_content,
            )
        except Exception as e:
            logger.warning("Failed to log contradiction: %s", e)

        # Emit claim-level contradiction linkage and events (best-effort).
        new_claim_id = _materialize_claim_for_record(
            new_records[new_idx],
            topic_id=existing["id"],
            source_episode_ids=cluster_ep_ids,
            source_record_id=None,
            confidence=confidence,
            valid_from=now_ts,
            event_type=None,
        )
        old_claim_id = None
        if old_content_dict:
            old_claim_id = _materialize_claim_for_record(
                old_content_dict,
                topic_id=existing["id"],
                source_episode_ids=[],
                source_record_id=ex_id,
                confidence=float(old_rec.get("confidence", confidence) or confidence),
                valid_from=old_rec.get("valid_from") or old_rec.get("created_at"),
                event_type=None,
            )

        if new_claim_id and old_claim_id:
            contradiction_details = {
                "topic_id": existing["id"],
                "topic_filename": existing["filename"],
                "old_record_id": ex_id,
                "new_record_index": new_idx,
            }
            if contradiction_id:
                contradiction_details["contradiction_id"] = contradiction_id
            try:
                insert_claim_edge(
                    from_claim_id=new_claim_id,
                    to_claim_id=old_claim_id,
                    edge_type="contradicts",
                    details=contradiction_details,
                )
            except Exception as e:
                logger.warning(
                    "Failed to insert contradiction edge (%s -> %s): %s",
                    new_claim_id,
                    old_claim_id,
                    e,
                )

            for contradiction_claim_id, role in ((new_claim_id, "new"), (old_claim_id, "old")):
                try:
                    insert_claim_event(
                        contradiction_claim_id,
                        event_type="contradiction",
                        details={**contradiction_details, "role": role},
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to insert contradiction claim event for %s: %s",
                        contradiction_claim_id,
                        e,
                    )

    # Reduce confidence by 10% when contradictions are detected
    if contradicted_existing_ids:
        confidence = confidence * 0.9

    # Expire contradicted records instead of soft-deleting them
    for ex_id in contradicted_existing_ids:
        expire_record(ex_id, valid_until=now_ts)
        logger.info("Expired contradicted record %s in topic %s", ex_id, existing["filename"])
        old_rec = existing_by_id.get(ex_id, {})
        old_content_dict = _coerce_content_dict(old_rec.get("content", {}))
        if not old_content_dict:
            continue
        expired_claim_id = _materialize_claim_for_record(
            old_content_dict,
            topic_id=existing["id"],
            source_episode_ids=[],
            source_record_id=ex_id,
            confidence=float(old_rec.get("confidence", confidence) or confidence),
            valid_from=old_rec.get("valid_from") or old_rec.get("created_at"),
            event_type=None,
        )
        if not expired_claim_id:
            continue
        try:
            expire_claim(expired_claim_id, valid_until=now_ts)
        except Exception as e:
            logger.warning("Failed to expire claim %s for record %s: %s", expired_claim_id, ex_id, e)
        try:
            insert_claim_event(
                expired_claim_id,
                event_type="expire",
                details={
                    "topic_id": existing["id"],
                    "topic_filename": existing["filename"],
                    "record_id": ex_id,
                    "reason": "contradiction",
                },
            )
        except Exception as e:
            logger.warning("Failed to insert expire claim event for %s: %s", expired_claim_id, e)

    # Soft-delete remaining old records (non-contradicted ones replaced by merge)
    non_contradicted_ids = [
        r["id"] for r in existing_db_records if r["id"] not in contradicted_existing_ids
    ]
    # Build merged record rows.
    # When contradictions were detected, mark ALL merged records with valid_from.
    # Rationale: the LLM rewrites and deduplicates records during merge, so there
    # is no reliable 1:1 mapping between new_records indices and merged_records.
    # Marking all merged records timestamps when this version of the topic was
    # established, which is the correct conservative approach for temporal tracking.
    has_contradictions = len(contradicted_existing_ids) > 0
    record_rows = []
    for rec in merged_records:
        row = {
            "record_type": rec.get("type", "fact"),
            "content": rec,
            "embedding_text": _embedding_text_for_record(rec),
            "confidence": confidence,
        }
        if has_contradictions:
            row["valid_from"] = now_ts
        record_rows.append(row)
    # Replace non-contradicted records and insert merged ones atomically so a
    # failure during insert does not leave the topic with deleted records.
    with get_connection() as conn:
        if non_contradicted_ids:
            placeholders = ",".join("?" for _ in non_contradicted_ids)
            conn.execute(
                f"UPDATE knowledge_records SET deleted = 1, updated_at = ? WHERE id IN ({placeholders}) AND deleted = 0",  # nosec B608
                [now_ts, *non_contradicted_ids],
            )
        inserted_record_ids = insert_knowledge_records(
            existing["id"],
            record_rows,
            source_episodes=cluster_ep_ids,
            scope=cluster_scope,
            conn=conn,
        )
    _emit_claims_for_records(
        merged_records,
        topic_id=existing["id"],
        source_episode_ids=cluster_ep_ids,
        source_record_ids=inserted_record_ids,
        confidence=confidence,
        default_valid_from=now_ts if has_contradictions else None,
        event_type="update",
    )

    # Render markdown
    if get_config().RENDER_MARKDOWN:
        _version_knowledge_file(filepath)
        md = _render_markdown_from_records(
            merged_title, merged_summary, merged_tags, confidence, merged_records
        )
        filepath.write_text(md, encoding="utf-8")

    upsert_knowledge_topic(
        filename=existing["filename"],
        title=merged_title,
        summary=merged_summary,
        source_episodes=cluster_ep_ids,
        fact_count=len(merged_records),
        confidence=confidence,
        scope=cluster_scope,
    )
    mark_consolidated(cluster_ep_ids, existing["filename"])
    topic_cache.invalidate()
    record_cache.invalidate()
    claim_cache.invalidate()
    logger.info(
        "Merged into existing topic: %s (%d records)", existing["filename"], len(merged_records)
    )
    get_plugin_manager().fire(
        "on_topic_updated",
        filename=existing["filename"],
        title=merged_title,
        record_count=len(merged_records),
    )
    return "updated", merge_calls


# ── Cluster processing ────────────────────────────────────────────────────────


def _process_cluster(
    cluster_id: int,
    cluster_items: list[tuple[dict, int]],
    sim_matrix: np.ndarray,
    cluster_confidences: list[float],
) -> dict:
    """Process a single cluster: extract records via LLM, create or merge topic.

    Returns:
        Dict with keys: status ('created'|'updated'|'failed'), api_calls (int),
        and optionally failed_ep_ids (list[str]).
    """
    cfg = get_config()
    if len(cluster_items) > cfg.CONSOLIDATION_MAX_CLUSTER_SIZE:
        sorted_items = sorted(
            cluster_items,
            key=lambda item: item[0].get("surprise_score", 0.5),
            reverse=True,
        )
        kept = sorted_items[: cfg.CONSOLIDATION_MAX_CLUSTER_SIZE]
        dropped = sorted_items[cfg.CONSOLIDATION_MAX_CLUSTER_SIZE :]
        dropped_ids = [ep["id"] for ep, _ in dropped]
        increment_consolidation_attempts(dropped_ids)
        logger.warning(
            "Cluster %d truncated from %d to %d episodes; %d dropped episodes "
            "had consolidation_attempts incremented",
            cluster_id,
            len(cluster_items),
            len(kept),
            len(dropped_ids),
        )
        cluster_items = kept

    cluster_episodes = [ep for ep, _ in cluster_items]
    cluster_indices = [idx for _, idx in cluster_items]

    confidence = _compute_cluster_confidence(
        cluster_episodes,
        sim_matrix,
        cluster_indices,
    )
    cluster_confidences.append(confidence)

    all_tags: list[str] = []
    for ep in cluster_episodes:
        all_tags.extend(parse_json_list(ep["tags"]))
    tag_counts = Counter(all_tags).most_common(5)
    tag_summary = ", ".join(f"{t}({c})" for t, c in tag_counts) if tag_counts else "none"

    prompt = _build_extraction_prompt(cluster_episodes, confidence, tag_summary)

    logger.info(
        "Extracting records from cluster %d (%d episodes)...", cluster_id, len(cluster_episodes)
    )
    cluster_ep_ids = [ep["id"] for ep in cluster_episodes]
    cluster_scope = _episode_scope_row(cluster_episodes[0]) if cluster_episodes else _default_scope_row()
    api_calls = 0

    try:
        extraction_data, calls = _llm_extract_with_validation(prompt, cluster_episodes)
        api_calls += calls
    except Exception as e:
        logger.error("LLM extraction failed for cluster %d: %s", cluster_id, e, exc_info=True)
        increment_consolidation_attempts(cluster_ep_ids)
        return {"status": "failed", "api_calls": api_calls, "failed_ep_ids": cluster_ep_ids}

    title = extraction_data.get("title", f"Topic {cluster_id}")
    summary = extraction_data.get("summary", "")
    tags = extraction_data.get("tags", [t for t, _ in tag_counts])
    records = extraction_data.get("records", [])

    existing = _find_similar_topic(title, summary, tags, scope=cluster_scope)

    if existing:
        try:
            status, merge_calls = _merge_into_existing(
                existing,
                extraction_data,
                cluster_episodes,
                cluster_ep_ids,
                confidence,
                cluster_scope,
            )
            api_calls += merge_calls
            if status == "failed":
                return {"status": "failed", "api_calls": api_calls, "failed_ep_ids": cluster_ep_ids}
            return {"status": status, "api_calls": api_calls}
        except Exception as e:
            logger.error("Merge failed for topic %s: %s", existing["filename"], e, exc_info=True)
            increment_consolidation_attempts(cluster_ep_ids)
            return {"status": "failed", "api_calls": api_calls, "failed_ep_ids": cluster_ep_ids}
    else:
        scope_prefix = _scope_filename_prefix(cluster_scope)
        base_slug = _slugify(title)
        if scope_prefix:
            base_slug = f"{scope_prefix}__{base_slug}"
        filename = base_slug + ".md"
        filepath = cfg.KNOWLEDGE_DIR / filename
        counter = 2
        while filepath.exists():
            filename = f"{base_slug}_{counter}.md"
            filepath = cfg.KNOWLEDGE_DIR / filename
            counter += 1

        # Store records in DB
        topic_id = upsert_knowledge_topic(
            filename=filename,
            title=title,
            summary=summary,
            source_episodes=cluster_ep_ids,
            fact_count=len(records),
            confidence=confidence,
            scope=cluster_scope,
        )
        record_rows = []
        for rec in records:
            record_rows.append(
                {
                    "record_type": rec.get("type", "fact"),
                    "content": rec,
                    "embedding_text": _embedding_text_for_record(rec),
                    "confidence": confidence,
                }
            )
        inserted_record_ids = insert_knowledge_records(
            topic_id,
            record_rows,
            source_episodes=cluster_ep_ids,
            scope=cluster_scope,
        )
        _emit_claims_for_records(
            records,
            topic_id=topic_id,
            source_episode_ids=cluster_ep_ids,
            source_record_ids=inserted_record_ids,
            confidence=confidence,
            event_type="create",
        )

        # Render markdown file
        if cfg.RENDER_MARKDOWN:
            md = _render_markdown_from_records(title, summary, tags, confidence, records)
            filepath.write_text(md, encoding="utf-8")

        mark_consolidated(cluster_ep_ids, filename)
        topic_cache.invalidate()
        record_cache.invalidate()
        claim_cache.invalidate()
        logger.info("Created new topic: %s (%d records)", filename, len(records))
        get_plugin_manager().fire(
            "on_topic_created",
            filename=filename,
            title=title,
            record_count=len(records),
        )
        return {"status": "created", "api_calls": api_calls}


def _build_scope_isolated_similarity(
    valid_episodes: list[dict],
    vectors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build similarity/distance matrices with scope isolation applied."""
    sim_matrix = vectors @ vectors.T
    np.fill_diagonal(sim_matrix, 1.0)
    dist_matrix = 1.0 - sim_matrix
    dist_matrix = np.clip(dist_matrix, 0, 2)

    # Scope isolation guard: never cluster episodes from different
    # namespace/project/client/agent/session scopes together.
    scope_keys = [_scope_key(_episode_scope_row(ep)) for ep in valid_episodes]
    for i in range(len(scope_keys)):
        for j in range(i + 1, len(scope_keys)):
            if scope_keys[i] != scope_keys[j]:
                dist_matrix[i, j] = 2.0
                dist_matrix[j, i] = 2.0
                sim_matrix[i, j] = 0.0
                sim_matrix[j, i] = 0.0

    return sim_matrix, dist_matrix


def _build_clusters_from_distance(
    valid_episodes: list[dict],
    dist_matrix: np.ndarray,
) -> tuple[dict[int, list[tuple[dict, int]]], dict[int, list[tuple[dict, int]]]]:
    """Build all clusters and the subset meeting min-size threshold."""
    cfg = get_config()
    condensed = squareform(dist_matrix, checks=False)
    linkage_matrix = linkage(condensed, method="average")
    labels = fcluster(
        linkage_matrix,
        t=1.0 - cfg.CONSOLIDATION_CLUSTER_THRESHOLD,
        criterion="distance",
    )

    clusters: dict[int, list[tuple[dict, int]]] = {}
    for idx, (ep, label) in enumerate(zip(valid_episodes, labels)):
        clusters.setdefault(int(label), []).append((ep, idx))

    valid_clusters = {
        cluster_id: items
        for cluster_id, items in clusters.items()
        if len(items) >= cfg.CONSOLIDATION_MIN_CLUSTER_SIZE
    }
    return clusters, valid_clusters


def _run_cluster_processing_loop(
    valid_clusters: dict[int, list[tuple[dict, int]]],
    sim_matrix: np.ndarray,
    cfg,
) -> tuple[int, int, int, int, list[float], list[str], float]:
    """Process clusters and aggregate counters for the consolidation report."""
    topics_created = 0
    topics_updated = 0
    clusters_failed = 0
    consecutive_failures = 0
    api_calls = 0
    run_start = time.monotonic()
    cluster_confidences: list[float] = []
    all_failed_ep_ids: list[str] = []

    for cluster_id, cluster_items in valid_clusters.items():
        elapsed = time.monotonic() - run_start
        if elapsed > cfg.CONSOLIDATION_MAX_DURATION:
            logger.warning(
                "Consolidation max duration (%.0fs) exceeded after %.0fs, stopping early",
                cfg.CONSOLIDATION_MAX_DURATION,
                elapsed,
            )
            break

        if consecutive_failures >= 3:
            logger.warning(
                "3 consecutive cluster failures — aborting consolidation "
                "(backend likely unavailable)"
            )
            break

        cluster_result = _process_cluster(
            cluster_id,
            cluster_items,
            sim_matrix,
            cluster_confidences,
        )
        api_calls += cluster_result["api_calls"]
        if cluster_result["status"] == "created":
            topics_created += 1
            consecutive_failures = 0
        elif cluster_result["status"] == "updated":
            topics_updated += 1
            consecutive_failures = 0
        elif cluster_result["status"] == "failed":
            clusters_failed += 1
            consecutive_failures += 1
            all_failed_ep_ids.extend(cluster_result.get("failed_ep_ids", []))

    return (
        topics_created,
        topics_updated,
        clusters_failed,
        api_calls,
        cluster_confidences,
        all_failed_ep_ids,
        run_start,
    )


def _maybe_prune_and_compact(vs: VectorStore, cfg) -> list[dict]:
    """Handle pruning/tombstone compaction and return pruned episode rows."""
    prunable: list[dict] = []
    if cfg.CONSOLIDATION_PRUNE_ENABLED:
        prunable = get_prunable_episodes(days=cfg.CONSOLIDATION_PRUNE_AFTER_DAYS)
        if prunable:
            prune_ids = [ep["id"] for ep in prunable]
            mark_pruned(prune_ids)
            removed = vs.remove_batch(prune_ids)
            logger.info(
                "Pruned %d old episodes (%d vectors tombstoned)",
                len(prunable),
                removed,
            )
            get_plugin_manager().fire("on_prune", episode_ids=prune_ids)
    else:
        logger.debug("Pruning disabled (set prune_enabled = true in config to enable)")

    logger.info(
        "FAISS tombstone ratio: %.1f%% (compaction threshold: %.1f%%)",
        vs.tombstone_ratio * 100,
        cfg.FAISS_COMPACTION_THRESHOLD * 100,
    )
    if vs.tombstone_ratio >= cfg.FAISS_COMPACTION_THRESHOLD:
        compacted = vs.compact()
        logger.info("Compacted %d tombstoned vectors from FAISS index", compacted)

    VectorStore.signal_reload()
    return prunable


# ── Main consolidation loop ───────────────────────────────────────────────────


def run_consolidation(vector_store: VectorStore | None = None) -> ConsolidationReport:
    """Main consolidation loop.

    Args:
        vector_store: Existing VectorStore instance to reuse. If None, creates
            a new one (backwards compatible for CLI/scheduled task usage).
    """
    cfg = get_config()
    ensure_schema()
    cfg.KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    cfg.KNOWLEDGE_VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
    cfg.CONSOLIDATION_LOG_DIR.mkdir(parents=True, exist_ok=True)

    topic_cache.invalidate()
    record_cache.invalidate()
    claim_cache.invalidate()

    # Reset episodes stuck at max attempts whose last retry was >24h ago,
    # so they get another chance after backend recovery.
    reset_count = reset_stale_consolidation_attempts(max_attempts=cfg.CONSOLIDATION_MAX_ATTEMPTS)
    if reset_count:
        logger.info("Reset consolidation_attempts for %d stale episodes", reset_count)

    run_id = start_consolidation_run()
    logger.info("Consolidation run %s started", run_id)

    try:
        episodes = get_unconsolidated_episodes(
            limit=cfg.CONSOLIDATION_MAX_EPISODES_PER_RUN,
            max_attempts=cfg.CONSOLIDATION_MAX_ATTEMPTS,
        )

        get_plugin_manager().fire(
            "on_consolidation_start",
            run_id=run_id,
            episode_count=len(episodes),
        )

        if len(episodes) < cfg.CONSOLIDATION_MIN_CLUSTER_SIZE:
            logger.info("Only %d episodes — nothing to consolidate.", len(episodes))
            complete_consolidation_run(
                run_id, status=RUN_STATUS_COMPLETED, episodes_processed=len(episodes)
            )
            early_report: ConsolidationReport = {
                "status": "nothing_to_consolidate",
                "episodes": len(episodes),
            }
            get_plugin_manager().fire("on_consolidation_complete", report=early_report)
            return early_report

        logger.info("Loaded %d unconsolidated episodes", len(episodes))

        local_vs = vector_store is None
        vs = vector_store if vector_store is not None else VectorStore()
        try:
            episode_ids = [ep["id"] for ep in episodes]
            batch_result = vs.reconstruct_batch(episode_ids)

            if batch_result is None:
                logger.warning("No vectors found for episodes — aborting.")
                complete_consolidation_run(
                    run_id, status=RUN_STATUS_FAILED, error_message="No vectors in FAISS"
                )
                early_report_nv: ConsolidationReport = {
                    "status": "error",
                    "message": "No vectors found",
                    "run_id": run_id,
                }
                get_plugin_manager().fire("on_consolidation_complete", report=early_report_nv)
                return early_report_nv

            found_ids, vectors = batch_result

            id_to_episode = {ep["id"]: ep for ep in episodes}
            valid_episodes = [id_to_episode[uid] for uid in found_ids if uid in id_to_episode]

            sim_matrix, dist_matrix = _build_scope_isolated_similarity(valid_episodes, vectors)

            if len(valid_episodes) < 2:
                logger.info("Only 1 valid episode — skipping clustering.")
                complete_consolidation_run(
                    run_id, status=RUN_STATUS_COMPLETED, episodes_processed=1
                )
                early_report_fe: ConsolidationReport = {"status": "too_few_episodes"}
                get_plugin_manager().fire("on_consolidation_complete", report=early_report_fe)
                return early_report_fe

            clusters, valid_clusters = _build_clusters_from_distance(valid_episodes, dist_matrix)

            logger.info(
                "Formed %d clusters, %d valid (>=%d episodes)",
                len(clusters),
                len(valid_clusters),
                cfg.CONSOLIDATION_MIN_CLUSTER_SIZE,
            )

            (
                topics_created,
                topics_updated,
                clusters_failed,
                api_calls,
                cluster_confidences,
                all_failed_ep_ids,
                _run_start,
            ) = _run_cluster_processing_loop(valid_clusters, sim_matrix, cfg)

            _update_index()

            surprise_adjusted = _adjust_surprise_scores()

            prunable = _maybe_prune_and_compact(vs, cfg)

            report_ts = datetime.now(timezone.utc)
            report: ConsolidationReport = {
                "run_id": run_id,
                "timestamp": report_ts.isoformat(),
                "episodes_loaded": len(episodes),
                "episodes_with_vectors": len(valid_episodes),
                "clusters_total": len(clusters),
                "clusters_valid": len(valid_clusters),
                "clusters_failed": clusters_failed,
                "topics_created": topics_created,
                "topics_updated": topics_updated,
                "episodes_pruned": len(prunable),
                "surprise_adjusted": surprise_adjusted,
                "api_calls": api_calls,
                "failed_episode_ids": all_failed_ep_ids,
            }

            report_path = cfg.CONSOLIDATION_LOG_DIR / (
                f"{report_ts.strftime('%Y-%m-%dT%H-%M-%S')}.json"
            )
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

            complete_consolidation_run(
                run_id,
                status=RUN_STATUS_COMPLETED,
                episodes_processed=len(valid_episodes),
                clusters_formed=len(valid_clusters),
                topics_created=topics_created,
                topics_updated=topics_updated,
                episodes_pruned=len(prunable),
            )

            avg_conf = 0.0
            if cluster_confidences:
                avg_conf = round(sum(cluster_confidences) / len(cluster_confidences), 4)

            try:
                insert_consolidation_metrics(
                    run_id=run_id,
                    clusters_succeeded=topics_created + topics_updated,
                    clusters_failed=clusters_failed,
                    avg_confidence=avg_conf,
                    episodes_processed=len(episodes),
                    duration_seconds=round(time.monotonic() - _run_start, 2),
                    api_calls=api_calls,
                    topics_created=topics_created,
                    topics_updated=topics_updated,
                    episodes_pruned=report.get("episodes_pruned") or 0,
                )
            except Exception as e:
                logger.warning("Failed to write consolidation metrics: %s", e)

            logger.info(
                "Consolidation complete: %d episodes -> %d clusters -> "
                "%d new topics, %d updated, %d failed, %d pruned, "
                "%d surprise-adjusted",
                len(valid_episodes),
                len(valid_clusters),
                topics_created,
                topics_updated,
                clusters_failed,
                len(prunable),
                surprise_adjusted,
            )
            get_plugin_manager().fire("on_consolidation_complete", report=report)
            return report
        finally:
            if local_vs:
                vs._save()

    except Exception as e:
        logger.exception("Consolidation failed: %s", e)
        complete_consolidation_run(run_id, status=RUN_STATUS_FAILED, error_message=str(e))
        error_report: ConsolidationReport = {
            "status": "error",
            "message": str(e),
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "topics_created": 0,
            "topics_updated": 0,
            "episodes_pruned": 0,
            "clusters_failed": 0,
            "api_calls": 0,
            "episodes_loaded": 0,
            "episodes_with_vectors": 0,
            "clusters_total": 0,
            "clusters_valid": 0,
            "surprise_adjusted": 0,
            "failed_episode_ids": [],
        }
        get_plugin_manager().fire("on_consolidation_complete", report=error_report)
        return error_report


# ── Index update ──────────────────────────────────────────────────────────────


def _update_index() -> None:
    topics = get_all_knowledge_topics()
    lines = ["# Knowledge Base Index\n"]
    lines.append(f"*Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*\n")
    lines.append(f"*{len(topics)} topics*\n")

    for topic in topics:
        lines.append(f"## [{topic['title']}]({topic['filename']})")
        lines.append(f"{topic['summary']}")
        lines.append(
            f"*Updated: {topic['updated_at'][:10]} | "
            f"{topic['fact_count']} facts | "
            f"Confidence: {topic['confidence']} | "
            f"Accessed: {topic['access_count']}x*\n"
        )

    index_path = get_config().KNOWLEDGE_DIR / "index.md"
    index_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Updated index.md with %d topics", len(topics))
