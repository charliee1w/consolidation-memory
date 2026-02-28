"""Main consolidation orchestration engine.

Includes: knowledge-file versioning, contradiction detection, topic merging,
cluster processing, the run_consolidation entry point, and index updating.
"""

import json
import logging
import re
import shutil
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from consolidation_memory import record_cache, topic_cache
from consolidation_memory.config import get_config
from consolidation_memory.database import (
    complete_consolidation_run,
    ensure_schema,
    expire_record,
    get_all_knowledge_topics,
    get_prunable_episodes,
    get_records_by_topic,
    get_unconsolidated_episodes,
    increment_consolidation_attempts,
    insert_consolidation_metrics,
    insert_knowledge_records,
    mark_consolidated,
    mark_pruned,
    reset_stale_consolidation_attempts,
    soft_delete_records_by_ids,
    start_consolidation_run,
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
)
from consolidation_memory.consolidation.scoring import _adjust_surprise_scores
from consolidation_memory.types import ConsolidationReport

logger = logging.getLogger(__name__)


# ── Knowledge versioning ─────────────────────────────────────────────────────


def _version_knowledge_file(filepath: Path) -> None:
    if not filepath.exists():
        return

    get_config().KNOWLEDGE_VERSIONS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
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
    for old in existing_versions[get_config().KNOWLEDGE_MAX_VERSIONS:]:
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
        return [
            (new_idx, existing_records[ex_idx]["id"])
            for new_idx, ex_idx in candidate_pairs
        ]

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
        raw = raw.strip()
        raw = re.sub(r"^```(?:json)?\s*\n?", "", raw)
        raw = re.sub(r"\n?```\s*$", "", raw)
        verdicts = json.loads(raw.strip())
    except Exception as e:
        logger.warning("LLM contradiction verification failed: %s", e)
        return []

    if not isinstance(verdicts, list) or len(verdicts) != len(candidate_pairs):
        logger.warning(
            "LLM returned %d verdicts for %d pairs; expected exact match. Skipping.",
            len(verdicts) if isinstance(verdicts, list) else 0,
            len(candidate_pairs),
        )
        return []

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


# ── Topic merging ─────────────────────────────────────────────────────────────


def _merge_into_existing(
    existing: dict,
    extraction_data: dict,
    cluster_episodes: list[dict],
    cluster_ep_ids: list[str],
    confidence: float,
) -> tuple[str, int]:
    """Merge new extracted records into an existing topic.

    Returns:
        Tuple of (status, api_calls) where status is 'updated' or 'failed'.

    Raises:
        Exception: Propagated from LLM or file I/O failures (caller handles).
    """
    # Load existing records from DB
    existing_db_records = get_records_by_topic(existing["id"])
    existing_records = []
    for r in existing_db_records:
        try:
            content = json.loads(r["content"]) if isinstance(r["content"], str) else r["content"]
        except (json.JSONDecodeError, TypeError):
            content = {"type": "fact", "subject": "?", "info": r.get("content", "")}
        existing_records.append(content)

    new_records = extraction_data.get("records", [])

    # Parse existing topic metadata
    existing_tags = []
    filepath = get_config().KNOWLEDGE_DIR / existing["filename"]
    if filepath.exists():
        existing_content = filepath.read_text(encoding="utf-8")
        parsed_fm = _parse_frontmatter(existing_content)
        existing_tags = parsed_fm["meta"].get("tags", [])

    merge_prompt = _build_merge_extraction_prompt(
        existing_records=existing_records,
        new_records=new_records,
        existing_title=existing["title"],
        existing_summary=existing["summary"],
        existing_tags=existing_tags,
    )

    try:
        merged_data, merge_calls = _llm_extract_with_validation(merge_prompt, cluster_episodes)
    except ValueError as e:
        logger.error("LLM merge returned invalid JSON for %s: %s", existing["filename"], e)
        increment_consolidation_attempts(cluster_ep_ids)
        return "failed", 1

    merged_title = merged_data.get("title", existing["title"])
    merged_summary = merged_data.get("summary", existing["summary"])
    merged_tags = merged_data.get("tags", existing_tags)
    merged_records = merged_data.get("records", [])

    if not merged_records:
        logger.error(
            "LLM merge produced no records for %s; original preserved.", existing["filename"]
        )
        increment_consolidation_attempts(cluster_ep_ids)
        return "failed", merge_calls

    # Detect contradictions between new extraction and existing records
    new_for_detection = []
    for rec in new_records:
        det = dict(rec)
        det["embedding_text"] = _embedding_text_for_record(rec)
        new_for_detection.append(det)

    contradictions = _detect_contradictions(new_for_detection, existing_db_records)
    contradicted_existing_ids = {ex_id for _, ex_id in contradictions}
    contradicting_new_indices = {new_idx for new_idx, _ in contradictions}

    now_ts = datetime.now(timezone.utc).isoformat()

    # Expire contradicted records instead of soft-deleting them
    for ex_id in contradicted_existing_ids:
        expire_record(ex_id, valid_until=now_ts)
        logger.info("Expired contradicted record %s in topic %s", ex_id, existing["filename"])

    # Soft-delete remaining old records (non-contradicted ones replaced by merge)
    non_contradicted_ids = [
        r["id"] for r in existing_db_records if r["id"] not in contradicted_existing_ids
    ]
    if non_contradicted_ids:
        soft_delete_records_by_ids(non_contradicted_ids)

    # Build merged record rows, marking those that contradict with valid_from
    record_rows = []
    for i, rec in enumerate(merged_records):
        row = {
            "record_type": rec.get("type", "fact"),
            "content": rec,
            "embedding_text": _embedding_text_for_record(rec),
            "confidence": confidence,
        }
        if contradicting_new_indices:
            row["valid_from"] = now_ts
        record_rows.append(row)
    insert_knowledge_records(existing["id"], record_rows, source_episodes=cluster_ep_ids)

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
    )
    mark_consolidated(cluster_ep_ids, existing["filename"])
    topic_cache.invalidate()
    record_cache.invalidate()
    logger.info(
        "Merged into existing topic: %s (%d records)", existing["filename"], len(merged_records)
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
        cluster_items = sorted(
            cluster_items,
            key=lambda item: item[0].get("surprise_score", 0.5),
            reverse=True,
        )[:cfg.CONSOLIDATION_MAX_CLUSTER_SIZE]

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
        raw = ep["tags"]
        all_tags.extend(json.loads(raw) if isinstance(raw, str) else raw)
    tag_counts = Counter(all_tags).most_common(5)
    tag_summary = ", ".join(f"{t}({c})" for t, c in tag_counts) if tag_counts else "none"

    prompt = _build_extraction_prompt(cluster_episodes, confidence, tag_summary)

    logger.info(
        "Extracting records from cluster %d (%d episodes)...", cluster_id, len(cluster_episodes)
    )
    cluster_ep_ids = [ep["id"] for ep in cluster_episodes]
    api_calls = 0

    try:
        extraction_data, calls = _llm_extract_with_validation(prompt, cluster_episodes)
        api_calls += calls
    except (Exception, ValueError) as e:
        logger.error(
            "LLM extraction failed for cluster %d: %s", cluster_id, e, exc_info=True
        )
        increment_consolidation_attempts(cluster_ep_ids)
        return {"status": "failed", "api_calls": api_calls, "failed_ep_ids": cluster_ep_ids}

    title = extraction_data.get("title", f"Topic {cluster_id}")
    summary = extraction_data.get("summary", "")
    tags = extraction_data.get("tags", [t for t, _ in tag_counts])
    records = extraction_data.get("records", [])

    existing = _find_similar_topic(title, summary, tags)

    if existing:
        try:
            status, merge_calls = _merge_into_existing(
                existing,
                extraction_data,
                cluster_episodes,
                cluster_ep_ids,
                confidence,
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
        base_slug = _slugify(title)
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
        insert_knowledge_records(topic_id, record_rows, source_episodes=cluster_ep_ids)

        # Render markdown file
        if cfg.RENDER_MARKDOWN:
            md = _render_markdown_from_records(title, summary, tags, confidence, records)
            filepath.write_text(md, encoding="utf-8")

        mark_consolidated(cluster_ep_ids, filename)
        topic_cache.invalidate()
        record_cache.invalidate()
        logger.info("Created new topic: %s (%d records)", filename, len(records))
        return {"status": "created", "api_calls": api_calls}


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

    # Reset episodes stuck at max attempts whose last retry was >24h ago,
    # so they get another chance after backend recovery.
    reset_count = reset_stale_consolidation_attempts(max_attempts=cfg.CONSOLIDATION_MAX_ATTEMPTS)
    if reset_count:
        logger.info("Reset consolidation_attempts for %d stale episodes", reset_count)

    run_id = start_consolidation_run()
    logger.info("Consolidation run %s started", run_id)

    try:
        episodes = get_unconsolidated_episodes(
            limit=cfg.CONSOLIDATION_MAX_EPISODES_PER_RUN, max_attempts=cfg.CONSOLIDATION_MAX_ATTEMPTS
        )

        if len(episodes) < cfg.CONSOLIDATION_MIN_CLUSTER_SIZE:
            logger.info("Only %d episodes — nothing to consolidate.", len(episodes))
            complete_consolidation_run(
                run_id, status="completed", episodes_processed=len(episodes)
            )
            return {"status": "nothing_to_consolidate", "episodes": len(episodes)}

        logger.info("Loaded %d unconsolidated episodes", len(episodes))

        vs = vector_store if vector_store is not None else VectorStore()
        episode_ids = [ep["id"] for ep in episodes]
        batch_result = vs.reconstruct_batch(episode_ids)

        if batch_result is None:
            logger.warning("No vectors found for episodes — aborting.")
            complete_consolidation_run(
                run_id, status="failed", error_message="No vectors in FAISS"
            )
            return {"status": "error", "message": "No vectors found"}

        found_ids, vectors = batch_result

        id_to_episode = {ep["id"]: ep for ep in episodes}
        valid_episodes = [id_to_episode[uid] for uid in found_ids if uid in id_to_episode]

        sim_matrix = vectors @ vectors.T
        np.fill_diagonal(sim_matrix, 1.0)
        dist_matrix = 1.0 - sim_matrix
        dist_matrix = np.clip(dist_matrix, 0, 2)

        if len(valid_episodes) < 2:
            logger.info("Only 1 valid episode — skipping clustering.")
            complete_consolidation_run(run_id, status="completed", episodes_processed=1)
            return {"status": "too_few_episodes"}

        condensed = squareform(dist_matrix, checks=False)
        Z = linkage(condensed, method="average")
        labels = fcluster(Z, t=1.0 - cfg.CONSOLIDATION_CLUSTER_THRESHOLD, criterion="distance")

        clusters: dict[int, list[tuple[dict, int]]] = {}
        for idx, (ep, label) in enumerate(zip(valid_episodes, labels)):
            clusters.setdefault(int(label), []).append((ep, idx))

        valid_clusters = {
            k: v for k, v in clusters.items() if len(v) >= cfg.CONSOLIDATION_MIN_CLUSTER_SIZE
        }

        logger.info(
            "Formed %d clusters, %d valid (>=%d episodes)",
            len(clusters),
            len(valid_clusters),
            cfg.CONSOLIDATION_MIN_CLUSTER_SIZE,
        )

        topics_created = 0
        topics_updated = 0
        clusters_failed = 0
        consecutive_failures = 0
        api_calls = 0
        _run_start = time.monotonic()
        cluster_confidences: list[float] = []
        all_failed_ep_ids: list[str] = []

        for cluster_id, cluster_items in valid_clusters.items():
            elapsed = time.monotonic() - _run_start
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

        _update_index()

        surprise_adjusted = _adjust_surprise_scores()

        prunable = []
        if cfg.CONSOLIDATION_PRUNE_ENABLED:
            prunable = get_prunable_episodes(days=cfg.CONSOLIDATION_PRUNE_AFTER_DAYS)
            if prunable:
                prune_ids = [ep["id"] for ep in prunable]
                mark_pruned(prune_ids)
                removed = vs.remove_batch(prune_ids)
                logger.info(
                    "Pruned %d old episodes (%d vectors tombstoned)", len(prunable), removed
                )
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

        report: ConsolidationReport = {
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "episodes_loaded": len(episodes),
            "episodes_with_vectors": len(valid_episodes),
            "clusters_total": len(clusters),
            "clusters_valid": len(valid_clusters),
            "clusters_failed": clusters_failed,
            "topics_created": topics_created,
            "topics_updated": topics_updated,
            "episodes_pruned": len(prunable) if prunable else 0,
            "surprise_adjusted": surprise_adjusted,
            "api_calls": api_calls,
            "failed_episode_ids": all_failed_ep_ids,
        }

        report_path = cfg.CONSOLIDATION_LOG_DIR / (
            f"{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H-%M-%S')}.json"
        )
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        complete_consolidation_run(
            run_id,
            episodes_processed=len(valid_episodes),
            clusters_formed=len(valid_clusters),
            topics_created=topics_created,
            topics_updated=topics_updated,
            episodes_pruned=len(prunable) if prunable else 0,
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
            "Consolidation complete: %d episodes -> %d clusters -> %d new topics, "
            "%d updated, %d failed, %d pruned, %d surprise-adjusted",
            len(valid_episodes),
            len(valid_clusters),
            topics_created,
            topics_updated,
            clusters_failed,
            len(prunable) if prunable else 0,
            surprise_adjusted,
        )
        return report

    except Exception as e:
        logger.exception("Consolidation failed: %s", e)
        complete_consolidation_run(run_id, status="failed", error_message=str(e))
        return {"status": "error", "message": str(e)}


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
