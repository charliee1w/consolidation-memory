"""Consolidation engine.

Clusters unconsolidated episodes, summarizes via LLM backend,
writes knowledge base files, prunes old episodes.
Can run standalone or as a background thread inside the MCP server.
"""

import json
import logging
import re
import shutil
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from consolidation_memory.config import (
    FAISS_COMPACTION_THRESHOLD,
    LLM_VALIDATION_RETRY,
    CONSOLIDATION_CLUSTER_THRESHOLD,
    CONSOLIDATION_LOG_DIR,
    CONSOLIDATION_MAX_CLUSTER_SIZE,
    CONSOLIDATION_MAX_EPISODES_PER_RUN,
    CONSOLIDATION_MIN_CLUSTER_SIZE,
    CONSOLIDATION_PRUNE_ENABLED,
    CONSOLIDATION_PRUNE_AFTER_DAYS,
    KNOWLEDGE_DIR,
    KNOWLEDGE_VERSIONS_DIR,
    KNOWLEDGE_MAX_VERSIONS,
    SURPRISE_BOOST_PER_ACCESS,
    SURPRISE_DECAY_INACTIVE_DAYS,
    SURPRISE_DECAY_RATE,
    SURPRISE_MIN,
    SURPRISE_MAX,
)
from consolidation_memory.database import (
    complete_consolidation_run,
    ensure_schema,
    get_all_active_episodes,
    get_all_knowledge_topics,
    get_prunable_episodes,
    get_unconsolidated_episodes,
    mark_consolidated,
    mark_pruned,
    start_consolidation_run,
    update_surprise_scores,
    upsert_knowledge_topic,
)
from consolidation_memory.vector_store import VectorStore
from consolidation_memory.backends import encode_documents, get_llm_backend

logger = logging.getLogger(__name__)

# LLM system prompt for consolidation
_LLM_SYSTEM_PROMPT = (
    "You are a precise knowledge extractor. You output structured "
    "markdown documents with YAML frontmatter. Never add information "
    "not present in the source material. Preserve all specific details: "
    "file paths, version numbers, commands, error messages."
)


from consolidation_memory import topic_cache  # noqa: E402


# ── Utilities ────────────────────────────────────────────────────────────────

def _slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^\w\s-]", "", text.lower())
    text = re.sub(r"[-\s]+", "_", text).strip("_")
    return text[:60]


def _call_llm(prompt: str, max_retries: int = 3) -> str:
    """Call LLM backend with retry."""
    llm = get_llm_backend()
    if llm is None:
        raise RuntimeError("LLM backend is disabled. Cannot run consolidation.")

    import time
    last_err = None
    for attempt in range(max_retries):
        try:
            return llm.generate(_LLM_SYSTEM_PROMPT, prompt)
        except Exception as e:
            last_err = e
            logger.warning("LLM attempt %d/%d failed: %s", attempt + 1, max_retries, e)
            if attempt < max_retries - 1:
                time.sleep(2.0 * (attempt + 1))

    raise ConnectionError(f"LLM failed after {max_retries} attempts: {last_err}")


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:ya?ml|markdown|md)?\s*\n", "", text)
    text = re.sub(r"\n```\s*$", "", text)
    return text.strip()


def _normalize_output(text: str) -> str:
    text = _strip_code_fences(text)

    if not text.startswith("---"):
        return text

    lines = text.split("\n")
    has_closing = False
    for i, line in enumerate(lines[1:], 1):
        if line.strip() == "---":
            has_closing = True
            break
        if line.strip().startswith("##") or (line.strip() == "" and i > 1):
            break

    if not has_closing:
        new_lines = [lines[0]]
        for i, line in enumerate(lines[1:], 1):
            if line.strip() == "" or line.strip().startswith("##"):
                new_lines.append("---")
                new_lines.append("")
                new_lines.extend(lines[i:])
                break
            new_lines.append(line)
        else:
            new_lines.append("---")
        text = "\n".join(new_lines)
        logger.info("Fixed missing closing --- in LLM output")

    return text


def _parse_frontmatter(text: str) -> dict:
    text = _strip_code_fences(text)

    if not text.startswith("---"):
        return {"meta": {}, "body": text}

    match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)", text, re.DOTALL)
    if not match:
        match = re.match(r"^---\s*\n(.*?)(?:\n\s*\n|\n(?=##))(.*)", text, re.DOTALL)

    if not match:
        return {"meta": {}, "body": text}

    meta = _parse_fm_lines(match.group(1))
    if not meta.get("title"):
        return {"meta": {}, "body": text}

    return {"meta": meta, "body": match.group(2)}


def _parse_fm_lines(block: str) -> dict:
    meta = {}
    for line in block.strip().splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower()
        value = value.strip()

        if key == "tags":
            value = value.strip("[]")
            meta["tags"] = [t.strip().strip("'\"") for t in value.split(",") if t.strip()]
        elif key == "confidence":
            try:
                meta["confidence"] = float(value)
            except ValueError:
                meta["confidence"] = 0.8
        else:
            meta[key] = value
    return meta


def _count_facts(text: str) -> int:
    return len(re.findall(r"^[\s]*[-*\d+.]\s+", text, re.MULTILINE))


def _compute_cluster_confidence(
    cluster_episodes: list[dict],
    sim_matrix: np.ndarray,
    cluster_indices: list[int],
) -> float:
    if len(cluster_indices) < 2:
        coherence = 0.8
    else:
        pairwise_sims = []
        for i_idx in range(len(cluster_indices)):
            for j_idx in range(i_idx + 1, len(cluster_indices)):
                pairwise_sims.append(sim_matrix[cluster_indices[i_idx], cluster_indices[j_idx]])
        coherence = float(np.mean(pairwise_sims))

    surprises = [ep.get("surprise_score", 0.5) for ep in cluster_episodes]
    source_quality = float(np.mean(surprises))

    confidence = coherence * 0.6 + source_quality * 0.4
    return round(max(0.5, min(0.95, confidence)), 2)


# ── LLM output validation ───────────────────────────────────────────────────

def _validate_llm_output(text: str, cluster_episodes: list[dict]) -> tuple[bool, list[str]]:
    failures = []
    parsed = _parse_frontmatter(text)
    meta = parsed["meta"]
    body = parsed["body"]

    if not meta.get("title"):
        failures.append("Missing or empty title in frontmatter")
    if not meta.get("summary"):
        failures.append("Missing or empty summary in frontmatter")

    summary = meta.get("summary", "")
    vague_patterns = [
        r"^(this document |discusses |describes |covers |details |provides )",
        r"^(a document about|an overview of|information about)",
    ]
    for pattern in vague_patterns:
        if re.match(pattern, summary.lower()):
            failures.append(f"Summary is vague/meta-descriptive: '{summary[:80]}'")
            break

    if "##" not in body:
        failures.append("No section headings (## Facts, ## Solutions, etc.) found in body")

    fact_count = _count_facts(body)
    if fact_count == 0:
        failures.append("No bullet points or numbered items found in body")

    source_specifics = set()
    for ep in cluster_episodes:
        content = ep.get("content", "")
        paths = re.findall(r'[A-Z]:\\[\w\\./\-]+|/[\w./\-]{5,}|~/[\w./\-]+', content)
        source_specifics.update(paths[:5])
        versions = re.findall(r'\d+\.\d+(?:\.\d+)+', content)
        source_specifics.update(versions[:3])

    if source_specifics:
        preserved = sum(1 for s in source_specifics if s in text)
        preservation_ratio = preserved / len(source_specifics)
        if preservation_ratio < 0.3:
            failures.append(
                f"Low specifics preservation: {preserved}/{len(source_specifics)} "
                f"key details from source episodes found in output"
            )

    return (len(failures) == 0, failures)


def _llm_with_validation(prompt: str, cluster_episodes: list[dict]) -> tuple[str, int]:
    response_text = _normalize_output(_call_llm(prompt))
    api_calls = 1

    is_valid, failures = _validate_llm_output(response_text, cluster_episodes)
    if not is_valid and LLM_VALIDATION_RETRY:
        logger.warning("Output failed validation: %s. Retrying...", "; ".join(failures))
        retry_addendum = (
            "\n\nPREVIOUS ATTEMPT FAILED VALIDATION. Fix these issues:\n"
            + "\n".join(f"- {f}" for f in failures)
            + "\n\nOutput the corrected document:"
        )
        try:
            response_text = _normalize_output(_call_llm(prompt + retry_addendum))
            api_calls += 1
            is_valid_2, failures_2 = _validate_llm_output(response_text, cluster_episodes)
            if not is_valid_2:
                logger.warning("Retry still failed: %s. Using best-effort output.", "; ".join(failures_2))
        except Exception as e:
            logger.error("LLM retry failed: %s", e)

    return response_text, api_calls


# ── Knowledge versioning ────────────────────────────────────────────────────

def _version_knowledge_file(filepath: Path) -> None:
    if not filepath.exists():
        return

    KNOWLEDGE_VERSIONS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    stem = filepath.stem
    versioned_name = f"{stem}.{timestamp}.md"
    versioned_path = KNOWLEDGE_VERSIONS_DIR / versioned_name

    shutil.copy2(str(filepath), str(versioned_path))
    logger.info("Versioned %s -> %s", filepath.name, versioned_name)

    pattern = f"{stem}.*.md"
    existing_versions = sorted(
        KNOWLEDGE_VERSIONS_DIR.glob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for old in existing_versions[KNOWLEDGE_MAX_VERSIONS:]:
        old.unlink()
        logger.debug("Pruned old version: %s", old.name)


# ── Adaptive surprise scoring ───────────────────────────────────────────────

def _adjust_surprise_scores() -> int:
    episodes = get_all_active_episodes()
    if not episodes:
        return 0

    access_counts = [ep["access_count"] for ep in episodes]
    median_access = float(np.median(access_counts))

    now = datetime.now(timezone.utc)
    updates = []

    for ep in episodes:
        original = ep["surprise_score"]
        new_score = original
        access = ep["access_count"]

        if access > median_access and median_access > 0:
            excess = access - median_access
            boost = min(excess * SURPRISE_BOOST_PER_ACCESS, 0.15)
            new_score += boost

        try:
            last_update = datetime.fromisoformat(ep["updated_at"])
            days_inactive = (now - last_update).total_seconds() / 86400.0
        except (ValueError, TypeError):
            days_inactive = 0

        # Only decay episodes that have been consolidated (their knowledge is
        # captured in a topic document).  Unconsolidated episodes are the sole
        # record of their information and should never be decayed into
        # obscurity.  This breaks the positive feedback loop where low-access
        # episodes decay → rank lower → get accessed even less → decay more.
        is_consolidated = ep.get("consolidated", 0) == 1
        if access == 0 and days_inactive >= SURPRISE_DECAY_INACTIVE_DAYS and is_consolidated:
            new_score -= SURPRISE_DECAY_RATE

        new_score = max(SURPRISE_MIN, min(SURPRISE_MAX, new_score))

        if abs(new_score - original) >= 0.005:
            updates.append((round(new_score, 4), ep["id"]))

    if updates:
        update_surprise_scores(updates)
        logger.info(
            "Adjusted surprise scores for %d/%d episodes (median_access=%.1f)",
            len(updates), len(episodes), median_access,
        )

    return len(updates)


# ── Topic matching ───────────────────────────────────────────────────────────

def _find_similar_topic(title: str, summary: str, tags: list[str]) -> dict | None:
    topics, existing_vecs = topic_cache.get_topic_vecs()
    if not topics:
        return None

    try:
        new_text = f"{title}. {summary}"
        new_vec = encode_documents([new_text])
        if existing_vecs is not None:
            sims = (new_vec @ existing_vecs.T).flatten()
            best_idx = int(np.argmax(sims))
            if sims[best_idx] >= 0.75:
                logger.info(
                    "Semantic topic match: '%.40s' -> '%.40s' (sim=%.3f)",
                    title, topics[best_idx]["title"], sims[best_idx],
                )
                return topics[best_idx]
    except Exception as e:
        logger.warning("Semantic topic matching failed, falling back to word overlap: %s", e)

    stopwords = {"the", "a", "an", "and", "or", "of", "in", "on", "for", "to", "with", "is", "at"}
    title_words = set(title.lower().split()) - stopwords
    if not title_words:
        return None

    for topic in topics:
        existing_words = set(topic["title"].lower().split()) - stopwords
        if not existing_words:
            continue
        overlap = len(title_words & existing_words)
        min_len = min(len(title_words), len(existing_words))
        if min_len > 0 and overlap / min_len > 0.5:
            return topic

    return None


# ── Main consolidation loop ─────────────────────────────────────────────────

def run_consolidation(vector_store: VectorStore | None = None) -> dict:
    """Main consolidation loop.

    Args:
        vector_store: Existing VectorStore instance to reuse. If None, creates
            a new one (backwards compatible for CLI/scheduled task usage).
    """
    ensure_schema()
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    KNOWLEDGE_VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
    CONSOLIDATION_LOG_DIR.mkdir(parents=True, exist_ok=True)

    topic_cache.invalidate()

    run_id = start_consolidation_run()
    logger.info("Consolidation run %s started", run_id)

    try:
        episodes = get_unconsolidated_episodes(limit=CONSOLIDATION_MAX_EPISODES_PER_RUN)

        if len(episodes) < CONSOLIDATION_MIN_CLUSTER_SIZE:
            logger.info("Only %d episodes — nothing to consolidate.", len(episodes))
            complete_consolidation_run(run_id, status="completed", episodes_processed=len(episodes))
            return {"status": "nothing_to_consolidate", "episodes": len(episodes)}

        logger.info("Loaded %d unconsolidated episodes", len(episodes))

        vs = vector_store if vector_store is not None else VectorStore()
        episode_ids = [ep["id"] for ep in episodes]
        result = vs.reconstruct_batch(episode_ids)

        if result is None:
            logger.warning("No vectors found for episodes — aborting.")
            complete_consolidation_run(run_id, status="failed",
                                       error_message="No vectors in FAISS")
            return {"status": "error", "message": "No vectors found"}

        found_ids, vectors = result

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
        labels = fcluster(Z, t=1.0 - CONSOLIDATION_CLUSTER_THRESHOLD, criterion="distance")

        clusters: dict[int, list[tuple[dict, int]]] = {}
        for idx, (ep, label) in enumerate(zip(valid_episodes, labels)):
            clusters.setdefault(int(label), []).append((ep, idx))

        valid_clusters = {k: v for k, v in clusters.items()
                         if len(v) >= CONSOLIDATION_MIN_CLUSTER_SIZE}

        logger.info("Formed %d clusters, %d valid (>=%d episodes)",
                     len(clusters), len(valid_clusters), CONSOLIDATION_MIN_CLUSTER_SIZE)

        topics_created = 0
        topics_updated = 0
        api_calls = 0

        for cluster_id, cluster_items in valid_clusters.items():
            if len(cluster_items) > CONSOLIDATION_MAX_CLUSTER_SIZE:
                cluster_items.sort(
                    key=lambda item: item[0].get("surprise_score", 0.5),
                    reverse=True,
                )
                cluster_items = cluster_items[:CONSOLIDATION_MAX_CLUSTER_SIZE]

            cluster_episodes = [ep for ep, _ in cluster_items]
            cluster_indices = [idx for _, idx in cluster_items]

            confidence = _compute_cluster_confidence(
                cluster_episodes, sim_matrix, cluster_indices,
            )

            all_tags = []
            for ep in cluster_episodes:
                tags = json.loads(ep["tags"]) if isinstance(ep["tags"], str) else ep["tags"]
                all_tags.extend(tags)
            tag_counts = Counter(all_tags).most_common(5)
            tag_summary = ", ".join(f"{t}({c})" for t, c in tag_counts) if tag_counts else "none"

            episode_texts = []
            for ep in cluster_episodes:
                episode_texts.append(
                    f"[{ep['created_at']}] [{ep['content_type']}] {ep['content']}"
                )

            prompt = f"""Distill {len(cluster_episodes)} episodes into a reference document. This document will be retrieved months later to avoid re-learning the same information.

STRICT RULES:
- Extract ONLY information explicitly stated in the episodes. NEVER add advice, recommendations, or inferences not directly present in the source text.
- Preserve specific details: file paths, version numbers, line numbers, command syntax, error messages. These are the most valuable parts. Vague summaries are worthless.
- A "fact" is a static piece of information (what exists, what is configured, what version).
- A "solution" is a problem->fix pair. Always state WHAT PROBLEM it solves, then the fix steps. If multiple distinct problems exist, use separate subsections.
- A "preference" is an explicitly stated user choice or workflow habit. Do NOT invent preferences from facts.
- The summary must be a dense factual statement, not a description of the document. BAD: "Discusses VR setup." GOOD: "VR stack uses SteamVR with SpaceCalibrator for multi-tracker calibration and VRCFaceTracking for face tracking."
- Omit any section (Facts, Solutions, Preferences) that would be empty.
- Do NOT wrap output in code fences. Output raw markdown only.

Common tags: {tag_summary}

Episodes:
{chr(10).join(episode_texts)}

Output format (raw markdown, no code fences):

---
title: Short Descriptive Title
summary: Dense factual summary with key specifics.
tags: [tag1, tag2]
confidence: {confidence}
---

## Facts
- Specific fact with concrete details preserved

## Solutions
### Problem Name
1. Step with exact commands/paths
2. Next step

## Preferences
- Explicitly stated user preference"""

            logger.info("Summarizing cluster %d (%d episodes)...", cluster_id, len(cluster_episodes))
            try:
                response_text, calls = _llm_with_validation(prompt, cluster_episodes)
                api_calls += calls
            except Exception as e:
                logger.error("LLM API call failed for cluster %d: %s", cluster_id, e)
                continue

            parsed = _parse_frontmatter(response_text)
            meta = parsed["meta"]
            title = meta.get("title", f"Topic {cluster_id}")
            summary = meta.get("summary", "")
            tags = meta.get("tags", [t for t, _ in tag_counts])

            existing = _find_similar_topic(title, summary, tags)
            cluster_ep_ids = [ep["id"] for ep in cluster_episodes]

            if existing:
                filepath = KNOWLEDGE_DIR / existing["filename"]
                existing_content = ""
                if filepath.exists():
                    existing_content = filepath.read_text(encoding="utf-8")

                merge_prompt = f"""Merge new knowledge into an existing reference document.

STRICT RULES:
- If a fact appears in both documents, keep the MORE SPECIFIC version (the one with more detail, exact paths, version numbers).
- If new information contradicts existing information, keep the NEWER information (from NEW KNOWLEDGE) and remove the old.
- Do NOT add commentary, advice, or inferences not present in either source.
- Preserve all file paths, commands, version numbers, and error messages exactly.
- Combine tags from both documents, deduplicated.
- Update the summary to cover the merged content. Keep it dense and factual.
- Maintain section structure: Facts, Solutions (with ### subsections per problem), Preferences.
- Omit empty sections.
- Do NOT wrap output in code fences. Output raw markdown only.

EXISTING DOCUMENT:
{existing_content}

NEW KNOWLEDGE TO MERGE:
{response_text}

Output the complete merged document starting with --- frontmatter:"""

                try:
                    merged_text, merge_calls = _llm_with_validation(merge_prompt, cluster_episodes)
                    api_calls += merge_calls
                    _version_knowledge_file(filepath)
                    filepath.write_text(merged_text, encoding="utf-8")
                    merged_parsed = _parse_frontmatter(merged_text)
                    upsert_knowledge_topic(
                        filename=existing["filename"],
                        title=merged_parsed["meta"].get("title", existing["title"]),
                        summary=merged_parsed["meta"].get("summary", existing["summary"]),
                        source_episodes=cluster_ep_ids,
                        fact_count=_count_facts(merged_text),
                        confidence=float(merged_parsed["meta"].get("confidence", confidence)),
                    )
                    mark_consolidated(cluster_ep_ids, existing["filename"])
                    topic_cache.invalidate()
                    topics_updated += 1
                    logger.info("Merged into existing topic: %s", existing["filename"])
                except Exception as e:
                    logger.error("Merge failed for topic %s: %s", existing["filename"], e)
                    continue
            else:
                filename = _slugify(title) + ".md"
                filepath = KNOWLEDGE_DIR / filename
                _version_knowledge_file(filepath)
                filepath.write_text(response_text, encoding="utf-8")
                upsert_knowledge_topic(
                    filename=filename,
                    title=title,
                    summary=summary,
                    source_episodes=cluster_ep_ids,
                    fact_count=_count_facts(response_text),
                    confidence=confidence,
                )
                mark_consolidated(cluster_ep_ids, filename)
                topic_cache.invalidate()
                topics_created += 1
                logger.info("Created new topic: %s", filename)

        _update_index()

        surprise_adjusted = _adjust_surprise_scores()

        prunable = []
        if CONSOLIDATION_PRUNE_ENABLED:
            prunable = get_prunable_episodes(days=CONSOLIDATION_PRUNE_AFTER_DAYS)
            if prunable:
                prune_ids = [ep["id"] for ep in prunable]
                mark_pruned(prune_ids)
                removed = vs.remove_batch(prune_ids)
                logger.info("Pruned %d old episodes (%d vectors tombstoned)", len(prunable), removed)
        else:
            logger.debug("Pruning disabled (set prune_enabled = true in config to enable)")

        if vs.tombstone_ratio >= FAISS_COMPACTION_THRESHOLD:
            compacted = vs.compact()
            logger.info("Compacted %d tombstoned vectors from FAISS index", compacted)

        VectorStore.signal_reload()

        report = {
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "episodes_loaded": len(episodes),
            "episodes_with_vectors": len(valid_episodes),
            "clusters_total": len(clusters),
            "clusters_valid": len(valid_clusters),
            "topics_created": topics_created,
            "topics_updated": topics_updated,
            "episodes_pruned": len(prunable) if prunable else 0,
            "surprise_adjusted": surprise_adjusted,
            "api_calls": api_calls,
        }

        report_path = CONSOLIDATION_LOG_DIR / f"{datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        complete_consolidation_run(
            run_id,
            episodes_processed=len(valid_episodes),
            clusters_formed=len(valid_clusters),
            topics_created=topics_created,
            topics_updated=topics_updated,
            episodes_pruned=len(prunable) if prunable else 0,
        )

        logger.info(
            "Consolidation complete: %d episodes -> %d clusters -> %d new topics, "
            "%d updated, %d pruned, %d surprise-adjusted",
            len(valid_episodes), len(valid_clusters), topics_created, topics_updated,
            len(prunable) if prunable else 0, surprise_adjusted,
        )
        return report

    except Exception as e:
        logger.exception("Consolidation failed: %s", e)
        complete_consolidation_run(run_id, status="failed", error_message=str(e))
        return {"status": "error", "message": str(e)}


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

    index_path = KNOWLEDGE_DIR / "index.md"
    index_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Updated index.md with %d topics", len(topics))
