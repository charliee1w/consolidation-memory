"""Hierarchical clustering logic and cluster formation.

Includes: cluster confidence computation and topic similarity matching
used to decide whether a new cluster merges into an existing topic.
"""

import logging

import numpy as np

from consolidation_memory.config import get_config
from consolidation_memory import topic_cache

logger = logging.getLogger(__name__)


def _matches_scope(
    row: dict,
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


def _compute_cluster_confidence(
    cluster_episodes: list[dict],
    sim_matrix: np.ndarray,
    cluster_indices: list[int],
) -> float:
    cfg = get_config()
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

    confidence = (
        coherence * cfg.CONSOLIDATION_CONFIDENCE_COHERENCE_W
        + source_quality * cfg.CONSOLIDATION_CONFIDENCE_SURPRISE_W
    )
    return round(max(0.5, min(0.95, confidence)), 2)


def _find_similar_topic(
    title: str,
    summary: str,
    tags: list[str],
    *,
    scope: dict[str, str | None] | None = None,
) -> dict | None:
    from consolidation_memory.backends import encode_documents

    topics, existing_vecs = topic_cache.get_topic_vecs()
    if not topics:
        return None

    if scope:
        filtered_indices = [i for i, topic in enumerate(topics) if _matches_scope(topic, scope)]
        topics = [topics[i] for i in filtered_indices]
        if existing_vecs is not None:
            existing_vecs = existing_vecs[filtered_indices] if filtered_indices else None
        if not topics:
            return None

    try:
        new_text = f"{title}. {summary}"
        new_vec = encode_documents([new_text])
        if existing_vecs is not None:
            sims = (new_vec @ existing_vecs.T).flatten()
            best_idx = int(np.argmax(sims))
            if sims[best_idx] >= get_config().CONSOLIDATION_TOPIC_SEMANTIC_THRESHOLD:
                logger.info(
                    "Semantic topic match: '%.40s' -> '%.40s' (sim=%.3f)",
                    title,
                    topics[best_idx]["title"],
                    sims[best_idx],
                )
                return topics[best_idx]
    except Exception as e:
        logger.warning(
            "Semantic topic matching failed, falling back to word overlap: %s",
            e,
            exc_info=True,
        )

    stopwords = get_config().CONSOLIDATION_STOPWORDS
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
