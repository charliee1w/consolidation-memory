"""Hierarchical clustering logic and cluster formation.

Includes: cluster confidence computation and topic similarity matching
used to decide whether a new cluster merges into an existing topic.
"""

import logging
import re

import numpy as np

from consolidation_memory.config import get_config
from consolidation_memory import topic_cache

logger = logging.getLogger(__name__)


def _title_tokens(title: str, stopwords: set[str] | frozenset[str]) -> set[str]:
    """Tokenize a title into lowercase alphanumerics, excluding stopwords."""
    return {
        token
        for token in re.findall(r"[a-z0-9]+", title.lower())
        if token and token not in stopwords
    }


def _title_overlap_ratio(
    lhs_title: str,
    rhs_title: str,
    stopwords: set[str] | frozenset[str],
) -> float:
    """Return overlap ratio using the smaller title token set as denominator."""
    lhs = _title_tokens(lhs_title, stopwords)
    rhs = _title_tokens(rhs_title, stopwords)
    if not lhs or not rhs:
        return 0.0
    return len(lhs & rhs) / min(len(lhs), len(rhs))


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
    cfg = get_config()
    stopwords = cfg.CONSOLIDATION_STOPWORDS

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
            best_sim = float(sims[best_idx])
            if best_sim >= cfg.CONSOLIDATION_TOPIC_SEMANTIC_THRESHOLD:
                overlap = _title_overlap_ratio(
                    title,
                    str(topics[best_idx].get("title", "")),
                    stopwords,
                )
                if (
                    overlap < cfg.CONSOLIDATION_TOPIC_TITLE_OVERLAP_THRESHOLD
                    and best_sim < cfg.CONSOLIDATION_TOPIC_FORCE_SEMANTIC_THRESHOLD
                ):
                    logger.info(
                        "Rejected semantic topic match: '%.40s' -> '%.40s' "
                        "(sim=%.3f, title_overlap=%.3f; min_overlap=%.3f, force=%.3f)",
                        title,
                        topics[best_idx]["title"],
                        best_sim,
                        overlap,
                        cfg.CONSOLIDATION_TOPIC_TITLE_OVERLAP_THRESHOLD,
                        cfg.CONSOLIDATION_TOPIC_FORCE_SEMANTIC_THRESHOLD,
                    )
                    return None
                logger.info(
                    "Semantic topic match: '%.40s' -> '%.40s' (sim=%.3f)",
                    title,
                    topics[best_idx]["title"],
                    best_sim,
                )
                return topics[best_idx]
    except Exception as e:
        logger.warning(
            "Semantic topic matching failed, falling back to word overlap: %s",
            e,
            exc_info=True,
        )

    title_words = _title_tokens(title, stopwords)
    if not title_words:
        return None

    for topic in topics:
        existing_words = _title_tokens(str(topic.get("title", "")), stopwords)
        if not existing_words:
            continue
        overlap = len(title_words & existing_words)
        min_len = min(len(title_words), len(existing_words))
        if min_len > 0 and overlap / min_len > 0.5:
            return topic

    return None
