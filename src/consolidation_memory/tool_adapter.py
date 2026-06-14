"""Shared transport adapter logic for memory tool calls.

MCP and REST differ only in async lifecycle and JSON/HTTP framing. Recall
deadlines, deferred-knowledge warnings, and timeout fallback payloads live here
so transports do not reimplement them.
"""

from __future__ import annotations

import logging
import math
import os
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

_DEFERRED_KNOWLEDGE_RETRY_SECONDS_ENV = "CONSOLIDATION_MEMORY_DEFERRED_KNOWLEDGE_RETRY_SECONDS"
_DEFAULT_DEFERRED_KNOWLEDGE_RETRY_SECONDS = 3.0
_DEFERRED_KNOWLEDGE_POLL_INTERVAL_SECONDS = 0.05

_DEFERRED_KNOWLEDGE_WARNING = (
    "Knowledge/records/claims deferred: record embedding cache still warming. "
    "Episodes returned now; call memory_recall again shortly for full knowledge."
)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    token = raw.strip()
    if not token:
        return default
    try:
        value = float(token)
    except ValueError:
        return default
    return value if math.isfinite(value) else default


def recall_timeout_seconds() -> float:
    configured = _env_float("CONSOLIDATION_MEMORY_RECALL_TIMEOUT_SECONDS", 90.0)
    return configured if configured > 0 else 90.0


def recall_deadline_margin_ratio() -> float:
    margin = _env_float("CONSOLIDATION_MEMORY_RECALL_DEADLINE_MARGIN_RATIO", 0.85)
    if margin <= 0 or margin > 1:
        return 0.85
    return margin


def recall_deadline_monotonic(timeout_seconds: float) -> float:
    budget = max(0.5, float(timeout_seconds) * recall_deadline_margin_ratio())
    return time.monotonic() + budget


def inject_recall_deadline(arguments: dict[str, Any], *, timeout_seconds: float) -> None:
    arguments["_recall_deadline_monotonic"] = recall_deadline_monotonic(timeout_seconds)


def recall_knowledge_cache_ready() -> bool:
    from consolidation_memory import record_cache

    return record_cache.is_unexpired_cache_warm()


def deferred_knowledge_requested(include_knowledge: bool) -> bool:
    return bool(include_knowledge and not recall_knowledge_cache_ready())


def effective_include_knowledge(include_knowledge: bool) -> bool:
    return bool(include_knowledge and not deferred_knowledge_requested(include_knowledge))


def deferred_knowledge_retry_seconds() -> float:
    value = _env_float(
        _DEFERRED_KNOWLEDGE_RETRY_SECONDS_ENV,
        _DEFAULT_DEFERRED_KNOWLEDGE_RETRY_SECONDS,
    )
    return max(0.0, value)


def result_has_deferred_knowledge_warning(result: dict[str, Any]) -> bool:
    warnings_raw = result.get("warnings")
    if not isinstance(warnings_raw, list):
        return False
    return any(_DEFERRED_KNOWLEDGE_WARNING in str(item) for item in warnings_raw)


def append_deferred_knowledge_warning(result: dict[str, Any]) -> dict[str, Any]:
    warnings_raw = result.get("warnings")
    warnings: list[str] = (
        list(warnings_raw) if isinstance(warnings_raw, list) else []
    )
    warnings.insert(0, _DEFERRED_KNOWLEDGE_WARNING)
    updated = dict(result)
    updated["warnings"] = warnings
    return updated


def recall_result_needs_background_warm(warnings: list[str]) -> bool:
    return any("deadline" in str(item).lower() for item in warnings)


def build_recall_search_arguments(recall_arguments: dict[str, Any]) -> dict[str, Any]:
    n_results = recall_arguments.get("n_results", 10)
    if not isinstance(n_results, int):
        try:
            n_results = int(n_results)
        except (TypeError, ValueError):
            n_results = 10
    return {
        "query": recall_arguments.get("query"),
        "content_types": recall_arguments.get("content_types"),
        "tags": recall_arguments.get("tags"),
        "after": recall_arguments.get("after"),
        "before": recall_arguments.get("before"),
        "limit": max(1, min(n_results, 50)),
        "scope": recall_arguments.get("scope"),
    }


def build_recall_timeout_fallback_result(
    keyword_result: dict[str, Any],
    *,
    recall_timeout_seconds: float,
    include_knowledge: bool,
) -> dict[str, Any]:
    warnings = [
        f"Recall timed out after {recall_timeout_seconds:g}s; returned episodes-only fallback."
    ]
    if include_knowledge:
        warnings.append("Knowledge retrieval skipped in fallback mode.")

    episodes_value = keyword_result.get("episodes")
    fallback_episodes = episodes_value if isinstance(episodes_value, list) else []
    total_matches_value = keyword_result.get("total_matches")
    if isinstance(total_matches_value, int):
        total_episodes = total_matches_value
    elif isinstance(total_matches_value, str):
        try:
            total_episodes = int(total_matches_value)
        except ValueError:
            total_episodes = len(fallback_episodes)
    else:
        total_episodes = len(fallback_episodes)

    return {
        "episodes": fallback_episodes,
        "knowledge": [],
        "records": [],
        "claims": [],
        "total_episodes": total_episodes,
        "total_knowledge_topics": 0,
        "message": "Semantic recall timed out; returned keyword episodes-only fallback.",
        "warnings": warnings,
    }


def maybe_complete_deferred_recall(
    result: dict[str, Any],
    *,
    include_knowledge: bool,
    recall_executor: Callable[[], dict[str, Any]],
    client: object | None = None,
    retry_seconds: float | None = None,
) -> dict[str, Any]:
    """Warm record caches and retry recall once when knowledge was deferred."""
    if not include_knowledge or not result_has_deferred_knowledge_warning(result):
        return result

    budget = (
        deferred_knowledge_retry_seconds()
        if retry_seconds is None
        else max(0.0, retry_seconds)
    )
    if budget <= 0:
        return result

    warm_recall_caches(client)
    deadline = time.monotonic() + budget
    while time.monotonic() < deadline:
        if recall_knowledge_cache_ready():
            break
        time.sleep(_DEFERRED_KNOWLEDGE_POLL_INTERVAL_SECONDS)

    if not recall_knowledge_cache_ready():
        return result

    try:
        completed = recall_executor()
    except Exception as exc:
        logger.debug("Deferred knowledge retry failed: %s", exc)
        return result

    if result_has_deferred_knowledge_warning(completed):
        return result
    return completed


def warm_recall_caches(client: object | None = None) -> dict[str, int | bool]:
    """Prime recall caches without blocking interactive tool calls."""
    from consolidation_memory import claim_cache, record_cache, topic_cache
    from consolidation_memory.backends import get_embedding_backend
    from consolidation_memory.client import _resolved_scope_to_query_filter
    from consolidation_memory.config import get_config

    cfg = get_config()
    get_embedding_backend()

    warmed_topics = 0
    if cfg.WARMUP_PRIME_TOPIC_CACHE:
        topics, _ = topic_cache.get_topic_vecs()
        warmed_topics = len(topics)

    warmed_records = 0
    if cfg.WARMUP_PRIME_RECORD_CACHE:
        records, _ = record_cache.get_record_vecs(include_expired=False)
        warmed_records = len(records)
        if client is not None:
            try:
                default_scope_filter = _resolved_scope_to_query_filter(
                    client.resolve_scope()  # type: ignore[attr-defined]
                )
                scoped_records, _ = record_cache.get_record_vecs(
                    include_expired=False,
                    scope=default_scope_filter,
                )
                warmed_records = max(warmed_records, len(scoped_records))
            except Exception as exc:
                logger.debug("Scoped warmup skipped: %s", exc)

    warmed_claims = 0
    if cfg.WARMUP_PRIME_CLAIM_CACHE:
        warmed_claims = claim_cache.warm_active_claim_vecs(
            limit=max(1, int(cfg.WARMUP_CLAIM_LIMIT)),
        )

    stats: dict[str, int | bool] = {
        "topics": warmed_topics,
        "records": warmed_records,
        "claims": warmed_claims,
        "knowledge_cache_ready": recall_knowledge_cache_ready(),
    }
    logger.info(
        "Warmup complete (topics=%d, records=%d, claims=%d, ready=%s)",
        warmed_topics,
        warmed_records,
        warmed_claims,
        stats["knowledge_cache_ready"],
    )
    return stats