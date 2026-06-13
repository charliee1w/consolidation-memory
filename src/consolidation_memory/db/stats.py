"""Aggregate stats counters."""

from __future__ import annotations

from typing import Any, Mapping

from consolidation_memory.db.connection import get_connection
from consolidation_memory.db.scope import _apply_scope_filters
from consolidation_memory.types import StatsDict

def get_stats(scope: Mapping[str, Any] | None = None) -> StatsDict:
    with get_connection() as conn:
        episode_conditions: list[str] = []
        episode_params: list[Any] = []
        _apply_scope_filters(episode_conditions, episode_params, scope)
        episode_where = f"WHERE {' AND '.join(episode_conditions)}" if episode_conditions else ""
        ep_counts = conn.execute(
            f"""SELECT
                 COUNT(*) FILTER (WHERE deleted = 0) as total,
                 COUNT(*) FILTER (WHERE consolidated = 0 AND deleted = 0) as pending,
                 COUNT(*) FILTER (WHERE consolidated = 1 AND deleted = 0) as consolidated,
                 COUNT(*) FILTER (WHERE consolidated = 2 OR deleted = 1) as pruned
               FROM episodes {episode_where}""",
            episode_params,
        ).fetchone()
        topic_conditions: list[str] = []
        topic_params: list[Any] = []
        _apply_scope_filters(topic_conditions, topic_params, scope)
        topic_where = f"WHERE {' AND '.join(topic_conditions)}" if topic_conditions else ""
        kt_counts = conn.execute(
            f"""SELECT COUNT(*) as total_topics,
                      COALESCE(SUM(fact_count), 0) as total_facts
               FROM knowledge_topics {topic_where}""",
            topic_params,
        ).fetchone()
        record_conditions: list[str] = []
        record_params: list[Any] = []
        _apply_scope_filters(record_conditions, record_params, scope)
        record_where = f"WHERE {' AND '.join(record_conditions)}" if record_conditions else ""
        rec_counts = conn.execute(
            f"""SELECT
                 COUNT(*) FILTER (WHERE deleted = 0) as total_records,
                 COUNT(*) FILTER (WHERE deleted = 0 AND record_type = 'fact') as facts,
                 COUNT(*) FILTER (WHERE deleted = 0 AND record_type = 'solution') as solutions,
                 COUNT(*) FILTER (WHERE deleted = 0 AND record_type = 'preference') as preferences,
                 COUNT(*) FILTER (WHERE deleted = 0 AND record_type = 'procedure') as procedures,
                 COUNT(*) FILTER (WHERE deleted = 0 AND record_type = 'strategy') as strategies
               FROM knowledge_records {record_where}""",
            record_params,
        ).fetchone()

    return {
        "episodic_buffer": {
            "total": ep_counts["total"],
            "pending_consolidation": ep_counts["pending"],
            "consolidated": ep_counts["consolidated"],
            "pruned": ep_counts["pruned"],
        },
        "knowledge_base": {
            "total_topics": kt_counts["total_topics"],
            "total_facts": kt_counts["total_facts"],
            "total_records": rec_counts["total_records"],
            "records_by_type": {
                "facts": rec_counts["facts"],
                "solutions": rec_counts["solutions"],
                "preferences": rec_counts["preferences"],
                "procedures": rec_counts["procedures"],
                "strategies": rec_counts["strategies"],
            },
        },
    }
