"""Read-only data layer for the TUI dashboard.

Queries SQLite directly via database.py to avoid initializing
FAISS/embedding backends. All methods return plain dicts/lists.
"""

from __future__ import annotations

import json

from consolidation_memory.config import get_config as _get_config
from consolidation_memory.database import ensure_schema, get_connection
from consolidation_memory.utils import parse_json_list


class DashboardData:
    """Lightweight read-only data access for the dashboard."""

    def __init__(self) -> None:
        ensure_schema()

    def get_episodes(
        self,
        sort_by: str = "created_at",
        desc: bool = True,
        content_type: str | None = None,
        limit: int = 500,
    ) -> list[dict]:
        """Fetch episodes for the browser table.

        Returns dicts with: id, content_preview, content_type, tags,
        surprise_score, created_at, consolidated.
        """
        allowed_sorts = {
            "created_at", "content_type", "surprise_score", "consolidated",
        }
        if sort_by not in allowed_sorts:
            sort_by = "created_at"

        direction = "DESC" if desc else "ASC"
        conditions = ["deleted = 0"]
        params: list = []

        if content_type:
            conditions.append("content_type = ?")
            params.append(content_type)

        where = " AND ".join(conditions)
        # Use rowid as a deterministic tie-breaker for equal timestamps/scores.
        if sort_by == "created_at":
            order_by = f"{sort_by} {direction}, rowid {direction}"
        else:
            order_by = f"{sort_by} {direction}, created_at DESC, rowid DESC"

        sql = (
            f"SELECT id, content, content_type, tags, surprise_score, "
            f"created_at, consolidated "
            f"FROM episodes WHERE {where} "
            f"ORDER BY {order_by} LIMIT ?"
        )
        params.append(limit)

        with get_connection() as conn:
            rows = conn.execute(sql, params).fetchall()

        results = []
        for row in rows:
            r = dict(row)
            content = r.pop("content")
            r["content_preview"] = (
                content[:80] + "..." if len(content) > 80 else content
            )
            r["tags"] = parse_json_list(r["tags"])
            results.append(r)
        return results

    def get_knowledge_topics(self) -> list[dict]:
        """Fetch all knowledge topics with source episode counts."""
        with get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM knowledge_topics ORDER BY updated_at DESC"
            ).fetchall()

        results = []
        for row in rows:
            r = dict(row)
            r["source_episode_count"] = len(parse_json_list(r.get("source_episodes")))
            results.append(r)
        return results

    def get_records_for_topic(self, topic_id: str) -> list[dict]:
        """Fetch active records for a specific knowledge topic."""
        with get_connection() as conn:
            rows = conn.execute(
                "SELECT record_type, content, confidence, created_at "
                "FROM knowledge_records WHERE topic_id = ? AND deleted = 0",
                (topic_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_consolidation_runs(self, limit: int = 100) -> list[dict]:
        """Fetch consolidation run history, newest first."""
        with get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM consolidation_runs ORDER BY started_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_stats(self) -> dict:
        """Aggregate memory statistics for the stats tab."""
        with get_connection() as conn:
            # Episode counts by type
            type_rows = conn.execute(
                "SELECT content_type, COUNT(*) as cnt "
                "FROM episodes WHERE deleted = 0 "
                "GROUP BY content_type"
            ).fetchall()
            episodes_by_type = {row["content_type"]: row["cnt"] for row in type_rows}

            total_episodes = sum(episodes_by_type.values())

            # Knowledge topics
            kt_row = conn.execute(
                "SELECT COUNT(*) as cnt FROM knowledge_topics"
            ).fetchone()
            knowledge_topic_count = kt_row["cnt"] if kt_row else 0

            # Knowledge records
            rec_row = conn.execute(
                "SELECT COUNT(*) as cnt FROM knowledge_records WHERE deleted = 0"
            ).fetchone()
            record_count = rec_row["cnt"] if rec_row else 0

            # Last consolidation
            last_run = conn.execute(
                "SELECT * FROM consolidation_runs "
                "ORDER BY started_at DESC LIMIT 1"
            ).fetchone()

        # DB size
        db_size_mb = 0.0
        if _get_config().DB_PATH.exists():
            db_size_mb = round(
                _get_config().DB_PATH.stat().st_size / (1024 * 1024), 2
            )

        return {
            "episodes_by_type": episodes_by_type,
            "total_episodes": total_episodes,
            "knowledge_topic_count": knowledge_topic_count,
            "record_count": record_count,
            "db_size_mb": db_size_mb,
            "last_consolidation": dict(last_run) if last_run else None,
        }

    def get_faiss_stats(self) -> dict:
        """Read FAISS metadata without importing faiss.

        Parses the JSON sidecar files to get index size and tombstone info.
        """
        id_map_path = _get_config().FAISS_ID_MAP_PATH
        tombstone_path = _get_config().FAISS_TOMBSTONE_PATH

        total = 0
        tombstones = 0

        if id_map_path.exists():
            try:
                ids = json.loads(id_map_path.read_text(encoding="utf-8"))
                total = len(ids)
            except (json.JSONDecodeError, OSError):
                pass

        if tombstone_path.exists():
            try:
                tombs = json.loads(tombstone_path.read_text(encoding="utf-8"))
                tombstones = len(tombs)
            except (json.JSONDecodeError, OSError):
                pass

        active = max(0, total - tombstones)
        ratio = tombstones / total if total > 0 else 0.0

        return {
            "index_size": active,
            "tombstone_count": tombstones,
            "tombstone_ratio": ratio,
        }
