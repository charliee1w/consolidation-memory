"""Corpus hygiene scan and apply for noisy episodes and orphaned claims."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

from consolidation_memory.database import (
    detach_claim_sources_for_episode,
    ensure_schema,
    expire_claims_without_sources,
    get_connection,
    soft_delete_episode,
)
from consolidation_memory.episode_embedding import solution_store_shape_warnings
from consolidation_memory.vector_store import VectorStore

_TEMP_PATTERNS = (
    "coding_agent_eval_test",
    ".tmp_coding_agent_eval",
    "drift_000",
    "drift_001",
    "novelty_eval_test",
)

_NOISE_RE = re.compile(
    r"(?is)^(?:"
    r"completed(?:\s+(?:item|multi-agent|de-claude|prompt|data layer|mcp|the|a))?\b"
    r"|implemented(?:\s+(?:builder-readiness|roadmap item|permanent|automatic|mcp|fixes for four))?\b"
    r"|validated\s+consolidation"
    r"|provided\s+in-depth"
    r"|final\s+audit\b"
    r"|consolidation-memory\s+v0\.\d"
    r"|all\s+\d+\s+tests\s+pass"
    r"|faiss\s+index\s+health\s+check"
    r"|multi-agent\s+hardening\s+pass"
    r"|researched\s+and\s+documented"
    r"|architecture\s+assessment"
    r")",
)

_PROTECT_RE = re.compile(
    r"(?is)^(?:fixed|patched|diagnosed|solved|problem:)",
)

_MAX_SAMPLES = 8


def _is_temp_episode(content: str, tags: str) -> bool:
    text = (content or "").lower()
    if any(pattern in text for pattern in _TEMP_PATTERNS):
        return True
    try:
        parsed = json.loads(tags) if tags else []
    except json.JSONDecodeError:
        parsed = []
    if isinstance(parsed, list):
        joined = " ".join(str(tag).lower() for tag in parsed)
        if "smoke" in joined or "test-run" in joined:
            return True
    return False


def _is_noise_journal(content: str) -> bool:
    text = (content or "").strip()
    if not text:
        return True
    if _PROTECT_RE.match(text):
        return False
    if not solution_store_shape_warnings(text):
        return False
    return bool(_NOISE_RE.match(text))


def _episode_sample(episode: dict[str, Any]) -> dict[str, str]:
    content = str(episode.get("content") or "")
    preview = content[:120] + ("…" if len(content) > 120 else "")
    return {
        "id": str(episode["id"]),
        "content_type": str(episode.get("content_type") or ""),
        "preview": preview,
    }


def _load_active_episodes() -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, content_type, content, tags, created_at
            FROM episodes
            WHERE deleted = 0
            ORDER BY created_at DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def classify_episode_candidates(
    episodes: list[dict[str, Any]] | None = None,
) -> dict[str, list[str]]:
    """Classify active episodes into hygiene cleanup buckets."""
    source = episodes if episodes is not None else _load_active_episodes()
    temp_ids: list[str] = []
    exchange_ids: list[str] = []
    noise_journal_ids: list[str] = []

    for episode in source:
        episode_id = str(episode["id"])
        content_type = str(episode.get("content_type") or "")
        content = str(episode.get("content") or "")
        tags = str(episode.get("tags") or "[]")

        if _is_temp_episode(content, tags):
            temp_ids.append(episode_id)
            continue
        if content_type == "exchange":
            exchange_ids.append(episode_id)
            continue
        if content_type == "solution" and _is_noise_journal(content):
            noise_journal_ids.append(episode_id)

    recommended = sorted(set(temp_ids + exchange_ids + noise_journal_ids))
    return {
        "temp_ids": temp_ids,
        "exchange_ids": exchange_ids,
        "noise_journal_ids": noise_journal_ids,
        "recommended_cleanup_ids": recommended,
    }


def _orphan_claim_rows() -> list[dict[str, str]]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT c.id,
                   c.claim_type,
                   c.canonical_text,
                   CASE
                       WHEN NOT EXISTS (
                           SELECT 1 FROM claim_sources cs WHERE cs.claim_id = c.id
                       ) THEN 'no_sources'
                       ELSE 'deleted_episode_only'
                   END AS reason
              FROM claims c
             WHERE c.status = 'active'
               AND (c.valid_until IS NULL OR julianday(c.valid_until) > julianday('now'))
               AND (
                   NOT EXISTS (
                       SELECT 1 FROM claim_sources cs WHERE cs.claim_id = c.id
                   )
                   OR (
                       NOT EXISTS (
                           SELECT 1
                             FROM claim_sources cs
                            WHERE cs.claim_id = c.id
                              AND (
                                  cs.source_record_id IS NOT NULL
                                  OR cs.source_topic_id IS NOT NULL
                              )
                       )
                       AND NOT EXISTS (
                           SELECT 1
                             FROM claim_sources cs
                             JOIN episodes e ON e.id = cs.source_episode_id
                            WHERE cs.claim_id = c.id
                              AND cs.source_episode_id IS NOT NULL
                              AND e.deleted = 0
                       )
                   )
               )
             ORDER BY c.id ASC
            """
        ).fetchall()
    return [
        {
            "id": str(row["id"]),
            "claim_type": str(row["claim_type"] or ""),
            "canonical_text": str(row["canonical_text"] or ""),
            "reason": str(row["reason"] or ""),
        }
        for row in rows
    ]


def _deleted_episodes_with_claim_sources() -> list[str]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT cs.source_episode_id AS episode_id
              FROM claim_sources cs
              JOIN episodes e ON e.id = cs.source_episode_id
             WHERE cs.source_episode_id IS NOT NULL
               AND e.deleted = 1
             ORDER BY episode_id ASC
            """
        ).fetchall()
    return [str(row["episode_id"]) for row in rows if row["episode_id"]]


def scan_corpus_hygiene() -> dict[str, object]:
    """Read-only hygiene report for episode noise and orphaned active claims."""
    ensure_schema()
    episodes = _load_active_episodes()
    buckets = classify_episode_candidates(episodes)
    episode_by_id = {str(ep["id"]): ep for ep in episodes}

    def bucket_payload(ids: list[str], reason: str) -> dict[str, object]:
        samples = [
            {**_episode_sample(episode_by_id[episode_id]), "reason": reason}
            for episode_id in ids[:_MAX_SAMPLES]
            if episode_id in episode_by_id
        ]
        return {"count": len(ids), "ids": ids, "samples": samples}

    orphan_rows = _orphan_claim_rows()
    stale_episode_ids = _deleted_episodes_with_claim_sources()
    recommended = buckets["recommended_cleanup_ids"]
    total_active = len(episodes)

    return {
        "status": "ok",
        "episodes": {
            "total_active": total_active,
            "temp": bucket_payload(buckets["temp_ids"], "temp_or_test"),
            "exchange": bucket_payload(buckets["exchange_ids"], "exchange"),
            "noise_journal": bucket_payload(buckets["noise_journal_ids"], "noise_journal"),
            "recommended_cleanup_ids": recommended,
            "would_remain": total_active - len(recommended),
        },
        "orphaned_claims": {
            "count": len(orphan_rows),
            "ids": [row["id"] for row in orphan_rows],
            "samples": [
                {
                    "id": row["id"],
                    "claim_type": row["claim_type"],
                    "preview": row["canonical_text"][:120]
                    + ("…" if len(row["canonical_text"]) > 120 else ""),
                    "reason": row["reason"],
                }
                for row in orphan_rows[:_MAX_SAMPLES]
            ],
        },
        "stale_episode_sources": {
            "count": len(stale_episode_ids),
            "episode_ids": stale_episode_ids,
        },
    }


def repair_orphaned_claims(*, dry_run: bool = False) -> dict[str, object]:
    """Detach provenance for deleted episodes and expire sourceless active claims."""
    ensure_schema()
    stale_episode_ids = _deleted_episodes_with_claim_sources()
    orphan_before = _orphan_claim_rows()

    if dry_run:
        return {
            "status": "dry_run",
            "stale_episode_sources": len(stale_episode_ids),
            "orphaned_claims_before": len(orphan_before),
            "would_expire_claims": [row["id"] for row in orphan_before],
        }

    repaired_at = datetime.now(timezone.utc).isoformat()
    impacted_claim_ids: list[str] = []
    seen_claim_ids: set[str] = set()
    for episode_id in stale_episode_ids:
        for claim_id in detach_claim_sources_for_episode(episode_id):
            if claim_id not in seen_claim_ids:
                seen_claim_ids.add(claim_id)
                impacted_claim_ids.append(claim_id)

    expired_from_detach = expire_claims_without_sources(
        impacted_claim_ids,
        valid_until=repaired_at,
        reason="orphan_repair",
        details={"source": "corpus_hygiene", "stale_episode_sources": len(stale_episode_ids)},
    )

    remaining_orphans = _orphan_claim_rows()
    remaining_ids = [row["id"] for row in remaining_orphans]
    expired_no_sources = expire_claims_without_sources(
        remaining_ids,
        valid_until=repaired_at,
        reason="orphan_repair",
        details={"source": "corpus_hygiene", "reason": "no_sources"},
    )
    expired_claim_ids = sorted(set(expired_from_detach + expired_no_sources))

    from consolidation_memory import claim_cache as _cc

    if expired_claim_ids or stale_episode_ids:
        _cc.invalidate()

    return {
        "status": "ok",
        "stale_episode_sources": len(stale_episode_ids),
        "orphaned_claims_before": len(orphan_before),
        "expired_claims": len(expired_claim_ids),
        "expired_claim_ids": expired_claim_ids,
        "orphaned_claims_after": len(_orphan_claim_rows()),
    }


def apply_corpus_hygiene(
    episode_ids: list[str] | None = None,
    *,
    use_recommended: bool = False,
    expire_orphans: bool = False,
    dry_run: bool = False,
) -> dict[str, object]:
    """Forget selected episodes and optionally repair orphaned claims."""
    ensure_schema()
    scan = scan_corpus_hygiene()
    recommended = list(scan["episodes"]["recommended_cleanup_ids"])  # type: ignore[index]

    if use_recommended:
        targets = recommended
    elif episode_ids is not None:
        targets = sorted({str(episode_id).strip() for episode_id in episode_ids if str(episode_id).strip()})
    else:
        targets = []

    orphan_report: dict[str, object] | None = None
    if dry_run:
        if expire_orphans:
            orphan_report = repair_orphaned_claims(dry_run=True)
        return {
            "status": "dry_run",
            "episode_targets": len(targets),
            "episode_ids": targets,
            "expire_orphans": expire_orphans,
            "orphan_repair": orphan_report,
        }

    vector_store = VectorStore()
    forgotten = 0
    not_found = 0
    forgotten_at = datetime.now(timezone.utc).isoformat()

    for episode_id in targets:
        deleted = soft_delete_episode(episode_id, scope=None)
        if not deleted:
            not_found += 1
            continue
        vector_store.remove(episode_id)
        impacted_claim_ids = detach_claim_sources_for_episode(episode_id)
        expire_claims_without_sources(
            impacted_claim_ids,
            valid_until=forgotten_at,
            reason="episode_forgotten",
            details={"episode_id": episode_id, "source": "corpus_hygiene"},
        )
        forgotten += 1

    if forgotten:
        VectorStore.signal_reload()
        from consolidation_memory import claim_cache as _cc

        _cc.invalidate()

    if expire_orphans:
        orphan_report = repair_orphaned_claims(dry_run=False)

    return {
        "status": "ok",
        "forgotten": forgotten,
        "not_found": not_found,
        "episode_targets": len(targets),
        "expire_orphans": expire_orphans,
        "orphan_repair": orphan_report,
    }