"""Soft-delete irrelevant episodes from the active project corpus (maintainer tool)."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from consolidation_memory.config import get_config
from consolidation_memory.database import (
    detach_claim_sources_for_episode,
    ensure_schema,
    expire_claims_without_sources,
    soft_delete_episode,
)
from consolidation_memory.vector_store import VectorStore

logger = logging.getLogger("cleanup_corpus")


def main() -> None:
    parser = argparse.ArgumentParser(description="Forget irrelevant episodes from live memory")
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path("benchmarks/results/cleanup_candidates.json"),
        help="JSON file with recommended_cleanup_ids",
    )
    parser.add_argument("--dry-run", action="store_true", help="List IDs only, do not delete")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    payload = json.loads(args.candidates.read_text(encoding="utf-8"))
    ids: list[str] = sorted(set(payload.get("recommended_cleanup_ids", [])))
    if not ids:
        print("No cleanup candidates.")
        return

    print(f"Cleanup targets: {len(ids)} episodes")
    if args.dry_run:
        for eid in ids:
            print(eid)
        return

    ensure_schema()
    cfg = get_config()
    vector_store = VectorStore()

    forgotten = 0
    not_found = 0
    forgotten_at = datetime.now(timezone.utc).isoformat()

    for eid in ids:
        # Maintainer cleanup: no scope filter (cross-scope test artifacts + exchanges).
        deleted = soft_delete_episode(eid, scope=None)
        if not deleted:
            not_found += 1
            logger.warning("Not found: %s", eid)
            continue

        vector_store.remove(eid)
        impacted_claim_ids = detach_claim_sources_for_episode(eid)
        expire_claims_without_sources(
            impacted_claim_ids,
            valid_until=forgotten_at,
            reason="episode_forgotten",
            details={"episode_id": eid, "source": "cleanup_corpus"},
        )
        forgotten += 1

    VectorStore.signal_reload()
    print(f"Forgotten: {forgotten}, not_found: {not_found}")
    print(f"DB: {cfg.DB_PATH}")
    print("Run: python -m consolidation_memory reindex")


if __name__ == "__main__":
    main()