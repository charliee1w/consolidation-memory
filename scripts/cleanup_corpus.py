"""Soft-delete irrelevant episodes from the active project corpus (maintainer tool)."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from consolidation_memory.config import get_config
from consolidation_memory.corpus_hygiene import apply_corpus_hygiene

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
    parser.add_argument(
        "--expire-orphans",
        action="store_true",
        help="Also detach stale episode provenance and expire orphaned claims",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    payload = json.loads(args.candidates.read_text(encoding="utf-8"))
    ids: list[str] = sorted(set(payload.get("recommended_cleanup_ids", [])))
    if not ids and not args.expire_orphans:
        print("No cleanup candidates.")
        return

    print(f"Cleanup targets: {len(ids)} episodes")
    if args.dry_run:
        result = apply_corpus_hygiene(
            ids,
            expire_orphans=args.expire_orphans,
            dry_run=True,
        )
        for episode_id in result.get("episode_ids", []):
            print(episode_id)
        if args.expire_orphans and result.get("orphan_repair"):
            repair = result["orphan_repair"]
            print(f"Would expire {len(repair.get('would_expire_claims', []))} orphaned claims")
        return

    result = apply_corpus_hygiene(
        ids,
        expire_orphans=args.expire_orphans,
        dry_run=False,
    )
    print(f"Forgotten: {result.get('forgotten', 0)}, not_found: {result.get('not_found', 0)}")
    if args.expire_orphans and result.get("orphan_repair"):
        repair = result["orphan_repair"]
        print(f"Expired claims: {repair.get('expired_claims', 0)}")
    print(f"DB: {get_config().DB_PATH}")
    print("Run: python -m consolidation_memory reindex")


if __name__ == "__main__":
    main()