"""Analyze live corpus for cleanup candidates. Read-only."""

from __future__ import annotations

import json

from consolidation_memory.corpus_hygiene import scan_corpus_hygiene


def main() -> None:
    report = scan_corpus_hygiene()
    episodes = report["episodes"]
    assert isinstance(episodes, dict)

    print("Episodes by cleanup bucket:")
    print(f"  temp/test: {episodes['temp']['count']}")
    print(f"  exchange: {episodes['exchange']['count']}")
    print(f"  noise journal: {episodes['noise_journal']['count']}")
    print(f"Total active episodes: {episodes['total_active']}")
    print(f"Total cleanup targets: {len(episodes['recommended_cleanup_ids'])}")
    print(f"Would remain: {episodes['would_remain']}")
    print(f"Orphaned active claims: {report['orphaned_claims']['count']}")

    print("\nSample noise journal removals:")
    for sample in episodes["noise_journal"]["samples"][:12]:
        print(f"  {sample['id'][:8]}  {sample['preview']}")

    out = {
        "temp_ids": episodes["temp"]["ids"],
        "exchange_ids": episodes["exchange"]["ids"],
        "noise_journal_ids": episodes["noise_journal"]["ids"],
        "recommended_cleanup_ids": episodes["recommended_cleanup_ids"],
    }
    path = "benchmarks/results/cleanup_candidates.json"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()