"""Minimal Python API example for consolidation-memory.

Run from the repository root or any environment where
``consolidation-memory`` is installed.
"""

from __future__ import annotations

from consolidation_memory import MemoryClient


def main() -> None:
    with MemoryClient(auto_consolidate=False) as mem:
        mem.store(
            "The user prefers short pull request summaries with concrete file paths.",
            content_type="preference",
            tags=["demo", "workflow"],
        )

        result = mem.recall(
            "How should I summarize pull requests?",
            n_results=5,
            include_knowledge=True,
        )

    print(f"Episodes: {len(result.episodes)}")
    print(f"Knowledge topics: {len(result.knowledge)}")
    print(f"Structured records: {len(result.records)}")
    print(f"Claims: {len(result.claims)}")

    if result.episodes:
        print("\nTop episode:")
        print(result.episodes[0].content)


if __name__ == "__main__":
    main()
