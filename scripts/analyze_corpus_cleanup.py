"""Analyze live corpus for cleanup candidates. Read-only."""

from __future__ import annotations

import json
import re

from consolidation_memory.database import ensure_schema, get_connection
from consolidation_memory.episode_embedding import solution_store_shape_warnings

_TEMP_PATTERNS = (
    "coding_agent_eval_test",
    ".tmp_coding_agent_eval",
    "drift_000",
    "drift_001",
    "novelty_eval_test",
)

# Session/audit/completion journal noise — not actionable solutions.
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


def _is_temp_episode(content: str, tags: str) -> bool:
    text = (content or "").lower()
    if any(p in text for p in _TEMP_PATTERNS):
        return True
    try:
        parsed = json.loads(tags) if tags else []
    except json.JSONDecodeError:
        parsed = []
    if isinstance(parsed, list):
        joined = " ".join(str(t).lower() for t in parsed)
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


def main() -> None:
    ensure_schema()
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT content_type, COUNT(*) n FROM episodes WHERE deleted=0 GROUP BY content_type"
        ).fetchall()
        print("Episodes by type (before):", {r["content_type"]: r["n"] for r in rows})

        episodes = conn.execute(
            """
            SELECT id, content_type, content, tags, created_at
            FROM episodes WHERE deleted=0
            ORDER BY created_at DESC
            """
        ).fetchall()

    temp_ids: list[str] = []
    exchange_ids: list[str] = []
    noise_journal_ids: list[str] = []

    for ep in episodes:
        eid = ep["id"]
        ctype = ep["content_type"]
        content = ep["content"] or ""
        tags = ep["tags"] or "[]"

        if _is_temp_episode(content, tags):
            temp_ids.append(eid)
            continue

        if ctype == "exchange":
            exchange_ids.append(eid)
            continue

        if ctype == "solution" and _is_noise_journal(content):
            noise_journal_ids.append(eid)

    cleanup_ids = sorted(set(temp_ids + exchange_ids + noise_journal_ids))

    print(f"Total active episodes: {len(episodes)}")
    print(f"Temp/test episodes: {len(temp_ids)}")
    print(f"Exchange episodes: {len(exchange_ids)}")
    print(f"Noise journal solutions: {len(noise_journal_ids)}")
    print(f"Total cleanup targets: {len(cleanup_ids)}")
    print(f"Would remain: {len(episodes) - len(cleanup_ids)}")

    print("\nSample noise journal removals:")
    for eid in noise_journal_ids[:12]:
        ep = next(e for e in episodes if e["id"] == eid)
        print(f"  {eid[:8]}  {ep['content'][:90]}")

    out = {
        "temp_ids": temp_ids,
        "exchange_ids": exchange_ids,
        "noise_journal_ids": noise_journal_ids,
        "recommended_cleanup_ids": cleanup_ids,
    }
    path = "benchmarks/results/cleanup_candidates.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()