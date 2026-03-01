"""Load locomo10.json and ingest into MemoryClient instances."""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger("benchmark.ingestion")

# LoCoMo category mapping (from the actual JSON, NOT the paper ordering):
# 1 = multi-hop, 2 = temporal, 3 = open-domain, 4 = single-hop, 5 = adversarial
CATEGORY_MAP = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
    5: "adversarial",
}


def load_dataset(data_path: Path) -> list[dict]:
    """Load locomo10.json and return the list of conversations."""
    locomo_file = data_path / "locomo10.json"
    if not locomo_file.exists():
        raise FileNotFoundError(
            f"locomo10.json not found at {locomo_file}. "
            f"Download from https://github.com/snap-research/locomo "
            f"and place in {data_path}/"
        )
    with open(locomo_file, encoding="utf-8") as f:
        data = json.load(f)
    logger.info("Loaded %d conversations from locomo10.json", len(data))
    return data


def get_speakers(conversation: dict) -> tuple[str, str]:
    """Extract the two speaker names from a conversation."""
    speakers = set()
    for session in conversation.get("conversation", []):
        for turn in session.get("turns", session.get("dialogue", [])):
            speaker = turn.get("speaker", turn.get("role", ""))
            if speaker:
                speakers.add(speaker)
    speakers = sorted(speakers)
    if len(speakers) >= 2:
        return speakers[0], speakers[1]
    return "Speaker A", "Speaker B"


def ingest_conversation(
    client,
    conversation: dict,
    sample_id: str,
) -> int:
    """Ingest all turns from a conversation into a MemoryClient.

    Each dialogue turn becomes an episode with session timestamps prepended.

    Returns:
        Number of episodes ingested.
    """
    count = 0
    sessions = conversation.get("conversation", [])

    for session_idx, session in enumerate(sessions):
        session_date = session.get("date", session.get("session_date", ""))
        session_time = session.get("time", session.get("session_time", ""))
        timestamp = f"{session_date} {session_time}".strip() if session_date else ""

        turns = session.get("turns", session.get("dialogue", []))
        for turn in turns:
            speaker = turn.get("speaker", turn.get("role", "Unknown"))
            text = turn.get("text", turn.get("content", turn.get("utterance", "")))
            if not text:
                continue

            # Prepend timestamp for temporal question answering
            if timestamp:
                content = f"[{timestamp}] {speaker}: {text}"
            else:
                content = f"{speaker}: {text}"

            client.store(
                content=content,
                content_type="exchange",
                tags=[sample_id, f"session_{session_idx}"],
                surprise=0.5,
            )
            count += 1

    logger.info("Ingested %d turns for %s", count, sample_id)
    return count


def get_qa_pairs(conversation: dict) -> list[dict]:
    """Extract QA pairs from a conversation, skipping adversarial (category 5).

    Returns list of dicts with keys: question, answer, category (int), category_name (str).
    """
    pairs = []
    for qa in conversation.get("qa_pairs", conversation.get("questions", [])):
        category = qa.get("category", qa.get("type", 0))
        if isinstance(category, str):
            # Try to parse category from string
            for k, v in CATEGORY_MAP.items():
                if v == category.lower():
                    category = k
                    break
            else:
                continue

        if category == 5:  # Skip adversarial
            continue

        question = qa.get("question", qa.get("query", ""))
        answer = qa.get("answer", qa.get("gold_answer", ""))
        if not question or not answer:
            continue

        pairs.append({
            "question": question,
            "answer": answer,
            "category": category,
            "category_name": CATEGORY_MAP.get(category, f"unknown_{category}"),
        })

    return pairs
