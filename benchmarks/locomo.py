"""LoCoMo benchmark CLI — orchestrates ingestion, answering, and scoring.

Usage:
    python -m benchmarks.locomo --mode full
    python -m benchmarks.locomo --mode episodes-only
    python -m benchmarks.locomo --mode full-context
    python -m benchmarks.locomo --mode all
"""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI

from benchmarks.ingestion import (
    get_qa_pairs,
    get_speakers,
    ingest_conversation,
    load_dataset,
)
from benchmarks.answering import answer_question
from benchmarks.baselines import answer_full_context, build_transcript
from benchmarks.scoring import compute_scores

logger = logging.getLogger("benchmark")

# Mem0 reference numbers (LLM Judge) from published results
MEM0_REF = {
    "single-hop": 67.1,
    "multi-hop": 51.2,
    "temporal": 55.5,
    "open-domain": 72.9,
    "overall": 66.9,
}


def run_memory_mode(
    dataset: list[dict],
    openai_client: OpenAI,
    mode: str,
    model: str,
    results_dir: Path,
) -> dict:
    """Run a memory-based mode (full or episodes-only).

    Creates an isolated MemoryClient per conversation, ingests episodes,
    optionally consolidates, then answers questions.
    """
    from consolidation_memory import MemoryClient
    from consolidation_memory.config import reset_config
    from consolidation_memory import database
    from consolidation_memory.backends import reset_backends
    from consolidation_memory import topic_cache, record_cache

    all_predictions = []

    for conv_idx, conversation in enumerate(dataset):
        sample_id = conversation.get("id", conversation.get("sample_id", f"conv_{conv_idx}"))
        speaker_a, speaker_b = get_speakers(conversation)
        logger.info(
            "[%d/%d] Processing %s (%s & %s) in %s mode",
            conv_idx + 1, len(dataset), sample_id, speaker_a, speaker_b, mode,
        )

        # Fresh isolated environment per conversation
        with tempfile.TemporaryDirectory(prefix=f"locomo_{sample_id}_") as tmpdir:
            tmp_path = Path(tmpdir)
            for d in ["knowledge", "knowledge/versions", "consolidation_logs", "backups"]:
                (tmp_path / "projects" / "default" / d).mkdir(parents=True, exist_ok=True)

            database.close_all_connections()
            reset_backends()
            topic_cache.invalidate()
            record_cache.invalidate()
            reset_config(_base_data_dir=tmp_path, active_project="default")

            with MemoryClient(auto_consolidate=False) as client:
                # Ingest all turns
                n_turns = ingest_conversation(client, conversation, sample_id)
                logger.info("  Ingested %d turns", n_turns)

                # Consolidate if full mode
                if mode == "full":
                    logger.info("  Running consolidation...")
                    report = client.consolidate()
                    logger.info("  Consolidation: %s", report.get("status", report))

                # Answer questions
                qa_pairs = get_qa_pairs(conversation)
                logger.info("  Answering %d questions...", len(qa_pairs))

                for qa_idx, qa in enumerate(qa_pairs):
                    predicted = answer_question(
                        openai_client, client,
                        qa["question"], speaker_a, speaker_b, model,
                    )
                    all_predictions.append({
                        "sample_id": sample_id,
                        "question": qa["question"],
                        "gold": qa["answer"],
                        "predicted": predicted,
                        "category": qa["category"],
                        "category_name": qa["category_name"],
                    })

                    if (qa_idx + 1) % 20 == 0:
                        logger.info("    Answered %d/%d", qa_idx + 1, len(qa_pairs))

            database.close_all_connections()

    # Score
    logger.info("Scoring %d predictions...", len(all_predictions))
    scores = compute_scores(all_predictions, openai_client, model)

    return {
        "mode": mode,
        "model": model,
        "predictions": all_predictions,
        "scores": scores,
        "n_predictions": len(all_predictions),
    }


def run_full_context(
    dataset: list[dict],
    openai_client: OpenAI,
    model: str,
) -> dict:
    """Run the full-context baseline — no memory system."""
    all_predictions = []

    for conv_idx, conversation in enumerate(dataset):
        sample_id = conversation.get("id", conversation.get("sample_id", f"conv_{conv_idx}"))
        speaker_a, speaker_b = get_speakers(conversation)
        logger.info(
            "[%d/%d] Full-context baseline for %s",
            conv_idx + 1, len(dataset), sample_id,
        )

        transcript = build_transcript(conversation)
        qa_pairs = get_qa_pairs(conversation)

        for qa_idx, qa in enumerate(qa_pairs):
            predicted = answer_full_context(
                openai_client, transcript,
                qa["question"], speaker_a, speaker_b, model,
            )
            all_predictions.append({
                "sample_id": sample_id,
                "question": qa["question"],
                "gold": qa["answer"],
                "predicted": predicted,
                "category": qa["category"],
                "category_name": qa["category_name"],
            })

            if (qa_idx + 1) % 20 == 0:
                logger.info("    Answered %d/%d", qa_idx + 1, len(qa_pairs))

    logger.info("Scoring %d predictions...", len(all_predictions))
    scores = compute_scores(all_predictions, openai_client, model)

    return {
        "mode": "full-context",
        "model": model,
        "predictions": all_predictions,
        "scores": scores,
        "n_predictions": len(all_predictions),
    }


def save_results(result: dict, results_dir: Path) -> Path:
    """Save results to a JSON file."""
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = results_dir / f"{result['mode']}_{timestamp}.json"
    # Don't serialize predictions in the summary file to keep it manageable
    summary = {k: v for k, v in result.items() if k != "predictions"}
    summary["predictions_count"] = len(result.get("predictions", []))
    path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    logger.info("Results saved to %s", path)
    return path


def print_results_table(results: dict[str, dict]) -> None:
    """Print a markdown comparison table to stdout."""
    categories = ["single-hop", "multi-hop", "temporal", "open-domain", "overall"]

    # Column headers
    modes = list(results.keys())
    header = "| Category     |"
    separator = "|--------------|"
    for mode in modes:
        label = mode.replace("-", " ").title() + " (J)"
        header += f" {label:<20}|"
        separator += f"-{'':->20}|"
    header += " Mem0 (ref) |"
    separator += "------------|"

    print()
    print(header)
    print(separator)

    for cat in categories:
        is_overall = cat == "overall"
        row = f"| {'**' + cat.title() + '**' if is_overall else cat.title():<13}|"

        for mode in modes:
            scores = results[mode].get("scores", {})
            cat_scores = scores.get(cat, {})
            judge_pct = cat_scores.get("judge", 0) * 100
            cell = f"{judge_pct:.1f}"
            if is_overall:
                cell = f"**{cell}**"
            row += f" {cell:<20}|"

        ref = MEM0_REF.get(cat, "—")
        if is_overall:
            row += f" **{ref}**     |"
        else:
            row += f" {ref:<11}|"

        print(row)

    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LoCoMo benchmark for consolidation-memory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["full", "episodes-only", "full-context", "all"],
        default="all",
        help="Run mode (default: all)",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model for answering and judging (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent / "data",
        help="Directory containing locomo10.json",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).parent / "results",
        help="Directory for output JSON files",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load dataset
    dataset = load_dataset(args.data_dir)

    # Initialize OpenAI client
    openai_client = OpenAI()  # Uses OPENAI_API_KEY env var

    modes_to_run = []
    if args.mode == "all":
        modes_to_run = ["full", "episodes-only", "full-context"]
    else:
        modes_to_run = [args.mode]

    all_results = {}
    start = time.monotonic()

    for mode in modes_to_run:
        mode_start = time.monotonic()
        logger.info("=== Running mode: %s ===", mode)

        if mode == "full-context":
            result = run_full_context(dataset, openai_client, args.model)
        else:
            result = run_memory_mode(dataset, openai_client, mode, args.model, args.results_dir)

        elapsed = time.monotonic() - mode_start
        result["elapsed_seconds"] = round(elapsed, 1)
        logger.info("Mode %s completed in %.1fs", mode, elapsed)

        save_results(result, args.results_dir)
        all_results[mode] = result

    total_elapsed = time.monotonic() - start
    logger.info("All modes completed in %.1fs", total_elapsed)

    # Print comparison table
    print_results_table(all_results)


if __name__ == "__main__":
    main()
