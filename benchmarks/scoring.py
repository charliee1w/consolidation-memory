"""Scoring metrics: Token F1, BLEU-1, and LLM-as-Judge."""

from __future__ import annotations

import logging
from collections import Counter

from openai import OpenAI

from benchmarks.prompts import LLM_JUDGE_SYSTEM, LLM_JUDGE_USER

logger = logging.getLogger("benchmark.scoring")


def token_f1(predicted: str, gold: str) -> float:
    """Compute token-level F1 between predicted and gold answers.

    Lowercase, split on whitespace, compute precision/recall/F1.
    """
    pred_tokens = predicted.lower().split()
    gold_tokens = gold.lower().split()

    if not pred_tokens or not gold_tokens:
        return 0.0

    pred_counts = Counter(pred_tokens)
    gold_counts = Counter(gold_tokens)

    # Overlap = sum of min counts for each token
    overlap = sum((pred_counts & gold_counts).values())

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def bleu_1(predicted: str, gold: str) -> float:
    """Compute BLEU-1 (unigram precision with brevity penalty).

    Manual implementation — no nltk dependency required.
    """
    pred_tokens = predicted.lower().split()
    gold_tokens = gold.lower().split()

    if not pred_tokens:
        return 0.0

    gold_counts = Counter(gold_tokens)
    clipped = 0
    for token in pred_tokens:
        if gold_counts[token] > 0:
            clipped += 1
            gold_counts[token] -= 1

    precision = clipped / len(pred_tokens)

    # Brevity penalty
    bp = 1.0
    if len(pred_tokens) < len(gold_tokens):
        bp = min(1.0, len(pred_tokens) / len(gold_tokens)) if len(gold_tokens) > 0 else 0.0

    return bp * precision


def llm_judge(
    openai_client: OpenAI,
    predicted: str,
    gold: str,
    model: str = "gpt-4o-mini",
) -> int:
    """Use LLM to judge if predicted answer is correct.

    Returns:
        1 if CORRECT, 0 if INCORRECT.
    """
    prompt = LLM_JUDGE_USER.format(gold=gold, predicted=predicted)

    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": LLM_JUDGE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=10,
        )
        verdict = response.choices[0].message.content.strip().upper()
        return 1 if "CORRECT" in verdict else 0
    except Exception as e:
        logger.error("LLM judge error: %s", e)
        return 0


def compute_scores(
    predictions: list[dict],
    openai_client: OpenAI,
    model: str = "gpt-4o-mini",
) -> dict:
    """Compute all scores for a list of predictions.

    Args:
        predictions: List of dicts with 'predicted', 'gold', 'category', 'category_name'.
        openai_client: OpenAI client for LLM judge.
        model: Model for LLM judge.

    Returns:
        Dict with per-category and overall scores for each metric.
    """
    from collections import defaultdict

    by_category: dict[str, list[dict]] = defaultdict(list)

    for pred in predictions:
        f1 = token_f1(pred["predicted"], pred["gold"])
        b1 = bleu_1(pred["predicted"], pred["gold"])
        judge = llm_judge(openai_client, pred["predicted"], pred["gold"], model)

        scored = {**pred, "f1": f1, "bleu1": b1, "judge": judge}
        by_category[pred["category_name"]].append(scored)

    results = {}
    all_f1, all_b1, all_judge, total_count = [], [], [], 0

    for cat_name, items in sorted(by_category.items()):
        f1s = [it["f1"] for it in items]
        b1s = [it["bleu1"] for it in items]
        judges = [it["judge"] for it in items]
        n = len(items)

        results[cat_name] = {
            "count": n,
            "f1": round(sum(f1s) / n, 4) if n else 0,
            "bleu1": round(sum(b1s) / n, 4) if n else 0,
            "judge": round(sum(judges) / n, 4) if n else 0,
        }

        # Weight by category size for overall
        all_f1.extend(f1s)
        all_b1.extend(b1s)
        all_judge.extend(judges)
        total_count += n

    if total_count:
        results["overall"] = {
            "count": total_count,
            "f1": round(sum(all_f1) / total_count, 4),
            "bleu1": round(sum(all_b1) / total_count, 4),
            "judge": round(sum(all_judge) / total_count, 4),
        }
    else:
        results["overall"] = {"count": 0, "f1": 0, "bleu1": 0, "judge": 0}

    return results
