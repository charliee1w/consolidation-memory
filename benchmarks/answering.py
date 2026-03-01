"""Query memory and generate answers using gpt-4o-mini."""

from __future__ import annotations

import logging

from openai import OpenAI

from benchmarks.prompts import ANSWER_SYSTEM, ANSWER_USER

logger = logging.getLogger("benchmark.answering")


def answer_question(
    openai_client: OpenAI,
    client,
    question: str,
    speaker_a: str,
    speaker_b: str,
    model: str = "gpt-4o-mini",
) -> str:
    """Recall memories and generate an answer for a single question.

    Args:
        openai_client: OpenAI client instance.
        client: MemoryClient to recall from.
        question: The question to answer.
        speaker_a: First speaker name.
        speaker_b: Second speaker name.
        model: OpenAI model to use.

    Returns:
        The predicted answer string.
    """
    # Recall relevant memories
    result = client.recall(
        query=question,
        n_results=20,
        include_knowledge=True,
        include_expired=True,
    )

    # Build context from retrieved episodes and knowledge
    context_parts = []

    for ep in result.episodes:
        content = ep.get("content", "")
        context_parts.append(content)

    for k in result.knowledge:
        title = k.get("title", "")
        summary = k.get("summary", "")
        if title or summary:
            context_parts.append(f"[Knowledge: {title}] {summary}")

    for r in result.records:
        content = r.get("content", {})
        if isinstance(content, dict):
            # Flatten record content
            parts = []
            for key, val in content.items():
                if val:
                    parts.append(f"{key}: {val}")
            context_parts.append(" | ".join(parts))
        elif isinstance(content, str):
            context_parts.append(content)

    context = "\n".join(context_parts) if context_parts else "No relevant memories found."

    # Call gpt-4o-mini
    prompt = ANSWER_USER.format(
        speaker_a=speaker_a,
        speaker_b=speaker_b,
        context=context,
        question=question,
    )

    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": ANSWER_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=50,
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        logger.error("OpenAI API error: %s", e)
        answer = ""

    return answer
