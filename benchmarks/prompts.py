"""Prompt templates for the LoCoMo benchmark."""

ANSWER_SYSTEM = (
    "You are answering questions about conversations between two people. "
    "Use ONLY the provided memory context to answer. "
    "Pay close attention to timestamps and dates. "
    "Convert relative time references (e.g., 'last week', 'recently') to specific dates when possible. "
    "Answer concisely in under 6 words."
)

ANSWER_USER = (
    "Based on these memories from conversations between {speaker_a} and {speaker_b}, "
    "answer the question.\n\n"
    "MEMORY CONTEXT:\n{context}\n\n"
    "QUESTION: {question}\n\n"
    "Answer in under 6 words."
)

FULL_CONTEXT_SYSTEM = (
    "You are answering questions about conversations between two people. "
    "Use ONLY the provided conversation transcript to answer. "
    "Pay close attention to timestamps and dates. "
    "Convert relative time references to specific dates when possible. "
    "Answer concisely in under 6 words."
)

FULL_CONTEXT_USER = (
    "Here is the full conversation transcript between {speaker_a} and {speaker_b}:\n\n"
    "{transcript}\n\n"
    "QUESTION: {question}\n\n"
    "Answer in under 6 words."
)

LLM_JUDGE_SYSTEM = "You are a precise answer evaluator."

LLM_JUDGE_USER = (
    "Given the ground truth answer '{gold}', is the predicted answer '{predicted}' correct? "
    "Consider semantic equivalence — the prediction doesn't need to match word-for-word, "
    "but must convey the same information. "
    "Reply with exactly one word: CORRECT or INCORRECT."
)
