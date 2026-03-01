"""Protocol definitions for embedding and LLM backends."""

from typing import Protocol, runtime_checkable

import numpy as np


def normalize_l2(vecs: np.ndarray) -> np.ndarray:
    """L2-normalize embedding vectors. Zero vectors are left as-is."""
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    result: np.ndarray = vecs / norms
    return result


@runtime_checkable
class EmbeddingBackend(Protocol):
    """Interface for embedding providers."""

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        """Encode texts for storage. Returns L2-normalized (n, dim) float32 array."""
        ...

    def encode_query(self, text: str) -> np.ndarray:
        """Encode a single query for retrieval. Returns L2-normalized (1, dim) float32 array."""
        ...

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        ...


@runtime_checkable
class LLMBackend(Protocol):
    """Interface for LLM providers (used for consolidation summarization)."""

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a completion given system and user prompts."""
        ...
