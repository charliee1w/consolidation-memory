"""LM Studio backend — embeddings and LLM via OpenAI-compatible API.

Embedding: Uses nomic-embed-text-v1.5 with task-specific prefixes.
LLM: Uses chat completions for consolidation summarization.
"""

import logging

import httpx
import numpy as np

from consolidation_memory.backends.base import normalize_l2

logger = logging.getLogger(__name__)

_TRANSIENT = (httpx.HTTPError, httpx.TimeoutException, ConnectionError, TimeoutError, OSError)


class LMStudioEmbeddingBackend:
    """Embedding via LM Studio's /v1/embeddings endpoint."""

    def __init__(self, api_base: str, model_name: str, dimension: int):
        self._api_base = api_base.rstrip("/")
        self._model_name = model_name
        self._dim = dimension
        self._embed_url = f"{self._api_base}/embeddings"

    def _embed(self, texts: list[str]) -> np.ndarray:
        from consolidation_memory.backends import retry_with_backoff

        def _do() -> list[dict]:  # type: ignore[type-arg]
            response = httpx.post(
                self._embed_url,
                json={"input": texts, "model": self._model_name},
                timeout=30.0,
            )
            response.raise_for_status()
            result: list[dict] = response.json()["data"]  # type: ignore[type-arg]
            return result

        data = retry_with_backoff(
            _do,
            transient_exceptions=_TRANSIENT,
            context="LM Studio embedding",
        )
        data.sort(key=lambda x: x["index"])
        vecs = np.array([d["embedding"] for d in data], dtype=np.float32)
        return normalize_l2(vecs)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        prefixed = [f"search_document: {t}" for t in texts]
        return self._embed(prefixed)

    def encode_query(self, text: str) -> np.ndarray:
        prefixed = [f"search_query: {text}"]
        return self._embed(prefixed)

    @property
    def dimension(self) -> int:
        return self._dim


class LMStudioLLMBackend:
    """LLM via LM Studio's /v1/chat/completions endpoint."""

    def __init__(
        self,
        api_base: str,
        model: str,
        max_tokens: int = 2048,
        temperature: float = 0.3,
        min_p: float = 0.05,
    ):
        self._api_base = api_base.rstrip("/")
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._min_p = min_p

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        from consolidation_memory.backends import retry_with_backoff

        def _do() -> str:
            response = httpx.post(
                f"{self._api_base}/chat/completions",
                json={
                    "model": self._model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "max_tokens": self._max_tokens,
                    "temperature": self._temperature,
                    "min_p": self._min_p,
                },
                timeout=120.0,
            )
            response.raise_for_status()
            return str(response.json()["choices"][0]["message"]["content"])

        result: str = retry_with_backoff(
            _do,
            transient_exceptions=_TRANSIENT,
            context="LM Studio LLM",
        )
        return result
