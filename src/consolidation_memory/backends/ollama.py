"""Ollama backend — embeddings and LLM via Ollama's API.

Embedding default: nomic-embed-text (768-dim).
LLM: Any chat model served by Ollama.
"""

import logging

import httpx
import numpy as np

from consolidation_memory.backends import retry_with_backoff

logger = logging.getLogger(__name__)

_TRANSIENT = (httpx.HTTPError, httpx.TimeoutException, ConnectionError, TimeoutError, OSError)


class OllamaEmbeddingBackend:
    def __init__(self, api_base: str, model_name: str, dimension: int):
        # Strip trailing /v1 or /v1/ if present (common misconfiguration from LM Studio defaults)
        base = api_base.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        self._api_base = base
        self._model_name = model_name
        self._dim = dimension

    def _normalize(self, vecs: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized: np.ndarray = vecs / norms
        return normalized

    def _embed_single(self, text: str) -> list[float]:
        response = httpx.post(
            f"{self._api_base}/api/embed",
            json={"model": self._model_name, "input": text},
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()
        result: list[float] = data["embeddings"][0]
        return result

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        def _do():
            response = httpx.post(
                f"{self._api_base}/api/embed",
                json={"model": self._model_name, "input": texts},
                timeout=60.0,
            )
            response.raise_for_status()
            return response.json()

        data = retry_with_backoff(
            _do, transient_exceptions=_TRANSIENT, context="Ollama embedding",
        )
        vecs = np.array(data["embeddings"], dtype=np.float32)
        if self._dim == 0:
            self._dim = vecs.shape[1]
        return self._normalize(vecs)

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode_documents([text])

    @property
    def dimension(self) -> int:
        return self._dim


class OllamaLLMBackend:
    def __init__(
        self,
        api_base: str = "http://localhost:11434",
        model: str = "qwen2.5:7b",
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ):
        base = api_base.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        self._api_base = base
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        def _do():
            response = httpx.post(
                f"{self._api_base}/api/chat",
                json={
                    "model": self._model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "stream": False,
                    "options": {
                        "num_predict": self._max_tokens,
                        "temperature": self._temperature,
                    },
                },
                timeout=120.0,
            )
            response.raise_for_status()
            return response.json()["message"]["content"]

        result: str = retry_with_backoff(
            _do, transient_exceptions=_TRANSIENT, context="Ollama LLM",
        )
        return result
