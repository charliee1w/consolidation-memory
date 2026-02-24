"""OpenAI backend — embeddings and LLM via OpenAI SDK.

Embedding default: text-embedding-3-small (1536-dim).
LLM default: gpt-4o-mini.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class OpenAIEmbeddingBackend:
    def __init__(self, model_name: str, dimension: int, api_key: str, api_base: str | None = None):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai is required for the openai backend. "
                "Install it with: pip install consolidation-memory[openai]"
            )
        kwargs = {"api_key": api_key}
        if api_base:
            kwargs["base_url"] = api_base
        self._client = OpenAI(**kwargs)
        self._model_name = model_name
        self._dim = dimension

    def _normalize(self, vecs: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        response = self._client.embeddings.create(input=texts, model=self._model_name)
        data = sorted(response.data, key=lambda x: x.index)
        vecs = np.array([d.embedding for d in data], dtype=np.float32)
        return self._normalize(vecs)

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode_documents([text])

    @property
    def dimension(self) -> int:
        return self._dim


class OpenAILLMBackend:
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str = "",
        api_base: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai is required for the openai LLM backend. "
                "Install it with: pip install consolidation-memory[openai]"
            )
        kwargs = {"api_key": api_key}
        if api_base:
            kwargs["base_url"] = api_base
        self._client = OpenAI(**kwargs)
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )
        return response.choices[0].message.content
