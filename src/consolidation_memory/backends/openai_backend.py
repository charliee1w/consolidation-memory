"""OpenAI backend — embeddings and LLM via OpenAI SDK.

Embedding default: text-embedding-3-small (1536-dim).
LLM default: gpt-4o-mini.
"""

import logging

import numpy as np

from consolidation_memory.backends import retry_with_backoff
from consolidation_memory.backends.base import normalize_l2

logger = logging.getLogger(__name__)


class OpenAIEmbeddingBackend:
    def __init__(self, model_name: str, dimension: int, api_key: str, api_base: str | None = None):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai is required for the openai embedding backend. "
                "Install it with: pip install consolidation-memory[openai]"
            ) from None
        kwargs: dict[str, str] = {"api_key": api_key}
        if api_base:
            kwargs["base_url"] = api_base
        self._client = OpenAI(**kwargs)  # type: ignore[arg-type]
        self._model_name = model_name
        self._dim = dimension

    def _get_transient_exceptions(self) -> tuple:
        """Return OpenAI SDK transient exception types (import-time safe)."""
        try:
            from openai import APIConnectionError, APIError, APITimeoutError
            return (APIError, APITimeoutError, APIConnectionError, ConnectionError, TimeoutError)
        except ImportError:
            return (ConnectionError, TimeoutError, OSError)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        transient = self._get_transient_exceptions()

        def _do():
            response = self._client.embeddings.create(input=texts, model=self._model_name)
            data = sorted(response.data, key=lambda x: x.index)
            return np.array([d.embedding for d in data], dtype=np.float32)

        vecs = retry_with_backoff(
            _do, transient_exceptions=transient, context="OpenAI embedding",
        )
        return normalize_l2(vecs)

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
            ) from None
        kwargs: dict[str, str] = {"api_key": api_key}
        if api_base:
            kwargs["base_url"] = api_base
        self._client = OpenAI(**kwargs)  # type: ignore[arg-type]
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature

    def _get_transient_exceptions(self) -> tuple:
        try:
            from openai import APIConnectionError, APIError, APITimeoutError
            return (APIError, APITimeoutError, APIConnectionError, ConnectionError, TimeoutError)
        except ImportError:
            return (ConnectionError, TimeoutError, OSError)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        transient = self._get_transient_exceptions()

        def _do():
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )
            content = response.choices[0].message.content
            if content is None:
                raise ValueError("LLM returned empty response (message.content is None)")
            return content

        result: str = retry_with_backoff(
            _do, transient_exceptions=transient, context="OpenAI LLM",
        )
        return result

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict,
    ) -> str:
        transient = self._get_transient_exceptions()

        def _do():
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=self._max_tokens,
                    temperature=self._temperature,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "consolidation_extraction",
                            "strict": True,
                            "schema": json_schema,
                        },
                    },
                )
            except Exception as e:
                # Some OpenAI-compatible endpoints don't support json_schema format.
                if "response_format" in str(e).lower() or "json_schema" in str(e).lower():
                    logger.warning(
                        "Structured output unsupported by OpenAI-compatible endpoint; "
                        "retrying without response_format"
                    )
                    response = self._client.chat.completions.create(
                        model=self._model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        max_tokens=self._max_tokens,
                        temperature=self._temperature,
                    )
                else:
                    raise
            content = response.choices[0].message.content
            if content is None:
                raise ValueError("LLM returned empty response (message.content is None)")
            return content

        result: str = retry_with_backoff(
            _do, transient_exceptions=transient, context="OpenAI LLM JSON",
        )
        return result
