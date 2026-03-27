"""LM Studio backend — embeddings and LLM via OpenAI-compatible API.

Embedding: Uses nomic-embed-text-v1.5 with task-specific prefixes.
LLM: Uses chat completions for consolidation summarization.
"""

import logging

import httpx
import numpy as np

from consolidation_memory.backends.base import normalize_l2

logger = logging.getLogger(__name__)

# Retry only transport/timeouts. HTTPStatusError for 4xx/5xx should be handled
# by caller logic, not blindly retried as transient.
_TRANSIENT = (httpx.TransportError, httpx.TimeoutException, ConnectionError, TimeoutError, OSError)


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

    def _build_payload(self, system_prompt: str, user_prompt: str) -> dict:
        return {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
            "min_p": self._min_p,
            "top_p": 1.0,
            "top_k": 0,
            "repeat_penalty": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        }

    def _chat_completion(
        self,
        payload: dict,
        *,
        allow_reasoning_content_fallback: bool = False,
    ) -> str:
        response = httpx.post(
            f"{self._api_base}/chat/completions",
            json=payload,
            timeout=120.0,
        )
        response.raise_for_status()
        payload_json = response.json()
        try:
            message = payload_json["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("LLM returned malformed response (missing choices[0].message)") from exc

        content = message.get("content")
        if isinstance(content, str):
            if content.strip():
                return content
            if allow_reasoning_content_fallback:
                reasoning = message.get("reasoning_content")
                if isinstance(reasoning, str) and reasoning.strip():
                    logger.warning(
                        "LM Studio returned empty message.content for structured output; "
                        "using message.reasoning_content fallback."
                    )
                    return reasoning
            raise ValueError("LLM returned empty response (message.content is empty)")

        if content is None:
            if allow_reasoning_content_fallback:
                reasoning = message.get("reasoning_content")
                if isinstance(reasoning, str) and reasoning.strip():
                    logger.warning(
                        "LM Studio returned message.content=None for structured output; "
                        "using message.reasoning_content fallback."
                    )
                    return reasoning
            raise ValueError("LLM returned empty response (message.content is None)")

        return str(content)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        from consolidation_memory.backends import retry_with_backoff

        payload = self._build_payload(system_prompt, user_prompt)

        def _do() -> str:
            return self._chat_completion(payload)

        result: str = retry_with_backoff(
            _do,
            transient_exceptions=_TRANSIENT,
            context="LM Studio LLM",
        )
        return result

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict,
    ) -> str:
        from consolidation_memory.backends import retry_with_backoff

        payload = self._build_payload(system_prompt, user_prompt)
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "consolidation_extraction",
                "strict": True,
                "schema": json_schema,
            },
        }

        def _do() -> str:
            try:
                return self._chat_completion(payload, allow_reasoning_content_fallback=True)
            except httpx.HTTPStatusError as e:
                if e.response.status_code in {400, 404, 415, 422}:
                    body_snippet = e.response.text[:300] if e.response.text else ""
                    raise ValueError(
                        "LM Studio rejected response_format=json_schema "
                        f"(HTTP {e.response.status_code}). "
                        "This backend requires structured output for extraction. "
                        f"Response body: {body_snippet}"
                    )
                raise

        result: str = retry_with_backoff(
            _do,
            transient_exceptions=_TRANSIENT,
            context="LM Studio LLM JSON",
        )
        return result
