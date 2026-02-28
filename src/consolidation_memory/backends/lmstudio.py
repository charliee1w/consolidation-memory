"""LM Studio backend — embeddings and LLM via OpenAI-compatible API.

Embedding: Uses nomic-embed-text-v1.5 with task-specific prefixes.
LLM: Uses chat completions for consolidation summarization.
"""

import json
import logging
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

import numpy as np

logger = logging.getLogger(__name__)


class LMStudioEmbeddingBackend:
    """Embedding via LM Studio's /v1/embeddings endpoint."""

    def __init__(self, api_base: str, model_name: str, dimension: int):
        self._api_base = api_base.rstrip("/")
        self._model_name = model_name
        self._dim = dimension
        self._embed_url = f"{self._api_base}/embeddings"

    def _embed(self, texts: list[str]) -> np.ndarray:
        payload = json.dumps({"input": texts, "model": self._model_name}).encode()
        req = Request(self._embed_url, data=payload, headers={"Content-Type": "application/json"})

        last_err = None
        for attempt in range(3):
            try:
                with urlopen(req, timeout=30) as resp:
                    body = json.loads(resp.read())
                data = body["data"]
                break
            except (URLError, ConnectionError, TimeoutError, KeyError) as e:
                last_err = e
                logger.warning("LM Studio embedding attempt %d failed: %s", attempt + 1, e)
                if attempt < 2:
                    time.sleep(1.0 * (attempt + 1))
        else:
            raise ConnectionError(
                f"LM Studio embedding API unreachable after 3 attempts: {last_err}. "
                f"Is LM Studio running with {self._model_name} loaded?"
            )

        data.sort(key=lambda x: x["index"])
        vecs = np.array([d["embedding"] for d in data], dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized: np.ndarray = vecs / norms
        return normalized

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
        import httpx

        last_err = None
        for attempt in range(3):
            try:
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
                        "stop": ["<|im_end|>"],
                    },
                    timeout=120.0,
                )
                response.raise_for_status()
                return str(response.json()["choices"][0]["message"]["content"])
            except (httpx.HTTPError, httpx.TimeoutException, KeyError, ConnectionError) as e:
                last_err = e
                logger.warning("LM Studio LLM attempt %d/3 failed: %s", attempt + 1, e)
                if attempt < 2:
                    time.sleep(2.0 * (attempt + 1))

        raise ConnectionError(
            f"LM Studio LLM API failed after 3 attempts: {last_err}. "
            f"Is LM Studio running with {self._model} loaded?"
        )
