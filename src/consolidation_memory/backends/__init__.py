"""Backend factory and compatibility layer.

Provides drop-in replacements for the old embeddings.py functions:
  encode_documents(texts) -> np.ndarray
  encode_query(text) -> np.ndarray
  get_dimension() -> int

Backend is lazily instantiated on first use based on config.EMBEDDING_BACKEND.
"""

import logging
import threading
import time
from typing import Callable, TypeVar

import numpy as np

from consolidation_memory.backends.base import EmbeddingBackend, LLMBackend
from consolidation_memory.circuit_breaker import CircuitBreaker

logger = logging.getLogger(__name__)

T = TypeVar("T")


def retry_with_backoff(
    fn: Callable[[], T],
    *,
    max_retries: int = 3,
    base_delay: float = 2.0,
    transient_exceptions: tuple[type[Exception], ...] = (ConnectionError, TimeoutError, OSError),
    context: str = "backend call",
) -> T:
    """Retry a callable with exponential backoff on transient errors.

    Args:
        fn: Zero-argument callable to retry.
        max_retries: Maximum number of attempts.
        base_delay: Base delay in seconds (multiplied by attempt number).
        transient_exceptions: Exception types considered transient.
        context: Description for log messages.

    Returns:
        Result of fn().

    Raises:
        The last caught exception if all retries fail.
    """
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            return fn()
        except transient_exceptions as e:
            last_err = e
            logger.warning("%s attempt %d/%d failed: %s", context, attempt + 1, max_retries, e)
            if attempt < max_retries - 1:
                time.sleep(base_delay * (attempt + 1))
    raise last_err  # type: ignore[misc]

_backend_lock = threading.Lock()
_embedding_backend: EmbeddingBackend | None = None
_llm_backend: LLMBackend | None = None
_embed_circuit: CircuitBreaker | None = None


def _get_embed_circuit() -> CircuitBreaker:
    global _embed_circuit
    if _embed_circuit is None:
        with _backend_lock:
            if _embed_circuit is None:
                from consolidation_memory.config import get_config
                cfg = get_config()
                _embed_circuit = CircuitBreaker(
                    threshold=cfg.CIRCUIT_BREAKER_THRESHOLD,
                    cooldown=cfg.CIRCUIT_BREAKER_COOLDOWN,
                    name="embedding",
                )
    return _embed_circuit


def _create_embedding_backend() -> EmbeddingBackend:
    from consolidation_memory.config import get_config

    config = get_config()
    backend = config.EMBEDDING_BACKEND
    logger.info("Initializing embedding backend: %s", backend)

    if backend == "fastembed":
        from consolidation_memory.backends.fastembed_backend import FastEmbedEmbeddingBackend
        return FastEmbedEmbeddingBackend(model_name=config.EMBEDDING_MODEL_NAME)

    elif backend == "lmstudio":
        from consolidation_memory.backends.lmstudio import LMStudioEmbeddingBackend
        return LMStudioEmbeddingBackend(
            api_base=config.EMBEDDING_API_BASE,
            model_name=config.EMBEDDING_MODEL_NAME,
            dimension=config.EMBEDDING_DIMENSION,
        )

    elif backend == "openai":
        from consolidation_memory.backends.openai_backend import OpenAIEmbeddingBackend
        return OpenAIEmbeddingBackend(
            model_name=config.EMBEDDING_MODEL_NAME,
            dimension=config.EMBEDDING_DIMENSION,
            api_key=config.EMBEDDING_API_KEY,
            api_base=config.EMBEDDING_API_BASE if "localhost" not in config.EMBEDDING_API_BASE else None,
        )

    elif backend == "ollama":
        from consolidation_memory.backends.ollama import OllamaEmbeddingBackend
        return OllamaEmbeddingBackend(
            api_base=config.EMBEDDING_API_BASE,
            model_name=config.EMBEDDING_MODEL_NAME,
            dimension=config.EMBEDDING_DIMENSION,
        )

    else:
        raise ValueError(f"Unknown embedding backend: {backend!r}. "
                         f"Choose from: fastembed, lmstudio, openai, ollama")


def _create_llm_backend() -> LLMBackend | None:
    from consolidation_memory.config import get_config

    config = get_config()
    backend = config.LLM_BACKEND
    if backend == "disabled":
        logger.info("LLM backend disabled — consolidation will not run.")
        return None

    logger.info("Initializing LLM backend: %s", backend)

    if backend == "lmstudio":
        from consolidation_memory.backends.lmstudio import LMStudioLLMBackend
        return LMStudioLLMBackend(
            api_base=config.LLM_API_BASE,
            model=config.LLM_MODEL,
            max_tokens=config.LLM_MAX_TOKENS,
            temperature=config.LLM_TEMPERATURE,
            min_p=config.LLM_MIN_P,
        )

    elif backend == "openai":
        from consolidation_memory.backends.openai_backend import OpenAILLMBackend
        return OpenAILLMBackend(
            model=config.LLM_MODEL,
            api_key=config.LLM_API_KEY,
            api_base=config.LLM_API_BASE if "localhost" not in config.LLM_API_BASE else None,
            max_tokens=config.LLM_MAX_TOKENS,
            temperature=config.LLM_TEMPERATURE,
        )

    elif backend == "ollama":
        from consolidation_memory.backends.ollama import OllamaLLMBackend
        return OllamaLLMBackend(
            api_base=config.LLM_API_BASE,
            model=config.LLM_MODEL,
            max_tokens=config.LLM_MAX_TOKENS,
            temperature=config.LLM_TEMPERATURE,
        )

    else:
        raise ValueError(f"Unknown LLM backend: {backend!r}. "
                         f"Choose from: lmstudio, openai, ollama, disabled")


def get_embedding_backend() -> EmbeddingBackend:
    global _embedding_backend
    if _embedding_backend is None:
        with _backend_lock:
            if _embedding_backend is None:
                _embedding_backend = _create_embedding_backend()
    return _embedding_backend


def get_llm_backend() -> LLMBackend | None:
    global _llm_backend
    if _llm_backend is None:
        with _backend_lock:
            if _llm_backend is None:
                _llm_backend = _create_llm_backend()
    return _llm_backend


def reset_backends() -> None:
    """Reset cached backends (for testing or config reload)."""
    global _embedding_backend, _llm_backend, _embed_circuit
    with _backend_lock:
        _embedding_backend = None
        _llm_backend = None
        _embed_circuit = None


# ── Drop-in compatibility functions ──────────────────────────────────────────

def encode_documents(texts: list[str]) -> np.ndarray:
    """Encode texts for storage. Drop-in replacement for old embeddings.py."""
    cb = _get_embed_circuit()
    cb.check()
    try:
        result = get_embedding_backend().encode_documents(texts)
        cb.record_success()
        return result
    except ConnectionError:
        cb.record_failure()
        raise
    except Exception as e:
        cb.record_failure()
        raise ConnectionError(str(e)) from e


def encode_query(text: str) -> np.ndarray:
    """Encode a single query for retrieval. Drop-in replacement for old embeddings.py."""
    cb = _get_embed_circuit()
    cb.check()
    try:
        result = get_embedding_backend().encode_query(text)
        cb.record_success()
        return result
    except ConnectionError:
        cb.record_failure()
        raise
    except Exception as e:
        cb.record_failure()
        raise ConnectionError(str(e)) from e


def get_dimension() -> int:
    """Return the embedding dimension."""
    return get_embedding_backend().dimension
