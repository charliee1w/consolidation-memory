"""FastEmbed backend — zero-config, downloads model on first use.

Uses Qdrant's fastembed library with ONNX Runtime.
Default model: BAAI/bge-small-en-v1.5 (384-dim, ~32MB quantized).
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

import numpy as np
from platformdirs import user_cache_dir

from consolidation_memory.backends.base import normalize_l2
from consolidation_memory.config import APP_NAME

logger = logging.getLogger(__name__)

_FASTEMBED_CACHE_ENV = "CONSOLIDATION_MEMORY_FASTEMBED_CACHE_DIR"
_CACHE_RECOVERY_MARKERS = (
    "NO_SUCHFILE",
    "File doesn't exist",
    "Files have been corrupted during downloading process",
)


def _default_fastembed_cache_dir() -> Path:
    env_override = os.environ.get(_FASTEMBED_CACHE_ENV)
    if env_override:
        return Path(env_override).expanduser()
    return Path(user_cache_dir(APP_NAME)) / "fastembed"


def _prepare_fastembed_cache_dir(cache_dir: str | Path | None = None) -> Path:
    resolved = Path(cache_dir) if cache_dir is not None else _default_fastembed_cache_dir()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _looks_like_recoverable_cache_error(exc: Exception) -> bool:
    message = str(exc)
    return any(marker in message for marker in _CACHE_RECOVERY_MARKERS)


def _reset_fastembed_cache(cache_dir: Path) -> None:
    if not cache_dir.exists():
        return
    for child in cache_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            try:
                child.unlink()
            except FileNotFoundError:
                continue


class FastEmbedEmbeddingBackend:
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        cache_dir: str | Path | None = None,
    ):
        try:
            from fastembed import TextEmbedding
        except ImportError:
            raise ImportError(
                "fastembed is required for the fastembed backend. "
                "Install it with: pip install consolidation-memory[fastembed]"
            )

        resolved_cache_dir = _prepare_fastembed_cache_dir(cache_dir)
        logger.info(
            "Loading FastEmbed model '%s' from cache '%s' (first run downloads ~32MB)...",
            model_name,
            resolved_cache_dir,
        )
        self._model = self._load_model(
            TextEmbedding=TextEmbedding,
            model_name=model_name,
            cache_dir=resolved_cache_dir,
        )
        self._model_name = model_name
        self._cache_dir = resolved_cache_dir
        # Probe actual dimension from model (don't rely on config fallback)
        probe = list(self._model.embed(["dimension probe"]))
        self._dim: int = len(probe[0])

    @staticmethod
    def _load_model(*, TextEmbedding, model_name: str, cache_dir: Path):
        try:
            return TextEmbedding(model_name, cache_dir=str(cache_dir))
        except Exception as exc:
            if not _looks_like_recoverable_cache_error(exc):
                raise
            logger.warning(
                "FastEmbed cache at '%s' appears incomplete or corrupt for model '%s'; "
                "clearing cache and retrying once",
                cache_dir,
                model_name,
            )
            _reset_fastembed_cache(cache_dir)
            return TextEmbedding(model_name, cache_dir=str(cache_dir))

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        embeddings = list(self._model.embed(texts))
        vecs = np.array(embeddings, dtype=np.float32)
        return normalize_l2(vecs)

    def encode_query(self, text: str) -> np.ndarray:
        embeddings = list(self._model.query_embed(text))
        vecs = np.array(embeddings, dtype=np.float32)
        return normalize_l2(vecs)

    @property
    def dimension(self) -> int:
        return self._dim
