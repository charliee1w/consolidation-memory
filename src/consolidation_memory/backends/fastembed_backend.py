"""FastEmbed backend — zero-config, downloads model on first use.

Uses Qdrant's fastembed library with ONNX Runtime.
Default model: BAAI/bge-small-en-v1.5 (384-dim, ~32MB quantized).
"""

import logging

import numpy as np

from consolidation_memory.backends.base import normalize_l2

logger = logging.getLogger(__name__)


class FastEmbedEmbeddingBackend:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        try:
            from fastembed import TextEmbedding
        except ImportError:
            raise ImportError(
                "fastembed is required for the fastembed backend. "
                "Install it with: pip install consolidation-memory[fastembed]"
            )
        logger.info("Loading FastEmbed model '%s' (first run downloads ~32MB)...", model_name)
        self._model = TextEmbedding(model_name)
        self._model_name = model_name
        self._dim: int | None = None

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        embeddings = list(self._model.embed(texts))
        vecs = np.array(embeddings, dtype=np.float32)
        if self._dim is None:
            self._dim = vecs.shape[1]
        return normalize_l2(vecs)

    def encode_query(self, text: str) -> np.ndarray:
        embeddings = list(self._model.query_embed(text))
        vecs = np.array(embeddings, dtype=np.float32)
        if self._dim is None:
            self._dim = vecs.shape[1]
        return normalize_l2(vecs)

    @property
    def dimension(self) -> int:
        if self._dim is not None:
            return self._dim
        from consolidation_memory.config import get_config
        return get_config().EMBEDDING_DIMENSION
