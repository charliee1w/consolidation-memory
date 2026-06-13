"""Shared test helper functions.

Import these in test modules:
    from tests.helpers import (
        make_normalized_vec,
        make_normalized_batch,
        mock_encode,
        mock_embed_items_incremental,
    )
"""

import numpy as np


def make_normalized_vec(dim=384, seed=None):
    """Create a random L2-normalized vector."""
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return vec


def make_normalized_batch(n, dim=384, seed=None):
    """Create n random L2-normalized vectors."""
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


def mock_encode(texts):
    """Deterministic mock: hash text to seed a reproducible vector."""
    vecs = []
    for t in texts:
        seed = hash(t) % (2**31)
        vecs.append(make_normalized_vec(seed=seed))
    return np.array(vecs, dtype=np.float32)


def mock_embed_items_incremental(items, *, namespace, retain_ids=None):
    """Mock for ``embed_items_incremental`` — delegates to :func:`mock_encode`."""
    ordered_items = [(str(item_id), text) for item_id, text in items]
    if not ordered_items:
        return None
    texts = [text for _, text in ordered_items]
    if not any(text.strip() for text in texts):
        return None
    return mock_encode(texts)
