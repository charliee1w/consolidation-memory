"""Shared claim embedding cache for recall.

Caches claim text embeddings for semantic claim ranking. The cache is keyed by
the exact ordered claim snapshot (id + updated_at + text), so updates naturally
invalidate without sacrificing ranking fidelity.
"""

from __future__ import annotations

import logging
import threading

import numpy as np

from consolidation_memory import backends
from consolidation_memory.query_semantics import parse_claim_payload

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_version: int = 0
_cache: dict[str, object] = {
    "version": -1,
    "signature": None,
    "vecs": None,
}


def invalidate() -> None:
    """Force re-embedding on next get_claim_vecs() call."""
    global _version
    with _lock:
        _version += 1
        _cache["signature"] = None
        _cache["vecs"] = None


def build_claim_texts(claims: list[dict]) -> tuple[list[dict[str, object]], list[str]]:
    """Return parsed payloads and embedding texts for claim rows."""
    payloads: list[dict[str, object]] = []
    texts: list[str] = []
    for claim in claims:
        payload = parse_claim_payload(claim.get("payload"))
        payloads.append(payload)
        payload_text = " ".join(f"{k} {v}" for k, v in sorted(payload.items()))
        texts.append(f"{claim.get('canonical_text', '')} {payload_text}".strip())
    return payloads, texts


def _snapshot_signature(claims: list[dict], texts: list[str]) -> tuple[tuple[str, str, str], ...]:
    """Build a deterministic signature for a claim snapshot."""
    signature: list[tuple[str, str, str]] = []
    for claim, text in zip(claims, texts):
        signature.append((
            str(claim.get("id", "")),
            str(claim.get("updated_at", "")),
            text,
        ))
    return tuple(signature)


def get_claim_vecs(claims: list[dict], texts: list[str]) -> np.ndarray | None:
    """Return claim embedding matrix with snapshot-aware caching."""
    if not claims:
        return None
    if len(claims) != len(texts):
        raise ValueError("claims/texts length mismatch")

    signature = _snapshot_signature(claims, texts)
    with _lock:
        if (
            _cache["version"] == _version
            and _cache["signature"] == signature
            and _cache["vecs"] is not None
        ):
            return _cache["vecs"]  # type: ignore[return-value]
        fetch_version = _version

    try:
        vecs = backends.encode_documents(texts)
    except Exception as e:
        logger.warning("Failed to embed claim texts: %s", e, exc_info=True)
        return None

    with _lock:
        if _version == fetch_version:
            _cache["version"] = fetch_version
            _cache["signature"] = signature
            _cache["vecs"] = vecs
        elif _cache["signature"] == signature and _cache["vecs"] is not None:
            return _cache["vecs"]  # type: ignore[return-value]

    return vecs


def warm_active_claim_vecs(limit: int) -> int:
    """Warm claim embedding cache for active claims. Returns warmed claim count."""
    from consolidation_memory.database import get_active_claims

    claims = get_active_claims(limit=max(1, int(limit)))
    if not claims:
        return 0
    _payloads, texts = build_claim_texts(claims)
    vecs = get_claim_vecs(claims, texts)
    if vecs is None:
        return 0
    return len(claims)
