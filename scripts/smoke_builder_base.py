#!/usr/bin/env python3
"""Deterministic builder-baseline smoke test.

Runs a fast end-to-end check against a fresh temporary project and validates
the core Python + tool-dispatch surfaces without relying on external models.
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from consolidation_memory import MemoryClient
from consolidation_memory.config import reset_config
from consolidation_memory.database import close_all_connections, ensure_schema
from consolidation_memory.schemas import dispatch_tool_call, openai_tools

_SMOKE_DIM = 64


def _seed_for_text(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _normalized_vector(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(_SMOKE_DIM).astype(np.float32)
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        vec[0] = 1.0
        return vec
    return vec / norm


def _encode_documents(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, _SMOKE_DIM), dtype=np.float32)
    rows = [_normalized_vector(_seed_for_text(text)) for text in texts]
    return np.vstack(rows).astype(np.float32)


def _encode_query(text: str) -> np.ndarray:
    return _normalized_vector(_seed_for_text(text)).reshape(1, -1).astype(np.float32)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def run_smoke() -> None:
    with tempfile.TemporaryDirectory(prefix="cm-builder-smoke-") as tmp:
        base_dir = Path(tmp) / "data"
        reset_config(
            _base_data_dir=base_dir,
            active_project="builder-smoke",
            EMBEDDING_BACKEND="fastembed",
            EMBEDDING_DIMENSION=_SMOKE_DIM,
            LLM_BACKEND="disabled",
            CONSOLIDATION_AUTO_RUN=False,
        )
        close_all_connections()
        ensure_schema()

        with (
            patch("consolidation_memory.backends.encode_documents", side_effect=_encode_documents),
            patch("consolidation_memory.backends.encode_query", side_effect=_encode_query),
        ):
            with MemoryClient(auto_consolidate=False) as client:
                s1 = client.store(
                    "Builder smoke: plugin hook review workflow",
                    content_type="fact",
                    tags=["smoke", "builder"],
                )
                _assert(s1.status == "stored", "store() did not return status=stored")

                s2 = client.store(
                    "Builder smoke: run pytest and mypy before release",
                    content_type="solution",
                    tags=["smoke", "ci"],
                )
                _assert(s2.status == "stored", "second store() did not return status=stored")

                recall_result = client.recall(
                    "Builder smoke: plugin hook review workflow",
                    n_results=5,
                    include_knowledge=False,
                )
                _assert(recall_result.total_episodes >= 1, "recall() returned no episodes")

                search_result = client.search("run pytest and mypy", limit=5)
                _assert(search_result.total_matches >= 1, "search() returned no matches")

                status_result = client.status()
                _assert(
                    status_result.episodic_buffer["total"] >= 2,
                    "status() episodic buffer count is unexpectedly low",
                )

                _assert(len(openai_tools) >= 10, "openai_tools list is unexpectedly short")
                tool_result = dispatch_tool_call(
                    client,
                    "memory_recall",
                    {
                        "query": "plugin hook review workflow",
                        "n_results": 3,
                        "include_knowledge": False,
                    },
                )
                episodes = tool_result.get("episodes", [])
                _assert(bool(episodes), "dispatch_tool_call(memory_recall) returned no episodes")

                exported = client.export()
                _assert(exported.path is not None, "export() did not return a path")
                export_path = Path(str(exported.path))
                _assert(export_path.exists(), "export() path does not exist on disk")

        close_all_connections()


def main() -> None:
    run_smoke()
    print("builder smoke: ok")


if __name__ == "__main__":
    main()
