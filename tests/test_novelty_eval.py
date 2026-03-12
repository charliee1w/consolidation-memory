"""Regression tests for novelty benchmark harness behavior."""

from __future__ import annotations

import tempfile
from pathlib import Path

from benchmarks.novelty_eval import (
    _local_embedding_patches,
    _reset_eval_environment,
    evaluate_temporal_belief_reconstruction,
)


def test_temporal_belief_reconstruction_passes_with_scoped_claim_provenance():
    local_tmp_base = Path.cwd() / ".tmp_novelty_eval_runtime"
    local_tmp_base.mkdir(parents=True, exist_ok=True)
    tmp_root = Path(tempfile.mkdtemp(prefix="novelty_eval_test_", dir=str(local_tmp_base)))

    _reset_eval_environment(tmp_root)
    with _local_embedding_patches():
        result = evaluate_temporal_belief_reconstruction(query_limit=4)

    assert result["pass"] is True
    assert result["measured"]["overall_macro_precision_at_5"] == 1.0
    for query in result["queries"]:
        assert len(query["top5_claim_ids"]) == 5
        assert query["relevant_hits_in_top5"] == 5
