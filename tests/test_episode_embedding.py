"""Tests for episode embedding and solution-shape helpers."""

from consolidation_memory.episode_embedding import (
    distinctive_token_set,
    embedding_text_for_episode,
    problem_query_from_content,
    solution_store_shape_warnings,
)


class TestProblemQueryFromContent:
    def test_strips_fix_section(self):
        content = (
            "MCP recall slowness on cold start. "
            "Fix: added disk-backed embedding cache at src/consolidation_memory/embedding_disk_cache.py"
        )
        query = problem_query_from_content(content)
        assert "Fix:" not in query
        assert "MCP recall slowness" in query

    def test_truncates_long_problem(self):
        content = " ".join(["word"] * 60) + " Fix: done"
        query = problem_query_from_content(content, max_len=40)
        assert len(query) <= 40


class TestSolutionStoreShapeWarnings:
    def test_warns_on_unstructured_journal(self):
        warnings = solution_store_shape_warnings(
            "Built a benchmark harness and saved baseline metrics without structure."
        )
        assert len(warnings) == 1
        assert "Problem:" in warnings[0]

    def test_no_warning_with_problem_line(self):
        warnings = solution_store_shape_warnings(
            "Problem: MCP recall times out\nFix: enable embedding cache"
        )
        assert warnings == []

    def test_no_warning_with_structured_json(self):
        warnings = solution_store_shape_warnings(
            '{"type": "solution", "problem": "timeouts", "fix": "cache embeddings"}'
        )
        assert warnings == []

    def test_no_warning_with_path_anchor(self):
        warnings = solution_store_shape_warnings(
            "Tests fail in tests/test_client.py until schema migration runs."
        )
        assert warnings == []


class TestDistinctiveTokenSet:
    def test_extracts_paths_and_leaf_names(self):
        content = "Fix in src/consolidation_memory/context_assembler.py for recall."
        tokens = distinctive_token_set(content)
        assert "src/consolidation_memory/context_assembler.py" in tokens
        assert "context_assembler.py" in tokens

    def test_extracts_commit_hashes(self):
        tokens = distinctive_token_set("Regression fixed in commit abc1234def.")
        assert "abc1234def" in tokens


class TestEmbeddingTextForEpisode:
    def test_solution_uses_problem_skew(self):
        content = (
            "Built benchmarks/real_world_eval.py for live evaluation. "
            "Fix: added Ollama health probe in src/consolidation_memory/client_runtime.py"
        )
        embed = embedding_text_for_episode(
            content=content,
            content_type="solution",
            tags=["benchmark"],
        )
        assert embed.startswith("Problem:")
        assert "Built benchmarks" in embed
        assert "Fix:" not in embed or "client_runtime.py" in embed
        assert "benchmark" in embed

    def test_non_solution_uses_full_content(self):
        content = "Routine exchange about project setup."
        embed = embedding_text_for_episode(content=content, content_type="exchange")
        assert embed == content