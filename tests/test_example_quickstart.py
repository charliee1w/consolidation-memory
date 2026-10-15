"""The documented quickstart examples must actually run.

`README.md` and `examples/python-quickstart/quickstart.py` are the first code a
new user executes. Nothing else in CI covers them, so a snippet that reads
`.episodes[0].content` while `RecallResult.episodes` holds plain dicts shipped
green until someone ran it by hand. These tests execute the snippets instead of
restating them, so the shape they teach stays the shape the client returns.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

from consolidation_memory.types import RecallResult

_ROOT = Path(__file__).resolve().parents[1]


def _episode_rows(client: Any) -> RecallResult:
    """Store and recall one episode, the way the README snippet does."""
    client.store(
        "Fix: rerun pytest with -p no:randomly when the fixture order flakes",
        content_type="solution",
        tags=["pytest", "ci"],
    )
    return client.recall("pytest fixture order flake")


def test_readme_quickstart_snippet_runs() -> None:
    """`episodes[0]["content"]` is the access the README quickstart prints."""
    from consolidation_memory import MemoryClient

    with MemoryClient(auto_consolidate=False) as client:
        result = _episode_rows(client)

    assert result.episodes, "the quickstart stores an episode and recalls it"
    assert isinstance(result.episodes[0], dict), (
        "the README teaches subscript access, so an episode must be a dict; "
        f"got {type(result.episodes[0]).__name__}"
    )
    content = result.episodes[0]["content"]
    assert isinstance(content, str)
    assert "pytest" in content


def test_quickstart_example_module_runs() -> None:
    """`examples/python-quickstart/quickstart.py` executes end to end."""
    path = _ROOT / "examples" / "python-quickstart" / "quickstart.py"
    spec = importlib.util.spec_from_file_location("example_python_quickstart", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        module.main()
    finally:
        sys.modules.pop(spec.name, None)


def test_langgraph_example_reads_episode_content() -> None:
    """The LangGraph node's comprehension works on real recall rows.

    `langgraph` is not a project dependency, so the module is not imported; the
    comprehension it documents is executed against a live result instead.
    """
    from consolidation_memory import MemoryClient

    with MemoryClient(auto_consolidate=False) as client:
        result = _episode_rows(client)

    memory_hits = [str(episode["content"]) for episode in result.episodes]

    assert memory_hits
    assert all(isinstance(hit, str) and hit for hit in memory_hits)


@pytest.mark.parametrize(
    "relative_path",
    ["README.md", "examples/python-quickstart/quickstart.py"],
)
def test_quickstart_sources_use_dict_access(relative_path: str) -> None:
    """No published snippet reaches into an episode row as an object."""
    text = (_ROOT / relative_path).read_text(encoding="utf-8")

    for marker in ("episodes[0].", "episode.content", "for episode in result.episodes"):
        assert marker not in text, (
            f"{relative_path} reads an episode row with {marker!r}; "
            'RecallResult.episodes holds dicts, so use ["content"]'
        )