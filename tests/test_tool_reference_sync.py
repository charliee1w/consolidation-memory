"""docs/TOOLS.md must stay in sync with the published tool schemas."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "generate_tool_reference.py"
    spec = importlib.util.spec_from_file_location("generate_tool_reference_script", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_tool_reference_is_up_to_date() -> None:
    module = _load_module()
    rendered = module.render()
    committed_path = Path(__file__).resolve().parents[1] / "docs" / "TOOLS.md"
    committed = committed_path.read_text(encoding="utf-8")
    assert rendered == committed, (
        "docs/TOOLS.md does not match the published tool schemas; "
        "run scripts/generate_tool_reference.py and commit the result"
    )


def test_render_forces_the_full_profile(monkeypatch) -> None:
    """A `simple` environment must not produce a three-tool reference.

    The profile is read when tools are registered, i.e. at server import, so a
    stale `CONSOLIDATION_MEMORY_MCP_TOOL_PROFILE=simple` would otherwise rewrite
    docs/TOOLS.md with three tools under a header that claims the full profile.
    """
    monkeypatch.setenv("CONSOLIDATION_MEMORY_MCP_TOOL_PROFILE", "simple")
    module = _load_module()

    rendered = module.render()

    assert "memory_hygiene_apply" in rendered
    assert "memory_detect_drift" in rendered
    assert os.environ[module._PROFILE_ENV] == "full"


def test_render_refuses_a_partial_listing() -> None:
    """The guard is the assertion, not the environment variable."""
    module = _load_module()

    class _Tool:
        name = "memory_recall"

    with pytest.raises(RuntimeError, match="full MCP profile"):
        module._assert_full_profile([_Tool()])
