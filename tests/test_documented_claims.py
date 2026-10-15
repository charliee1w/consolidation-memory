"""Documentation claims that drift silently.

CLAUDE.md's guardrail 4 exists because a dated or wrong statement in the docs
rots without any test noticing: nothing imports a markdown file, so the only way
a claim stays true is if something checks it. These are the claims from that
class which can be checked against code rather than re-read by a reviewer.

The rest of the docs are prose about behaviour. A claim here is one where the
code says something different, or where a stated mechanism is unreachable.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

from consolidation_memory import ui_ops, web_ui

_ROOT = Path(__file__).resolve().parents[1]


def _doc(relative: str) -> str:
    return (_ROOT / relative).read_text(encoding="utf-8")


def test_httpx_is_a_core_dependency() -> None:
    """Both example READMEs told readers to install `httpx` separately.

    It is a main dependency, so the instruction sent people to install something
    they already had and implied it was optional.
    """
    pyproject = _doc("pyproject.toml")
    assert 'httpx>=' in pyproject

    for relative in ("examples/README.md", "examples/rest-api/README.md"):
        text = _doc(relative)
        assert "not part of any project extra" not in text, relative
        assert "no project extra provides" not in text, relative


def test_ui_metrics_route_reads_the_eval_report_not_the_dashboard() -> None:
    """UI.md grouped `metrics` with the `DashboardData` read helpers.

    It does not: the route calls `load_metrics_for_ui`, which reads a
    `real_world_eval` JSON report and falls back to the bundled
    `web/published_metrics.json`. Nothing in that path touches the database.
    """
    source = inspect.getsource(web_ui)
    assert "load_metrics_for_ui()" in source
    assert "build_ops_overview" in source, "overview should still use DashboardData"

    assert "def load_metrics_for_ui" in inspect.getsource(ui_ops)
    assert "real_world_eval" in inspect.getsource(ui_ops)

    ui_doc = _doc("docs/UI.md")
    assert "ui_ops.load_metrics_for_ui()" in ui_doc
    assert "`overview`, `recent` and `metrics`" not in ui_doc


def test_consolidation_complete_fires_on_failure_too() -> None:
    """PLUGIN_DEVELOPMENT.md said the hook fires on 'success or partial'.

    The engine also fires it from the run's exception handler and from three
    early error exits, so a plugin counting successful runs overcounted.
    """
    from consolidation_memory.consolidation import engine

    source = inspect.getsource(engine)
    fires = source.count('"on_consolidation_complete"')
    assert fires >= 2, f"expected the hook to fire on more than one path, saw {fires}"

    plugin_doc = _doc("docs/PLUGIN_DEVELOPMENT.md")
    assert "success or partial" not in plugin_doc
    assert "exception handler" in plugin_doc


def test_recall_deadline_ratio_is_the_spend_not_the_reserve() -> None:
    """MCP_GUIDE described 0.85 as the share reserved before fallback.

    It is the share the internal deadline spans; the reserve is the remainder.
    """
    from consolidation_memory.tool_adapter import recall_deadline_margin_ratio

    ratio = recall_deadline_margin_ratio()
    assert 0 < ratio < 1

    guide = _doc("docs/MCP_GUIDE.md")
    line = next(
        line for line in guide.splitlines() if "RECALL_DEADLINE_MARGIN_RATIO" in line
    )
    assert "reserved" not in line, (
        "the margin ratio is the fraction the deadline spans, not the reserve: "
        f"{line!r}"
    )


def test_protected_file_list_matches_the_guard_paths() -> None:
    """The guard's own list had six entries; the code and the notes say seven."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "check_release_docs_under_test", _ROOT / "scripts" / "check_release_docs.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    automation_doc = _doc("docs/RELEASE_AUTOMATION.md")
    listed = re.search(
        r"release automation scripts or workflows \(([^)]*)\)", automation_doc
    )
    assert listed, "the protected-file sentence is gone; this test needs updating"
    names = {part.strip().strip("`") for part in listed.group(1).split(",")}
    # The doc names workflows bare, the guard's set carries .github/workflows/.
    bare = {name.rsplit("/", 1)[-1] for name in module.RELEASE_AUTOMATION_PATHS}

    assert names == bare, (
        "the documented list and RELEASE_AUTOMATION_PATHS disagree: "
        f"{sorted(names ^ bare)}"
    )
    assert "the seven in" in automation_doc