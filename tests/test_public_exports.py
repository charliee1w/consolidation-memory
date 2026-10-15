"""The package root must export the result types callers need, lazily.

``consolidation_memory.__init__`` resolves public names through a
``_LAZY_IMPORTS`` table so a bare ``import consolidation_memory`` never pulls
faiss, numpy or SciPy. That table is also the package's public API list, so a
result type that is missing from it is a type consumers cannot import from the
root even though the tool that returns it is published.

``HygieneApplyResult`` and ``ConsolidationLogResult`` are the two result types
this PR added, and both were missing: a caller had to reach into
``consolidation_memory.types`` for a type the published output contract
(``tool_contracts``) already mirrors.
"""

from __future__ import annotations

import dataclasses
import subprocess
import sys
from pathlib import Path

import pytest

import consolidation_memory
from consolidation_memory import types

_SRC = Path(__file__).resolve().parents[1] / "src"

NEW_RESULT_TYPES = ["HygieneApplyResult", "ConsolidationLogResult"]


class TestRootExportsNewResultTypes:
    @pytest.mark.parametrize("name", NEW_RESULT_TYPES)
    def test_exported_from_the_package_root(self, name: str):
        assert name in consolidation_memory.__all__
        exported = getattr(consolidation_memory, name)
        assert exported is getattr(types, name)

    @pytest.mark.parametrize("name", NEW_RESULT_TYPES)
    def test_exported_type_is_the_real_result_dataclass(self, name: str):
        exported = getattr(consolidation_memory, name)
        assert dataclasses.is_dataclass(exported)
        assert exported.__module__ == "consolidation_memory.types"

    def test_hygiene_apply_result_payload_matches_the_mirrored_contract(self):
        """The export is the producer of the published MCP/REST contract."""
        from consolidation_memory.tool_contracts import HygieneApplyOutput

        payload = consolidation_memory.HygieneApplyResult(status="applied", forgotten=2).as_payload()
        assert set(payload) == set(HygieneApplyOutput.model_fields)

    def test_consolidation_log_result_is_the_client_return_type(self):
        """The other new public tool, ``memory_consolidation_log``."""
        result = consolidation_memory.ConsolidationLogResult(
            entries=[{"run_id": "r1"}], total=1, message="ok"
        )
        assert result.entries == [{"run_id": "r1"}]
        assert result.total == 1

    def test_unknown_names_still_raise_attribute_error(self):
        with pytest.raises(AttributeError, match="no attribute 'NotAResultType'"):
            consolidation_memory.NotAResultType  # noqa: B018


class TestLazyImportContract:
    def test_table_points_at_the_declaring_module(self):
        import importlib

        for name, module_name in consolidation_memory._LAZY_IMPORTS.items():
            assert module_name.startswith("consolidation_memory.")
            assert getattr(consolidation_memory, name) is getattr(
                importlib.import_module(module_name), name
            )

    def test_bare_import_pulls_no_heavy_or_result_modules(self):
        """The table must stay lazy: resolution happens on attribute access."""
        code = """
import sys
import consolidation_memory
heavy = [m for m in sys.modules if m.split(".")[0] in {"faiss", "numpy", "scipy"}]
assert not heavy, heavy
assert "consolidation_memory.types" not in sys.modules
assert consolidation_memory.HygieneApplyResult is not None
assert "consolidation_memory.types" in sys.modules
print("ok")
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env={"PYTHONPATH": str(_SRC), "PATH": "/usr/bin:/bin"},
            timeout=120,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "ok" in result.stdout
