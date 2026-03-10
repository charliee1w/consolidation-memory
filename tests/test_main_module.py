"""Tests for `python -m consolidation_memory` entrypoint behavior."""

from __future__ import annotations

import runpy
from unittest.mock import patch


def test_main_module_invokes_cli_main_once():
    with patch("consolidation_memory.cli.main") as mock_main:
        runpy.run_module("consolidation_memory.__main__", run_name="__main__")
    mock_main.assert_called_once_with()
