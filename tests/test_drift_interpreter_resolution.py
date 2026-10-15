"""The drift worker must run on the interpreter the server itself runs on.

Regression cover for the reported failure: in a virtualenv whose ``bin/python``
is a symlink to a base interpreter that does not have the distribution installed,
resolving that symlink handed the worker the base interpreter. The child could
then import a different dependency set, and importing ``consolidation_memory``
died on the version lookup, surfacing as
``RuntimeError: Isolated drift detection failed: ... PackageNotFoundError``.

These tests build the reported environment for real (a throwaway virtualenv) and
run the actual child command inside it, rather than mocking the subprocess.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest

import consolidation_memory
from consolidation_memory import drift_subprocess

_REPO_SRC = Path(__file__).resolve().parents[1] / "src"
_INIT_PATH = _REPO_SRC / "consolidation_memory" / "__init__.py"
_VERSION_PROBE = "import importlib.metadata as m; print(m.version('consolidation-memory'))"


def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":  # pragma: no cover - exercised on Windows only
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _venv_prefix(python_path: Path) -> Path:
    return python_path.parent.parent


@pytest.fixture(scope="module")
def symlinked_venv(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Virtualenv whose ``bin/python`` is a symlink to a bare base interpreter.

    This is the standard POSIX layout and the exact precondition of the report.
    """
    venv_dir = tmp_path_factory.mktemp("symlinked-venv")
    try:
        subprocess.run(
            [sys.executable, "-m", "venv", "--without-pip", str(venv_dir)],
            check=True,
            capture_output=True,
            timeout=180,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        pytest.skip(f"virtualenv creation unavailable on this platform: {exc}")

    python_path = _venv_python(venv_dir)
    if not python_path.exists():  # pragma: no cover - defensive
        pytest.skip(f"virtualenv has no interpreter at {python_path}")
    if not python_path.is_symlink():
        pytest.skip(
            f"{python_path} is not a symlink; the reported collapse cannot happen "
            "on a platform where venv copies the interpreter binary"
        )
    return python_path


@pytest.fixture(scope="module")
def clean_source_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Sources reachable by ``PYTHONPATH`` with no distribution metadata beside them.

    A develop-mode install (``pip install -e .`` with the legacy backend) leaves
    ``src/consolidation_memory.egg-info`` in the tree, and that directory alone
    lets ``importlib.metadata`` resolve a version. Copying the package to a
    scratch root keeps the precondition reproducible on any machine.
    """
    root = tmp_path_factory.mktemp("clean-source") / "src"
    shutil.copytree(
        _REPO_SRC / "consolidation_memory",
        root / "consolidation_memory",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info", "*.dist-info"),
    )
    return root


def test_symlinked_venv_keeps_its_own_environment(symlinked_venv: Path) -> None:
    """Precondition: the symlinked interpreter is a real, separate environment."""
    probe = subprocess.run(
        [str(symlinked_venv), "-c", "import sys; print(sys.prefix); print(sys.base_prefix)"],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    venv_prefix_value, base_prefix_value = probe.stdout.split()

    assert Path(venv_prefix_value) == _venv_prefix(symlinked_venv)
    assert Path(base_prefix_value) != Path(venv_prefix_value)


def test_base_interpreter_has_no_distribution_metadata(
    symlinked_venv: Path, clean_source_root: Path
) -> None:
    """Precondition: the dist-info stays invisible even with the sources reachable.

    This is what made the old code crash with ``PackageNotFoundError`` once the
    package itself became importable through ``PYTHONPATH``.
    """
    probe = subprocess.run(
        [str(symlinked_venv), "-c", _VERSION_PROBE],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env={**os.environ, "PYTHONPATH": str(clean_source_root)},
    )

    assert probe.returncode != 0
    assert "PackageNotFoundError" in probe.stderr


def test_clean_source_root_really_has_no_distribution_metadata(
    clean_source_root: Path,
) -> None:
    """The precondition above is hermetic, not an accident of the working copy.

    A develop-mode install leaves a ``src/*.egg-info`` behind, and that directory
    is enough for ``importlib.metadata`` to resolve a version from sources alone.
    Depending on its absence would make this suite machine-dependent — which is
    exactly how the original bug hid.
    """
    assert not list(clean_source_root.glob("*.egg-info"))
    assert not list(clean_source_root.glob("*.dist-info"))
    assert (clean_source_root / "consolidation_memory" / "__init__.py").is_file()


def test_resolve_python_executable_keeps_the_virtualenv(
    monkeypatch: pytest.MonkeyPatch, symlinked_venv: Path
) -> None:
    """The venv interpreter is returned, not the base interpreter behind it."""
    base_interpreter = symlinked_venv.resolve()

    monkeypatch.setattr(sys, "executable", str(symlinked_venv))
    resolved = Path(drift_subprocess._resolve_python_executable())

    assert resolved == symlinked_venv
    assert str(_venv_prefix(symlinked_venv)) in str(resolved)
    assert str(resolved) != str(base_interpreter), (
        "resolution collapsed the virtualenv down to the base interpreter"
    )


def test_drift_command_targets_the_virtualenv_interpreter(
    monkeypatch: pytest.MonkeyPatch, symlinked_venv: Path
) -> None:
    monkeypatch.setattr(sys, "executable", str(symlinked_venv))
    monkeypatch.setattr(
        "consolidation_memory.config.get_active_project", lambda: "default"
    )

    cmd = drift_subprocess._build_drift_command(base_ref=None, repo_path=None)

    assert cmd[0] == str(symlinked_venv)
    assert cmd[1:3] == ["-m", "consolidation_memory.drift_worker"]


def test_child_environment_imports_the_package_without_metadata(
    monkeypatch: pytest.MonkeyPatch, symlinked_venv: Path
) -> None:
    """The child can import the package and read a usable ``__version__``.

    No dist-info is reachable here, so the version lookup fails; the child must
    still start. This is the exact exception the drift worker used to die with.
    """
    monkeypatch.delenv("PYTHONPATH", raising=False)
    child_env = drift_subprocess._build_child_env()

    probe = subprocess.run(
        [
            str(symlinked_venv),
            "-c",
            "import consolidation_memory as m; print(m.__version__); print(m.__file__)",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env=child_env,
    )

    assert probe.returncode == 0, probe.stderr
    version, module_file = probe.stdout.split()
    assert version, "child reported an empty version"
    assert Path(module_file).is_relative_to(_REPO_SRC)


def test_drift_worker_command_completes_in_the_virtualenv(
    monkeypatch: pytest.MonkeyPatch, symlinked_venv: Path
) -> None:
    """The real worker command starts and completes in the reported layout.

    ``drift_worker --help`` exercises the same import chain a full scan uses
    (``consolidation_memory`` -> ``types`` -> argparse) with only stdlib, so it
    runs inside a dependency-free throwaway virtualenv.
    """
    monkeypatch.delenv("PYTHONPATH", raising=False)
    child_env = drift_subprocess._build_child_env()

    probe = subprocess.run(
        [str(symlinked_venv), "-m", "consolidation_memory.drift_worker", "--help"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env=child_env,
    )

    assert probe.returncode == 0, probe.stderr
    assert "consolidation_memory.drift_worker" in probe.stdout


def test_child_environment_forwards_only_the_import_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The parent environment is inherited, and only PYTHONPATH is adjusted."""
    inherited = tmp_path / "inherited"
    inherited.mkdir()
    monkeypatch.setenv(
        "PYTHONPATH", os.pathsep.join([str(inherited), str(_REPO_SRC)])
    )
    monkeypatch.setenv("CONSOLIDATION_MEMORY_PROJECT", "probe-sentinel")

    env = drift_subprocess._build_child_env()
    entries = env["PYTHONPATH"].split(os.pathsep)

    assert entries[0] == str(_REPO_SRC), "the package root must come first"
    assert entries[1:] == [str(inherited)], "the inherited PYTHONPATH was not preserved"
    assert len(entries) == len(set(entries)), "PYTHONPATH entries were duplicated"
    assert env["CONSOLIDATION_MEMORY_PROJECT"] == "probe-sentinel"
    assert set(env) == set(os.environ), "unrelated parent state was dropped"


def test_run_detect_drift_subprocess_completes_with_child_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """End-to-end: the subprocess returns a payload from a real repository."""
    if shutil.which("git") is None:  # pragma: no cover - git is required for drift
        pytest.skip("git is unavailable; drift detection cannot run")

    repo = tmp_path / "repo"
    repo.mkdir()
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "probe",
        "GIT_AUTHOR_EMAIL": "probe@example.com",
        "GIT_COMMITTER_NAME": "probe",
        "GIT_COMMITTER_EMAIL": "probe@example.com",
    }
    subprocess.run(["git", "init", "-q", str(repo)], check=True, env=env, timeout=120)
    (repo / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True, env=env, timeout=120)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-qm", "init"],
        check=True,
        env=env,
        timeout=120,
    )

    monkeypatch.setenv("CONSOLIDATION_MEMORY_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("CONSOLIDATION_MEMORY_PROJECT", "default")
    monkeypatch.delenv("PYTHONPATH", raising=False)

    result = asyncio.run(
        drift_subprocess.run_detect_drift_subprocess(
            base_ref=None,
            repo_path=str(repo),
            timeout_seconds=180.0,
        )
    )

    assert isinstance(result, dict)
    assert set(result) >= {
        "checked_anchors",
        "impacted_claim_ids",
        "challenged_claim_ids",
        "impacts",
    }
    assert json.dumps(result)


def test_analysed_repository_cannot_shadow_the_worker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A module in the analysed repo must not be imported by the worker.

    ``python -m`` puts the working directory at ``sys.path[0]``, ahead of
    ``PYTHONPATH`` and the stdlib. With the worker started inside the analysed
    repository, a ``consolidation_memory/`` package there took precedence and its
    code ran in place of the real one. The marker file records whether any
    shadowing module was imported.
    """
    repo = tmp_path / "analysed"
    shadow = repo / "consolidation_memory"
    shadow.mkdir(parents=True)
    marker = tmp_path / "shadow-marker.txt"
    (shadow / "__init__.py").write_text(
        "from pathlib import Path\n"
        "Path(__import__('os').environ['CM_SHADOW_MARKER']).write_text('executed')\n",
        encoding="utf-8",
    )
    (repo / "json.py").write_text("raise RuntimeError('shadowed stdlib json')\n", encoding="utf-8")

    monkeypatch.setenv("CM_SHADOW_MARKER", str(marker))
    monkeypatch.setenv("CONSOLIDATION_MEMORY_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("CONSOLIDATION_MEMORY_PROJECT", "default")
    monkeypatch.delenv("PYTHONPATH", raising=False)

    # The real end-to-end path: whatever cwd it picks, the worker must import the
    # real package rather than the one sitting in the analysed repository.
    # A repository without commits makes the worker fail; the import chain is what
    # this test is about, and it has already run by then.
    with contextlib.suppress(RuntimeError):
        asyncio.run(
            drift_subprocess.run_detect_drift_subprocess(
                base_ref=None,
                repo_path=str(repo),
                timeout_seconds=180.0,
            )
        )

    assert not marker.exists(), "the analysed repository shadowed the real package"


def _load_init_module(name: str) -> ModuleType:
    """Execute ``__init__.py`` standalone under a throwaway module name."""
    spec = importlib.util.spec_from_file_location(name, _INIT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_version_is_available_from_distribution_metadata() -> None:
    module = _load_init_module("_cm_version_metadata_probe")

    assert module.__version__ == consolidation_memory.__version__


def test_version_survives_missing_distribution_metadata() -> None:
    with patch(
        "importlib.metadata.version",
        side_effect=PackageNotFoundError("consolidation-memory"),
    ):
        module = _load_init_module("_cm_version_missing_probe")

    assert isinstance(module.__version__, str)
    assert module.__version__ == "0.0.0"
    assert module.__version__ < "0.1"


def test_version_lookup_does_not_swallow_other_errors() -> None:
    with (
        patch("importlib.metadata.version", side_effect=OSError("broken metadata")),
        pytest.raises(OSError, match="broken metadata"),
    ):
        _load_init_module("_cm_version_broken_probe")
