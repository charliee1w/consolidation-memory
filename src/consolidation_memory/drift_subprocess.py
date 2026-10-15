"""Isolated subprocess execution for drift detection.

Running drift scans in a fresh interpreter avoids shared MCP worker starvation
or poisoned in-process state after prior timeouts.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import sys
from pathlib import Path

from consolidation_memory.types import DriftOutput


def _absolute_without_resolving(path: str) -> Path:
    """Make ``path`` absolute while keeping symlinks intact.

    ``Path.resolve()`` must not be used here: a virtualenv layout
    (``.venv/bin/python -> python3.14 -> /usr/bin/python3.14``) resolves straight
    through the environment to the base interpreter, which has a different
    ``site-packages`` and usually no distribution metadata.
    """
    return Path(path).expanduser().absolute()


def _package_root() -> str:
    """Directory that must be importable for the child to find the package."""
    return str(_absolute_without_resolving(__file__).parent.parent)


def _resolve_python_executable() -> str:
    """Return the interpreter the server itself runs on.

    The drift worker must use the same environment as the server, otherwise it
    can import a different dependency set. Resolving symlinks would collapse a
    virtualenv down to its base interpreter, so the path is only made absolute.
    """
    executable = (sys.executable or "").strip()
    if executable:
        candidate = _absolute_without_resolving(executable)
        if candidate.exists():
            return str(candidate)

    discovered = shutil.which("python")
    if discovered:
        return str(_absolute_without_resolving(discovered))

    raise RuntimeError(
        "Unable to locate a Python executable for isolated drift detection."
    )


def _build_child_env() -> dict[str, str]:
    """Environment for the worker: parent environment plus a usable import path.

    The parent environment is inherited unchanged, as before; only ``PYTHONPATH``
    gains the directory that makes ``consolidation_memory`` importable, so the
    child can import the same sources the server is running.
    """
    env = dict(os.environ)
    inherited = env.get("PYTHONPATH", "").strip()
    candidates = [_package_root(), *inherited.split(os.pathsep)]

    deduped: list[str] = []
    for entry in candidates:
        entry = entry.strip()
        if entry and entry not in deduped:
            deduped.append(entry)

    env["PYTHONPATH"] = os.pathsep.join(deduped)
    return env


def _build_drift_command(*, base_ref: str | None, repo_path: str | None) -> list[str]:
    from consolidation_memory.config import get_active_project

    cmd = [
        _resolve_python_executable(),
        "-m",
        "consolidation_memory.drift_worker",
        "--project",
        get_active_project(),
    ]
    if base_ref:
        cmd.extend(["--base-ref", base_ref])
    if repo_path:
        cmd.extend(["--repo-path", repo_path])
    return cmd


def _decode_output(raw: bytes) -> str:
    text = raw.decode("utf-8", errors="replace").strip()
    return text


def _summarize_process_error(
    *,
    returncode: int,
    stdout: bytes,
    stderr: bytes,
) -> str:
    stderr_text = _decode_output(stderr)
    stdout_text = _decode_output(stdout)
    details = stderr_text or stdout_text or f"exit code {returncode}"
    if len(details) > 400:
        details = f"{details[:397]}..."
    return details


async def run_detect_drift_subprocess(
    *,
    base_ref: str | None = None,
    repo_path: str | None = None,
    timeout_seconds: float,
) -> DriftOutput:
    cmd = _build_drift_command(base_ref=base_ref, repo_path=repo_path)
    # No cwd: `python -m` puts the current directory at sys.path[0], ahead of
    # PYTHONPATH and the stdlib, so running inside the analysed repository let a
    # `consolidation_memory/` or `json.py` sitting in that repo shadow the real
    # module - the worker would then execute the analysed repository's code. The
    # repository is passed as `--repo-path` and resolved by the worker itself, so
    # the child needs nothing from the working directory.
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=_build_child_env(),
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(),
            timeout=max(0.001, float(timeout_seconds)),
        )
    except asyncio.TimeoutError:
        # The process already exited; killing it is a no-op.
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        await proc.wait()
        raise

    return_code = proc.returncode
    if return_code is None:  # pragma: no cover - process exited before communicate returned
        raise RuntimeError("Isolated drift detection process did not report an exit code.")

    if return_code != 0:
        details = _summarize_process_error(
            returncode=return_code,
            stdout=stdout,
            stderr=stderr,
        )
        raise RuntimeError(f"Isolated drift detection failed: {details}")

    raw_payload = _decode_output(stdout)
    if not raw_payload:
        raise RuntimeError("Isolated drift detection produced empty output.")

    try:
        payload = json.loads(raw_payload)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Isolated drift detection returned invalid JSON output."
        ) from exc

    if not isinstance(payload, dict):
        raise RuntimeError(  # noqa: TRY004 - payload shape assertion, not an argument validator
            "Isolated drift detection output must be a JSON object."
        )

    return payload  # type: ignore[return-value]


__all__ = ["run_detect_drift_subprocess"]
