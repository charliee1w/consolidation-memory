"""Isolated subprocess execution for drift detection.

Running drift scans in a fresh interpreter avoids shared MCP worker starvation
or poisoned in-process state after prior timeouts.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
from pathlib import Path

from consolidation_memory.types import DriftOutput


def _resolve_python_executable() -> str:
    executable = (sys.executable or "").strip()
    if executable:
        resolved = Path(executable).expanduser().resolve()
        if resolved.exists():
            return str(resolved)

    discovered = shutil.which("python")
    if discovered:
        return str(Path(discovered).expanduser().resolve())

    raise RuntimeError(
        "Unable to locate a Python executable for isolated drift detection."
    )


def _build_drift_command(*, base_ref: str | None, repo_path: str | None) -> list[str]:
    cmd = [
        _resolve_python_executable(),
        "-m",
        "consolidation_memory.cli",
        "detect-drift",
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
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(),
            timeout=max(0.001, float(timeout_seconds)),
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise

    if proc.returncode != 0:
        details = _summarize_process_error(
            returncode=int(proc.returncode),
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
        raise RuntimeError("Isolated drift detection output must be a JSON object.")

    return payload  # type: ignore[return-value]


__all__ = ["run_detect_drift_subprocess"]
