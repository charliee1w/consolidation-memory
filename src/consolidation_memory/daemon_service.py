"""Background maintenance daemon: consolidation scheduler separate from fast MCP."""

from __future__ import annotations

import logging
import os
import platform
import signal
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger("consolidation_memory.daemon")

_TASK_NAME_PREFIX = "consolidation-memory-maintenance"
_LAUNCH_AGENT_LABEL_PREFIX = "com.consolidation-memory.maintenance"
_SYSTEMD_UNIT_PREFIX = "consolidation-memory-maintenance"


def daemon_task_name(project: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in project)
    return f"{_TASK_NAME_PREFIX}-{safe}"


def daemon_launch_agent_label(project: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in project)
    return f"{_LAUNCH_AGENT_LABEL_PREFIX}.{safe}"


def daemon_systemd_unit(project: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in project)
    return f"{_SYSTEMD_UNIT_PREFIX}-{safe}.service"


def daemon_lock_path() -> Path:
    from consolidation_memory.config import get_config

    return get_config().DATA_DIR / ".maintenance_daemon.lock"


def daemon_wrapper_dir() -> Path:
    from consolidation_memory.config import get_config

    return get_config()._base_data_dir / "daemon"


def daemon_log_path(project: str) -> Path:
    from consolidation_memory.config import get_config

    cfg = get_config()
    log_dir = cfg.LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in project)
    return log_dir / f"maintenance_daemon_{safe}.log"


def daemon_launch_command(*, project: str | None = None) -> list[str]:
    from consolidation_memory.config import get_active_project

    active = project or get_active_project()
    python = sys.executable or "python"
    return [python, "-m", "consolidation_memory", "--project", active, "daemon"]


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        process_query_limited = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(process_query_limited, False, pid)
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    else:
        return True


def _read_lock_pid(lock_path: Path) -> int | None:
    if not lock_path.is_file():
        return None
    try:
        payload = lock_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for token in payload.replace(";", " ").split():
        if token.startswith("pid="):
            try:
                return int(token.split("=", 1)[1])
            except ValueError:
                return None
    return None


def is_daemon_running() -> bool:
    lock_path = daemon_lock_path()
    pid = _read_lock_pid(lock_path)
    if pid is not None and _pid_alive(pid):
        return True
    return False


def _platform_key() -> str:
    system = platform.system().lower()
    if system == "windows":
        return "windows"
    if system == "darwin":
        return "macos"
    return "linux"


def _write_wrapper_script(*, project: str) -> Path:
    wrapper_dir = daemon_wrapper_dir()
    wrapper_dir.mkdir(parents=True, exist_ok=True)
    command = daemon_launch_command(project=project)
    log_path = daemon_log_path(project)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if os.name == "nt":
        wrapper_path = wrapper_dir / f"run_{project}.cmd"
        quoted_cmd = subprocess.list2cmdline(command)
        content = (
            "@echo off\r\n"
            f"{quoted_cmd} >> \"{log_path}\" 2>&1\r\n"
        )
        wrapper_path.write_text(content, encoding="utf-8")
        return wrapper_path

    wrapper_path = wrapper_dir / f"run_{project}.sh"
    shell_cmd = " ".join(_shell_quote(part) for part in command)
    content = (
        "#!/bin/sh\n"
        "set -e\n"
        f'exec {shell_cmd} >> "{log_path}" 2>&1\n'
    )
    wrapper_path.write_text(content, encoding="utf-8", newline="\n")
    wrapper_path.chmod(wrapper_path.stat().st_mode | 0o111)
    return wrapper_path


def _shell_quote(value: str) -> str:
    if not value:
        return "''"
    if all(ch.isalnum() or ch in "/._-:" for ch in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


def install_daemon(*, project: str | None = None) -> dict[str, Any]:
    """Register a login-time maintenance daemon for the active project."""
    from consolidation_memory.config import get_active_project

    active = project or get_active_project()
    wrapper_path = _write_wrapper_script(project=active)
    platform_key = _platform_key()

    if platform_key == "windows":
        task_name = daemon_task_name(active)
        tr_value = f'"{wrapper_path}"'
        result = subprocess.run(
            [
                "schtasks",
                "/Create",
                "/TN",
                task_name,
                "/SC",
                "ONLOGON",
                "/TR",
                tr_value,
                "/F",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            message = (result.stderr or result.stdout or "schtasks failed").strip()
            return {
                "status": "error",
                "platform": platform_key,
                "project": active,
                "message": message,
            }
        started_now = False
        run_result = subprocess.run(
            ["schtasks", "/Run", "/TN", task_name],
            capture_output=True,
            text=True,
            check=False,
        )
        started_now = run_result.returncode == 0
        message = f"Scheduled task '{task_name}' registered."
        if started_now:
            message += " Started maintenance daemon now."
        else:
            message += " It will also start at next logon."
        return {
            "status": "installed",
            "platform": platform_key,
            "project": active,
            "task_name": task_name,
            "wrapper_path": str(wrapper_path),
            "log_path": str(daemon_log_path(active)),
            "started_now": started_now,
            "message": message,
        }

    if platform_key == "macos":
        label = daemon_launch_agent_label(active)
        plist_path = Path.home() / "Library" / "LaunchAgents" / f"{label}.plist"
        plist_path.parent.mkdir(parents=True, exist_ok=True)
        plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>{label}</string>
  <key>ProgramArguments</key>
  <array>
    <string>{wrapper_path}</string>
  </array>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>StandardOutPath</key>
  <string>{daemon_log_path(active)}</string>
  <key>StandardErrorPath</key>
  <string>{daemon_log_path(active)}</string>
</dict>
</plist>
"""
        plist_path.write_text(plist_content, encoding="utf-8")
        load = subprocess.run(
            ["launchctl", "load", "-w", str(plist_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if load.returncode != 0:
            message = (load.stderr or load.stdout or "launchctl load failed").strip()
            return {
                "status": "error",
                "platform": platform_key,
                "project": active,
                "plist_path": str(plist_path),
                "message": message,
            }
        return {
            "status": "installed",
            "platform": platform_key,
            "project": active,
            "plist_path": str(plist_path),
            "log_path": str(daemon_log_path(active)),
            "message": f"LaunchAgent '{label}' loaded.",
        }

    unit_name = daemon_systemd_unit(active)
    unit_path = Path.home() / ".config" / "systemd" / "user" / unit_name
    unit_path.parent.mkdir(parents=True, exist_ok=True)
    unit_content = f"""[Unit]
Description=Consolidation Memory maintenance daemon ({active})
After=network.target

[Service]
Type=simple
ExecStart={wrapper_path}
Restart=on-failure
RestartSec=30

[Install]
WantedBy=default.target
"""
    unit_path.write_text(unit_content, encoding="utf-8")
    enable = subprocess.run(
        ["systemctl", "--user", "daemon-reload"],
        capture_output=True,
        text=True,
        check=False,
    )
    if enable.returncode != 0:
        message = (enable.stderr or enable.stdout or "systemctl daemon-reload failed").strip()
        return {
            "status": "error",
            "platform": platform_key,
            "project": active,
            "unit_path": str(unit_path),
            "message": message,
        }
    enable = subprocess.run(
        ["systemctl", "--user", "enable", "--now", unit_name],
        capture_output=True,
        text=True,
        check=False,
    )
    if enable.returncode != 0:
        message = (enable.stderr or enable.stdout or "systemctl enable failed").strip()
        return {
            "status": "error",
            "platform": platform_key,
            "project": active,
            "unit_path": str(unit_path),
            "message": message,
        }
    return {
        "status": "installed",
        "platform": platform_key,
        "project": active,
        "unit_path": str(unit_path),
        "unit_name": unit_name,
        "log_path": str(daemon_log_path(active)),
        "message": f"systemd user unit '{unit_name}' enabled.",
    }


def uninstall_daemon(*, project: str | None = None) -> dict[str, Any]:
    from consolidation_memory.config import get_active_project

    active = project or get_active_project()
    platform_key = _platform_key()

    if platform_key == "windows":
        task_name = daemon_task_name(active)
        result = subprocess.run(
            ["schtasks", "/Delete", "/TN", task_name, "/F"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0 and "cannot find" not in (result.stderr or "").lower():
            message = (result.stderr or result.stdout or "schtasks delete failed").strip()
            return {"status": "error", "platform": platform_key, "project": active, "message": message}
        return {
            "status": "uninstalled",
            "platform": platform_key,
            "project": active,
            "task_name": task_name,
            "message": f"Removed scheduled task '{task_name}' (if present).",
        }

    if platform_key == "macos":
        label = daemon_launch_agent_label(active)
        plist_path = Path.home() / "Library" / "LaunchAgents" / f"{label}.plist"
        subprocess.run(
            ["launchctl", "unload", "-w", str(plist_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if plist_path.is_file():
            plist_path.unlink()
        return {
            "status": "uninstalled",
            "platform": platform_key,
            "project": active,
            "plist_path": str(plist_path),
            "message": f"Removed LaunchAgent '{label}' (if present).",
        }

    unit_name = daemon_systemd_unit(active)
    subprocess.run(
        ["systemctl", "--user", "disable", "--now", unit_name],
        capture_output=True,
        text=True,
        check=False,
    )
    unit_path = Path.home() / ".config" / "systemd" / "user" / unit_name
    if unit_path.is_file():
        unit_path.unlink()
    subprocess.run(
        ["systemctl", "--user", "daemon-reload"],
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "status": "uninstalled",
        "platform": platform_key,
        "project": active,
        "unit_name": unit_name,
        "message": f"Removed systemd user unit '{unit_name}' (if present).",
    }


def _is_daemon_installed(*, project: str) -> bool:
    platform_key = _platform_key()
    if platform_key == "windows":
        task_name = daemon_task_name(project)
        result = subprocess.run(
            ["schtasks", "/Query", "/TN", task_name],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.returncode == 0
    if platform_key == "macos":
        label = daemon_launch_agent_label(project)
        plist_path = Path.home() / "Library" / "LaunchAgents" / f"{label}.plist"
        return plist_path.is_file()
    unit_path = Path.home() / ".config" / "systemd" / "user" / daemon_systemd_unit(project)
    return unit_path.is_file()


def daemon_status(*, project: str | None = None) -> dict[str, Any]:
    from consolidation_memory.config import get_active_project, get_config

    active = project or get_active_project()
    cfg = get_config()
    running = is_daemon_running()
    installed = _is_daemon_installed(project=active)
    return {
        "project": active,
        "platform": _platform_key(),
        "installed": installed,
        "running": running,
        "lock_path": str(daemon_lock_path()),
        "log_path": str(daemon_log_path(active)),
        "launch_command": daemon_launch_command(project=active),
        "scheduler_enabled": bool(cfg.CONSOLIDATION_AUTO_RUN),
        "message": (
            "Maintenance daemon is running."
            if running
            else (
                "Maintenance daemon is installed but not running."
                if installed
                else "Maintenance daemon is not installed."
            )
        ),
    }


def run_daemon(*, stop_event: threading.Event | None = None) -> dict[str, Any]:
    """Foreground maintenance daemon with utility scheduler enabled."""
    from consolidation_memory.client import MemoryClient
    from consolidation_memory.config import get_active_project, get_config, override_config
    from consolidation_memory.process_write_lock import ProcessWriteLease

    project = get_active_project()
    cfg = get_config()
    lock_path = daemon_lock_path()
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    stop = stop_event or threading.Event()

    def _handle_signal(signum: int, _frame: object) -> None:
        logger.info("Received signal %s; shutting down maintenance daemon", signum)
        stop.set()

    previous_handlers: dict[int, Any] = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            previous_handlers[sig] = signal.signal(sig, _handle_signal)
        except (ValueError, OSError):
            pass

    lease_timeout = 2.0
    lease = ProcessWriteLease(lock_path, timeout_seconds=lease_timeout)
    try:
        with lease.acquire():
            logger.info(
                "Maintenance daemon started (project=%s, data_dir=%s, auto_run=%s)",
                project,
                cfg.DATA_DIR,
                cfg.CONSOLIDATION_AUTO_RUN,
            )
            with override_config(CONSOLIDATION_AUTO_RUN=True):
                client = MemoryClient(auto_consolidate=True)
                try:
                    from consolidation_memory.maintenance import warmup_recall_caches

                    try:
                        warmup_recall_caches()
                    except Exception:
                        logger.exception("Daemon warmup failed; continuing")

                    while not stop.wait(timeout=5.0):
                        if not client._consolidation_thread or not client._consolidation_thread.is_alive():
                            logger.error("Consolidation thread stopped unexpectedly")
                            break
                finally:
                    client.close()
    except TimeoutError:
        return {
            "status": "already_running",
            "project": project,
            "lock_path": str(lock_path),
            "message": "Another maintenance daemon already holds the lock.",
        }
    finally:
        for signum, handler in previous_handlers.items():
            try:
                signal.signal(signum, handler)
            except (ValueError, OSError):
                pass

    return {
        "status": "stopped",
        "project": project,
        "message": "Maintenance daemon exited.",
    }


def format_daemon_install_hint(*, project: str) -> str:
    command = subprocess.list2cmdline(daemon_launch_command(project=project))
    return (
        "Background maintenance (consolidation scheduler) runs separately from fast MCP:\n"
        f"  Foreground: {command}\n"
        f"  Install at logon: consolidation-memory --project {project} daemon install"
    )