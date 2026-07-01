"""First-run setup helpers shared by CLI and browser UI."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, cast

from consolidation_memory import __version__


def _toml_basic_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def recommended_mcp_fast_env() -> dict[str, str]:
    """Environment overrides for low-latency MCP tool calls."""
    return {
        "PYTHONUNBUFFERED": "1",
        "CONSOLIDATION_MEMORY_IDLE_TIMEOUT_SECONDS": "900",
        "CONSOLIDATION_MEMORY_EMBEDDING_BACKEND": "fastembed",
        "CONSOLIDATION_MEMORY_LLM_BACKEND": "disabled",
        "CONSOLIDATION_MEMORY_WARMUP_ON_START": "1",
        "CONSOLIDATION_MEMORY_PRELOAD_NUMERIC_BACKENDS_ON_START": "1",
        "CONSOLIDATION_MEMORY_STATUS_LIGHTWEIGHT": "1",
        "CONSOLIDATION_MEMORY_MCP_AUTO_CONSOLIDATE": "0",
        "CONSOLIDATION_MEMORY_CONSOLIDATION_AUTO_RUN": "0",
        "CONSOLIDATION_MEMORY_WARMUP_PRIME_TOPIC_CACHE": "1",
        "CONSOLIDATION_MEMORY_WARMUP_PRIME_RECORD_CACHE": "1",
        "CONSOLIDATION_MEMORY_WARMUP_PRIME_CLAIM_CACHE": "0",
        "CONSOLIDATION_MEMORY_WARMUP_AWAIT_SECONDS": "15",
        "CONSOLIDATION_MEMORY_CLIENT_INIT_TIMEOUT_SECONDS": "20",
        "CONSOLIDATION_MEMORY_RECALL_TIMEOUT_SECONDS": "25",
        "CONSOLIDATION_MEMORY_RECALL_FALLBACK_TIMEOUT_SECONDS": "10",
        "CONSOLIDATION_MEMORY_DEFERRED_KNOWLEDGE_RETRY_SECONDS": "0",
    }


def recommended_mcp_server_config(project: str) -> dict[str, object]:
    """Return the most stable MCP launch configuration for the current interpreter."""
    command = sys.executable or "python"
    return {
        "command": command,
        "args": ["-m", "consolidation_memory", "--project", project, "serve"],
        "env": recommended_mcp_fast_env(),
    }


def recommended_mcp_simple_server_config(project: str) -> dict[str, object]:
    """MCP config exposing only recall + remember + ask (newcomer-friendly)."""
    config = recommended_mcp_server_config(project)
    env = dict(cast(dict[str, str], config["env"]))
    env["CONSOLIDATION_MEMORY_MCP_TOOL_PROFILE"] = "simple"
    return {**config, "env": env}


def build_mcp_snippets(project: str) -> dict[str, object]:
    """JSON-ready MCP client snippets for UI and CLI."""
    return {
        "project": project,
        "full": {"mcpServers": {"consolidation_memory": recommended_mcp_server_config(project)}},
        "simple": {"mcpServers": {"consolidation_memory": recommended_mcp_simple_server_config(project)}},
    }


def build_daemon_snippet(project: str) -> dict[str, object]:
    """Foreground and install commands for the maintenance daemon."""
    from consolidation_memory.daemon_service import daemon_launch_command

    return {
        "foreground": daemon_launch_command(project=project),
        "install": ["consolidation-memory", "--project", project, "daemon", "install"],
        "status": ["consolidation-memory", "--project", project, "daemon", "status"],
    }


def print_adoption_hints(project: str) -> None:
    """Print MCP + maintenance daemon guidance after init."""
    import json

    from consolidation_memory.daemon_service import format_daemon_install_hint

    print("\n--- Add to your MCP client config (fast recall; full tools) ---")
    print(json.dumps(build_mcp_snippets(project)["full"], indent=2))
    print("\n--- Simple profile (recall + remember + ask only) ---")
    print(json.dumps(build_mcp_snippets(project)["simple"], indent=2))
    print("\n--- Background maintenance (consolidation scheduler) ---")
    print(format_daemon_install_hint(project=project))
    print(f"\nMCP project namespace: {project}")


def fastembed_available() -> bool:
    try:
        import fastembed  # noqa: F401
    except ImportError:
        return False
    return True


def assess_setup_status() -> dict[str, object]:
    """Return whether the browser UI should show the setup wizard."""
    from consolidation_memory.config import get_active_project, get_config_path, get_default_config_dir

    config_path = get_config_path()
    project = get_active_project()
    default_config = get_default_config_dir() / "config.toml"
    return {
        "needs_setup": config_path is None,
        "config_path": str(config_path) if config_path else None,
        "default_config_path": str(default_config),
        "project": project,
        "version": __version__,
        "fastembed_installed": fastembed_available(),
        "python_executable": sys.executable or "python",
    }


def write_quick_config(*, embed_backend: str = "fastembed", llm_backend: str = "disabled") -> Path:
    """Write starter config.toml and initialize data directories."""
    from consolidation_memory.config import get_config, get_default_config_dir
    from consolidation_memory.database import ensure_schema

    embed_config = f"backend = {_toml_basic_string(embed_backend)}"
    llm_config = f"backend = {_toml_basic_string(llm_backend)}"

    config_dir = get_default_config_dir()
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.toml"

    config_content = f"""# Consolidation Memory configuration
# Generated by: consolidation-memory init --quick

[embedding]
{embed_config}

[llm]
{llm_config}

[consolidation]
auto_run = true
interval_hours = 6

[dedup]
enabled = true
similarity_threshold = 0.95
"""
    config_path.write_text(config_content, encoding="utf-8")

    cfg = get_config()
    for directory in [cfg.DATA_DIR, cfg.KNOWLEDGE_DIR, cfg.LOG_DIR, cfg.BACKUP_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    ensure_schema()
    return config_path


def run_quick_setup() -> dict[str, Any]:
    """Programmatic equivalent of ``consolidation-memory init --quick``."""
    from consolidation_memory.config import get_active_project, get_config_path

    existing = get_config_path()
    if existing is not None:
        project = get_active_project()
        return {
            "status": "already_configured",
            "config_path": str(existing),
            "project": project,
            "mcp": build_mcp_snippets(project),
            "daemon": build_daemon_snippet(project),
            "message": "Config already exists; kept existing settings.",
        }

    if not fastembed_available():
        return {
            "status": "missing_dependency",
            "message": "Install fastembed first: pip install consolidation-memory[fastembed]",
            "fastembed_installed": False,
        }

    config_path = write_quick_config(embed_backend="fastembed", llm_backend="disabled")
    from consolidation_memory.config import get_config

    project = get_active_project()
    return {
        "status": "configured",
        "config_path": str(config_path),
        "data_dir": os.fspath(get_config().DATA_DIR),
        "project": project,
        "mcp": build_mcp_snippets(project),
        "daemon": build_daemon_snippet(project),
        "message": "Quick setup complete: fastembed embeddings, LLM disabled.",
    }