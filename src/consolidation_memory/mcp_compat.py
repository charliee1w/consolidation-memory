"""Compatibility seam for the private ``mcp`` SDK internals this project depends on.

Boundary contract
-----------------
The published ``outputSchema`` of every MCP tool and the strictness of tool
argument validation are load-bearing for the consolidation-memory MCP contract,
and the SDK only exposes them through private, unversioned surfaces. Every one
of those touches lives in this module; nothing else in the codebase may reach
into ``mcp`` internals. The private surfaces used are exactly:

1. ``mcp.server.mcpserver.utilities.func_metadata.ArgModelBase`` — the private
   ``utilities`` subpackage is not re-exported from ``mcp.server``. The class is
   patched with ``extra="forbid"`` by :mod:`consolidation_memory.server` so
   unknown tool arguments fail validation instead of being silently dropped
   (pydantic's default is ``extra="ignore"``).
2. ``MCPServer._tool_manager`` — the per-server tool registry.
3. ``ToolManager._tools`` — the registered-tool mapping (``name -> Tool``). The
   public :meth:`ToolManager.list_tools` returns a list, so the mapping is only
   reachable privately.
4. ``Tool.fn_metadata.output_schema`` — where a tool's derived JSON schema
   lives, and where the ``anyOf[success, error]`` contract is written back.
5. ``Tool.__dict__["output_schema"]`` — ``Tool.output_schema`` is a
   ``functools.cached_property``. Assigning ``fn_metadata.output_schema`` does
   **not** refresh it, so the cached entry has to be evicted or the tool keeps
   publishing its pre-``anyOf`` schema forever.

Supported range
---------------
``mcp[cli]>=2.3.0,<3`` (``pyproject.toml``). The range is wide enough that a
minor bump can rename or relocate any of the five surfaces above. When that
happens the failure mode used to be silent: a renamed ``_tools`` dropped every
tool's ``outputSchema`` with no error, and a renamed module killed the process
with a bare ``ImportError`` traceback. Both are now explicit
:class:`MCPCompatError` messages that name the installed version, the
supported range, and the remediation, and :func:`registered_tools` lists every
attribute it tried before giving up.

A minor ``mcp`` bump may break any of the five surfaces. If the SDK moves one of
them, fix it here — nowhere else — and re-run
``python scripts/generate_tool_reference.py`` plus ``pytest tests/ -q``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from importlib import metadata as importlib_metadata
from typing import Any, TypeAlias, cast

from mcp.server import MCPServer

#: Supported ``mcp`` range, kept in sync with ``pyproject.toml``.
MCP_COMPAT_REQUIREMENT = ">=2.3.0,<3"

#: Attribute holding the tool registry on ``MCPServer`` (private surface 2).
_TOOL_MANAGER_ATTR = "_tool_manager"

#: Attribute names holding the registered-tool mapping on the manager. The
#: second entry is a defensive alias probed only to turn a rename into a clear
#: message instead of a silent contract loss.
_TOOL_MAPPING_ATTRS = ("_tools", "_tool_map")

#: Cached-property key on ``Tool`` (private surface 5).
_TOOL_SCHEMA_CACHE_KEY = "output_schema"

#: Wrapper name used as the install guard for :func:`install_list_tools_heal`.
_HEAL_WRAPPER_NAME = "_list_tools_with_output_schema_heal"

# Tool objects are deliberately typed as ``Any``: annotating them would mean
# importing a second private module (``mcp.server.mcpserver.tools.base``) just
# for a type, doubling the rename exposure this seam exists to contain. The
# attribute names actually touched are listed in the module docstring.
ToolLike: TypeAlias = Any


class MCPCompatError(RuntimeError):
    """A private ``mcp`` SDK surface this project depends on is missing or changed."""


def installed_mcp_version() -> str:
    """Return the installed ``mcp`` distribution version, or ``"unknown"``."""
    try:
        return importlib_metadata.version("mcp")
    except importlib_metadata.PackageNotFoundError:  # pragma: no cover - defensive
        return "unknown"


def _rename_advice(detail: str) -> str:
    """Build the shared, actionable tail of every compat failure message."""
    return (
        f"{detail} — consolidation_memory depends on private mcp internals for the published "
        f"outputSchema and strict tool-argument validation. Installed mcp: "
        f"{installed_mcp_version()}; supported range: mcp {MCP_COMPAT_REQUIREMENT} (pyproject.toml). "
        "Remediation: pin mcp into the supported range, or update "
        "consolidation_memory/mcp_compat.py (see its module docstring for the exact private "
        "surfaces used) and re-run scripts/generate_tool_reference.py."
    )


try:
    from mcp.server.mcpserver.utilities.func_metadata import ArgModelBase
except ImportError as exc:  # pragma: no cover - depends on the installed SDK
    raise MCPCompatError(
        _rename_advice(
            "mcp.server.mcpserver.utilities.func_metadata.ArgModelBase is not importable "
            f"({exc.__class__.__name__}: {exc})"
        )
    ) from exc

__all__ = [
    "MCP_COMPAT_REQUIREMENT",
    "ArgModelBase",
    "MCPCompatError",
    "arg_model_extra_mode",
    "install_list_tools_heal",
    "installed_mcp_version",
    "output_schema_dict",
    "registered_tools",
    "set_output_schema",
]


def arg_model_extra_mode() -> str | None:
    """Return the ``extra`` policy patched onto :data:`ArgModelBase`.

    ``"forbid"`` is the load-bearing value: it makes unknown tool arguments a
    validation error and publishes ``additionalProperties: false`` in every
    input schema. A silent loss of it would let misspelled arguments disappear
    instead of erroring.
    """
    return cast("str | None", ArgModelBase.model_config.get("extra"))


def registered_tools(server: MCPServer[Any]) -> Mapping[str, Any]:
    """Return the ``name -> tool`` mapping registered on ``server``.

    Probes the known private attributes in order and raises
    :class:`MCPCompatError` — naming every attribute tried — when none exists,
    instead of publishing success-only schemas for a silently missed registry.
    """
    view = cast(Any, server)
    manager = getattr(view, _TOOL_MANAGER_ATTR, None)
    if manager is None:
        raise MCPCompatError(
            _rename_advice(
                f"MCPServer has no {_TOOL_MANAGER_ATTR!r} attribute, so the registered tools are "
                f"unreachable (tried {_TOOL_MANAGER_ATTR!r} on {type(server).__name__})"
            )
        )
    for attr in _TOOL_MAPPING_ATTRS:
        mapping = getattr(manager, attr, None)
        if isinstance(mapping, Mapping):
            return cast("Mapping[str, Any]", mapping)
    tried = ", ".join(repr(f"{_TOOL_MANAGER_ATTR}.{attr}") for attr in _TOOL_MAPPING_ATTRS)
    raise MCPCompatError(
        _rename_advice(
            f"the mcp tool registry has no known tool mapping (tried {tried} on "
            f"{type(manager).__name__})"
        )
    )


def _fn_metadata(tool: Any, action: str) -> Any:
    """Return ``Tool.fn_metadata`` or raise naming the tool and the action."""
    metadata = getattr(tool, "fn_metadata", None)
    if metadata is None:
        raise MCPCompatError(
            _rename_advice(
                f"tool {getattr(tool, 'name', '<unnamed>')!r} has no 'fn_metadata' attribute, so "
                f"its output schema cannot be {action}"
            )
        )
    return metadata


def output_schema_dict(tool: Any) -> dict[str, Any] | None:
    """Return the tool's live output-schema dict, or ``None`` when it has none.

    The SDK caches the published schema on ``Tool.output_schema``; the cached
    entry is evicted here so callers always observe ``fn_metadata`` as it is now
    (private surface 5).
    """
    schema = getattr(_fn_metadata(tool, "read"), "output_schema", None)
    if schema is not None and not isinstance(schema, dict):
        raise MCPCompatError(
            _rename_advice(
                f"tool {getattr(tool, 'name', '<unnamed>')!r} has a non-dict output schema "
                f"({type(schema).__name__})"
            )
        )
    tool.__dict__.pop(_TOOL_SCHEMA_CACHE_KEY, None)
    return cast("dict[str, Any] | None", schema)


def set_output_schema(tool: Any, schema: dict[str, Any]) -> None:
    """Publish ``schema`` for ``tool`` and drop the SDK's cached copy.

    Both steps are required: the assignment is what ``call_tool`` validates
    against, the eviction is what ``list_tools`` reads. Assigning only the field
    leaves ``Tool.output_schema`` serving a stale cached dict (private surface 5).
    """
    _fn_metadata(tool, "published").output_schema = schema
    tool.__dict__.pop(_TOOL_SCHEMA_CACHE_KEY, None)


def install_list_tools_heal(
    server: MCPServer[Any],
    heal: Callable[[], object],
) -> None:
    """Run ``heal`` before every tool listing so late registration self-repairs.

    ``MCPServer.list_tools`` is public and is the single funnel both
    ``tools/list`` requests and :mod:`scripts.generate_tool_reference` go
    through, so wrapping it turns one-shot import-time publication into a
    self-healing contract. ``heal``'s return value is ignored; installation is
    idempotent.
    """
    existing = server.list_tools
    if getattr(existing, "__name__", None) == _HEAL_WRAPPER_NAME:
        return

    async def _list_tools_with_output_schema_heal() -> list[Any]:
        heal()
        return await existing()

    server.list_tools = _list_tools_with_output_schema_heal  # type: ignore[assignment]
