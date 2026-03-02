"""Plugin system for consolidation-memory.

Provides a hook-based extensibility mechanism.  Plugins subclass
:class:`PluginBase` and override only the hooks they care about.
A :class:`PluginManager` singleton discovers, loads, and dispatches
lifecycle events to registered plugins.

Plugin discovery (in order):

1. **Entry points** — packages declare
   ``[project.entry-points."consolidation_memory.plugins"]`` in their
   ``pyproject.toml``.  Each entry point should reference a
   :class:`PluginBase` subclass.
2. **TOML config** — ``[plugins] enabled = ["dotted.path.ClassName"]``
   in the consolidation-memory config file.
3. **Programmatic** — call ``PluginManager.register(instance)`` at
   runtime (useful for testing).

Thread safety: hooks fire on the calling thread.  Plugins that
maintain mutable state must handle their own synchronization.
"""

from __future__ import annotations

import importlib
import logging
import sys
from typing import Any

logger = logging.getLogger(__name__)

# Entry-point group name
_EP_GROUP = "consolidation_memory.plugins"


# ── Plugin base class ────────────────────────────────────────────────────────


class PluginBase:
    """Base class for consolidation-memory plugins.

    Override any hook method you need.  All hooks are no-ops by default.
    Exceptions raised inside hooks are caught and logged — a failing
    plugin never crashes the host.

    Attributes:
        name: Human-readable plugin name (used in log messages).
    """

    name: str = "unnamed"

    # ── lifecycle ──

    def on_startup(self, client: Any) -> None:
        """Called once after MemoryClient finishes initialization.

        *client* is the :class:`~consolidation_memory.client.MemoryClient`
        instance.
        """

    def on_shutdown(self) -> None:
        """Called when MemoryClient.close() is invoked."""

    # ── store / recall / forget ──

    def on_store(
        self,
        episode_id: str,
        content: str,
        content_type: str,
        tags: list[str],
        surprise: float,
    ) -> None:
        """Called after an episode is successfully stored (not on duplicates)."""

    def on_recall(self, query: str, result: Any) -> None:
        """Called after recall completes. *result* is a RecallResult."""

    def on_forget(self, episode_id: str) -> None:
        """Called after an episode is successfully forgotten."""

    # ── consolidation ──

    def on_consolidation_start(self, run_id: str, episode_count: int) -> None:
        """Called at the start of a consolidation run."""

    def on_topic_created(
        self, filename: str, title: str, record_count: int
    ) -> None:
        """Called when a new knowledge topic is created."""

    def on_topic_updated(
        self, filename: str, title: str, record_count: int
    ) -> None:
        """Called when an existing topic is updated via merge."""

    def on_contradiction(
        self, topic_filename: str, old_content: str, new_content: str
    ) -> None:
        """Called for each detected contradiction during merge."""

    def on_consolidation_complete(self, report: dict[str, Any]) -> None:
        """Called after a consolidation run finishes (success or partial)."""

    def on_prune(self, episode_ids: list[str]) -> None:
        """Called after episodes are pruned."""


# ── Plugin manager ───────────────────────────────────────────────────────────


class PluginManager:
    """Discovers, loads, and dispatches events to plugins."""

    def __init__(self) -> None:
        self._plugins: list[PluginBase] = []

    @property
    def plugins(self) -> list[PluginBase]:
        """Return a copy of the registered plugin list."""
        return list(self._plugins)

    # ── registration ──

    def register(self, plugin: PluginBase) -> None:
        """Add a plugin instance. Silently ignores duplicates."""
        if plugin in self._plugins:
            return
        self._plugins.append(plugin)
        logger.info("Registered plugin: %s", plugin.name)

    def unregister(self, plugin: PluginBase) -> None:
        """Remove a plugin instance."""
        try:
            self._plugins.remove(plugin)
            logger.info("Unregistered plugin: %s", plugin.name)
        except ValueError:
            pass

    def clear(self) -> None:
        """Remove all registered plugins."""
        self._plugins.clear()

    # ── discovery & loading ──

    def load_plugins(self) -> None:
        """Discover and instantiate plugins from entry points and config."""
        self._load_from_entry_points()
        self._load_from_config()

    def _load_from_entry_points(self) -> None:
        from importlib.metadata import entry_points

        eps: Any
        if sys.version_info >= (3, 12):
            eps = entry_points(group=_EP_GROUP)
        else:
            all_eps = entry_points()
            eps = all_eps.get(_EP_GROUP, [])  # type: ignore[union-attr]

        for ep in eps:
            try:
                cls = ep.load()
                instance = cls()
                self.register(instance)
                logger.info("Loaded plugin from entry point: %s", ep.name)
            except Exception:
                logger.exception("Failed to load plugin entry point: %s", ep.name)

    def _load_from_config(self) -> None:
        from consolidation_memory.config import get_config

        cfg = get_config()
        paths = cfg.PLUGINS_ENABLED
        if not paths:
            return

        for dotted_path in paths:
            try:
                instance = _import_plugin(dotted_path)
                self.register(instance)
            except Exception:
                logger.exception("Failed to load plugin from config: %s", dotted_path)

    # ── dispatch ──

    def fire(self, hook_name: str, **kwargs: Any) -> None:
        """Dispatch *hook_name* to every registered plugin.

        Per-plugin exceptions are caught and logged — one failing plugin
        does not block the others.
        """
        for plugin in self._plugins:
            method = getattr(plugin, hook_name, None)
            if method is None:
                continue
            try:
                method(**kwargs)
            except Exception:
                logger.exception(
                    "Plugin %r raised in %s", plugin.name, hook_name,
                )


# ── helpers ──────────────────────────────────────────────────────────────────


def _import_plugin(dotted_path: str) -> PluginBase:
    """Import a PluginBase subclass from a dotted path like ``pkg.mod.Class``.

    Raises:
        ValueError: If the path doesn't contain a class name.
        ImportError: If the module can't be imported.
        TypeError: If the resolved object isn't a PluginBase subclass.
    """
    if "." not in dotted_path:
        raise ValueError(
            f"Plugin path must be a dotted path like 'pkg.mod.ClassName', "
            f"got: {dotted_path!r}"
        )

    module_path, class_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)

    if not (isinstance(cls, type) and issubclass(cls, PluginBase)):
        raise TypeError(
            f"{dotted_path!r} is not a PluginBase subclass "
            f"(got {type(cls).__name__})"
        )

    return cls()


# ── singleton ────────────────────────────────────────────────────────────────

_manager: PluginManager | None = None


def get_plugin_manager() -> PluginManager:
    """Return the PluginManager singleton, creating it on first access."""
    global _manager
    if _manager is None:
        _manager = PluginManager()
    return _manager


def reset_plugin_manager() -> PluginManager:
    """Reset the singleton (for tests). Returns the fresh instance."""
    global _manager
    _manager = PluginManager()
    return _manager
