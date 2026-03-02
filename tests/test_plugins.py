"""Tests for the plugin system.

Run with: python -m pytest tests/test_plugins.py -v
"""

from unittest.mock import patch

from helpers import make_normalized_vec

from consolidation_memory.plugins import (
    PluginBase,
    PluginManager,
    _import_plugin,
    get_plugin_manager,
    reset_plugin_manager,
)


# ── Test fixtures (inline plugin subclasses) ─────────────────────────────────


class TrackingPlugin(PluginBase):
    """Records every hook call for assertion."""

    name = "tracking"

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def _record(self, hook, **kw):
        self.calls.append((hook, kw))

    def on_startup(self, client):
        self._record("on_startup", client=client)

    def on_shutdown(self):
        self._record("on_shutdown")

    def on_store(self, episode_id, content, content_type, tags, surprise):
        self._record("on_store", episode_id=episode_id, content=content,
                     content_type=content_type, tags=tags, surprise=surprise)

    def on_recall(self, query, result):
        self._record("on_recall", query=query, result=result)

    def on_forget(self, episode_id):
        self._record("on_forget", episode_id=episode_id)

    def on_consolidation_start(self, run_id, episode_count):
        self._record("on_consolidation_start", run_id=run_id, episode_count=episode_count)

    def on_topic_created(self, filename, title, record_count):
        self._record("on_topic_created", filename=filename, title=title,
                     record_count=record_count)

    def on_topic_updated(self, filename, title, record_count):
        self._record("on_topic_updated", filename=filename, title=title,
                     record_count=record_count)

    def on_contradiction(self, topic_filename, old_content, new_content):
        self._record("on_contradiction", topic_filename=topic_filename,
                     old_content=old_content, new_content=new_content)

    def on_consolidation_complete(self, report):
        self._record("on_consolidation_complete", report=report)

    def on_prune(self, episode_ids):
        self._record("on_prune", episode_ids=episode_ids)


class ExplodingPlugin(PluginBase):
    """Raises on every hook — tests exception isolation."""

    name = "exploding"

    def on_startup(self, client):
        raise RuntimeError("boom in startup")

    def on_store(self, episode_id, content, content_type, tags, surprise):
        raise RuntimeError("boom in store")

    def on_recall(self, query, result):
        raise RuntimeError("boom in recall")


# ── PluginManager unit tests ─────────────────────────────────────────────────


class TestPluginManager:
    def test_register_and_plugins_list(self):
        mgr = PluginManager()
        p = TrackingPlugin()
        mgr.register(p)
        assert p in mgr.plugins
        assert len(mgr.plugins) == 1

    def test_register_deduplicates(self):
        mgr = PluginManager()
        p = TrackingPlugin()
        mgr.register(p)
        mgr.register(p)
        assert len(mgr.plugins) == 1

    def test_unregister(self):
        mgr = PluginManager()
        p = TrackingPlugin()
        mgr.register(p)
        mgr.unregister(p)
        assert len(mgr.plugins) == 0

    def test_unregister_nonexistent_is_noop(self):
        mgr = PluginManager()
        mgr.unregister(TrackingPlugin())  # should not raise

    def test_clear(self):
        mgr = PluginManager()
        mgr.register(TrackingPlugin())
        mgr.register(ExplodingPlugin())
        mgr.clear()
        assert len(mgr.plugins) == 0

    def test_fire_dispatches_to_all(self):
        mgr = PluginManager()
        p1 = TrackingPlugin()
        p2 = TrackingPlugin()
        mgr.register(p1)
        mgr.register(p2)
        mgr.fire("on_forget", episode_id="abc-123")
        assert len(p1.calls) == 1
        assert p1.calls[0] == ("on_forget", {"episode_id": "abc-123"})
        assert len(p2.calls) == 1

    def test_fire_unknown_hook_is_noop(self):
        mgr = PluginManager()
        mgr.register(TrackingPlugin())
        mgr.fire("on_nonexistent_hook", x=1)  # should not raise

    def test_fire_exception_isolation(self):
        """One plugin raising should not prevent other plugins from firing."""
        mgr = PluginManager()
        exploding = ExplodingPlugin()
        tracker = TrackingPlugin()
        mgr.register(exploding)
        mgr.register(tracker)

        mgr.fire("on_store", episode_id="x", content="c", content_type="fact",
                 tags=[], surprise=0.5)

        # tracker should still have been called despite exploding raising first
        assert len(tracker.calls) == 1
        assert tracker.calls[0][0] == "on_store"


# ── Singleton tests ──────────────────────────────────────────────────────────


class TestSingleton:
    def test_get_returns_same_instance(self):
        m1 = get_plugin_manager()
        m2 = get_plugin_manager()
        assert m1 is m2

    def test_reset_creates_new_instance(self):
        m1 = get_plugin_manager()
        m2 = reset_plugin_manager()
        assert m1 is not m2
        assert get_plugin_manager() is m2


# ── _import_plugin tests ─────────────────────────────────────────────────────


class TestImportPlugin:
    def test_import_valid_plugin(self):
        # Import our own TrackingPlugin
        instance = _import_plugin("test_plugins.TrackingPlugin")
        assert isinstance(instance, PluginBase)
        assert instance.name == "tracking"

    def test_import_no_dot_raises(self):
        import pytest
        with pytest.raises(ValueError, match="dotted path"):
            _import_plugin("NoDots")

    def test_import_nonexistent_module_raises(self):
        import pytest
        with pytest.raises(ImportError):
            _import_plugin("nonexistent.module.Class")

    def test_import_non_plugin_class_raises(self):
        import pytest
        with pytest.raises(TypeError, match="PluginBase subclass"):
            _import_plugin("test_plugins.PluginManager")  # not a PluginBase


# ── Config-based loading ─────────────────────────────────────────────────────


class TestConfigLoading:
    def test_load_from_config_enabled(self):
        from consolidation_memory.config import get_config
        cfg = get_config()
        cfg.PLUGINS_ENABLED = ["test_plugins.TrackingPlugin"]

        mgr = PluginManager()
        mgr._load_from_config()
        assert len(mgr.plugins) == 1
        assert mgr.plugins[0].name == "tracking"

    def test_load_from_config_empty(self):
        from consolidation_memory.config import get_config
        cfg = get_config()
        cfg.PLUGINS_ENABLED = []

        mgr = PluginManager()
        mgr._load_from_config()
        assert len(mgr.plugins) == 0

    def test_load_from_config_bad_path_logs_error(self):
        from consolidation_memory.config import get_config
        cfg = get_config()
        cfg.PLUGINS_ENABLED = ["nonexistent.module.BadPlugin"]

        mgr = PluginManager()
        mgr._load_from_config()  # should not raise
        assert len(mgr.plugins) == 0


# ── Integration with MemoryClient ────────────────────────────────────────────


class TestClientHooks:
    @patch("consolidation_memory.backends.encode_documents")
    def test_store_fires_on_store(self, mock_embed):
        from consolidation_memory.database import ensure_schema
        from consolidation_memory.client import MemoryClient

        ensure_schema()

        tracker = TrackingPlugin()
        mgr = reset_plugin_manager()
        mgr.register(tracker)

        vec = make_normalized_vec(seed=42)
        mock_embed.return_value = vec.reshape(1, -1)

        client = MemoryClient(auto_consolidate=False)

        # on_startup should have fired
        startup_calls = [c for c in tracker.calls if c[0] == "on_startup"]
        assert len(startup_calls) == 1

        result = client.store("test content", content_type="fact", tags=["py"], surprise=0.7)
        assert result.status == "stored"

        store_calls = [c for c in tracker.calls if c[0] == "on_store"]
        assert len(store_calls) == 1
        _, kwargs = store_calls[0]
        assert kwargs["episode_id"] == result.id
        assert kwargs["content"] == "test content"
        assert kwargs["content_type"] == "fact"
        assert kwargs["tags"] == ["py"]
        assert kwargs["surprise"] == 0.7

        client.close()

    @patch("consolidation_memory.backends.encode_documents")
    def test_store_duplicate_does_not_fire(self, mock_embed):
        from consolidation_memory.database import ensure_schema
        from consolidation_memory.client import MemoryClient

        ensure_schema()

        tracker = TrackingPlugin()
        mgr = reset_plugin_manager()
        mgr.register(tracker)

        vec = make_normalized_vec(seed=42)
        mock_embed.return_value = vec.reshape(1, -1)

        client = MemoryClient(auto_consolidate=False)

        client.store("same content", content_type="fact")
        # Store same again — should be duplicate
        result = client.store("same content", content_type="fact")
        assert result.status == "duplicate_detected"

        store_calls = [c for c in tracker.calls if c[0] == "on_store"]
        # Only one on_store (the first successful one)
        assert len(store_calls) == 1

        client.close()

    @patch("consolidation_memory.backends.encode_documents")
    @patch("consolidation_memory.backends.encode_query")
    def test_recall_fires_on_recall(self, mock_query, mock_embed):
        from consolidation_memory.database import ensure_schema
        from consolidation_memory.client import MemoryClient

        ensure_schema()

        tracker = TrackingPlugin()
        mgr = reset_plugin_manager()
        mgr.register(tracker)

        vec = make_normalized_vec(seed=42)
        mock_embed.return_value = vec.reshape(1, -1)
        mock_query.return_value = vec.reshape(1, -1)

        client = MemoryClient(auto_consolidate=False)
        client.store("recall test", content_type="fact")

        result = client.recall("recall test")
        recall_calls = [c for c in tracker.calls if c[0] == "on_recall"]
        assert len(recall_calls) == 1
        _, kwargs = recall_calls[0]
        assert kwargs["query"] == "recall test"
        assert kwargs["result"] is result

        client.close()

    @patch("consolidation_memory.backends.encode_documents")
    def test_forget_fires_on_forget(self, mock_embed):
        from consolidation_memory.database import ensure_schema
        from consolidation_memory.client import MemoryClient

        ensure_schema()

        tracker = TrackingPlugin()
        mgr = reset_plugin_manager()
        mgr.register(tracker)

        vec = make_normalized_vec(seed=42)
        mock_embed.return_value = vec.reshape(1, -1)

        client = MemoryClient(auto_consolidate=False)
        result = client.store("forget me", content_type="fact")

        forget_result = client.forget(result.id)
        assert forget_result.status == "forgotten"

        forget_calls = [c for c in tracker.calls if c[0] == "on_forget"]
        assert len(forget_calls) == 1
        assert forget_calls[0][1]["episode_id"] == result.id

        client.close()

    @patch("consolidation_memory.backends.encode_documents")
    def test_forget_not_found_does_not_fire(self, mock_embed):
        from consolidation_memory.database import ensure_schema
        from consolidation_memory.client import MemoryClient

        ensure_schema()

        tracker = TrackingPlugin()
        mgr = reset_plugin_manager()
        mgr.register(tracker)

        mock_embed.return_value = make_normalized_vec(seed=1).reshape(1, -1)

        client = MemoryClient(auto_consolidate=False)

        forget_result = client.forget("nonexistent-id")
        assert forget_result.status == "not_found"

        forget_calls = [c for c in tracker.calls if c[0] == "on_forget"]
        assert len(forget_calls) == 0

        client.close()

    def test_close_fires_on_shutdown(self):
        from consolidation_memory.database import ensure_schema
        from consolidation_memory.client import MemoryClient

        ensure_schema()

        tracker = TrackingPlugin()
        mgr = reset_plugin_manager()
        mgr.register(tracker)

        client = MemoryClient(auto_consolidate=False)
        client.close()

        shutdown_calls = [c for c in tracker.calls if c[0] == "on_shutdown"]
        assert len(shutdown_calls) == 1

    @patch("consolidation_memory.backends.encode_documents")
    def test_exploding_plugin_doesnt_crash_store(self, mock_embed):
        """An exception in a plugin hook must not break the host operation."""
        from consolidation_memory.database import ensure_schema
        from consolidation_memory.client import MemoryClient

        ensure_schema()

        mgr = reset_plugin_manager()
        mgr.register(ExplodingPlugin())

        vec = make_normalized_vec(seed=42)
        mock_embed.return_value = vec.reshape(1, -1)

        client = MemoryClient(auto_consolidate=False)
        result = client.store("should still work", content_type="fact")
        assert result.status == "stored"

        client.close()


# ── PluginBase default behavior ──────────────────────────────────────────────


class TestPluginBaseDefaults:
    def test_all_hooks_are_noop(self):
        """PluginBase hooks should not raise when called with valid args."""
        p = PluginBase()
        p.on_startup(client=None)
        p.on_shutdown()
        p.on_store(episode_id="x", content="c", content_type="fact",
                   tags=[], surprise=0.5)
        p.on_recall(query="q", result=None)
        p.on_forget(episode_id="x")
        p.on_consolidation_start(run_id="r", episode_count=10)
        p.on_topic_created(filename="f.md", title="T", record_count=3)
        p.on_topic_updated(filename="f.md", title="T", record_count=5)
        p.on_contradiction(topic_filename="f.md", old_content="a", new_content="b")
        p.on_consolidation_complete(report={})
        p.on_prune(episode_ids=["x"])

    def test_default_name(self):
        assert PluginBase.name == "unnamed"
