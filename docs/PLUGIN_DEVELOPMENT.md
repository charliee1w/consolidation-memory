# Plugin Development

consolidation-memory exposes a small hook surface for observability and custom
behavior. Plugins are optional — the trust stack works without them.

## Hook surface

Registered hooks, one row per entry of the `plugins.HOOK_NAMES` allowlist that
`fire()` validates against — 11 hooks, matching the 11 `PluginBase` methods
exactly. A plugin subclasses `PluginBase` and overrides only the hooks it needs;
every base hook is a no-op. Exceptions raised inside a hook are caught and
logged, so one failing plugin never blocks the host and never crashes recall.

| Hook | Signature | When it fires |
| --- | --- | --- |
| `on_startup` | `client` | Once after `MemoryClient` finishes initialization |
| `on_shutdown` | — | `MemoryClient.close()` |
| `on_store` | `episode_id, content, content_type, tags, surprise` | After an episode is stored (not on duplicates) |
| `on_recall` | `query, result` | After recall completes; `result` is a `types.RecallResult` |
| `on_forget` | `episode_id` | After an episode is forgotten |
| `on_consolidation_start` | `run_id, episode_count` | Before a consolidation run |
| `on_consolidation_complete` | `report` | Once per run, on every terminal outcome: the full report, the early exits (`nothing_to_consolidate`, `too_few_episodes`, `error` when there are no vectors or no valid episodes), and an `error` report from the run's exception handler |
| `on_topic_created` | `filename, title, record_count` | New knowledge topic written |
| `on_topic_updated` | `filename, title, record_count` | Existing topic merged/updated |
| `on_contradiction` | `topic_filename, old_content, new_content` | Contradiction detected during merge |
| `on_prune` | `episode_ids` | Episodes pruned after consolidation |

## Minimal plugin

Start from [examples/plugins/recall_audit_plugin.py](../examples/plugins/recall_audit_plugin.py)
or [docs/examples/minimal_plugin.py](examples/minimal_plugin.py).

```python
from consolidation_memory.plugins import PluginBase


class RecallAuditPlugin(PluginBase):
    name = "recall-audit"

    def on_recall(self, query: str, result: object) -> None:
        print(f"[recall] query={query!r} episodes={len(result.episodes)}")
```

## Enable a plugin

Discovery runs in this order, once per process, and `on_startup` fires only when
the first `MemoryClient` becomes active:

1. **Entry points** — a package declares
   `[project.entry-points."consolidation_memory.plugins"]` in its
   `pyproject.toml`. The group name is the loader's `plugins._EP_GROUP`; each
   entry point resolves to a `PluginBase` subclass that is instantiated with no
   arguments.
2. **Config file** — see below.
3. **Programmatic** — `get_plugin_manager().register(instance)`.

### Config file

```toml
[plugins]
enabled = ["examples.plugins.recall_audit_plugin.RecallAuditPlugin"]
```

Paths are import paths resolvable from your working directory (run from the
repository root when using `examples.plugins.*`).

### Programmatic registration

```python
from consolidation_memory.plugins import get_plugin_manager
from examples.plugins.recall_audit_plugin import RecallAuditPlugin

get_plugin_manager().register(RecallAuditPlugin())
```

Hooks fire on the calling thread, so a plugin with mutable state must
synchronize itself.

## Safety rules

- Plugins run in-process with full memory DB access — same trust boundary as MCP
  and the Python SDK (see [SECURITY.md](../SECURITY.md#trust-boundaries)).
- An unknown hook name raises `ValueError` listing the valid hooks; a hook that
  raises is logged and skipped.
- Do not perform blocking network I/O inside hooks; defer to background tasks.
- Replace demo `print()` calls with structured logging in production.

## Testing

```bash
pytest tests/test_plugins.py -q
```

When adding a hook consumer, add a regression that fires the hook on the code
path you extend. A hook that never fires logs nothing, so an unasserted hook is
an invisible one.

## Related docs

- [Examples: plugins](../examples/plugins/README.md)
- [Architecture](ARCHITECTURE.md)
- [Contributing: trust invariants](../CONTRIBUTING.md#trust-invariants)