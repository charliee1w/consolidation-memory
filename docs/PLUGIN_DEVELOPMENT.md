# Plugin Development

consolidation-memory exposes a small hook surface for observability and custom
behavior. Plugins are optional — the trust stack works without them.

## Hook surface

Registered hooks (see `consolidation_memory.plugins.PluginBase`):

| Hook | When it fires |
| --- | --- |
| `on_store` | After an episode is persisted |
| `on_recall` | After recall completes |
| `on_consolidation_start` | Before a consolidation run |
| `on_consolidation_complete` | After consolidation finishes (all exit paths) |
| `on_topic_created` | New knowledge topic written |
| `on_topic_updated` | Existing topic merged/updated |
| `on_contradiction` | Contradiction detected during consolidation |
| `on_prune` | Episode pruned after consolidation |

Plugins must subclass `PluginBase` and only implement hooks they need.

## Minimal plugin

Start from [examples/plugins/recall_audit_plugin.py](../examples/plugins/recall_audit_plugin.py)
or [docs/examples/minimal_plugin.py](examples/minimal_plugin.py).

```python
from consolidation_memory.plugins import PluginBase


class RecallAuditPlugin(PluginBase):
    def on_recall(self, query: str, result: dict) -> None:
        episodes = result.get("episodes") or []
        print(f"[recall] query={query!r} episodes={len(episodes)}")
```

## Enable a plugin

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

## Safety rules

- Plugins run in-process with full memory DB access — same trust boundary as MCP.
- Hook names are validated against a fixed allowlist before dispatch.
- Do not perform blocking network I/O inside hooks; defer to background tasks.
- Replace demo `print()` calls with structured logging in production.

## Testing

```bash
pytest tests/test_plugins.py -q
```

When adding a hook consumer, add a regression that fires the hook on the code
path you extend.

## Related docs

- [Examples: plugins](../examples/plugins/README.md)
- [Architecture](ARCHITECTURE.md)
- [Contributing: trust invariants](../CONTRIBUTING.md)