# Plugin Example

This plugin logs a short summary every time recall finishes.

Use it from the repository root so Python can import
`examples.plugins.recall_audit_plugin`.

## Option 1: load from config

Add this to your consolidation-memory config TOML:

```toml
[plugins]
enabled = ["examples.plugins.recall_audit_plugin.RecallAuditPlugin"]
```

Then start the server or client normally.

## Option 2: register programmatically

```python
from consolidation_memory.plugins import get_plugin_manager
from examples.plugins.recall_audit_plugin import RecallAuditPlugin

manager = get_plugin_manager()
manager.register(RecallAuditPlugin())
```

This is a debugging example, not a production logging sink. Replace `print()`
with your own structured logger or telemetry destination if you extend it.
