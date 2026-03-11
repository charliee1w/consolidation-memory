# Examples

These examples are the fastest path to a working `consolidation-memory`
integration from a clean checkout.

Prerequisites for most examples:

```bash
pip install -e ".[all,dev]"
```

If you only need the local-first default stack, this is enough:

```bash
pip install consolidation-memory[fastembed]
```

Examples in this directory:

- `python-quickstart/quickstart.py`
  - Smallest end-to-end Python API demo.
- `rest-api/`
  - Start the REST server, then store and recall with an HTTP client.
- `cursor-integration/`
  - Drop-in MCP config for Cursor.
- `continue-dev/`
  - Continue/Continue.dev MCP config.
- `langgraph-memory-node/`
  - LangGraph node example that reads from `MemoryClient`.
- `plugins/`
  - Minimal plugin that logs recall activity.

Notes:

- MCP configs use an exact Python interpreter path on purpose. That is more
  reliable than relying on a shell-installed console script.
- If you want examples to write into a dedicated project, set
  `CONSOLIDATION_MEMORY_PROJECT` before running them.
- The plugin example is easiest to use from the repository root so Python can
  import `examples.plugins.*` directly.
