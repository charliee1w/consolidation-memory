# Examples

These examples are the fastest path to a working `consolidation-memory`
integration from a clean checkout.

Prerequisites for most examples:

```bash
pip install -e ".[all,dev]"
```

If you only need the local-first default stack, this is enough:

```bash
pip install "consolidation-memory[fastembed]"
```

Examples in this directory:

- [python-quickstart/quickstart.py](python-quickstart/quickstart.py)
  - Smallest end-to-end Python API demo.
- [rest-api/](rest-api/README.md)
  - Start the REST server, then store and recall with an HTTP client.
- [cursor-integration/](cursor-integration/README.md)
  - Drop-in MCP config for Cursor.
- [continue-dev/](continue-dev/README.md)
  - Drop-in MCP config for Continue.
- [langgraph-memory-node/](langgraph-memory-node/README.md)
  - LangGraph node example that reads from `MemoryClient`.
- [plugins/](plugins/README.md)
  - Minimal plugin that logs recall activity.
- [trust-vs-rag/](trust-vs-rag/README.md)
  - Same bug twice: store a path-anchored solution, consolidate to claims, drift-challenge on refactor (`demo_flow.py`).

Notes:

- MCP configs use an exact Python interpreter path on purpose. That is more
  reliable than relying on a shell-installed console script.
- Set `CONSOLIDATION_MEMORY_PROJECT` to point the examples at a dedicated
  project instead of your active one. The `trust-vs-rag` demo is the exception:
  it builds a throwaway data dir and git repo of its own.
- The plugin example is easiest to use from the repository root so Python can
  import `examples.plugins.*` directly.
- `rest-api/client.py` imports `httpx`, which is a core dependency, so any
  install of the project already provides it.
