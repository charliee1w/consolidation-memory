# Continue MCP Integration

This example matches Continue's `mcpServers` list shape.

Steps:

1. Install `consolidation-memory` into the environment Continue should use.
2. Replace `/absolute/path/to/python` in `continue-config.json`.
3. Merge the JSON into your Continue config.
4. Restart Continue and verify the MCP server starts cleanly.

The config keeps the server on a shared `universal` project so multiple clients
can see the same memory state when that is what you want.
