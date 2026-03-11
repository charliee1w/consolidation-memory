# Cursor MCP Integration

1. Install the package in the environment you want Cursor to use.
2. Replace `/absolute/path/to/python` in `cursor-mcp.json` with that exact
   interpreter path.
3. Paste the JSON into Cursor's MCP settings.
4. Restart Cursor and verify the server with `consolidation-memory test`.

Why this config uses the interpreter path:

- it avoids PATH drift on Windows
- it keeps Cursor pinned to the environment where the package is installed
- it makes restart behavior more predictable for long-lived MCP sessions
