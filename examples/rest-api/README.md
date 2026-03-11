# REST API Example

This example assumes the REST server is running locally.

Start the server:

```bash
pip install -e ".[fastembed,rest]"
consolidation-memory serve --rest --host 127.0.0.1 --port 8080
```

If you enable auth for non-loopback binds, export the same token before running
the client:

```bash
export CONSOLIDATION_MEMORY_REST_AUTH_TOKEN="change-me"
```

Then run:

```bash
python examples/rest-api/client.py
```

Environment variables:

- `CONSOLIDATION_MEMORY_BASE_URL`
  - Defaults to `http://127.0.0.1:8080`
- `CONSOLIDATION_MEMORY_REST_AUTH_TOKEN`
  - Optional bearer token for authenticated servers
