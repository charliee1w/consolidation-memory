# REST API Example

This example assumes the REST server is running locally.

Start the server:

```bash
pip install -e ".[fastembed,rest]"
consolidation-memory serve --rest --host 127.0.0.1 --port 8080
```

Run the example. `client.py` imports `httpx`, which is a core dependency of the
project, so the `.[rest]` install above already provides it:

```bash
python examples/rest-api/client.py
```

(Install `httpx` by hand only if you run the script outside a project
environment.)

The script calls `GET /health`, `POST /memory/store`, `POST /memory/recall` and
`GET /memory/status`.

`127.0.0.1` needs no token. For a non-loopback bind the server refuses to start
without one, so export the same token on both sides:

```bash
export CONSOLIDATION_MEMORY_REST_AUTH_TOKEN="change-me"
```

```powershell
$env:CONSOLIDATION_MEMORY_REST_AUTH_TOKEN = "change-me"
```

Environment variables:

- `CONSOLIDATION_MEMORY_BASE_URL`
  - Defaults to `http://127.0.0.1:8080`
- `CONSOLIDATION_MEMORY_REST_AUTH_TOKEN`
  - Optional bearer token for authenticated servers
