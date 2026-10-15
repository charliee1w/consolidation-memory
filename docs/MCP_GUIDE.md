# MCP guide

The consolidation-memory MCP server exposes the memory engine to any MCP host:
agents get typed tools, discoverable schemas, structured results and a scope
system for safe sharing. This guide covers the wire contract, scopes and
policies, the error model, every environment variable, and end-to-end recipes.

- **Full tool reference** (all 29 tools, generated from the published schemas):
  [TOOLS.md](TOOLS.md)
- **Quick start / host config**: [README — Connect your agent](../README.md#connect-your-agent-mcp)

## Capabilities at a glance

| Area | What the server gives you |
| --- | --- |
| Tools | 29 tools (full profile): store/recall/search, claims graph, outcomes, consolidation, drift detection, hygiene, policies, scope discovery |
| Profiles | `full` (everything) or `simple` (3 conversational tools) via `CONSOLIDATION_MEMORY_MCP_TOOL_PROFILE` |
| Protocol | Handshake revisions `2024-11-05` … `2025-11-25`; `2026-07-28` only for modern-envelope hosts — the host proposes, the SDK settles ([details](#protocol-compatibility)) |
| Results | Typed `outputSchema` per tool, `structuredContent` object + UTF-8 JSON text, unknown input arguments rejected on every surface |
| Scopes | Envelope-based isolation (namespace / app / agent / session / project) with discovery and access policies |
| Reliability | Per-tool timeouts, recall and drift fallback chains, `isError` failures with actionable text |
| Surfaces | Same dispatch and semantics behind MCP, REST, Python SDK and OpenAI-compatible schemas |

## Connect

Use an **absolute Python path** — more reliable than a console script,
especially on Windows:

```json
{
  "mcpServers": {
    "consolidation_memory": {
      "command": "/absolute/path/to/python",
      "args": ["-m", "consolidation_memory", "--project", "default", "serve"],
      "env": {
        "PYTHONUNBUFFERED": "1",
        "CONSOLIDATION_MEMORY_IDLE_TIMEOUT_SECONDS": "900",
        "CONSOLIDATION_MEMORY_STATUS_LIGHTWEIGHT": "1",
        "CONSOLIDATION_MEMORY_MCP_AUTO_CONSOLIDATE": "0",
        "CONSOLIDATION_MEMORY_PRELOAD_SCIPY_ON_START": "1",
        "CONSOLIDATION_MEMORY_DEFERRED_KNOWLEDGE_RETRY_SECONDS": "0"
      }
    }
  }
}
```

`consolidation-memory init` prints the full recommended environment.

Drop-in host configs: [Cursor](../examples/cursor-integration/README.md) ·
[Continue](../examples/continue-dev/README.md)

### Profiles

| Profile | Tools | When |
| --- | --- | --- |
| `full` (default) | all 29 | Local agent with the whole memory toolkit |
| `simple` | `memory_recall`, `memory_remember`, `memory_ask` | Small context budget, chat-style memory only |

Select with `"CONSOLIDATION_MEMORY_MCP_TOOL_PROFILE": "simple"`.

## Protocol compatibility

The repository pins **no** protocol version. The host proposes one, the SDK
settles: `mcp/server/runner.py` answers
`negotiated = requested if requested in HANDSHAKE_PROTOCOL_VERSIONS else
LATEST_HANDSHAKE_VERSION`, so the server never chooses a version the host did
not ask for. The revision ladder comes from the installed library
(`mcp_types/version.py`), and the range this project supports is
`mcp[cli]>=2.3.0,<3` ([pyproject.toml](../pyproject.toml)).

The era is fixed by the client's **first** request
(`runner.serve_dual_era_loop`): an `initialize` handshake opens the handshake
era, any other request carrying the per-request `_meta` envelope opens the
modern era. A session cannot mix the two — an `initialize` on a modern
connection is `UNSUPPORTED_PROTOCOL_VERSION`, an enveloped request on a
handshake connection is `INVALID_REQUEST`.

| Revision | How it is reached |
| --- | --- |
| `2026-07-28` | Modern era only (`MODERN_PROTOCOL_VERSIONS`): no `initialize`; every request carries the envelope pair in `params._meta` — `io.modelcontextprotocol/protocolVersion` plus `io.modelcontextprotocol/clientCapabilities`. A missing companion key is `INVALID_PARAMS` naming it; any other version value is `UNSUPPORTED_PROTOCOL_VERSION` naming the served list. Results in this era also carry `resultType: "complete"` and a `serverInfo` `_meta` stamp |
| `2025-11-25` | Handshake ceiling (`LATEST_HANDSHAKE_VERSION`) — the newest revision an `initialize` can settle on, and the counter-offer for a requested version outside the ladder |
| `2025-06-18`, `2025-03-26` | Handshake era, served as offered |
| `2024-11-05` | Handshake floor (`OLDEST_SUPPORTED_VERSION`) — what `initialize` settles on for older hosts |

An `initialize` that requests `2026-07-28` is answered with `2025-11-25`:
`2026-07-28` is not in `HANDSHAKE_PROTOCOL_VERSIONS`, so the downgrade is
silent and deliberate. Read the negotiated revision from the `initialize`
result rather than assuming the offer survived — the stdio smoke gate asserts
exactly that against the ladder read out of the installed library.

Nothing host-side is required to benefit from structured results: every
version receives `content` text, and hosts that understand `structuredContent`
and `outputSchema` get the typed channel as well.

## Result contract

A tool that **executes** returns three synchronized pieces:

1. **`content`** — one text block. On a successful call it is the payload
   serialized as JSON; on `isError: true` it is an actionable prose message,
   not the payload (`_tool_error_result` sets `text=message`).
2. **`structuredContent`** — the same payload as a JSON object, for
   applications that consume data rather than text. Present on both the
   success and the `isError` paths (where it is `{"error": "..."}`); an
   argument rejected before the tool body runs has no `structuredContent` at
   all, only text.
3. **`outputSchema`** (published in `tools/list`) — a JSON Schema shaped as
   `anyOf[success, error]`: the success arm describes every field of the
   payload, the error arm is `{"error": "..."}`. Clients **should** validate
   `structuredContent` against it; the server validates every successful
   result against the success arm before it leaves the process.

The text block is the SDK's own serialization of the payload
(`pydantic_core.to_json(..., indent=2)`), so it is a JSON *string containing*
JSON — the enclosing frame escapes the quotes and newlines around it.
Non-ASCII is written raw in both channels (`Γειά`, never
`\u0393\u03b5\u03b9\u03ac`), so a store→read round trip returns the stored bytes
through either one, literal `\uXXXX` text in the stored content included. The
gap between the channels is one layer of encoding, not fidelity: `json.loads`
twice for the text block, once for `structuredContent`.

Unknown input arguments are **rejected** on every surface rather than dropped.
See [Surfaces parity](#surfaces-parity).

### Example — success

`tools/call` → `memory_search` with `{"query": "deploy", "limit": 5}`:

```json
{
  "content": [
    {
      "type": "text",
      "text": "{\n  \"episodes\": [\n    {\n      \"id\": \"259b0c75-...\",\n      \"content\": \"The deploy pipeline fails...\",\n      \"content_type\": \"exchange\",\n      \"tags\": [],\n      \"created_at\": \"2026-10-03T17:29:26.481902+00:00\",\n      \"surprise_score\": 0.5,\n      \"access_count\": 0\n    }\n  ],\n  \"total_matches\": 1,\n  \"query\": \"deploy\",\n  \"message\": null\n}"
    }
  ],
  "structuredContent": {
    "episodes": [
      {
        "id": "259b0c75-...",
        "content": "The deploy pipeline fails...",
        "content_type": "exchange",
        "tags": [],
        "created_at": "2026-10-03T17:29:26.481902+00:00",
        "surprise_score": 0.5,
        "access_count": 0
      }
    ],
    "total_matches": 1,
    "query": "deploy",
    "message": null
  },
  "isError": false
}
```

(The text block is the same object at `indent=2`; the frame's escapes around it
are folded onto one line above for readability.)

### Example — tool execution failure (timeout)

```json
{
  "content": [
    {
      "type": "text",
      "text": "memory_status timed out after 30s. Raise CONSOLIDATION_MEMORY_TIMEOUT_MEMORY_STATUS or CONSOLIDATION_MEMORY_TOOL_TIMEOUT_SECONDS."
    }
  ],
  "structuredContent": {
    "error": "memory_status timed out after 30s. Raise CONSOLIDATION_MEMORY_TIMEOUT_MEMORY_STATUS or CONSOLIDATION_MEMORY_TOOL_TIMEOUT_SECONDS."
  },
  "isError": true
}
```

The text is actionable on purpose: hosts pass it to the model so it can retry
with different parameters or raise the budget, per the MCP error-handling
model (tool execution errors are feedback, not protocol failures).

### Example — rejected input

`memory_status` called with an unknown argument. The SDK's pydantic model
validates before the body runs, so there is no `structuredContent` — the text
is the whole result, and it ends with a pydantic help URL that carries the
library version:

```json
{
  "content": [
    {
      "type": "text",
      "text": "Error executing tool memory_status: 1 validation error for memory_statusArguments\njunk\n  Extra inputs are not permitted [type=extra_forbidden, input_value=1, input_type=int]\n    For further information visit https://errors.pydantic.dev/2.13/v/extra_forbidden"
    }
  ],
  "isError": true
}
```

### Guarantees

| Guarantee | Meaning |
| --- | --- |
| Nothing lost | Undeclared payload keys pass through (`additionalProperties: true` on outputs); text and structured channels carry the same data |
| Typed outputs | Every success payload validates against the published `outputSchema` success arm (strict types, required fields per the dispatcher's shapes) |
| Honest failures | Timeouts, validation errors and dispatch failures are `isError: true`, never a soft `{"error": ...}` success |
| Stable bytes | Both channels leave the process as raw UTF-8 with no `\uXXXX` escaping of their own, so stored text survives a write→read round trip byte-for-byte, literal escape sequences in the content included; the text block costs one extra `json.loads` |
| Two arms, always | Every tool in `tools/list` publishes an `anyOf[success, error]` `outputSchema`; a tool registered after startup is healed into the contract on its next `tools/list` rather than publishing success-only |
| Fail loudly at startup | If a tool would publish a success-only schema, the self-check refuses to serve instead of shipping a contract that rejects every `{"error": ...}` payload |

The startup self-check is pure verification: it repairs what it can, then
raises, naming every tool whose published `outputSchema` is not an
`anyOf[success, error]` pair together with the installed SDK version and the
supported range ([mcp_compat.py](../src/consolidation_memory/mcp_compat.py)).
The five private `mcp` surfaces the contract rests on are probed at import, so
a rename among them is an explicit boot failure that names the version and the
range, not a wire contract that has stopped describing the results.

## Scopes and sharing

A **scope** is the identity envelope every row is stamped with. Reads
(recall, search, browse, claims, outcomes) are filtered by the resolved
scope; writes stamp the scope onto new rows. Without a scope argument the
legacy defaults apply: namespace `default`, app client
`legacy_client/python_sdk`, project from `CONSOLIDATION_MEMORY_PROJECT`.

### Envelope anatomy

| Section | Keys | Meaning |
| --- | --- | --- |
| `namespace` | `slug`, `sharing_mode`, `display_name` | Top-level sharing boundary (`private` / `shared` / `team` / `managed`) |
| `app_client` | `name`, `app_type`, `provider`, `external_key` | Calling application (`mcp`, `python_sdk`, `rest`, `openai_agents`, `langgraph`, `adk`, `letta`, `cli`, `other`) |
| `agent` | `name`, `external_key` | Logical agent inside the app (`null` when unused) |
| `session` | `external_key`, `session_kind` | Short-lived interaction context (`conversation`, `thread`, `workflow`, `job`) |
| `project` | `slug`, `display_name`, `root_uri`, `repo_remote`, `default_branch` | Repository/project identity |
| `policy` | `write_mode`, `read_visibility` | Inline ACL for this call only — the **base** policy, overridden by a matching persisted binding ([ACL.md](ACL.md#how-the-effective-policy-is-resolved)) |

This is the **input** envelope: every section additionally accepts an `id`, the
stored row id (`project.id` doubles as a slug fallback). The
`memory_scope_list` output is narrower — it stamps only the keys that decide
read and write matching, so see [Discovering existing scopes](#discovering-existing-scopes).

`scope` accepts three shapes:

1. **Canonical object** — any subset of the sections above, e.g.
   `{"project": {"slug": "billing"}}`.
2. **String shorthand** — mapped to a project scope: values containing `/`,
   `\` or a URI prefix become `project.root_uri`, anything else becomes
   `project.slug` (so `"billing"` and `"/home/me/billing"` both work).
3. **`null`/omitted** — legacy defaults (see above).

### Discovering existing scopes

Scopes are not stored in a dedicated table — they live on the data rows, so
an external registry is not required. `memory_scope_list` aggregates what
exists:

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "memory_scope_list",
    "arguments": { "limit": 50, "offset": 0 }
  }
}
```

```json
{
  "structuredContent": {
    "scopes": [
      {
        "scope": {
          "namespace": { "slug": "default", "sharing_mode": "private" },
          "app_client": { "name": "legacy_client", "app_type": "python_sdk", "provider": null, "external_key": null },
          "agent": null,
          "session": null,
          "project": { "slug": "billing", "display_name": null, "root_uri": null, "repo_remote": null, "default_branch": null }
        },
        "counts": { "episodes": 12, "records": 3, "topics": 1 },
        "last_used_at": "2026-10-03T17:29:26.481902+00:00"
      }
    ],
    "total": 1,
    "offset": 0,
    "message": null
  },
  "isError": false
}
```

The returned `namespace` carries `slug` and `sharing_mode` only — there is no
`display_name` to read. The `scope` *argument* still accepts one
(`coerce_scope_envelope` keeps it as optional input), so an envelope is
reusable in both directions.

- Ordering is deterministic: most recently used first, stable tie-break.
- Page with `limit` (1–1000) + `offset`; `message` tells you where the window
  sits (`"Showing N of T scopes at offset ..."`) and `total` is the full count.
- Each returned `scope` can be passed **verbatim** back as the `scope`
  argument of any other tool. The published scope schema accepts the `null`s
  that discovery emits (`agent`, `session`, and unset identity keys inside
  `app_client` and `project`); `null` means "not set" and coercion treats it
  exactly like an absent key. A host that validates arguments against the
  published schema therefore accepts the envelope too.
- Grouping uses all 11 canonical scope keys — the same set the **write** filter
  matches on — so two scopes that differ only in `agent_name`,
  `session_kind`, `app_client_external_key`, `app_client_provider` or
  `namespace_sharing_mode` stay separate entries with their own counts instead
  of being merged and summed.
- The **read** filter is deliberately narrower than the grouping, so a count is a
  per-group figure and a recall through that envelope may return more rows than
  the count. `_resolved_scope_to_query_filter` drops a key whose value is unset,
  and `namespace_sharing_mode` is never a read predicate — both are the read
  visibility rules (`private` visibility isolates by app client but not by
  provider), not an oversight in the envelope. Concretely: an envelope with
  `app_client.provider: null` in a project that also has `prov-a` and `prov-b`
  rows recalls all three, while its own entry counts only the `null` group.
  Passing `provider` as a real string narrows the recall to that group.
- Counts cover **live rows only**: `forget()` tombstones are excluded from
  `episodes` and `records`. `knowledge_topics` has no `deleted` column, so
  topic rows are counted as stored; there is no tombstone counter for it.
- Display-only project metadata (`display_name`, `root_uri`, `repo_remote`,
  `default_branch`) is aggregated with `MAX()`, so the value is a
  deterministic function of the group's rows rather than whichever row a
  bare column came from.

#### Discovery tools are not read-visibility-filtered

`memory_scope_list` and `memory_policy_list` are deployment-topology audit
tools and are **not** filtered by `read_visibility`. A caller that reaches them
learns the shape of the corpus — which namespaces, projects, app clients and
agents exist, and roughly how much each holds — but no read filter is bypassed
and no content is exposed. The reasoning and the deployment conditions that
make this acceptable:
**[ACL.md — Trust boundary](ACL.md#trust-boundary)**.

### Policies and ACL

Scopes decide *visibility*; policies decide *permission*:

- `memory_policy_list` — persisted access policies and ACL bindings; an audit
  tool, like scope discovery above.
- `memory_policy_grant` — create/update a binding for a principal with
  `write_mode` (`allow`/`deny`) and `read_visibility`
  (`private`/`namespace`/`project`).
- A denied write returns `status: "write_denied"` in the success payload —
  it is a business outcome, not a transport error.

Full reference — resolution order, principal tokens, CLI/REST/agent
configuration, worked examples: **[ACL.md](ACL.md)**.

## Errors, timeouts and fallbacks

### Error taxonomy (MCP)

| Kind | Transport shape | Example |
| --- | --- | --- |
| Protocol error | JSON-RPC `error` (no result) | Unknown method (`-32601`), an `initialize` on a modern-envelope connection, an enveloped request on a handshake connection |
| Tool execution error | Result with `isError: true` + actionable text + `structuredContent.error`; an unknown tool name and an argument rejected before the body runs carry the text only | Timeout, bad input, dispatch failure |
| Business outcome | Result with `isError: false` and a status field | `status: "not_found"`, `status: "write_denied"`, `status: "dry_run"` |

An unknown **tool** is an execution error, not a protocol error — the frame is
well-formed JSON-RPC and the answer is `isError: true` with the text
`Unknown tool: <name>`. Only the middle row is `isError: true`; success-shaped
payloads with a non-happy `status` are **not** failures — callers branch on
the status.

### `memory_hygiene_apply` status

`status` is `dry_run` (preview, corpus untouched) or `applied` (cleanup
committed) — it is never `ok`. Both modes send the **same** keys: `status`,
`episode_targets`, `episode_ids`, `forgotten`, `not_found`, `expire_orphans`,
`orphan_repair`. Only `status` and the `forgotten` / `not_found` counters
differ, and both counters are `0` on a dry run, so a client can parse one shape
regardless of `dry_run`.

### Timeout resolution

Per tool: `CONSOLIDATION_MEMORY_TIMEOUT_<TOOL>` → per-tool default →
`CONSOLIDATION_MEMORY_TOOL_TIMEOUT_SECONDS` (default **60s**).

| Tool | Default | Tool | Default |
| --- | --- | --- | --- |
| `memory_store`, `memory_remember`, `memory_search` | 30s | `memory_claim_browse`, `memory_outcome_record`, `memory_outcome_browse`, `memory_status` | 30s |
| `memory_forget`, `memory_protect`, `memory_read_topic` | 30s | `memory_contradictions`, `memory_consolidation_log` | 30s |
| `memory_claim_search`, `memory_decay_report`, `memory_timeline`, `memory_browse` | 45s | `memory_store_batch`, `memory_correct` | 60s |
| `memory_ask` | see below | | |
| `memory_hygiene_scan` | 60s | `memory_hygiene_apply`, `memory_export` | 180s |
| `memory_compact` | 120s | `memory_consolidate` | 600s |
| `memory_policy_list`, `memory_policy_grant` | 20s | `memory_recall`, `memory_detect_drift` | see below |

`memory_scope_list` is the one tool with no per-tool default: it runs on the
generic `CONSOLIDATION_MEMORY_TOOL_TIMEOUT_SECONDS` fallback (60s).

`memory_ask` is the exception to the whole per-tool scheme: it fixes its budget
to `CONSOLIDATION_MEMORY_RECALL_TIMEOUT_SECONDS` before dispatch, so
`CONSOLIDATION_MEMORY_TIMEOUT_MEMORY_ASK` and
`CONSOLIDATION_MEMORY_TOOL_TIMEOUT_SECONDS` do not affect it. Raise the recall
variable. A timeout message says which budget applied instead of naming a
variable the call ignores.

Dedicated budgets (own environment variables):

| Variable | Default | Purpose |
| --- | --- | --- |
| `CONSOLIDATION_MEMORY_RECALL_TIMEOUT_SECONDS` | 60s | Semantic recall phase of `memory_recall` / `memory_ask` |
| `CONSOLIDATION_MEMORY_RECALL_FALLBACK_TIMEOUT_SECONDS` | 10s | Keyword-only fallback after a recall timeout |
| `CONSOLIDATION_MEMORY_DRIFT_TIMEOUT_SECONDS` | 90s | Drift scan (first attempt with `base_ref`) |
| `CONSOLIDATION_MEMORY_CLIENT_INIT_TIMEOUT_SECONDS` | 30s | Lazy client/engine startup on first tool call |

Every default above is the code default, not a recommendation: `server.py`
reads the variable at import and falls back to the listed value. A budget of
`0` or a negative number is not "unbounded" — the resolver substitutes its own
ceiling instead (180s drift, 90s client init, 90s recall, 20s recall fallback,
60s for a tool with no dedicated default).

### Fallback chains

- **`memory_recall`**: semantic phase times out → keyword `memory_search`
  fallback (payload gains a `message`/`warnings` explaining the degradation)
  → if that also times out, `isError: true` with tuning advice
  (shorter query, lower `n_results`, raise the budget).
- **`memory_detect_drift`**: first scan with `base_ref` times out → retry
  without `base_ref` (payload carries a `message` about the fallback) → if
  that fails too, a degraded result: full `DriftScanOutput` shape with empty
  lists, plus `message` **and `error`**, reported as `isError: true`. The `error`
  key is what makes it validate against the error arm — without it the payload
  also matched the success arm, and a client reading `structuredContent` saw a
  successful empty scan.
- **`memory_consolidate`**: bounded by its own budget; timeouts and crashes
  are `isError: true` with the exact variable to raise.

### Troubleshooting

| Symptom | Fix |
| --- | --- |
| `... timed out after Ns. Raise CONSOLIDATION_MEMORY_TIMEOUT_...` | Raise the named variable or reduce work (`n_results`, corpus size) |
| `Extra inputs are not permitted` | Remove the unknown argument — the published `inputSchema` is the contract. MCP reports it as `isError: true`, REST as HTTP 422, the dispatch seam as `ToolContractError` ([details](#surfaces-parity)) |
| Recall answers feel stale right after startup | `CONSOLIDATION_MEMORY_DEFERRED_KNOWLEDGE_RETRY_SECONDS` (default `3`) delays knowledge inclusion; set `0` for immediate reads |
| First call is slow | Client init + warmup run lazily; prewarm with `CONSOLIDATION_MEMORY_WARMUP_ON_START=1` (default on) and `CONSOLIDATION_MEMORY_PRELOAD_SCIPY_ON_START=1` |
| Two server processes fight over the DB | The stdio singleton guard serializes per project; tune `CONSOLIDATION_MEMORY_STDIO_SINGLETON` / `..._TAKEOVER_TIMEOUT_SECONDS` |

## Environment reference

All variables are prefixed `CONSOLIDATION_MEMORY_`. Most are read directly
from `os.environ` by the MCP runtime; the rest are `Config` fields, and a
field is addressable as `CONSOLIDATION_MEMORY_<FIELD>` —
`CONSOLIDATION_AUTO_RUN` is the field, so its variable is
`CONSOLIDATION_MEMORY_CONSOLIDATION_AUTO_RUN`. The tables below name the
variable without that prefix.

### Paths and projects

| Variable | Default | Purpose |
| --- | --- | --- |
| `CONFIG` | unset | Path to an alternate `config.toml`; unset discovers one in the platform config dir |
| `DATA_DIR` | platform data dir | Base data directory — the per-project `DATA_DIR` is derived below it as `projects/<project>` |
| `PROJECT` | `default` | Active project slug |

### Backends

| Variable | Default | Purpose |
| --- | --- | --- |
| `EMBEDDING_BACKEND` | `fastembed` | `fastembed`, `lmstudio`, `openai`, `ollama` |
| `LLM_BACKEND` | `lmstudio` | `lmstudio`, `openai`, `ollama`, `disabled` |
| `FASTEMBED_CACHE_DIR` | platform cache | Where the embedding model is downloaded |

### Consolidation

| Variable | Default | Purpose |
| --- | --- | --- |
| `CONSOLIDATION_AUTO_RUN` | `true` | `Config` field gating automatic consolidation (`client.py`, `client_runtime.py`, the maintenance daemon) |
| `MCP_AUTO_CONSOLIDATE` | off | The MCP client's own `auto_consolidate` flag: consolidate after stores when idle |

### MCP runtime

| Variable | Default | Purpose |
| --- | --- | --- |
| `MCP_TOOL_PROFILE` | `full` | `full` or `simple` (3 tools) |
| `MCP_BLOCKING_WORKERS` | 16 | Worker threads for blocking tool bodies |
| `WARMUP_ON_START` | on | Warm caches at server startup |
| `WARMUP_START_DELAY_SECONDS` / `WARMUP_AWAIT_SECONDS` | 0.25 / 0 | Warmup scheduling knobs |
| `WARMUP_PRIME_TOPIC_CACHE` / `WARMUP_PRIME_RECORD_CACHE` | `true` | Prime recall caches at warmup |
| `WARMUP_PRIME_CLAIM_CACHE` | `false` | Prime the claim cache (heavier) |
| `PRELOAD_SCIPY_ON_START` / `PRELOAD_NUMERIC_BACKENDS_ON_START` | on / on | Import heavy numeric deps up front (avoids first-call stalls) |
| `STATUS_LIGHTWEIGHT` | on | Fallback for `memory_status(lightweight=...)` when the argument is omitted: skip markdown scans |
| `IDLE_TIMEOUT_SECONDS` | 900 | Exit stdio server after N idle seconds (`0` = never) |
| `IDLE_CHECK_INTERVAL_SECONDS` | 15 | Idle sweep cadence |
| `STDIO_SINGLETON` / `STDIO_SINGLETON_TAKEOVER_TIMEOUT_SECONDS` | on / 10 | One server process per project; takeover wait |
| `DUMP_STACKS_ON_CLIENT_INIT_TIMEOUT` | off | Thread dump when client init hangs |

### Timeouts and recall

| Variable | Default | Purpose |
| --- | --- | --- |
| `TOOL_TIMEOUT_SECONDS` | 60 | Fallback budget for tools without a dedicated default |
| `TIMEOUT_<TOOL>` | per-tool | Per-tool override, e.g. `TIMEOUT_MEMORY_STATUS`; ignored by `memory_ask`, which uses `RECALL_TIMEOUT_SECONDS` |
| `RECALL_TIMEOUT_SECONDS` | 60 | Semantic recall phase, and the whole `memory_ask` budget |
| `RECALL_FALLBACK_TIMEOUT_SECONDS` | 10 | Keyword fallback phase |
| `RECALL_DEADLINE_MARGIN_RATIO` | 0.85 | Fraction of the timeout the internal recall deadline spans; the remaining 0.15 is the headroom before the caller-side timeout fires |
| `DRIFT_TIMEOUT_SECONDS` | 90 | Drift scan budget |
| `CLIENT_INIT_TIMEOUT_SECONDS` | 30 | Client/engine startup budget |
| `DEFERRED_KNOWLEDGE_RETRY_SECONDS` | 3 | Wait before knowledge is guaranteed fresh in recall |

### REST (when the HTTP surface is enabled)

| Variable | Default | Purpose |
| --- | --- | --- |
| `REST_AUTH_TOKEN` | unset | Bearer token; required for any bind outside loopback |
| `REST_ALLOW_PUBLIC_BIND` | unset (falsy) | Disables **both** non-loopback refusals — the startup bind check and the 503 middleware gate — and with no token set leaves the whole memory API reachable unauthenticated. Not a supported configuration; see [SECURITY.md — REST API](../SECURITY.md#rest-api) |

## Recipes

Every block below is the request object of a `tools/call`; add your own
`jsonrpc` and `id` when sending it on the wire.

### Remember something, then ask about it

```json
{"method": "tools/call", "params": {"name": "memory_remember", "arguments": {"content": "We deploy on Tuesdays; use podman locally.", "kind": "preference"}}}
```

```json
{"method": "tools/call", "params": {"name": "memory_ask", "arguments": {"query": "how do we deploy?"}}}
```

### Store with tags, retrieve semantically

```json
{"method": "tools/call", "params": {"name": "memory_store", "arguments": {"content": "Fix: retry the migration checksum check before deploy", "content_type": "solution", "tags": ["deploy", "ci"]}}}
```

```json
{"method": "tools/call", "params": {"name": "memory_recall", "arguments": {"query": "what breaks the deploy?", "n_results": 5}}}
```

### Work inside a project scope

```json
{"method": "tools/call", "params": {"name": "memory_store", "arguments": {"content": "Billing uses quarterly schema bumps", "scope": {"project": {"slug": "billing"}}}}}
```

```json
{"method": "tools/call", "params": {"name": "memory_search", "arguments": {"query": "schema", "scope": {"project": {"slug": "billing"}}}}}
```

### Discover scopes, then page through them

```json
{"method": "tools/call", "params": {"name": "memory_scope_list", "arguments": {"limit": 20}}}
```

Take any `scopes[i].scope` and reuse it as the `scope` argument above; if
`message` reports more pages, call again with `"offset": 20`.

### Consolidation health loop

```json
{"method": "tools/call", "params": {"name": "memory_consolidate", "arguments": {}}}
{"method": "tools/call", "params": {"name": "memory_status", "arguments": {"lightweight": true}}}
{"method": "tools/call", "params": {"name": "memory_consolidation_log", "arguments": {"last_n": 5}}}
```

### Challenge claims against code changes

```json
{"method": "tools/call", "params": {"name": "memory_detect_drift", "arguments": {"repo_path": "/path/to/repo", "base_ref": "main"}}}
{"method": "tools/call", "params": {"name": "memory_claim_search", "arguments": {"query": "deploy pipeline"}}}
```

## Surfaces parity

MCP, REST, the Python SDK and the OpenAI-compatible schemas all route through
the same dispatch (`tool_dispatch.py`) and canonical query layer — tool names,
parameter names and payload shapes match across surfaces. The only surface-specific
parts are transport encoding and the MCP-only bits documented here
(`outputSchema` publication, `isError` mapping, stdio lifecycle).

Argument **names** are the shared contract; types and bounds are not uniform,
because each surface validates what its own transport can:

- MCP validates against the model derived from the handler signature and then
  **clamps**: `n_results: 0` becomes `1`, and `"5"` is coerced to `5`.
- Dispatch and REST reject the same values: `n_results: 0` returns
  `{"error": "n_results must be between 1 and 50"}` on dispatch and HTTP 422 on
  REST.
- `maxLength` and `enum` constraints in a published `inputSchema` are advisory on
  MCP, which does not read them from the schema; dispatch enforces the same
  bounds in its validators.

So a host that relays caller intent should not assume an out-of-range value is
refused everywhere. The startup self-check below compares the **argument-name**
set only.

### One input contract, four enforcement points

Every published `inputSchema` declares `additionalProperties: false`, and
every surface enforces it. The allowed set is derived once from the published
schemas (`tool_dispatch.accepted_argument_names`), and the same set is checked
where the request enters:

| Surface | How the rejection surfaces |
| --- | --- |
| MCP | `isError: true` with text naming the offending key, no `structuredContent` — the SDK's argument model validates before the tool body runs |
| REST | HTTP **422** from the request model (`extra="forbid"`), with the offending keys in `loc`/`msg`; a `ToolContractError` from dispatch also maps to 422 |
| OpenAI / dispatch | `ToolContractError` propagates instead of becoming a soft `{"error": ...}` payload — a silent drop would mean the call ran with different arguments than asked |
| Desktop | the dispatch seam's `ToolContractError`, from the same `execute_tool_call` |

Deliberate exception: the nested objects inside `episodes` and `code_anchors`
stay permissive, because the published schema types them as plain objects. The
top-level bodies are strict, the nested maps are not. The CLI has its own
command surface and does not take tool arguments, so it is not part of this
contract.

A startup self-check cross-checks the SDK-enforced argument set against the
dispatch set for every registered tool and refuses to serve on divergence, so
a signature that drifts from its published schema is a boot failure rather than
a surface that quietly accepts what the others reject.

- OpenAI tool schemas: [`src/consolidation_memory/schemas.py`](../src/consolidation_memory/schemas.py)
- REST surface: [README — REST](../README.md#rest-api)
- Architecture: [ARCHITECTURE.md](ARCHITECTURE.md)
