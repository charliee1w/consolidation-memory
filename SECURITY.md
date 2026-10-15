# Security Policy

## Supported Versions

| Version line | Supported |
| --- | --- |
| `0.21.x` | Yes |
| `0.20.x` | Best-effort (upgrade to `0.21.x`) |
| `<0.20.0` | No |

`0.21.x` is the line that carries the current tool contract. A breaking change
ships as a minor under the 0.x policy — the project stays on `0.x`, the contract
may break, and no `1.x` line is planned — so each release takes the "Yes" row and
the line beneath it drops to best-effort. The policy is stated in full, including
why commit subjects carry neither `!` nor `BREAKING CHANGE`, in
[CONTRIBUTING.md](CONTRIBUTING.md#versioning) and
[docs/RELEASE_AUTOMATION.md](docs/RELEASE_AUTOMATION.md#versioning).

## Trust Boundaries

Data-level sharing between principals (scope policies, ACL bindings, deny/read
visibility) is documented in [docs/ACL.md](docs/ACL.md). It is not a substitute for
transport authentication.

### MCP (stdio)

The default MCP server speaks JSON-RPC over stdio. **Any process that can launch the
server can read and write the full memory database** for the configured project.
There is no authentication layer on stdio transport.

Treat MCP as a **local trust boundary** (IDE, agent host, same user session). Do not
expose the MCP subprocess to untrusted multi-tenant environments without OS-level
isolation.

### The repository being analysed

`memory_detect_drift` takes a repository path and scans it in a subprocess. The
repository is **input, not a trusted peer**: the worker reads it through `git` and
the memory database, never executes its files, and its own working directory is
never the analysed path — a repository containing `consolidation_memory/` or a
`json.py` cannot shadow the running package. What the repository still controls
is its own git metadata, so a `.git/config` there can affect the git invocation
the worker makes. Do not point drift detection at a repository you would not read
by hand.

### REST API

Beyond loopback, every request needs the bearer token from
`CONSOLIDATION_MEMORY_REST_AUTH_TOKEN` except `/health` and CORS preflight
(`OPTIONS`). A non-loopback bind without a token is refused twice over:
`validate_rest_bind` fails startup, and the auth middleware answers **503** to any
request that reaches a non-loopback host anyway.
`CONSOLIDATION_MEMORY_REST_ALLOW_PUBLIC_BIND=true` disables both refusals and is not
a supported configuration.

The tool surface is **31 distinct paths** under `/memory` answering **34 method+path
pairs**: one route per tool (29), two convenience path aliases
(`DELETE /memory/episodes/{episode_id}` for `memory_forget` and
`GET /memory/topics/{filename}` for `memory_read_topic`), and a bodyless `GET`
twin for the three tools that take a `scope` (`status`, `browse`,
`decay-report`). It covers the same 29 tools as MCP and OpenAI dispatch
(per-tool schemas in [docs/TOOLS.md](docs/TOOLS.md)). `GET /health` adds one
path and one pair, for 32 paths and 35 pairs. Seven `/ops/*` routes (overview,
metrics, daemon status/install, consolidate, warmup, reindex) sit outside that
surface and are authenticated like the rest.

Bodyless routes can only answer for the deployment, never for a caller-supplied scope: `GET /memory/status`, `GET /memory/browse`, `GET /memory/decay-report`, `GET /memory/topics/{filename}`, `GET /memory/policy`, `GET /memory/hygiene/scan` and `GET /memory/scopes`. The `POST` variants of `status`, `browse` and `decay-report` accept an explicit `scope` and are the scoped forms; `POST /memory/policy/grant` and `POST /memory/hygiene/apply` are separate operations, not scoped twins.

#### Audit tools disclose deployment topology

`memory_scope_list` and `GET /memory/scopes` return the deployment's scope topology — namespaces, projects, app clients, agents, sessions — with per-table row counts (`episodes`, `records`, `topics`), most-recently-used first, pageable via `limit`/`offset`.

The result is **intentionally global**: not filtered by `read_visibility`, by policy, or by the caller's scope. A principal that can read only its own scope still learns that other scopes exist and roughly how large they are. Two reasons:

- It is a topology **audit** tool. The value is in showing what the deployment actually contains, including scopes the caller cannot read into.
- The transport already grants far more (see the stdio boundary above): the REST token authenticates the transport, not a tenant boundary.

`memory_policy_list` and `GET /memory/policy` are unfiltered for the same reason and return **every persisted ACL binding**. A filtered grant table hides exactly the misconfigured binding a topology audit is meant to surface.

`docs/ACL.md` separates principals *inside* one deployment. Treat both audit surfaces as deployment-scoped metadata, not tenant isolation, and do not expose either to mutually distrusting tenants. Per-tool rationale: [ACL.md — Tools intentionally outside the read filter](docs/ACL.md#tools-intentionally-outside-the-read-filter).

### Python SDK

Direct `MemoryClient` usage inherits the privileges of the calling process and
reads/writes the same on-disk project data as MCP.

## Policy Coverage (`write_mode`)

`write_mode='deny'` is enforced in [`client.py`](src/consolidation_memory/client.py) by `_write_denied_message`, at six points covering the seven write tools below — `memory_remember` re-translates onto the store path, and `memory_store_batch` is denied as a whole before any episode is written. A denial is a business outcome, not a transport error: the call succeeds and the payload carries `status: "write_denied"`.

Gated: `memory_store`, `memory_remember`, `memory_store_batch`, `memory_outcome_record`, `memory_forget`, `memory_correct`, `memory_protect`.

**Not gated by `write_mode`**: `memory_export`, `memory_compact`, `memory_consolidate`, `memory_hygiene_scan`, `memory_hygiene_apply`, `memory_detect_drift`, policy administration (`memory_policy_list`, `memory_policy_grant`) and every `/ops/*` route — `POST /ops/daemon/install` registers a login-time job on the host. These are maintenance and administration operations, and the transport itself is not gated: a `deny` policy does not stop an export or a hygiene apply. Scope them with the transport boundary, not with policy.

## Input Contract Enforcement

Unknown input arguments are rejected on every surface, from one shared
allowed-argument set derived from the published `inputSchema`
(`tool_dispatch.accepted_argument_names`):

- **MCP** — pydantic `extra="forbid"` on the SDK argument model, before the tool body
  runs.
- **REST** — request bodies derive from `rest.StrictRequestModel` (`extra="forbid"`);
  an unknown body key is HTTP 422 naming the offending keys. Query parameters and
  nested values inside `episodes` / `code_anchors` are not covered.
- **OpenAI / dispatch** — `tool_dispatch.reject_unknown_arguments` runs first in
  `execute_tool_call` and raises `ToolContractError`. There is no exemption list:
  the recall deadline a surface resolves for its own timeout budget travels as a
  keyword on `execute_tool_call`, not as an argument key, so a caller cannot smuggle
  it past the contract on any tool.

The guarantee is about argument **names**. Bounds and types are enforced where each
surface can enforce them: MCP validates against the schema-derived model but clamps
what it accepts (a `memory_recall` with `n_results: 0` becomes `1`), while dispatch
and REST reject the same value outright.

`server._verify_published_argument_contract()` cross-checks the SDK-enforced set
against the dispatch-side set for every registered tool at import and in `lifespan`,
raising `MCPCompatError` on divergence. A test asserts set equality between the
published properties, the enforced set and the MCP `input_schema` for all 29 tools.

## Output Contract Integrity

Every tool publishes `outputSchema` as `anyOf[success, error]`. Publication is
self-healing: `mcp_compat.install_list_tools_heal` wraps the SDK's `tools/list` funnel,
so a tool registered after startup gets its schema on the next listing, and the SDK's
cached `Tool.output_schema` is evicted whenever it would go stale. At import and in
`lifespan`, `server._verify_published_output_schemas()` fails loudly if any registered
tool would publish a success-only schema.

Private `mcp` SDK surfaces are isolated in `src/consolidation_memory/mcp_compat.py`;
the supported range is `mcp[cli]>=2.3.0,<3`, and a missing module or attribute raises
`MCPCompatError` naming the installed version, the supported range and what was
probed. An `MCPCompatError` at startup is a hard failure, not a degradation.

## Reporting a Vulnerability

Do not open public GitHub issues for security vulnerabilities.

Report vulnerabilities through GitHub Private Vulnerability Reporting:

- [https://github.com/charliee1w/consolidation-memory/security/advisories/new](https://github.com/charliee1w/consolidation-memory/security/advisories/new)

Include:

- A clear description of impact and affected component(s)
- Reproduction steps or proof-of-concept
- Any suggested mitigation

## Response Expectations

- Initial acknowledgement target: within 3 business days
- Triage/update target: within 7 business days
- Fix and disclosure timing depends on severity and release complexity

We will coordinate disclosure timing with reporters when possible.
