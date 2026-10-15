# Access control: scope policies and ACL

Two layers decide what an operation is allowed to do with memory data:

1. **Inline scope policy** — the optional `policy` section of the `scope`
   argument passed to a tool call. It travels with the request and acts as
   the *base* policy for that call.
2. **Persisted ACL bindings** — rows in `access_policies`,
   `policy_acl_entries`, and `policy_principals` (schema in
   `db/migrations.py`, resolution in `policy_engine.py`). They are matched
   against the caller's *principal tokens* and, when any binding matches,
   they **override** the inline policy for that operation.

Both layers control exactly two knobs:

| Knob | Values | Guards |
| --- | --- | --- |
| `write_mode` | `allow` (default), `deny` | Mutations |
| `read_visibility` | `private` (default), `project`, `namespace` | How widely reads see data |

## What is restricted

### Writes (`write_mode`)

When the effective `write_mode` is `deny`, these mutations are refused:

- `memory_store` / `memory_remember` and `memory_store_batch`
- `memory_forget`
- `memory_protect`
- `memory_correct`
- `memory_outcome_record`

Enforcement is **fail-closed**: a write only proceeds when the mutation's
scope row matches the *exact* resolved scope (namespace + project + app
client + agent/session). Widening `read_visibility` never grants
cross-project or cross-app write rights (see the mutation filter in
`client.py`).

A denied write is a **business outcome, not a transport error** on every
surface (Python, MCP, REST, OpenAI): the call succeeds and the payload
carries

```json
{ "status": "write_denied", "message": "Writes are denied by scope policy (write_mode='deny')." }
```

so agents branch on `status` and pipelines do not treat it as an outage.
A batch store is denied **as a whole**, before any episode is validated or
written: `memory_store_batch` returns one `results` entry describing the denial,
with `stored` and `duplicates` at `0`. There is no per-episode outcome row for a
denied batch.

Not gated by `write_mode`: policy administration itself, scope discovery,
maintenance operations (`memory_hygiene_*`, `memory_compact`, `memory_export`,
`memory_consolidate`, `memory_detect_drift`). Those are covered by the trust
boundary below.

### Reads (`read_visibility`)

Reads never fail; they return *less*. The effective visibility widens the
read filter additively (exact rules in `client.py`):

| Value | Sees |
| --- | --- |
| `private` (default) | Same namespace + project, and — unless the namespace `sharing_mode` is `shared`/`team`/`managed` — only the same app client |
| `project` | Same namespace + project across app clients |
| `namespace` | Everything in the namespace, across projects and app clients |

Agent and session selectors keep narrowing within whichever level is active:
if the scope carries an `agent` or `session`, only rows stamped with that
agent/session are visible even under `namespace` visibility.

## How the effective policy is resolved

For every operation the client builds the resolved scope
(`client.py` → `policy_engine.py`):

1. Start from defaults: `private` + `allow`.
2. Take the call's inline `scope.policy` (if provided) as the **base**.
3. Collect ACL rows matching **any** of the caller's principal tokens
   (`get_matching_policy_acl_entries`).
4. If rows matched, they win:
   - `write_mode`: **deny overrides allow** across matched rows;
   - `read_visibility`: **most restrictive wins**
     (`private` < `project` < `namespace`);
   - both conflicts are recorded on the resolution and logged
     (`write_mode_conflict_deny_overrides_allow`,
     `read_visibility_conflict_most_restrictive_wins`).
5. The resolution carries its origin (`policy_source`: `scope_policy` or
   `persisted_acl`) and how many bindings matched (`policy_acl_matches`).

If the ACL lookup itself fails, the engine logs a warning and applies the
inline `scope.policy` unchanged — the persisted rows are simply not consulted,
so a lookup failure widens the effective policy to whatever the call asked for.

### Principal tokens

A binding matches when its principal key equals any token derived from the
resolved scope:

| Token type | Example value | When it is emitted |
| --- | --- | --- |
| `any` | `*` | always (wildcard match-all) |
| `namespace_slug` | `team-a` | always |
| `project_slug` | `repo-a` | always |
| `app_client` | `python_sdk:legacy_client` | always, as `type:name` |
| `app_client_external_key` / `app_client_provider` | configured app identifiers | only when set on the envelope |
| `agent_external_key` / `agent_name` | agent identity inside the app | `agent_external_key` when set, otherwise `agent_name` — never both |
| `session_external_key` / `session_kind` | session identity | `session_external_key` when set, `session_kind` always, for a scope carrying a session |

Matching is **exact**: the stored `(principal_type, principal_key)` pair must
equal one of the tokens above, and `any` is the only wildcard. Those ten types
are the complete vocabulary — `grant` validates the mode and visibility values
but not the type, so an unrecognized type is stored and never matches. `agent`
is not an alias for `agent_name`, and an `app_client` key must be written as
`type:name` (e.g. `python_sdk:legacy_client`, `mcp:desktop`). Verify a grant
that does not behave with `policy list` plus a probe call.

## Configuring

### Manually — CLI

```bash
# Audit everything that is granted
consolidation-memory policy list

# Deny writes for the default Python SDK app client
consolidation-memory policy grant --principal-type app_client \
    --principal-key python_sdk:legacy_client --write-mode deny
```

`policy list` prints one row per binding:
`policy_id · namespace · project · principal · write · read`.

`policy grant` flags:

| Flag | Meaning |
| --- | --- |
| `--namespace` / `--project` | Scope selectors; **omitted = wildcard** (binding applies to any namespace/project) |
| `--principal-type` | Required. One of the ten token types in [Principal tokens](#principal-tokens), e.g. `any`, `app_client`, `agent_name` |
| `--principal-key` | Required. Concrete key, e.g. `python_sdk:legacy_client` |
| `--write-mode` | `allow` or `deny` |
| `--read-visibility` | `private`, `project`, or `namespace` |

At least one of `--write-mode` / `--read-visibility` is required (the
binding must grant something).

### Manually — REST

```http
GET  /memory/policy
POST /memory/policy/grant
```

```json
{
  "namespace": "team-a",
  "project": "repo-a",
  "principal_type": "agent_name",
  "principal_key": "ci-bot",
  "write_mode": "deny"
}
```

Omitted `namespace`/`project` act as wildcards. The body is
`rest.PolicyGrantRequest`: a strict model, so an undeclared key is HTTP 422
rather than a silently ignored field.

### Via an agent — MCP

Two tools cover the loop; both are in the full profile:

```json
{"method": "tools/call", "params": {"name": "memory_policy_grant", "arguments": {
  "principal_type": "agent_name",
  "principal_key": "ci-bot",
  "write_mode": "deny"
}}}
```

```json
{"method": "tools/call", "params": {"name": "memory_policy_list", "arguments": {}}}
```

Typical agent flow: *“Deny writes for the CI bot”* → the agent calls
`memory_policy_grant`, then calls `memory_policy_list` to confirm the new
row, and finally issues a probe write (for example `memory_remember` with
the denied principal's scope) expecting `status: "write_denied"`.

The binding above matches calls whose **scope carries that agent**
(`scope.agent.name = "ci-bot"`); principals are matched exactly, see
[Principal tokens](#principal-tokens).

### Inline policy for one call

Scopes accept a `policy` section directly — useful for one-off locks and
for tests:

```json
{
  "namespace": {"slug": "team-a"},
  "project": {"slug": "repo-a"},
  "policy": {"write_mode": "deny"}
}
```

Precedence, restated: a matching persisted row **overrides** this inline
policy.

## Worked examples

**1. Make one project read-only for every caller (CLI, wildcard):**

```bash
consolidation-memory policy grant --namespace team-a --project repo-a \
    --principal-type any --principal-key '*' --read-visibility project --write-mode deny
```

`any` / `*` is the match-everyone token pair; the namespace/project
selectors keep the rule inside `team-a/repo-a`.

**2. Open up reads across a namespace for one app:**

```json
{"name": "memory_policy_grant", "arguments": {
  "principal_type": "app_client", "principal_key": "mcp:desktop",
  "read_visibility": "namespace"
}}
```

Afterwards a `memory_recall` issued with a `namespace`-scoped envelope sees
rows from sibling projects; writes still require an exact-scope match and
an `allow` binding.

**3. Two bindings disagree (conflict):** grant `write_mode: allow` for
`app_client` / `mcp:desktop` and `write_mode: deny` for `agent_name` /
`ci-bot`; a call whose principals match both resolves to **deny** (deny
overrides allow) and the resolution conflict is logged.

## Reference architecture: parallel microservice development

A team builds two interacting microservices **at the same time** — say
`orders-service` and `payments-service` — with agents working both repos in
parallel. Three knowledge bodies exist, in one namespace `team-payments`:

```text
            ┌────────────── shared contract scope ──────────────┐
            │ event payloads · idempotency rules · rollout order │
            │ breaking-change warnings · integration how-tos     │
            └────────▲───────────────────────────▲───────────────┘
       publish / read │                           │ publish / read
┌─────────────────────┴────────────┐ ┌────────────┴──────────────────────┐
│ project: orders-service          │ │ project: payments-service         │
│ failure modes, migrations,       │ │ retry semantics deep-dives,       │
│ internal refactors, experiments  │ │ internal refactors, experiments   │
└──────────────────────────────────┘ └───────────────────────────────────┘
```

Scope layout:

| Scope | Identity | Default visibility |
| --- | --- | --- |
| Service internals | `namespace=team-payments`, `project=orders-service` (resp. `payments-service`) | `private` — isolated out of the box, **no grants needed** |
| Shared contract | `namespace=team-payments`, `project=contract` | same default; widened by the grants below |

### Grants that make the pattern work

1. **Let every participant read the shared scope across apps.** The policy
   selector pins the rule to the contract project; `any`/`*` matches any
   caller, and `read_visibility=project` removes the default app-level
   isolation *inside that scope only*:

   ```bash
   consolidation-memory policy grant \
       --namespace team-payments --project contract \
       --principal-type any --principal-key '*' \
       --read-visibility project
   ```

   Calls scoped to `orders-service` or `payments-service` do not match the
   selector — they keep default `private` isolation.

2. **Keep a consumer read-only in the shared scope** (it consumes contract
   docs but must not rewrite them):

   ```bash
   consolidation-memory policy grant \
       --namespace team-payments --project contract \
       --principal-type app_client --principal-key mcp:payments-agent \
       --write-mode deny
   ```

   Writers that *should* publish (the services' boundary agents) need no
   grant for writing — `write_mode` defaults to `allow` and writes are
   exact-scope — but giving them an explicit
   `--write-mode allow` binding documents the intent in `policy list`.

3. **Give each service a distinguishable identity.** Have each agent write
   with its own envelope, e.g. `scope = {"namespace": {"slug":
   "team-payments"}, "project": {"slug": "orders-service"}, "app_client":
   {"name": "orders-agent", "app_type": "mcp"}}`. Those identities are what
   per-principal grants bind to later.

### Publish, then verify

```json
{"name": "memory_remember", "arguments": {
  "content": "payments retries are idempotent by (order_id, attempt); max 3 attempts before DLQ",
  "kind": "fact",
  "scope": {"namespace": {"slug": "team-payments"}, "project": {"slug": "contract"}}
}}
```

Verification loop (run from the counterpart's envelope):

1. `memory_scope_list` — the `contract` scope shows up with its counts. The
   listing is not read-visibility-filtered, so this is a whole-deployment
   topology check, not a check of what this agent may read; counts cover live
   rows only, so a forgotten episode stops counting.
2. `memory_recall` with the contract scope — the published fact is visible.
3. `memory_recall` with a service scope — sibling internals are **not**.
4. A probe write with a `write_mode: deny` principal — `status: "write_denied"`.

What belongs in the shared scope is a deliberate team decision (contract
facts, rollout rules, published integration docs — not work-in-progress
notes), carried in practice by distinct principals per service. Neither
listing is gated by `read_visibility` or `write_mode`; see
[Trust boundary](#tools-intentionally-outside-the-read-filter).

## Seeing what is in effect

- `consolidation-memory policy list` / `GET /memory/policy` /
  `memory_policy_list` — the persisted bindings (what you granted).
- Server log lines — resolved-policy conflicts and ACL lookup fallbacks.
- A probe write — the only direct answer about `write_mode` in a given scope:
  `status: "write_denied"` versus a successful write. `read_visibility` has no
  probe, because a read that is too narrow returns less rather than failing.

## Trust boundary

ACL is **data-level sharing control between principals inside one
deployment** — it is not authentication, and it does not separate tenants.
On MCP stdio any process that can launch the server has full database access
(see [SECURITY.md](../SECURITY.md#trust-boundaries)); a REST bind outside
loopback requires a bearer token, or an explicit
`CONSOLIDATION_MEMORY_REST_ALLOW_PUBLIC_BIND` opt-in. Use ACL to separate
projects, agents and apps — use the transport and OS permissions to separate
tenants.

### Tools intentionally outside the read filter

Two tools are deployment-topology **audit** tools and are deliberately not
filtered by `read_visibility`:

| Tool | What it reports | Why it is unfiltered |
| --- | --- | --- |
| `memory_policy_list` / `policy list` / `GET /memory/policy` | Every persisted ACL binding | The question an operator asks is "what did we grant", not "what may this principal see"; filtering the grant table makes a misconfigured binding invisible to the audit that would find it |
| `memory_scope_list` / `GET /memory/scopes` | Every scope that has stored rows, with per-table counts | The question is "does the data layout match the intended topology"; hiding the scopes a principal cannot read would defeat the audit |

Stated plainly: on a shared deployment, a caller that can invoke these tools
learns the shape of the whole corpus (which namespaces, projects, app clients
and agents exist, and roughly how much each holds) even when its own
`read_visibility` is `private`. It does **not** learn the contents — no read
filter is bypassed, and no row is read through this path.

This is acceptable exactly when the transport is: the stdio subprocess is
inherited by whoever launched it, so the caller already has full database
access and the listing adds no privilege; a REST deployment is bounded by the
bearer token and the bind address. Do **not** expose the MCP subprocess to
untrusted multi-tenant environments without OS-level isolation (separate user,
container, VM or equivalent) — ACL will not do it for you.

## Executable specification

`tests/test_scope_policy_contracts.py` locks the behavior across Python,
MCP, REST and OpenAI surfaces:

- inline `write_mode: deny` honored by Python, OpenAI dispatch, MCP and REST;
- persisted ACL deny enforced across surfaces;
- persisted `read_visibility` enforced across surfaces;
- `memory_forget` / `memory_protect` / `memory_correct` deny paths.

The input contract is strict on every surface: unknown top-level
arguments are rejected from one shared allowed-argument set derived from the
published `inputSchema` — HTTP 422 on REST, `ToolContractError` on the
dispatch seam — so a body that MCP refuses cannot be replayed through another
surface. Nested objects in `episodes` / `code_anchors` stay permissive because
the published schema types them as plain objects. Wire details:
[MCP guide — Surfaces parity](MCP_GUIDE.md#surfaces-parity).

## Related

- [MCP guide — scopes and policies](MCP_GUIDE.md#scopes-and-sharing)
- [MCP guide — discovering existing scopes](MCP_GUIDE.md#discovering-existing-scopes)
- [Architecture](ARCHITECTURE.md) — `db/scope.py`, `policy_engine.py`
- [Security policy](../SECURITY.md)
