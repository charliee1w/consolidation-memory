# Contributing

Thanks for contributing to `consolidation-memory`.

## Development setup

```bash
git clone https://github.com/charliee1w/consolidation-memory
cd consolidation-memory
pip install -e ".[all,dev]"
```

## Trust invariants

Behavior changes must preserve:

1. **Temporal correctness** — `as_of` queries reflect knowledge at that time.
2. **Provenance traceability** — claims link to source episodes, topics, or records.
3. **Contradiction visibility** — conflicts are logged and surfaced; history is not silently overwritten.
4. **Drift challenge auditability** — `code_drift_detected` events and challenged-claim state stay inspectable.
5. **Scope isolation** — namespace/project/app/agent/session boundaries are not accidentally widened.
6. **Surface parity** — Python, MCP, REST, and OpenAI tool dispatch share the same semantics.

Schema changes must be additive migrations with tests that call `ensure_schema()`. Invalidate caches (`topic_cache`, `record_cache`, `claim_cache`) after graph or knowledge mutations.

## Scope vs global operations

Some tools are **scope-aware** by default; others are **global by design**.

**Scope-aware reads and writes** (use resolved default scope when omitted):

- `memory_store`, `memory_recall`, `memory_forget`, browse/search paths
- Audit reads: `memory_contradictions`, `memory_decay_report`, `memory_status`, `memory_consolidation_log` — same default as recall/browse. Pass an explicit `scope` to narrow further.

**Global by design** (intentionally corpus- or repo-wide):

- `memory_consolidate` / `consolidate()` — processes unconsolidated episodes across the DB
- `memory_compact` / FAISS compaction — rebuilds the shared vector index
- `memory_detect_drift` — git diff against a base ref (namespace/project scope only narrows challenged-claim attribution)
- `memory_policy_list` / `memory_policy_grant` — persisted ACL administration across the DB (CLI: `consolidation-memory policy list|grant`)
- `memory_hygiene_scan` / `memory_hygiene_apply` — corpus-wide noisy-episode scan and orphan-claim repair (CLI: `consolidation-memory hygiene scan|apply`; UI Hygiene tab)
- `memory_scope_list` — deployment-topology audit. It is **intentionally not filtered by `read_visibility`**: ACL separates principals inside one deployment, it does not separate tenants, and a scope inventory is only useful when it is complete. See the trust boundary in [docs/ACL.md](docs/ACL.md#trust-boundary).
- Audit reads with `global_scope=true` — corpus-wide ops dashboard view; `memory_status` caches per scope key (including global)

`memory_forget` is scope-aware but **also expires claims** that lose all provenance when episodes are forgotten. Use hygiene apply with `expire_orphans=true` for claims detached by batch cleanup.

Two contract rules for the tools above:

- `memory_hygiene_apply` returns one key set in both modes: `status` is `dry_run` or `applied` (never `ok`), and `episode_ids` / `forgotten` / `not_found` are always present (zeroed on a dry run). `types.HygieneApplyResult` is the producer type and `tool_contracts.HygieneApplyOutput` mirrors it under an import-time guard, so a contract can never reject a payload after the episodes were deleted.
- `memory_scope_list` groups on the 11 canonical exact-match scope keys — two scopes differing only in `namespace_sharing_mode`, `app_client_provider`, `app_client_external_key`, `agent_name` or `session_kind` are distinct rows, not one merged row — and counts live rows only. `namespace.display_name` is not part of the payload; the envelope is reusable as a `scope` argument.

### Policy administration

Self-hosted deployments can manage namespace/project ACL bindings through any surface:

| Surface | List | Grant |
| --- | --- | --- |
| CLI | `consolidation-memory policy list` | `consolidation-memory policy grant --principal-type ...` |
| MCP / OpenAI tools | `memory_policy_list` | `memory_policy_grant` |
| REST | `GET /memory/policy` | `POST /memory/policy/grant` |

Omitted `namespace` or `project` selectors act as wildcards. Grant requires at least one of
`write_mode` (`allow`/`deny`) or `read_visibility` (`private`/`project`/`namespace`).

When adding new tools, document whether they are scope-aware or global. Do not widen scope silently on read paths.

### Simple agent surface

For newcomers, prefer the plain-language aliases over raw store/recall parameters:

| Surface | Remember | Ask |
| --- | --- | --- |
| MCP / OpenAI tools | `memory_remember` (`kind`: note, fact, fix, preference) | `memory_ask` (compact recall envelope) |
| REST | `POST /memory/remember` | `POST /memory/ask` |
| Browser UI | `POST /ui/api/remember` | `POST /ui/api/ask` |

`memory_remember` maps `kind` → episode `content_type` (`fix` → `solution`, `note` → `exchange`). `memory_ask`
delegates to `memory_recall` and returns a trimmed preview-oriented payload. Agent hooks that need
`memory_recall` on the first turn keep working; `memory_ask` is the ergonomic follow-up.

#### MCP simple profile

Set `CONSOLIDATION_MEMORY_MCP_TOOL_PROFILE=simple` in the MCP server `env` to expose only
`memory_recall`, `memory_remember`, and `memory_ask`. Default is `full` (all tools). `init` prints
both configs; `cli._recommended_mcp_simple_server_config()` is the canonical JSON snippet.

## Episode `content_type` vs record `type`

Episodes accept ingest types (`types.ContentType`): `exchange`, `fact`, `solution`, `preference`, `procedure`.

Consolidation emits records of type `fact`, `solution`, `preference`, `procedure` or `strategy` (`types.RecordType`). Only `strategy` has no ingest type: store it as structured JSON (`{"type": "strategy", ...}`) with any ingest `content_type` — see [docs/FAST_PATH_EPISODES.md](docs/FAST_PATH_EPISODES.md).

## MCP host configuration (interactive agents)

Prefer the canonical snippet from `consolidation-memory init` / `setup_service.recommended_mcp_server_config()`:

- **`command`**: absolute path to the Python that has `consolidation-memory` installed (`sys.executable`), not a bare name that can drift on Windows PATH.
- **`CONSOLIDATION_MEMORY_STATUS_LIGHTWEIGHT=1`**: status skips markdown consistency scans (and SciPy-heavy imports).
- **`CONSOLIDATION_MEMORY_MCP_AUTO_CONSOLIDATE=0`**: do not consolidate on the interactive path.
- **`CONSOLIDATION_MEMORY_PRELOAD_SCIPY_ON_START=1`**: load SciPy on the MCP main thread so consolidate does not hang on Windows worker-thread native imports.
- **`CONSOLIDATION_MEMORY_DEFERRED_KNOWLEDGE_RETRY_SECONDS`**: seconds to poll for a warm record-embedding cache after a deferred-knowledge recall. Default in library is `3`; **recommended MCP env is `0`** so the first `memory_recall` returns episodes immediately with a warning — call again shortly for full knowledge.
- **Tool budgets**: `CONSOLIDATION_MEMORY_TOOL_TIMEOUT_SECONDS` (default 60), plus per-tool `CONSOLIDATION_MEMORY_TIMEOUT_<TOOL>` (e.g. `MEMORY_STATUS`, `MEMORY_CONSOLIDATE`). Recall uses `CONSOLIDATION_MEMORY_RECALL_TIMEOUT_SECONDS` — including `memory_ask`, which fixes that budget and ignores the per-tool override.

Agent gate smoke (stdio initialize → status → recall under budgets). This is a
**CI gate**, not just a convenience — `test.yml` runs it on 3.13/ubuntu:

```bash
python scripts/smoke_mcp_stdio.py
```

Full MCP profile is **29 tools** (28 published output contracts —
`memory_store` and `memory_remember` share `StoreOutput`); simple profile is
`memory_recall`, `memory_remember`, `memory_ask`.

## Tool argument contract

Every published tool `inputSchema` declares `additionalProperties: false`, and
that is enforced on **all** surfaces from one allowed-argument set derived from
`schemas.openai_tools`: `tool_dispatch.reject_unknown_arguments` on the dispatch
seam, pydantic `extra="forbid"` on the MCP argument model, and
`rest.StrictRequestModel` (`extra="forbid"` → HTTP 422) on every top-level REST
body. Nested `EpisodeInput` / `OutcomeAnchorInput` stay permissive because the
published schemas type them as plain objects. When you add or rename a tool
argument, change the published schema — do not fork a second list.

## Local validation

```bash
python scripts/pre_push_check.py
python scripts/smoke_mcp_stdio.py
pytest tests/ -q
ruff check src tests/ scripts/
mypy src/consolidation_memory/
bandit -q -ll -r src scripts -s B608,B110
```

`ruff` is pinned to `>=0.7.0,<0.17` and `[tool.ruff.lint] select` lists the
enforced rules in full, with no `ignore` and no `per-file-ignores`: a new rule
becomes active only by an explicit edit, and every suppression is a `# noqa`
carrying its own reason. BLE001 is deliberately not selected. CI lints
`src/ tests/`; the command above and the pre-push hook also cover `scripts/`.

`pre_push_check.py` lints `scripts/` too and runs the stdio smoke only with
`--mcp-smoke`:

```bash
python scripts/pre_push_check.py --mcp-smoke
```

### Pre-push hook (recommended)

Install once to catch CI-style failures (optional-import test collection, lint,
bandit) before `git push`:

```bash
python scripts/install_git_hooks.py
```

Skip for a single push with `git push --no-verify`.

For consolidation or claim changes, also run:

```bash
python -m pytest -q tests/test_fast_path_consolidation.py tests/test_claim_emission.py
```

## Pull requests

1. Create a focused branch from `main`.
2. Keep changes scoped and include tests for behavior changes.
3. Update user-facing docs when behavior or setup changes.
4. Open a PR with problem statement, summary, test evidence, and risk notes for trust or scope changes.

## Commit style

Use clear, imperative commit messages. Prefer small, reviewable commits.

## Versioning

The project stays on `0.x` for the foreseeable future. `0.x` is a public beta:
breaking the tool contract is allowed, and the `mcp`, REST and OpenAI surfaces
are re-published together in the same release. A `1.x` line is not planned.

That has one concrete consequence for commit subjects. Conventional Commits
reserves `!` and a `BREAKING CHANGE` body for a major bump, and
`scripts/release_criteria.py` implements that rule literally, so either marker in
a commit in the release range turns the next release into `1.0.0`. **Do not use
`!` or `BREAKING CHANGE` here**, even for a genuine contract break. Write an
ordinary `feat:` or `fix:` subject and let the minor bump carry it, as
[SECURITY.md](SECURITY.md) describes for the supported line.

Nothing enforces that automatically. Merge to `main` goes through review, and the
reviewer is what stops an outside contributor who marked a break by the
conventional-commits rules. If a stray `!` reaches `main`, override the bump with
`workflow_dispatch` ([Manual Override](docs/RELEASE_AUTOMATION.md#manual-override)).

`CHANGELOG.md` has no `### Breaking Changes` section. The `## Unreleased`
section is regenerated from commit headers on every push to `main`, so hand-written
prose inside it is discarded. Upgrade-critical notes go in the commit body and, for
a release, in the PR description.

## Reporting bugs and features

- [GitHub Issues](https://github.com/charliee1w/consolidation-memory/issues)
- [GitHub Discussions](https://github.com/charliee1w/consolidation-memory/discussions)
- Security: [SECURITY.md](SECURITY.md)

## Code of conduct

By participating, you agree to follow [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).