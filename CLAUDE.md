# consolidation-memory contributor context

Compact maintainer map, aligned with the codebase.

## Repository shape

```text
src/consolidation_memory/
  client.py            orchestration + tool-facing operations
  runtime.py           shared blocking-execution runtime for the client
  server.py            MCP surface: tool registration + startup contract self-checks
  rest.py              REST surface (StrictRequestModel bodies) + ops_routes
  tool_adapter.py      recall parity seam: deferred knowledge, deadlines, recall budgets
  tool_dispatch.py     canonical dispatch seam; one allowed-argument set
  tool_contracts.py    typed MCP output contracts (published outputSchema)
  schemas.py           OpenAI tools + MCP dispatch
  mcp_compat.py        ONLY file touching private `mcp` SDK surfaces
  simple_api.py        memory_remember / memory_ask re-translation, shared by 3 surfaces
  policy_engine.py     write_mode / read_visibility evaluation
  corpus_hygiene.py    noisy-episode scan + orphan-claim repair
  query_service.py     canonical query envelopes
  context_assembler.py hybrid recall
  claim_graph.py       deterministic claim canonicalization
  drift.py             git-based drift challenge flow (+ drift_subprocess/worker)
  database.py          backward-compatible facade (re-exports db/)
  db/                  SQLite schema/migrations + domain CRUD
  db/scope.py          scope resolution, exact-match filters, scope discovery
  consolidation/       engine, fast_path, prompting, clustering, scoring, utility_scheduler
  types.py             enums and result types shared across surfaces
  setup_service.py     recommended MCP server config shared by CLI and web UI
```

## Important facts

- Package version: `pyproject.toml`; schema version: `db/migrations.py`
  (`CURRENT_SCHEMA_VERSION`).
- Tool surface: `schemas.openai_tools` (module-level list) and `schemas.dispatch_tool_call`. **29 tools, 104 published parameters, 28 output contracts** — `memory_store` and `memory_remember` share `StoreOutput`, `memory_store_batch` has its own `BatchStoreOutput`. A handler's return annotation in `tool_contracts.py` is the published `outputSchema`, validated at runtime with `extra="allow"` so unknown payload keys survive.
- Tool descriptions have **one** source: `schemas.openai_tools`. `_tracked_tool` passes the
  published text to `mcp.tool(description=...)`, and `_verify_tool_descriptions()` fails at
  import if a handler docstring diverges — MCP publishing a shorter summary is a silent
  contract loss, since the description is what a host reads to choose a tool.
- MCP SDK: `mcp[cli]>=2.3.0,<3` (upstream renamed `FastMCP` to `MCPServer` in 2.x).
  Every private SDK touch is isolated in `mcp_compat.py` — `ArgModelBase`,
  `MCPServer._tool_manager`, `ToolManager._tools`, `fn_metadata.output_schema` and the
  `Tool.__dict__["output_schema"]` eviction. **An `mcp` bump breaks there first**; a
  missing module or attribute raises `MCPCompatError` naming the installed version,
  the range and what was probed. No protocol version is pinned: the host proposes, the
  SDK negotiates, the handshake set is `2024-11-05`…`2025-11-25` (`2026-07-28` is
  modern-envelope only, so requesting it downgrades to `2025-11-25`).
- Output schema publication: `anyOf[success, error]` under a root `type` (pre-2026
  protocol validators require it), and self-healing —
  `mcp_compat.install_list_tools_heal` wraps the public `MCPServer.list_tools`
  (`tools/list` funnel) and evicts the SDK's cached `Tool.output_schema`, so a
  late-registered tool heals on the next listing. `server` runs
  `_verify_published_output_schemas()` and `_verify_published_argument_contract()` at
  import and in `lifespan`; both fail loudly rather than degrade.
- MCP error model: tool execution failures return `isError: true` via
  `_tool_error_result` (actionable text plus a `{"error": ...}` structured payload).
- Input contract: unknown arguments are rejected on all three surfaces from **one**
  allowed-argument set — `tool_dispatch.accepted_argument_names`, derived from the
  published `inputSchema`, enforced by `reject_unknown_arguments` first in
  `execute_tool_call`, by pydantic `extra="forbid"` on the MCP argument model, and by
  `rest.StrictRequestModel` (`extra="forbid"` → HTTP 422) on every top-level body.
  Nested values inside `episodes` / `code_anchors` and REST query params are outside
  the guard; `rest.EpisodeInput` / `rest.OutcomeAnchorInput` stay permissive so REST
  does not reject what the other surfaces accept.
- The contract is about argument **names**, not types or bounds: MCP validates
  against the schema-derived model but clamps what it accepts (`n_results: 0` → `1`,
  `"5"` → `5`), while dispatch and REST reject the same value.
- Transport state is a keyword, never an argument key. The recall deadline used to
  ride in as `_recall_deadline_monotonic`, which needed a global exemption in
  `reject_unknown_arguments` and so was accepted by every tool. It is now
  `recall_deadline_monotonic=` on `execute_tool_call` /
  `dispatch_tool_call`, and the exemption list is gone.
- Generated docs: `docs/TOOLS.md` comes from `scripts/generate_tool_reference.py`;
  `tests/test_tool_reference_sync.py` fails when it drifts.

## Local verification

```bash
pip install -e ".[all,dev]"
python scripts/smoke_builder_base.py        # builder smoke: ok
pytest tests/ -q                            # no failures
python scripts/smoke_mcp_stdio.py           # smoke_mcp_stdio PASS: 8/8 checks
ruff check src tests/ scripts/              # All checks passed!
mypy src/consolidation_memory/              # Success: no issues found
python scripts/generate_tool_reference.py --check   # docs/TOOLS.md is up to date
```

CI runs the same set, with two differences: `ruff` is `src/ tests/` only, and the
smoke, lint, type and security steps are 3.13/ubuntu while the matrix spans 3.10–3.13
× ubuntu/windows.

## Lint and typing facts

- `[tool.ruff.lint]` writes the enforced `select` out in full; there is no `ignore` and
  no `per-file-ignores`. Every suppression is a `# noqa` on the line it applies to,
  carrying its reason. Bandit is the security gate:
  `bandit -q -ll -r src scripts -s B608,B110`.
- `ruff>=0.16.10,<0.17` — the floor is the enforced `select` itself: it names rules
  that do not exist in older ruff (RUF068 is one), and ruff aborts on an unknown
  selector rather than skipping it, so a lower bound produced a config that could
  not be read. The upper bound is deliberate too: the rule set is explicit, but a
  bump can still add, rename or reclassify rules.
- BLE001 is **not** selected: `except Exception` in `tests/` is deliberate and not a
  lint failure. Do not claim BLE001 is enforced. TRY004 **is** enforced per site, not
  project-wide; REST maps `(TypeError, ValueError)` to 422, and the 400 handlers catch
  `RuntimeError`.
- `mcp` is not in `[tool.mypy.overrides] ignore_missing_imports`: the SDK ships
  `py.typed` and the type checker sees the real signatures. Only genuinely untyped
  distributions are listed.

## Guardrails

1. Keep semantics aligned across Python, MCP, REST, and OpenAI surfaces.
2. Preserve trust invariants (see [CONTRIBUTING.md](CONTRIBUTING.md)).
3. Update user-facing docs when behavior changes.
4. Do not hard-code test counts or dated status statements in docs; they rot silently.
5. Changed a tool signature or output contract? Run `scripts/generate_tool_reference.py`
   and commit `docs/TOOLS.md`.
6. Changed the allowed-argument set? One derivation (`schemas.openai_tools`) feeds the
   dispatch guard, the REST models and the SDK check — do not fork a second list.

## Debt ledger

Blind spots and blind-spot history, ranked P0 (trust/scope) → P2. Check before large
refactors; strike items off as they close.

**Open**

- **P1 Unvalidated contract subtrees**: 32 of 234 success-arm properties publish as
  `dict[str, Any]` / `list[dict[str, Any]]`, so their subtrees are validated as "is
  an object" and nothing more. `tests/test_output_contract_payload_typing.py` ratchets
  the number; do not raise it. The walk resolves `$ref` into the root `$defs`, so
  nested output models (`HealthOutput`, `ScopeUsageEntry`, `DriftClaimImpactOutput`)
  are counted too — 50 of the 234 live there.
- **P1 Changelog ranking rule**: `changelog_builder` ranks before capping
  (`DEFAULT_SUBJECT_LIMIT = 200`), so user-visible commits are never displaced by
  `docs`/`chore`. Two consequences to remember: `--limit` exists on
  `update_changelog.py` and `release.py`, and dependency-scope commits get their own
  `### Dependencies` section instead of `Internal`.
- **P2 `mcp` private-surface blast radius**: `mcp_compat.py` concentrates five private
  probes and the supported range is `<3`, so the next major is a known break. Watch
  dependabot for it; the import-time self-checks fail loudly, they do not degrade.

**Closed** (kept as a record of what these blind spots were)

- ~~`memory_scope_list` description~~: `schemas.openai_tools` is the single source for
  every tool description; `_tracked_tool` passes it to `mcp.tool(description=...)` and
  `server._verify_tool_descriptions()` fails at import if a docstring diverges. Both
  discovery tools state that they are not read-visibility filtered, on both surfaces.

- ~~Scope on audit APIs~~: `contradictions`, `decay_report`, `consolidation_log`, `status` take resolved default scope; explicit `scope` narrows further; `global_scope=True` for the corpus-wide view.
- ~~`content_type` validation~~: shared `validate_episode_content_type()` in `types.py`.
- ~~`trust_profile` in scoped `status()`~~: `get_claim_trust_stats`, `count_active_challenged_claims`, `get_recently_contradicted_topic_ids` accept `scope`.
- ~~Benchmark eval gates~~: `novelty_eval`, `coding_agent_eval` (`quick`) and `real_world_eval` (`ci`, seeded fixture) all gate in the `novelty_gates` job; the `full` runs are nightly and release-time.
- ~~`embedding_disk_cache` cross-process lock~~: `.embedding_cache_write.lock` via `process_write_lock.py`.
- ~~MCP trust boundary~~: a documented policy in `SECURITY.md`; the supported line moves with each minor.
- ~~`docs/MCP_GUIDE.md` drift~~: protocol ladder including the silent `2026-07-28` → `2025-11-25` downgrade, the 30s client-init budget, `STATUS_LIGHTWEIGHT`, `CONSOLIDATION_MEMORY_CONSOLIDATION_AUTO_RUN`, and the absent `namespace.display_name`.
- ~~Hygiene surface parity~~: `memory_hygiene_scan` / `memory_hygiene_apply` on MCP, REST, OpenAI dispatch, CLI, web UI, desktop.
- ~~`rest.py` E402~~: imports ordered above type aliases.
- ~~`tool_adapter` recall parity~~: shared deferred-knowledge and deadline semantics across MCP/REST/OpenAI.
- ~~`ContentType` vs `RecordType`~~: `procedure` is an ingest type, `strategy` is JSON-only at store time (see CONTRIBUTING + FAST_PATH_EPISODES).
- ~~`database.py` god-module~~: split into `db/` domain modules; `database.py` is a thin re-export facade.
- ~~Migrations v17–v20~~: additive tests in `tests/test_core.py`.
- ~~Multi-process lock~~: `tests/test_process_write_lock.py` for `ProcessWriteLease`.
- ~~Positioning vs RAG~~: `examples/trust-vs-rag/` with `demo_flow.py`; plugin guide in `docs/PLUGIN_DEVELOPMENT.md`; LoCoMo narrative in `docs/LOCOMO_BENCHMARK.md` (a full run needs an API key).
- ~~`memory_hygiene_apply` contract~~: `types.HygieneApplyResult` and `HygieneApplyOutput` mirror each other under an import-time guard, and both modes emit the same key set, so a contract cannot reject a payload after the episodes were deleted.
- ~~`memory_scope_list` merge + liveness~~: grouping on the 11 canonical scope keys, `deleted = 0` live-row counts, display metadata via `MAX()`, no `namespace.display_name`, no `ensure_schema()` on the read path.
- ~~Unknown-argument enforcement~~: one shared allowed-argument set feeds MCP, REST and dispatch, and startup fails if the SDK-enforced set diverges from it.
- ~~`mcp` private surfaces~~: isolated in `mcp_compat.py`, with self-healing `outputSchema` publication and a startup self-check.
- ~~Drift subprocess interpreter~~: not resolved through a venv symlink, with `PYTHONPATH` forwarded package root first, and **no `cwd`**: `python -m` puts the working directory at `sys.path[0]` ahead of `PYTHONPATH` and the stdlib, so running inside the analysed repository let its modules shadow the real package. The repository travels as `--repo-path` and the worker resolves it, so the child needs no working directory. Regression: `tests/test_drift_interpreter_resolution.py::test_analysed_repository_cannot_shadow_the_worker`.
- **P1 Analysed repository and `git` config are outside the documented boundary**: `memory_detect_drift` takes a caller-supplied `repo_path` and `drift.py` runs `git` with `cwd=repo_dir` and `-c safe.directory=…`. The repository's Python modules are no longer importable by the worker, but a `.git/config` in that repository can still influence the git invocation, and `SECURITY.md` does not say what the analysed repository is trusted to do. Decide the position, then write it down.
- ~~Lint/typing config~~: explicit `select`, no `ignore`/`per-file-ignores`, `scripts/` clean rather than excluded, `mcp` typed rather than silenced.

**Runtime residual (P0-1)**

- **Windows SciPy import hang on MCP workers**: mitigated by the lazy `consolidation`
  package export, lightweight `status` import isolation, a main-thread SciPy preload
  and bounded MCP tool timeouts. Residual: the first consolidate still needs SciPy —
  preload on the main thread or call `_ensure_scipy_for_consolidate()` before the
  worker. Regression: `tests/test_import_isolation.py`, `scripts/smoke_mcp_stdio.py`.

## Maintainer reminders (not debt)

- Full MCP profile ships **29 tools**; the simple profile exposes 3
  (`memory_recall`, `memory_remember`, `memory_ask`).
- `forget()` expires claims that lose all provenance (`expire_claims_without_sources`),
  so a forgotten episode leaves no claim behind. Consolidated knowledge can lag code —
  use `memory_correct`, or new episodes plus consolidate.
- `memory_hygiene_apply` is not gated by `write_mode`. Read its payload as
  `status: "applied"` (never `ok`) and expect `episode_ids` / `forgotten` /
  `not_found` in both modes.
- `memory_scope_list` counts live rows only (`deleted = 0` on `episodes` and
  `knowledge_records`; `knowledge_topics` has no such column) and is not read-filtered
  — see [SECURITY.md](SECURITY.md).
- Run `memory_hygiene_scan` on noisy corpora; verify with `ruff check src tests/ scripts/`,
  `mypy src/consolidation_memory/`, `pytest tests/ -q` and
  `python scripts/smoke_mcp_stdio.py`.

## Keep (do not rewrite)

- Episodes → records → claims → topics stack; the `tool_dispatch` seam; the FAISS write
  lease; fast-path before LLM; `query_service` envelopes.

## Core docs

- [README.md](README.md)
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [docs/MCP_GUIDE.md](docs/MCP_GUIDE.md) — wire contract, scopes, errors, environment, recipes
- [docs/TOOLS.md](docs/TOOLS.md) — generated input/output schemas for every tool
- [docs/ACL.md](docs/ACL.md) — policies, principals, multi-service scope pattern
- [docs/UI.md](docs/UI.md) — browser UI, TUI dashboard, desktop app
- [docs/FAST_PATH_EPISODES.md](docs/FAST_PATH_EPISODES.md)
- [CONTRIBUTING.md](CONTRIBUTING.md)
