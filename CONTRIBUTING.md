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
- Audit reads by default: `memory_contradictions`, `memory_decay_report`, `memory_status`, `memory_consolidation_log` use resolved default scope (same as recall/browse). Pass an explicit `scope` to narrow further.

**Global by design** (intentionally corpus- or repo-wide):

- `memory_consolidate` / `consolidate()` — processes unconsolidated episodes across the DB
- `memory_compact` / FAISS compaction — rebuilds the shared vector index
- `memory_detect_drift` — git diff against a base ref (namespace/project scope only narrows challenged-claim attribution)
- `memory_policy_list` / `memory_policy_grant` — persisted ACL administration across the DB (CLI: `consolidation-memory policy list|grant`)
- Audit reads with `global_scope=true` — corpus-wide ops dashboard view; `memory_status` caches per scope key (including global)

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

## Episode `content_type` vs record `type`

Episodes accept ingest types: `exchange`, `fact`, `solution`, `preference`, `procedure`.

Consolidation may emit knowledge records with additional types (`procedure`, `strategy`). Store `strategy` episodes as structured JSON (`{"type": "strategy", ...}`) with any ingest `content_type` — see [docs/FAST_PATH_EPISODES.md](docs/FAST_PATH_EPISODES.md).

## Local validation

```bash
pytest tests/ -q
ruff check src tests
mypy src/consolidation_memory/
bandit -q -r src scripts -s B608,B110
```

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

## Reporting bugs and features

- [GitHub Issues](https://github.com/charliee1w/consolidation-memory/issues)
- [GitHub Discussions](https://github.com/charliee1w/consolidation-memory/discussions)
- Security: [SECURITY.md](SECURITY.md)

## Code of conduct

By participating, you agree to follow [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).