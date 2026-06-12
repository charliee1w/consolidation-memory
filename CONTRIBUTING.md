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