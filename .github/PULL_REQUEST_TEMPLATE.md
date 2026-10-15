## Summary

- problem being solved:
- approach taken:

## Validation

Commands run locally:

```bash
python scripts/pre_push_check.py
python scripts/smoke_mcp_stdio.py
pytest tests/ -q
ruff check src tests/ scripts/
mypy src/consolidation_memory/
bandit -q -ll -r src scripts -s B608,B110
```

If not all commands were run, explain what was skipped and why.

## Docs And Examples

- [ ] I updated docs/examples for user-visible behavior changes
- [ ] I updated release-facing notes when needed
- [ ] No docs changes were needed

## Risk Review

- [ ] Temporal or trust semantics changed
- [ ] Scope/policy behavior changed
- [ ] Adapter parity changed (Python/MCP/REST/OpenAI)
- [ ] Storage/export/import behavior changed
- [ ] No special risk areas

## Project Rules

- [ ] Change preserves the trust invariants in `CONTRIBUTING.md` (temporal correctness, provenance traceability, contradiction visibility, drift challenge auditability, scope isolation, surface parity)
- [ ] New or changed tools keep Python / MCP / REST / OpenAI dispatch semantics aligned
- [ ] `CHANGELOG.md` and `docs/ROADMAP.md` updated when shipped behavior or a known blocker changed

## Checklist

- [ ] Tests were added or updated for behavior changes
- [ ] Backward compatibility was considered
- [ ] Security/privacy impact was reviewed
- [ ] I included enough context for a reviewer to reproduce the change
