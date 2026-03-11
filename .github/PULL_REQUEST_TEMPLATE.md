## Summary

- problem being solved:
- approach taken:

## Validation

Commands run locally:

```bash
pytest -q
ruff check src tests
mypy src
bandit -q -r src scripts
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

## Checklist

- [ ] Tests were added or updated for behavior changes
- [ ] Backward compatibility was considered
- [ ] Security/privacy impact was reviewed
- [ ] I included enough context for a reviewer to reproduce the change
