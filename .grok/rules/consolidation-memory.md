# consolidation-memory (Grok project rules)

## Memory first

See [00-memory-first.md](00-memory-first.md). `memory_recall` is the mandatory first
tool on every user turn when `consolidation_memory` MCP is available.

Read [CONTRIBUTING.md](../../CONTRIBUTING.md) and [docs/ARCHITECTURE.md](../../docs/ARCHITECTURE.md) before code changes.

## Fresh working copy

After `memory_recall`, run `python scripts/sync_working_copy.py` as the first shell step
when the tree is clean. It fetches and rebases onto `origin/<branch>`. If the tree is
dirty, report it and ask whether to commit, stash (`--stash`), or continue on the
current base. Re-run sync before pushing or after long idle gaps.

## Workflow

- Preserve trust invariants (temporal correctness, provenance, contradictions, drift auditability, scope isolation, surface parity).
- One focused slice per session.
- Update user-facing docs when behavior changes.
- Read the **Debt ledger** in [CLAUDE.md](../../CLAUDE.md) before large refactors.

## Where the contracts live

[CONTRIBUTING.md](../../CONTRIBUTING.md) owns these; read it before changing a tool:

- the trust invariants and the surface-parity requirement for a new tool
- which tools are scope-aware and which are global by design
- the one allowed-argument set behind unknown-argument rejection on MCP, REST and
  OpenAI dispatch
- the `memory_hygiene_apply` status values and key set

## Checks before done

```bash
python scripts/pre_push_check.py            # collection gate, ruff (src/ tests/ scripts/), bandit
pytest tests/ -q
mypy src/consolidation_memory/
python scripts/smoke_mcp_stdio.py           # 8-check MCP wire smoke
```

## Mantra

Deterministic belief maintenance first; LLMs only for unstructured residue; tests prove trust.