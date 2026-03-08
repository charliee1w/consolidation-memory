# Final Audit Prompt Template

Use this prompt when requesting a final independent audit pass on this repository.

## Prompt

```text
Audit this consolidation-memory branch for longevity, correctness, and trust-semantics regressions.

Required focus:
1) temporal correctness (`as_of`) for records and claims,
2) provenance and claim event integrity,
3) contradiction handling behavior,
4) drift challenge behavior,
5) cross-surface consistency (Python/MCP/REST/OpenAI schemas + dispatch),
6) migration/export/import safety,
7) documentation accuracy against code.

Output format:
- Findings first, ordered by severity.
- Include file paths and exact line references.
- For each finding: impact, reproduction, and concrete fix recommendation.
- Explicitly list residual testing gaps.
```

## Minimum Verification Commands For Auditor

```bash
python -m consolidation_memory --help
python -m consolidation_memory serve --help
pytest tests/ -q
ruff check src/ tests/
mypy src/consolidation_memory/
python -m benchmarks.novelty_eval --mode quick --output benchmarks/results/novelty_eval_audit_quick.json
```

## Optional Release-Grade Validation

```bash
python -m benchmarks.novelty_eval --mode full --output benchmarks/results/novelty_eval_audit_full.json
python scripts/verify_release_gates.py \
  --novelty-result benchmarks/results/novelty_eval_audit_full.json \
  --scope-use-case "Drift-aware debugging memory" \
  --output benchmarks/results/release_gate_report_audit.json
```
