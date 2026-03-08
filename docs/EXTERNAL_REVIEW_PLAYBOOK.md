# External Review Playbook

Use this playbook when requesting an external audit or independent reviewer pass.

## Reviewer Setup

```bash
git clone https://github.com/charliee1w/consolidation-memory
cd consolidation-memory
pip install -e ".[all,dev]"
python scripts/smoke_builder_base.py
```

## Reviewer Focus Areas

1. Trust invariants:
- temporal `as_of` correctness
- provenance completeness
- contradiction lifecycle correctness
- drift challenge behavior

2. Cross-surface consistency:
- Python, MCP, REST, OpenAI schemas/dispatch produce equivalent semantics

3. Data integrity:
- schema migration safety
- export/import behavior
- vector/store consistency and compaction behavior

4. Operational robustness:
- failure modes
- timeout handling
- error contracts

## Expected Review Output

- Severity-ranked findings with file/line references.
- Reproduction steps for each finding.
- Suggested fixes or risk mitigations.
- Explicit list of untested areas.

## Exit Criteria

- No untriaged critical/high findings.
- Agreed plan for medium findings.
- Updated tests/docs for accepted behavior changes.
