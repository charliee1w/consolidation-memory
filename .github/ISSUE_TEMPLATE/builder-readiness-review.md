---
name: Builder Readiness Review
about: Report onboarding and extension-building findings from an external review pass.
title: "[Builder Review] "
labels: ["builder-readiness", "triage"]
---

## Reviewer Environment

- OS:
- Python version:
- Install command used:

## Onboarding Results

- `python scripts/smoke_builder_base.py`: pass/fail
- `python -m pytest tests/ -q -W error::ResourceWarning`: pass/fail
- Notes:

## Extension Exercise

- Extension attempted:
- Did it work from docs only? yes/no
- If no, where did it fail:

## Finding

- Severity: `P0` / `P1` / `P2` / `P3`
- Confidence: `high` / `medium` / `low`
- Expected behavior:
- Actual behavior:

## Reproduction

```bash
# copy-paste exact commands
```

## Proposed Fix Direction

- 
