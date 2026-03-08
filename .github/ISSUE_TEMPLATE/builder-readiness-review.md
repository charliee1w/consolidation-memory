---
name: Builder readiness review
about: Request an external builder-style readiness review before merge or release
title: "[Builder Review] <scope>"
labels: ["review", "readiness"]
assignees: []
---

## Scope

What behavior or surface is being reviewed?

## Risk Areas

Which areas need close attention?

- [ ] Temporal semantics (`as_of`)
- [ ] Claim/provenance integrity
- [ ] Drift challenge behavior
- [ ] Adapter surface consistency (Python/MCP/REST/OpenAI)
- [ ] Schema/migration/export-import integrity

## Required Evidence

- Commit/PR:
- Local verification commands run:
- CI run links:
- Relevant novelty/release-gate artifacts (if applicable):

## Reviewer Deliverables

- Severity-ranked findings with file/line references
- Reproduction notes
- Gaps not reviewed
