# Fresh-Chat Prompt Pack

Use these prompts when starting a new implementation or review session.

## Session Bootstrap

```text
You are working in the consolidation-memory repository.
Before changes:
1) run memory_recall for the user goal,
2) identify current API/storage contracts touched by this task,
3) list verification commands you will run before finishing.
```

## Tool-Surface Consistency Prompt

```text
Implement the change in canonical service/client logic first, then propagate to all adapter surfaces (MCP, REST, OpenAI schemas/dispatch, CLI if applicable).
Do not ship partial surface support.
```

## Trust-Invariant Prompt

```text
Treat these as non-negotiable invariants:
- temporal correctness for as_of queries,
- provenance traceability for claims,
- contradiction visibility,
- drift challenge auditability,
- scope isolation correctness.
If any change weakens an invariant, stop and redesign.
```

## Docs-Accuracy Prompt

```text
Update documentation in the same PR for any changed behavior.
Avoid fragile counts (for example fixed test totals).
Prefer commands and source-of-truth references that remain accurate over time.
```

## Release-Readiness Prompt

```text
When behavior affects novelty or release gates:
1) run novelty_eval,
2) run verify_release_gates,
3) include artifact paths and pass/fail summary in the handoff.
```

## Final Handoff Prompt

```text
Return:
- changes made,
- verification commands and outcomes,
- residual risks,
- exact follow-up actions if anything is incomplete.
```
