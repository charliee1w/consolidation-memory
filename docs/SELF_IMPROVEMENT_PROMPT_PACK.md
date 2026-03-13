# Self-Improvement Prompt Pack

Use these prompts one by one in fresh Codex threads to evolve `consolidation-memory` into a trust-calibrated self-improving memory system for coding agents.

## How To Use This Pack

- Run these prompts in order.
- Use a fresh thread for each prompt.
- Do not batch multiple prompts into one request.
- If a prompt uncovers a prerequisite bug, fix that before moving on.
- Treat correctness, provenance, temporal semantics, and rollback safety as non-negotiable.

## Global Prefix

Paste this before any individual prompt if you want maximum consistency:

```text
You are working in the consolidation-memory repository.

Product goal:
Make this system the trust layer for self-improving coding-agent memory.
It should help agents preserve what worked, understand why it worked, detect when it is stale, and safely reuse only well-supported lessons.

Non-negotiable invariants:
- no SQLite/FAISS or cross-store divergence
- no claim/record/topic lifecycle drift
- provenance must stay inspectable
- temporal semantics must remain correct
- drift/challenge/expiry transitions must stay auditable
- scope isolation must not regress

Required workflow:
1. Start with memory_recall for the task goal.
2. Inspect the existing implementation before changing code.
3. Implement the change fully, not partially.
4. Add focused regression tests.
5. Run `python -m ruff check src tests`.
6. Run only the targeted pytest files needed for the change unless explicitly asked to run the full suite.
7. Store a concise memory summary of what changed.
8. Before final response, run memory_recall once more for consistency.

Return:
- root cause
- exact code changes
- verification commands and outcomes
- residual risks
```

## Prompt 1: Outcome Tracking

```text
Implement first-class outcome tracking for agent actions in consolidation-memory.

Goal:
The system currently stores observations, claims, records, and lifecycle events. It now needs to track whether a proposed action or strategy actually worked. Without outcome tracking, the system cannot become self-improving; it can only become self-accumulating.

Build this as a production-grade feature, not a demo.

Requirements:
- Add a durable model for action outcomes such as success, failure, partial_success, reverted, superseded.
- Support linking an outcome to:
  - one or more source claims
  - one or more source records
  - one or more episodes
  - optional code anchors or issue/PR identifiers
- Preserve timestamps and provenance for each outcome.
- Ensure outcome writes are transactional and auditable.
- Expose the feature through canonical client/service logic first, then propagate to MCP/REST/OpenAI tool surfaces if applicable.
- Do not break export/import or scoped behavior.

High-value design constraints:
- outcomes should be reusable evidence for future ranking and trust scoring
- outcomes should support repeated observations over time, not just a single final status
- the schema should make it possible to later answer:
  - which strategies worked repeatedly?
  - which fixes failed?
  - which lessons were later reverted?

Tests required:
- happy path outcome creation
- multiple outcomes linked to the same claim over time
- rollback behavior on write failure
- export/import round-trip if the feature touches snapshot schema
- scope enforcement and query visibility if surfaced through public APIs

Verification:
- run `python -m ruff check src tests`
- run only targeted pytest files

At the end, summarize how this changes the system from “memory of events” toward “memory of validated outcomes”.
```

## Prompt 2: Strategy Memory

```text
Build first-class strategy memory for consolidation-memory.

Goal:
The system should not only remember facts about the world or repo. It should remember reusable ways of solving problems, such as debugging approaches, patching tactics, testing sequences, and investigation workflows.

Requirements:
- Add a durable representation for a strategy/approach/heuristic distinct from plain fact claims.
- Support fields like:
  - problem pattern
  - strategy text
  - preconditions
  - expected signals
  - failure modes
  - evidence/outcomes supporting it
- The design must support multiple validations and contradictions over time.
- Strategies must be queryable and rankable separately from generic facts.
- Maintain provenance, temporal semantics, and scope behavior.
- Keep surface/API behavior clean and coherent across canonical service layers.

Important:
- do not just add another record type without thinking through retrieval and trust semantics
- make sure strategies can later be ranked by repeated success and penalized by failure/revert evidence

Tests required:
- create and retrieve strategy records
- repeated supporting outcomes improve evidence density
- contradicted or failed strategies remain inspectable but are no longer treated as equally reusable
- scoped filtering and export behavior

Verification:
- run `python -m ruff check src tests`
- run targeted pytest only

At the end, explain how strategy memory differs from fact memory in this system.
```

## Prompt 3: Reliability Scoring

```text
Implement reliability scoring for reusable claims and strategies in consolidation-memory.

Goal:
The system needs an evidence-based score that answers “how safe is this memory item to reuse?” Relevance alone is not enough.

Requirements:
- Add a computed reliability/trust score for claims and strategy-like memories.
- The score should consider signals such as:
  - supporting outcome count
  - success/failure ratio
  - recency
  - provenance richness
  - code-anchor support
  - contradiction history
  - drift/challenge state
  - reversion/supersession
- Keep the model explainable. The system should be able to expose why a score is high or low.
- Do not hide challenged/expired items; downgrade them appropriately.
- Avoid fake precision. The score can be bounded and coarse if that is more honest.

Implementation expectations:
- prefer a deterministic scoring model first
- make score inputs inspectable for debugging and future tuning
- do not introduce ranking behavior that silently ignores temporal validity

Tests required:
- high-support memories score above low-support ones
- failures and reversions lower trust
- challenged/drifted items are penalized
- unsupported items are not ranked as strongly reusable
- explanation payloads stay stable and testable

Verification:
- run `python -m ruff check src tests`
- run targeted pytest only

At the end, summarize the scoring model and why it is appropriate for a trust-calibrated system.
```

## Prompt 4: Retrieval That Prefers Durable Lessons

```text
Upgrade retrieval/ranking in consolidation-memory so the system prefers durable lessons over merely similar text.

Goal:
When an agent asks for help, the system should prioritize validated claims and strategies with strong support over semantically similar chatter.

Requirements:
- Update canonical retrieval/ranking logic to combine:
  - semantic similarity
  - relevance
  - reliability score
  - temporal validity
  - drift/challenge penalties
  - outcome support
- Keep query-time behavior explainable.
- Preserve existing scope and temporal semantics.
- Avoid pathological query cost growth.
- Keep adapters aligned with canonical service behavior.

Do not:
- bolt this into one adapter only
- bypass existing trust invariants
- bury unsupported stale items without leaving them inspectable

Tests required:
- validated strategies outrank weaker but semantically similar text
- stale/challenged items are demoted
- active, supported claims beat unsupported matches
- scoped/temporal filtering still behaves correctly

Verification:
- run `python -m ruff check src tests`
- run targeted pytest only

At the end, explain how ranking changed and why this is a major step toward self-improvement instead of generic RAG.
```

## Prompt 5: Learning Loop From Work Sessions

```text
Implement a post-task learning loop for consolidation-memory.

Goal:
After a coding session, the system should be able to convert raw task history into reusable lessons with outcomes, not just store another transcript.

Requirements:
- Add a canonical flow that can ingest a completed work session and extract:
  - durable claims
  - strategies used
  - outcomes
  - contradictions or reversions
- Make the flow idempotent or safely repeatable.
- Preserve provenance back to the original episodes/records.
- Record when extracted lessons are uncertain or weakly supported.
- Ensure failures in this flow do not corrupt existing data.

High-value behavior:
- prefer extracting reusable engineering lessons over verbose summaries
- distinguish “what happened” from “what should be reused later”

Tests required:
- end-to-end ingestion from a small synthetic session
- duplicate/safe re-run behavior
- failure rollback behavior
- correct provenance links between raw episodes and derived lessons

Verification:
- run `python -m ruff check src tests`
- run targeted pytest only

At the end, summarize the learning loop and identify remaining gaps before it can support real self-improvement.
```

## Prompt 6: Decay, Supersession, And Lesson Retirement

```text
Strengthen lesson decay and retirement semantics in consolidation-memory.

Goal:
A self-improving system must know when to stop trusting old lessons. Build rigorous retirement/supersession logic for claims and strategies.

Requirements:
- Extend lifecycle semantics so lessons can be:
  - active
  - challenged
  - expired
  - superseded
  - reverted
- Add clear transitions driven by:
  - new contradictory evidence
  - failed outcomes
  - code drift
  - replacement strategies
- Ensure search/browse/recall reflect these states without destroying auditability.
- Keep historical queries (`as_of`) correct.

Important:
- do not solve this by deleting history
- preserve why an item was downgraded or retired
- make the transition path testable and inspectable

Tests required:
- successively superseded lessons
- failed strategy leading to downgrade
- as_of reconstruction across multiple lifecycle transitions
- retrieval excluding or demoting retired items appropriately

Verification:
- run `python -m ruff check src tests`
- run targeted pytest only

At the end, explain how belief retirement now works and how it reduces recursive error accumulation.
```

## Prompt 7: Self-Improvement Evaluation Harness

```text
Build an evaluation harness that measures whether consolidation-memory actually improves coding-agent performance over time.

Goal:
Do not rely on intuition. Add a benchmark/evaluation loop that can prove whether memory with outcomes and strategies makes an agent better.

Requirements:
- Define measurable metrics such as:
  - repeated-failure reduction
  - stale-advice reduction
  - successful strategy reuse rate
  - unsupported-memory reuse rate
  - contradiction-aware retrieval quality
- Add a reproducible evaluation harness using synthetic or replayable scenarios.
- Make the harness cheap enough to run regularly in development.
- Avoid vanity metrics that only measure retrieval volume.

High-value benchmark tasks:
- repeating a known bug class
- selecting among prior successful and failed strategies
- avoiding stale advice after code drift
- reusing validated lessons across multiple sessions

Tests required:
- metric computation correctness
- stable fixture-driven benchmark scenarios
- explicit pass/fail or comparison outputs

Verification:
- run `python -m ruff check src tests`
- run only the relevant targeted checks for the new evaluation code

At the end, summarize what would count as real evidence that the system is becoming self-improving.
```

## Prompt 8: Belief Repair And Audit Tooling

```text
Implement repair and audit tooling for the self-improvement layer in consolidation-memory.

Goal:
As the system becomes more complex, it needs tools to inspect, audit, and repair belief-state corruption or inconsistency.

Requirements:
- Add tooling to detect:
  - claims without valid sources
  - strategies with contradictory outcome histories
  - outcome records linked to expired/nonexistent entities
  - stale reliability scores or lifecycle mismatches
- Add safe repair actions where appropriate.
- Produce operator-facing summaries that help debug trust failures quickly.
- Keep these tools deterministic and non-destructive by default.

Tests required:
- detection of each inconsistency class
- safe repair behavior
- no false deletion of still-valid data

Verification:
- run `python -m ruff check src tests`
- run targeted pytest only

At the end, explain how this tooling protects the system from silently compounding bad memory state.
```

## Prompt 9: Agent-Facing Trust UX

```text
Improve the agent-facing trust UX of consolidation-memory.

Goal:
An agent should not just receive a memory item. It should receive enough trust metadata to decide whether to use it.

Requirements:
- Improve response surfaces so retrieved lessons can expose:
  - reliability score
  - support/outcome summary
  - drift/challenge status
  - provenance summary
  - lifecycle state
  - timestamps
- Keep responses concise but inspectable.
- Do this through canonical response shaping first, then align adapters.
- Avoid cluttering every response with noise; make the trust layer useful.

Tests required:
- serialization/response contract coverage
- stable trust metadata presence on relevant surfaces
- no regression in scope/temporal filtering

Verification:
- run `python -m ruff check src tests`
- run targeted pytest only

At the end, summarize how an agent would now distinguish safe reusable memory from weak memory.
```

## Prompt 10: Product Hardening Pass

```text
Do a focused hardening pass on the self-improvement architecture now present in consolidation-memory.

Goal:
After outcome tracking, strategy memory, reliability scoring, retrieval updates, learning loops, and repair tooling are in place, perform a rigorous end-to-end review and close the most dangerous gaps.

Requirements:
- Review the full flow:
  - raw episode
  - derived claim/strategy
  - outcome capture
  - trust scoring
  - retrieval
  - decay/supersession
  - export/import
  - audit/repair
- Identify correctness gaps, rollback holes, lifecycle mismatches, and missing tests.
- Implement the highest-severity fixes you confirm.
- Add regression coverage for every bug fixed.
- Update docs to reflect the actual design, not the aspirational design.

Verification:
- run `python -m ruff check src tests`
- run targeted pytest for the touched files
- do not run the full suite unless explicitly asked

At the end, give a blunt assessment:
- what is now strong
- what is still weak
- what must be true before this can honestly be called a self-improving system
```

## Prompt 11: Narrow Productization

```text
Refine consolidation-memory around one product promise:
"This system makes coding agents better over time by remembering what worked, proving why it worked, and knowing when that knowledge is no longer safe."

Goal:
Turn the architecture into a clear product shape for coding agents.

Requirements:
- Audit naming, docs, status surfaces, and user-facing descriptions.
- Remove or de-emphasize broad generic-memory framing where it weakens the product story.
- Make the coding-agent trust layer the clear center of gravity.
- Keep the work honest: do not claim capabilities the implementation does not yet support.
- Update docs and surface text to reflect the actual system.

Deliverables:
- doc updates
- API/status wording updates if needed
- concise product rubric for future development

Verification:
- run `python -m ruff check src tests` if any code changes
- run targeted pytest if behavior changes

At the end, provide the updated product statement and the strict non-goals.
```

## Prompt 12: Final Readiness Review

```text
Do a final readiness review of consolidation-memory as a self-improving coding-agent memory system.

Goal:
Evaluate the current codebase against the intended product: trust-calibrated, outcome-aware, strategy-aware, drift-aware memory that helps coding agents get better over time without hardening mistakes.

Review focus:
- correctness
- trust semantics
- lifecycle integrity
- retrieval usefulness
- auditability
- outcome support
- strategy reuse
- resistance to stale or bad lesson reuse

Output format:
- findings first, ordered by severity
- exact file references
- residual testing gaps
- short change summary only after findings

If no major findings remain, explicitly say so and state what evidence supports that conclusion.
```
