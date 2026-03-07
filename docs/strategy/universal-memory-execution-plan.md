# Universal Memory Execution Plan

## Purpose

This plan converts the universal-memory transformation goal into an execution sequence that can be implemented in small, testable steps without weakening the repo's current trust features:

- temporal recall
- provenance and source traceability
- contradiction handling
- drift-aware invalidation
- local-first inspectability

It is intentionally grounded in the current repository state, not a greenfield redesign.

## Repo Audit Summary

### Current strengths to preserve

- The repo already exposes one memory engine through Python, MCP, REST, CLI, and OpenAI-style tool schemas.
- Trust-heavy behavior already exists in product code and tests:
  - temporal recall via `as_of`
  - claim provenance and lifecycle tracking
  - contradiction logging
  - anchor extraction
  - drift-triggered claim challenge
  - uncertainty signaling
- The current docs already include the strategic inputs needed for execution:
  - `docs/strategy/universal-memory-gap-analysis.md`
  - `docs/strategy/CODEX_UNIVERSAL_MEMORY_PROMPTS.txt`

### Current bottlenecks to resolve

- `project` is the only durable scope primitive; there is no first-class namespace, app, agent, session, or principal model.
- `MemoryClient` acts as both engine and service boundary, which will make future adapter work drift-prone.
- Existing external surfaces wrap the current client directly instead of depending on one canonical internal service/query layer.
- There is no native adapter framework yet for OpenAI Agents SDK, LangGraph-style stores, or other ecosystem-specific integrations.
- The trust wedge is strong, but the product surface still tells a narrower "coding-agent memory engine" story than the target universal shared-memory platform.

### Execution assumptions

- Backward compatibility for current single-project usage remains mandatory unless a later step explicitly documents a breaking change and migration.
- Shared memory must be explicit and opt-in. Isolation remains the default.
- Local-first/self-hosted must remain a supported deployment mode throughout the refactor.
- This baseline step is documentation-only. Product-code changes start in later milestones.
- The working tree already contains unrelated product-code edits, so future implementation steps must edit narrowly and avoid reverting pre-existing work.

## Milestone Types

- `Documentation-only`: strategy, object model, migration plans, packaging, positioning, and evidence docs.
- `Schema change`: persistent storage changes, migrations, backward-compatibility handling, and scope-model persistence.
- `Service-layer change`: internal semantic ownership for recall, provenance, contradictions, drift, and policy behavior.
- `Adapter/integration work`: ecosystem abstractions, transport rewiring, native integrations, demos, and end-to-end shared-memory flows.

## Milestone Sequence

| Milestone | Work Type | Outcome | Primary Surfaces | Depends On |
| --- | --- | --- | --- | --- |
| M1. Establish execution baseline | Documentation-only | One execution plan and one running log for the transformation | `docs/strategy/universal-memory-execution-plan.md`, `docs/strategy/execution-log.md` | None |
| M2. Define the canonical domain model | Documentation-only | Explicit shared-memory object model and backward-compatible migration strategy | `docs/strategy/memory-object-model.md`, `docs/strategy/schema-migration-plan.md` | M1 |
| M3. Introduce the canonical domain model skeleton | Service-layer change | Minimal code-level types and service seams for scopes, provenance, and lifecycle without changing single-project behavior | `src/consolidation_memory/types.py`, `src/consolidation_memory/client.py`, `src/consolidation_memory/schemas.py` | M2 |
| M4. Add persistent shared-scope support | Schema change, Service-layer change | Durable namespace/app/agent/session/project scope support with backward-compatible defaults | `src/consolidation_memory/database.py`, `src/consolidation_memory/config.py`, `src/consolidation_memory/client.py`, new migration logic, isolation tests | M3 |
| M5. Refactor to a canonical trust-preserving service layer | Service-layer change | One internal query/service layer that owns temporal, provenance, contradiction, drift, and scope semantics | new internal service module(s), `src/consolidation_memory/client.py`, `src/consolidation_memory/context_assembler.py`, `src/consolidation_memory/server.py`, `src/consolidation_memory/rest.py`, `src/consolidation_memory/schemas.py` | M4 |
| M6. Introduce the adapter framework | Service-layer change, Adapter/integration work | Stable abstraction for external ecosystems, with at least one current surface rewired through it | `src/consolidation_memory/adapters/`, affected surface modules, adapter docs/tests | M5 |
| M7. Ship first native universal integrations | Adapter/integration work | Native integrations that prove serious shared-memory interoperability instead of generic wrappers | initial adapters for MCP and OpenAI Agents SDK; LangGraph next | M6 |
| M8. Re-verify trust guarantees across shared scopes and adapters | Service-layer change, Adapter/integration work | Proof that the platform refactor preserved the trust wedge across scopes and surfaces | trust regression suites, parity tests, scope-sharing/isolation tests | M7 |
| M9. Close benchmark and release-gate evidence gaps | Documentation-only, Service-layer change | Evidence and release-gate enforcement that support the new product claims honestly | `docs/NOVELTY_*`, `docs/RELEASE_GATES.md`, `benchmarks/`, `scripts/verify_release_gates.py`, test coverage | M8 |
| M10. Package for private alpha | Documentation-only, Adapter/integration work | Design-partner-ready install path, onboarding docs, and flagship multi-surface demo | `docs/strategy/private-alpha-plan.md`, `docs/strategy/demo-scenarios.md`, `examples/` | M7, M8, M9 |
| M11. Rewrite public positioning around shipped capabilities | Documentation-only | README and outward-facing story aligned to real functionality and measured evidence | `README.md`, strategy/product docs, example references | M9, ideally M10 |

## Milestone Details And Validation Expectations

### M1. Establish execution baseline

- Classification: `Documentation-only`
- Goal: create the execution log and milestone plan so future prompts operate against one sequence instead of parallel strategy tracks.
- Validation expectation:
  - doc review for completeness and internal consistency
  - `git diff --check` on changed markdown files
  - heading/section presence check for required deliverables

### M2. Define the canonical domain model

- Classification: `Documentation-only`
- Goal: define first-class entities for namespace, principal, app/client, agent, session, project/repo, memory, claim, provenance link, anchor, contradiction, lifecycle event, and access scope.
- Validation expectation:
  - doc review against existing schema and current architecture
  - explicit mapping of current tables/modules to future entities
  - migration plan must identify what remains backward-compatible and what requires staged rollout

### M3. Introduce the canonical domain model skeleton

- Classification: `Service-layer change`
- Goal: add minimal internal types and seams so later scope and adapter work can land without another public-API rewrite.
- Status: completed on 2026-03-07 via `Prompt B2` with backward-compatible scope types and client/schema service seams.
- Validation expectation:
  - targeted unit tests for new types and request/response models
  - no behavior regressions in current single-project flows
  - run at least:
    - `tests/test_client.py`
    - `tests/test_schemas.py`
    - `tests/test_project_isolation.py`

### M4. Add persistent shared-scope support

- Classification: `Schema change`, `Service-layer change`
- Goal: persist intentional sharing scopes while preserving legacy project-only defaults.
- Status: completed on 2026-03-07 via `Prompt A6` with schema v13 scope persistence, scope-aware client/query semantics, and compatibility defaults.
- Implementation note: shared deployments also require one DB-backed consolidation scheduler lease so multiple clients can coordinate auto-consolidation without duplicate runs.
- Validation expectation:
  - migration tests for upgrade from current installs
  - backward-compatibility tests for legacy single-project behavior
  - explicit scope isolation and shared-scope access tests
  - scheduler lease contention tests for multi-client auto-consolidation safety
  - run at least:
    - `tests/test_project_isolation.py`
    - relevant migration tests
    - `tests/test_temporal_belief_queries.py`
    - `tests/test_claim_recall.py`

### M5. Refactor to a canonical trust-preserving service layer

- Classification: `Service-layer change`
- Goal: make one layer responsible for temporal recall, provenance-aware recall, contradiction-aware recall, drift-aware challenge behavior, and shared-scope semantics.
- Validation expectation:
  - semantic parity tests across Python, MCP, REST, and schema-dispatch surfaces
  - trust regression suites remain green
  - run at least:
    - `tests/test_temporal_belief_queries.py`
    - `tests/test_contradictions.py`
    - `tests/test_source_traceability.py`
    - `tests/test_drift_invalidation.py`
    - `tests/test_server.py`
    - `tests/test_rest.py`
    - `tests/test_schemas.py`

### M6. Introduce the adapter framework

- Classification: `Service-layer change`, `Adapter/integration work`
- Goal: separate ecosystem-specific integration code from canonical memory semantics.
- Validation expectation:
  - adapter abstraction tests
  - regression coverage for any rewired existing surface
  - adapter invariants documented and enforced by tests
  - run at least:
    - new adapter test module(s)
    - affected surface tests such as `tests/test_server.py` or `tests/test_schemas.py`

### M7. Ship first native universal integrations

- Classification: `Adapter/integration work`
- Goal: prove the platform works in real multi-surface workflows, starting with MCP and OpenAI Agents SDK, then LangGraph.
- Validation expectation:
  - adapter-specific regression tests
  - end-to-end example flows covering shared namespace usage
  - docs/examples must match actual runnable paths
  - run at least:
    - new integration-specific test modules
    - example smoke tests
    - affected surface suites

### M8. Re-verify trust guarantees across shared scopes and adapters

- Classification: `Service-layer change`, `Adapter/integration work`
- Goal: show the universal refactor preserved the repo's differentiators.
- Validation expectation:
  - matrix coverage for:
    - `as_of` correctness
    - provenance completeness
    - contradiction visibility
    - drift challenge behavior
    - isolation defaults and explicit sharing behavior
  - broader validation required because this milestone touches shared infrastructure
  - run the full trust-oriented suite plus affected integration suites

### M9. Close benchmark and release-gate evidence gaps

- Classification: `Documentation-only`, `Service-layer change`
- Goal: ensure universal-memory claims are backed by metrics, enforced gates, and current evidence.
- Validation expectation:
  - `tests/test_release_gates.py`
  - benchmark harness tests
  - docs/code consistency review so every claim maps to an enforced metric or clearly labeled future evidence

### M10. Package for private alpha

- Classification: `Documentation-only`, `Adapter/integration work`
- Goal: produce a design-partner-ready package instead of a purely repo-internal milestone set.
- Validation expectation:
  - onboarding dry run from a clean environment
  - example/demo validation with at least two different surfaces sharing memory
  - docs review for install, deployment mode, and feedback loop completeness

### M11. Rewrite public positioning around shipped capabilities

- Classification: `Documentation-only`
- Goal: align public story with what has actually shipped and been measured.
- Validation expectation:
  - README and product docs reviewed against the latest evidence
  - no unsupported claims
  - quick-start examples remain runnable

## Broader Validation Triggers

Use targeted validation by default, but run broader validation when a milestone changes shared infrastructure.

- If schema or migration code changes:
  - run migration, project isolation, client, and temporal/claim recall suites.
- If the canonical service layer changes:
  - run trust suites plus all affected external surfaces.
- If adapter or integration code changes:
  - run adapter tests, affected surface tests, and at least one end-to-end shared-memory flow.
- If only docs change:
  - limit validation to doc integrity and consistency checks unless the docs claim behavior that is not yet implemented.

## Recommended Prompt Order

This is the execution order to use after `Prompt B1`.

1. `Prompt A4: Canonical Memory Object Model`
2. `Prompt B2: Introduce The Canonical Domain Model Skeleton`
3. `Prompt A6: Shared Memory Scopes, Identity, And Namespacing`
4. `Prompt B3: Add Persistent Shared-Scope Support`
5. `Prompt A7: Trust-Preserving Universal Query Layer`
6. `Prompt B4: Refactor To A Canonical Memory Service Layer`
7. `Prompt A5: Adapter Layer Refactor`
8. `Prompt B5: Add The Adapter Framework`
9. `Prompt B6: Implement The First Two Universal Integrations`
10. `Prompt B7: Harden Shared-Memory Trust Semantics Across Surfaces`
11. `Prompt A8: Benchmark And Proof Of Novelty`
12. `Prompt B8: Close The Benchmark And Release-Gate Evidence Gap`
13. `Prompt A9: Private Alpha Distribution And Flagship Demo`
14. `Prompt B9: Build A Credible Private-Alpha Distribution`
15. `Prompt A2: Product Thesis And Competitive Map`
16. `Prompt B10: Rewrite The External Story Around Universal Shared Memory Trust`

## Immediate Next Step

Run `Prompt A7: Trust-Preserving Universal Query Layer`.

That step should consolidate scope, temporal, provenance, contradiction, and drift semantics behind one canonical internal service/query layer.
