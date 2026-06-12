# Agent Goal — Living Backlog

**North star:** Build a trust-calibrated persistent memory system where **deterministic belief maintenance** does the heavy lifting and LLMs handle only unstructured residue.

**Invariant to optimize toward:**

> With LLM consolidation disabled, the system still stores episodes, materializes claims, merges structured knowledge, ranks by trust, and degrades gracefully on drift.

**Rules:** [VIBECODING.md](VIBECODING.md) · **Architecture:** [ARCHITECTURE.md](ARCHITECTURE.md)

---

## Current milestone: M4 — Hypothesis competition

**Status:** in progress  
**Done when:** novelty gates pass with `CONTRADICTION_RESOLUTION_MODE=compete` enabled.

| Done | Task | Primary files |
| --- | --- | --- |
| ⬜ | `CONTRADICTION_RESOLUTION_MODE=compete\|expire_old` (default unchanged) | `config.py`, `query_semantics.py`, `engine.py` |
| ⬜ | Competing claims stay active with lowered precision + `contradicts` edges | `claim_graph.py`, `database.py` |
| ⬜ | Novelty eval scenario for compete mode | `benchmarks/`, `docs/NOVELTY_METRICS.md` |

**Commercial track (paid surface):** [MONETIZATION.md](MONETIZATION.md) — C0 published; C1 = policy admin API + trust dashboard for first pilot.

## Completed: M3 — Prediction-error-driven consolidation

**Status:** complete  
**Done when:** a failed outcome measurably raises consolidation priority for related claims.

| Done | Task | Primary files |
| --- | --- | --- |
| ✅ | `outcome_failure_rate` signal in utility scheduler | `consolidation/utility_scheduler.py`, `database.py`, `client_runtime.py`, `config.py` |
| ✅ | Consolidation prioritizes clusters linked to failed `action_outcomes` | `consolidation/engine.py`, `database.py` |
| ✅ | Status explains *why* consolidation ran (utility breakdown) | `client.py`, `types.py` |

## Completed: M2 — Claims as source of truth

**Status:** complete  
**Done when:** editing markdown alone does not change trusted recall behavior.

## Completed: M1 — LLM-optional substrate

**Status:** complete  
**Done when:** consolidation demo works with LLM off for preferences + anchor-rich solutions.

| Done | Task | Primary files |
| --- | --- | --- |
| ✅ | Fast-path consolidation module | `consolidation/fast_path.py`, `consolidation/engine.py` |
| ✅ | Fast-path metrics in consolidation report + DB | `database.py` (schema v18), `engine.py` |
| ✅ | Fast-path tests | `tests/test_fast_path_consolidation.py` |
| ✅ | Expose `fast_path_hits` / `llm_fallbacks` in `memory_status` | `client.py`, `schemas.py`, `server.py`, `rest.py` |
| ✅ | Fast-path parsers: structured JSON facts, procedures | `consolidation/fast_path.py`, tests |
| ✅ | Integration test: `LLM` backend disabled → structured consolidation succeeds | `tests/` |
| ✅ | Document fast-path episode shapes in `docs/` (user-facing) | `docs/FAST_PATH_EPISODES.md`, README |

---

## M2 — Claims as source of truth

| Done | Task | Primary files |
| --- | --- | --- |
| ✅ | Persist `precision` on claims (schema migration) | `database.py`, `claim_graph.py` |
| ✅ | Update precision from outcomes / drift / contradictions | `query_semantics.py`, `database.py` |
| ✅ | Fast-path topic markdown from records only (no LLM merge) | `engine.py` |
| ✅ | Recall ranking uses precision | `context_assembler.py`, `query_semantics.py` |

---

## Session pick-up order (agents)

When the user says **"go"**, **"continue"**, or starts a new task without specifics:

1. Read this file — take the **first unchecked task** in the current milestone (M2).
2. Read [VIBECODING.md](VIBECODING.md) §7 task template — plan one slice.
3. `memory_recall` with: `agent goal M1 consolidation-memory next task`.
4. Implement **one** unchecked task only.
5. Run verification from [VIBECODING.md](VIBECODING.md) §8.
6. `memory_store` what was done; update this file (check off task or note blockers).
7. `memory_recall` before final response.

**Do not** jump to M2/M3/M5 unless the user explicitly redirects.

---

## Default session prompt (user can paste or say)

```text
Continue consolidation-memory toward the agent goal.
Read docs/AGENT_GOAL.md and docs/VIBECODING.md.
Pick the first unchecked M1 task, implement one slice, test it, update AGENT_GOAL.md.
Follow AGENTS.md session workflow.
```

---

## Progress log

| Date | Agent / session | Completed |
| --- | --- | --- |
| 2026-06-12 | Initial vibecoding setup | Fast-path consolidation, VIBECODING guide, agent bootstrap docs |
| 2026-06-12 | M1 status metrics | `memory_status` exposes `fast_path_hits` / `llm_fallbacks` from last run + aggregates in `consolidation_quality` / `recent_activity` |
| 2026-06-12 | M1 fast-path parsers | Structured JSON validation for facts/procedures; text procedure parser (`Trigger:`/`Steps:`); cluster tests without LLM |
| 2026-06-12 | M1 LLM-off integration | `test_integration.py::TestLlmDisabledStructuredConsolidation` — preference + solution + fact via `run_consolidation`, `api_calls=0` |
| 2026-06-12 | M1 complete | `docs/FAST_PATH_EPISODES.md` + README/ARCHITECTURE links; milestone done-when met |
| 2026-06-12 | M2 claim precision | Schema v19 `claims.precision` (default 1.0); `upsert_claim` preserves precision on refresh; `claim_from_record` emits default precision |
| 2026-06-12 | M2 precision updates | `claim_precision_from_evidence()`; auto-recompute on outcomes, drift/challenge, contradiction events |
| 2026-06-12 | M2 fast-path markdown | `_render_topic_markdown_from_db()`; fast-path create/merge renders from persisted records; tests in `test_fast_path_consolidation.py` |
| 2026-06-12 | M2 precision ranking | `claim_precision_multiplier()` in ranking profile; claims + records recall apply persisted precision; `query_service` row mapping fixed |
| 2026-06-12 | M3 outcome_failure_rate | `get_outcome_failure_rate_since()`; utility scheduler signal + config weights; tests in `test_utility_scheduler.py` |
| 2026-06-12 | M3 failure cluster priority | `get_failure_linked_episode_ids_since()`; episode load + cluster processing order; tests in `test_outcome_cluster_priority.py` |
| 2026-06-12 | M3 status trigger explanation | `build_consolidation_trigger_explanation()`; persisted `last_trigger_breakdown` (schema v20); `utility_scheduler.last_run_trigger` + `run_decision.explanation` in `memory_status` |

---

## After M3

Proceed M4 → M5 on the engineering track. Parallel commercial milestones (C1–C3) are scoped in [MONETIZATION.md](MONETIZATION.md) and [ROADMAP.md](ROADMAP.md) — do not paywall open-core trust semantics.

See full milestone list in [VIBECODING.md §6](VIBECODING.md#6-milestone-roadmap-vibe-in-this-order).