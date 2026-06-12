# Vibecoding Guide

**Audience:** AI agents and humans making iterative changes to `consolidation-memory`.

**Purpose:** Keep fast, agent-assisted development from breaking trust semantics, surface parity, or long-term memory correctness.

Read this before writing code. If instructions in a task conflict with this guide, **this guide wins** unless the user explicitly overrides a specific rule.

---

## 1. What you are building

`consolidation-memory` is a **trust-calibrated belief layer for coding agents** — not a generic vector store and not a chat log archiver.

| Unit | Role |
| --- | --- |
| **Episode** | Raw evidence (what happened) |
| **Knowledge record** | Typed structured knowledge (`fact`, `solution`, `preference`, `procedure`, `strategy`) |
| **Claim** | Deterministic, hash-stable belief derived from a record |
| **Topic** | Human-readable consolidated view (markdown + DB rows) |

**North-star invariant (optimize toward this):**

> With LLM consolidation disabled, the system can still store episodes, materialize claims via deterministic paths, merge structured knowledge, rank by trust signals, and degrade gracefully on drift — without catastrophic recall failure.

---

## 2. Non-negotiable trust invariants

Every change must preserve these. If you cannot preserve them, **stop and ask** or scope the change smaller.

1. **Temporal correctness** — `as_of` queries return the knowledge state at that time, including superseded/expired records and claims.
2. **Provenance traceability** — claims link to source episodes/topics/records; do not create orphan beliefs.
3. **Contradiction visibility** — conflicts are logged and surfaced; never silently overwrite contradictory history.
4. **Drift challenge auditability** — `code_drift_detected` events and challenged-claim state remain inspectable.
5. **Scope isolation** — namespace/project/app/agent/session boundaries are never accidentally widened.
6. **Surface parity** — Python, MCP, REST, and OpenAI tool dispatch must share the same semantics (`MemoryClient` → `CanonicalQueryService` → `query_semantics`).

**Fail closed:** When uncertain, prefer explicit warnings, challenged state, or skipping consolidation over guessing.

---

## 3. Vibecoding rules

### Always do

- **Start small** — one behavior, one module, one test file per session when possible.
- **Prefer deterministic code** over LLM prompts for core memory behavior.
- **Read surrounding code** before editing; match naming, patterns, and error handling.
- **Add tests** for every trust-impacting or consolidation behavior change.
- **Run targeted verification** before claiming done (see §8).
- **Use additive schema migrations** — bump `CURRENT_SCHEMA_VERSION` in `database.py`.
- **Invalidate caches** after graph/knowledge mutations (`topic_cache`, `record_cache`, `claim_cache`).
- **Update docs in the same change** when user-visible behavior changes.

### Never do (without explicit user approval)

- Rewrite `MemoryClient`, `consolidation/engine.py`, or `database.py` in one shot.
- Change retrieval ranking without checking `query_semantics.py` and `context_assembler.py`.
- Replace LLM consolidation with “smarter prompts” as the only fix.
- Delete or weaken novelty/release gate checks.
- Break backward compatibility for default single-project usage.
- Store secrets in episodes, claims, or markdown topics.
- Ship schema changes without migration SQL and a test that calls `ensure_schema()`.
- Add features only to one transport (e.g. MCP-only) without updating `schemas.py`, `rest.py`, `server.py`, and tests.

### Vibecoding-friendly work (high success rate)

| Area | Module(s) | Why |
| --- | --- | --- |
| Fast-path consolidation parsers | `consolidation/fast_path.py` | Pure logic, unit-testable, no LLM |
| Claim precision / edges | `claim_graph.py`, `database.py`, `query_semantics.py` | Deterministic, eval-backed |
| Utility scheduler signals | `consolidation/utility_scheduler.py`, `client.py` | Small additive signals |
| Anchor / entity extraction | `anchors.py`, new thin helpers | Parser-only, no side effects |
| Tests & benchmarks | `tests/`, `benchmarks/` | Raises the safety floor |
| Examples & adapter docs | `examples/`, `docs/` | Low trust risk |

### Vibecoding-hostile work (split into many PRs)

| Area | Risk |
| --- | --- |
| Full consolidation rewrite | Hidden trust regressions |
| New retrieval paradigm without eval | “Feels better”, measures worse |
| Contradiction resolution behavior change | Needs novelty harness update |
| Policy/ACL semantics | Security + scope blast radius |
| FAISS index format changes | Data loss risk |

---

## 4. Architecture map — where to change what

```
Store/recall orchestration     → client.py
Canonical query envelopes      → query_service.py
Trust filters & ranking        → query_semantics.py, context_assembler.py
Consolidation orchestration    → consolidation/engine.py
Deterministic extraction       → consolidation/fast_path.py
LLM extract/merge (fallback)   → consolidation/prompting.py
Claim identity                 → claim_graph.py
Persistence & migrations       → database.py
Vectors                        → vector_store.py
Drift                          → drift.py, drift_worker.py
MCP tools                      → server.py
REST                           → rest.py
OpenAI tool schemas            → schemas.py, tool_dispatch.py
Config                         → config.py
Plugins                        → plugins.py
```

**Rule:** Business semantics belong in service/query modules. Transports are thin adapters.

---

## 5. Consolidation decision flow

```text
unconsolidated episodes
    → cluster (embedding + scope isolation)
    → try_fast_path_extraction()     # deterministic, no LLM
        ├─ hit  → extraction_data + deterministic merge only
        └─ miss → _llm_extract_with_validation()   # LLM fallback
    → find similar topic or create new
    → merge records → emit/update claims + provenance
    → contradiction detection (embedding + optional LLM verify)
    → mark episodes consolidated
```

**Agent rules for consolidation changes:**

1. Add new structured episode shapes to **fast-path first**, not new LLM prompts.
2. Fast-path must return `None` on ambiguity — fall back to LLM, do not guess.
3. Track `fast_path_hits` / `llm_fallbacks` when touching consolidation metrics.
4. Do not remove deterministic merge fallback paths in `engine.py`.
5. Contradiction handling changes require tests in `tests/test_contradictions.py` and consideration for `benchmarks/novelty_eval.py`.

---

## 6. Milestone roadmap (vibe in this order)

Do not skip ahead to “world model” work before the substrate is solid.

### M1 — LLM-optional substrate (current focus)

- [x] Fast-path consolidation (`consolidation/fast_path.py`)
- [x] More parsers: structured facts, procedures, tag-derived preference keys
- [x] `memory_status` exposes `fast_path_hits` / `llm_fallbacks` from last run
- [x] Integration test: consolidation succeeds with LLM backend disabled for structured episodes
- [x] User-facing fast-path episode shapes ([FAST_PATH_EPISODES.md](FAST_PATH_EPISODES.md))

**Done when:** demo consolidation with LLM off for preferences + anchor-rich solutions.

### M2 — Claims as source of truth

- [ ] Persisted `precision` (or equivalent) on claims
- [ ] Deterministic precision updates from outcomes, drift, contradictions
- [x] Topic markdown rendered from claim/record state (not LLM rewrite) on fast-path
- [x] Recall ranking incorporates precision

**Done when:** editing markdown alone does not change trusted recall behavior.

### M3 — Prediction-error-driven consolidation

- [x] `outcome_failure_rate` signal in `utility_scheduler.py`
- [x] Consolidation prioritizes clusters linked to failed `action_outcomes`
- [x] Status explains *why* consolidation ran (utility breakdown)

**Done when:** a failed outcome measurably raises consolidation priority for related claims.

### M4 — Hypothesis competition (current)

- [ ] `CONTRADICTION_RESOLUTION_MODE=compete|expire_old` (default unchanged for compat)
- [ ] Competing claims stay active with lowered precision + `contradicts` edges
- [ ] Novelty eval scenario for compete mode

**Done when:** novelty gates pass with compete mode enabled.

### M5 — Thin structural layer

- [ ] Entity records derived from anchors + record subjects
- [ ] Entity-centric recall expansion
- [ ] No separate storage paradigm — graph traversal over existing tables

**Done when:** entity-centric recall beats raw vector-only on a fixed eval fixture.

### Commercial track (C1–C3)

Paid surface is **team governance, sync, and compliance ops** — not core trust semantics. See [MONETIZATION.md](MONETIZATION.md).

- [x] C0 — publish monetization plan + roadmap alignment
- [ ] C1 — policy admin API + read-only trust dashboard (first paid pilot)
- [ ] C2 — sync transport + hosted workers (Team tier beta)
- [ ] C3 — SSO + evidence packs (Enterprise pilot)

---

## 7. Task sizing template

A vibe session should fit this template. If it does not, split it.

```markdown
## Task
[One sentence behavior change]

## Scope
- Files: [max 2–4 primary files]
- Out of scope: [explicit list]

## Trust impact
- [ ] temporal  [ ] provenance  [ ] contradiction  [ ] drift  [ ] scope  [ ] surface parity

## Implementation notes
- [Where in the flow this hooks in]

## Tests
- [ ] New test file or test name
- [ ] Existing tests that must still pass

## Verification
pytest -q tests/<file>.py
ruff check src tests
[optional] python -m benchmarks.novelty_eval --mode quick
```

**Maximum recommended diff:** ~300 lines for trust-impacting work. Larger changes need explicit user sign-off.

---

## 8. Verification matrix

Run the minimum row for your change type. Do not claim “done” without command output.

| Change type | Required checks |
| --- | --- |
| Any code change | `ruff check src tests` |
| Consolidation / claims / drift | `pytest -q tests/test_fast_path_consolidation.py tests/test_claim_emission.py tests/test_contradictions.py` |
| Query / recall / ranking | `pytest -q tests/test_query_semantics.py tests/test_query_service.py` (and related) |
| Schema migration | `pytest -q tests/test_core.py` + test calling `ensure_schema()` |
| REST/MCP/schema surface | `pytest -q tests/test_rest.py tests/test_schemas.py tests/test_server.py` |
| Release-impacting trust change | `python -m benchmarks.novelty_eval --mode quick` |
| Pre-release | Full builder baseline in `docs/BUILDER_BASELINE.md` + novelty full mode |

**Desktop / resource-constrained agents:** run targeted pytest first; ask before full `pytest tests/ -q`.

---

## 9. Schema & migration rules

1. Increment `CURRENT_SCHEMA_VERSION` in `database.py`.
2. Add migration SQL to `MIGRATIONS` dict for the new version.
3. Migrations must be **additive** when possible (`ALTER TABLE ADD COLUMN`, new tables).
4. Never drop columns or rewrite history without a documented migration plan in `docs/strategy/schema-migration-plan.md`.
5. New columns on metrics tables need defaults for backward-compatible inserts.
6. Test with a fresh `ensure_schema()` and, when feasible, upgrade from prior version fixtures.

---

## 10. Surface parity checklist

If you add or change a memory operation, verify alignment across:

- [ ] `client.py` (`MemoryClient` method)
- [ ] `query_service.py` (canonical envelope if query-shaped)
- [ ] `schemas.py` (`openai_tools` + `dispatch_tool_call`)
- [ ] `server.py` (MCP tool)
- [ ] `rest.py` (endpoint, if REST-exposed)
- [ ] `types.py` (result types)
- [ ] Tests for at least Python + schema dispatch

**Canonical semantics live in** `query_service.py` and `query_semantics.py` — adapters do not reimplement trust logic.

---

## 11. Config & feature flags

- Scalar config: `config.py` dataclass + TOML mapping + `CONSOLIDATION_MEMORY_<FIELD>` env override.
- Prefer **feature flags defaulting to current behavior** for trust behavior changes (e.g. `CONTRADICTION_RESOLUTION_MODE`).
- Document new config keys in `docs/ARCHITECTURE.md` or README configuration section when user-facing.

**Existing consolidation flags agents should know:**

| Flag | Default | Meaning |
| --- | --- | --- |
| `CONSOLIDATION_FAST_PATH_ENABLED` | `true` | Deterministic extraction before LLM |
| `CONSOLIDATION_MIN_CLUSTER_SIZE` | `2` | Min episodes per cluster (`1` allows singleton fast-path tests) |
| `CONTRADICTION_LLM_ENABLED` | `true` | LLM verification for contradiction pairs |
| `CONSOLIDATION_AUTO_RUN` | `true` | Background consolidation scheduler |

---

## 12. Prompt templates for agent sessions

### Add fast-path parser

```text
Add a deterministic parser in consolidation/fast_path.py for [episode shape].
Hook it in _extract_record_from_episode before LLM fallback.
Return None on ambiguity.
Add tests in tests/test_fast_path_consolidation.py proving LLM is not called.
Do not change query_semantics or retrieval ranking.
Run: pytest -q tests/test_fast_path_consolidation.py && ruff check src tests
```

### Add claim trust signal

```text
Add [signal] to claim reliability in query_semantics.py claim_reliability_profile().
Use existing evidence from get_claim_outcome_evidence / claim_events — no new LLM calls.
Add unit tests with fixture evidence payloads.
Preserve backward compatibility of profile keys.
Run: pytest -q tests/test_query_semantics.py && ruff check src tests
```

### Add utility scheduler signal

```text
Add [signal] to consolidation/utility_scheduler.py compute_utility_score().
Wire raw signal collection in client.py _compute_consolidation_utility().
Add tests for score breakdown with and without the signal.
Do not change consolidation engine clustering.
Run: pytest -q tests/test_adaptive_consolidation.py && ruff check src tests
```

### Trust behavior change (high risk)

```text
Implement [behavior] behind config flag defaulting to current behavior.
Add tests for old and new mode.
Add or extend novelty_eval scenario if contradiction/drift/temporal/provenance affected.
Document flag in docs/VIBECODING.md and docs/ARCHITECTURE.md.
Run targeted tests + novelty quick mode.
```

---

## 13. Red flags — stop and reassess

Stop the session if you notice:

- Tests pass only because LLM calls are mocked to return fixtures unrelated to the code path
- Consolidation creates claims without `claim_sources` rows
- `as_of` tests start failing or are deleted instead of fixed
- Retrieval ranking changed with no test and no eval
- Schema version not bumped but SQL expects new columns
- MCP tool added without `schemas.py` entry
- “Refactor while I'm here” expanding scope beyond the task
- Silent deletion of contradicted claims without `claim_events`
- Fast-path returns records for unstructured prose (over-parsing)

---

## 14. Using project memory while developing

This repo uses its own MCP server (`consolidation_memory`). Agents should:

1. `memory_recall` with the task goal before coding.
2. `memory_store` concise facts/decisions after meaningful progress (tag with module names).
3. `memory_detect_drift` after editing anchored source files — **once**; if timeout, report and continue.
4. `memory_recall` before final response to align with current claims.

Valid `memory_store` content types: `exchange`, `fact`, `preference`, `solution`.

Pass `tags` as a JSON array, not a comma-separated string.

---

## 15. Git & PR discipline

- Ask before commits, push, or PR workflows unless the user explicitly requested them.
- One logical change per commit/PR when possible.
- PR must include: problem, summary, test commands + output, trust-impact notes.
- Do not amend/rebase unless asked.

---

## 16. Definition of done

A vibe session is complete only when:

1. Scope matched the task template (§7) — no drive-by refactors.
2. Trust invariants (§2) preserved or explicitly flagged with user approval.
3. Tests added/updated for behavior changes.
4. Verification commands run and output cited.
5. Docs updated if behavior is user-visible.
6. No secrets, no telemetry, no hidden network calls introduced.

---

## 17. Related docs

| Doc | Use when |
| --- | --- |
| [AGENT_GOAL.md](AGENT_GOAL.md) | **Session start** — current milestone and next task |
| [../AGENTS.md](../AGENTS.md) | Per-session workflow checklist |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Understanding current system layout |
| [ROADMAP.md](ROADMAP.md) | Product priorities |
| [NOVELTY_METRICS.md](NOVELTY_METRICS.md) | Trust metric definitions |
| [NOVELTY_EVAL_GUIDE.md](NOVELTY_EVAL_GUIDE.md) | Running eval harness |
| [RELEASE_GATES.md](RELEASE_GATES.md) | Release requirements |
| [BUILDER_BASELINE.md](BUILDER_BASELINE.md) | Full local CI parity |
| [strategy/memory-object-model.md](strategy/memory-object-model.md) | Conceptual objects |
| [../CLAUDE.md](../CLAUDE.md) | Compact maintainer context |
| [../AGENTS.md](../AGENTS.md) | Session workflow + MCP usage |

---

## 18. One-line mantra

**Vibe deterministic belief maintenance into the codebase; use LLMs only for unstructured residue; let tests and novelty eval prove trust.**