# Monetization Plan

**Strategy:** Open-core adoption, paid operations and governance.

The trust layer (claims, drift, provenance, temporal correctness) stays **MIT and local-first** so builders can inspect, adopt, and extend it. Revenue comes from **running memory for teams** and **governing shared belief state** — not from locking the SQLite substrate.

**Wedge (unchanged):** [Drift-aware debugging memory](NOVELTY_WEDGE.md) for coding agents.

---

## Product tiers

| Tier | Audience | Price posture | What they get |
| --- | --- | --- | --- |
| **Community** | Solo devs, OSS adopters | Free forever | `pip install`, MCP/REST/Python, local SQLite+FAISS, drift challenge, novelty-eval reproducibility, examples |
| **Team** | Small eng teams sharing agent memory | Paid subscription (target) | Hosted or self-hosted **sync control plane**, namespace/project scope admin, policy/ACL UI, audit exports, priority support |
| **Enterprise** | Regulated or multi-repo orgs | Contract + SLA | SSO/SAML, compliance evidence packs, custom retention, dedicated drift workers, onboarding |

Community tier is the **distribution engine**. Team and Enterprise are the **monetization surface**.

---

## Open core (always free, MIT)

Ship and maintain in the public repo:

- Episode store, consolidation (fast-path + LLM fallback), claim graph
- Trust-aware recall, `as_of`, contradiction visibility, drift challenge
- Scope envelopes + persisted ACL primitives (runtime enforcement)
- MCP, REST, Python SDK, OpenAI tool parity
- Release gates, novelty harness, changelog/release automation
- Documentation, examples, smoke tests

**Rule:** If a feature proves *trust correctness* or *builder adoption*, it belongs in open core unless it requires multi-tenant ops by definition.

---

## Paid surface (build deliberately, not by accident)

### P1 — Team memory control plane (first revenue target)

**Problem:** Teams want shared agent memory without scope leaks or ungoverned belief drift.

**Paid capabilities:**

- Web/admin API for namespace, project, policy, and principal management
- Cross-machine memory sync with conflict-safe merge semantics
- Consolidation + drift job orchestration (scheduled workers, lease visibility)
- Usage and trust dashboards (challenged-claim backlog, provenance coverage, drift watch)

**Open-core boundary:** Enforcement primitives stay in `database.py` / `policy_engine.py`; **admin UX, sync transport, and hosted workers** are the paid layer.

**Done when:** A paying pilot team can onboard a namespace, connect two agent clients, and audit who wrote which claim — without forking core.

### P2 — Compliance and evidence packs

**Problem:** Teams need proof that agent memory did not silently rot.

**Paid capabilities:**

- Signed export bundles (claims, events, outcomes, drift challenges) for audit
- Release-gate / novelty-eval reports branded for customer change windows
- Retention and tombstone policies with legal hold

**Open-core boundary:** Export APIs and gate scripts stay open; **scheduled reports, signing, retention policy admin** are paid.

### P3 — Enterprise identity and operations

**Problem:** Large orgs need IAM integration and operational guarantees.

**Paid capabilities:**

- SSO/SAML, SCIM-style principal sync
- Per-tenant isolation, rate limits, backup/restore SLAs
- Dedicated support channel and upgrade playbooks

**Done when:** Enterprise security review can pass on documented data flows without reading proprietary core code.

---

## Roadmap alignment

| Phase | Engineering milestone | Commercial milestone |
| --- | --- | --- |
| **Now (M4)** | Hypothesis competition (`CONTRADICTION_RESOLUTION_MODE`) | Publish this plan; collect design-partner interest |
| **M5** | Entity-centric recall (thin structural layer) | Team tier **spec + pricing page** (even if waitlist) |
| **C1** | Policy admin REST + minimal web console (read-only first) | First **paid pilot** (manual billing OK) |
| **C2** | Sync transport + hosted drift/consolidation workers | Team tier **self-serve beta** |
| **C3** | Evidence packs + retention admin | Enterprise pilot + compliance one-pager |

Technical milestones M1–M3 are complete. See [AGENT_GOAL.md](AGENT_GOAL.md) for the engineering backlog; commercial milestones **C1–C3** track revenue readiness.

---

## What we will not monetize

- Core recall/store/consolidate semantics
- MCP tool schemas and transport parity
- Novelty eval harness (keeps marketing honest)
- Local-only single-developer workflows
- Security fixes and trust-invariant patches

---

## Pricing principles (pre-numbers)

1. **Free for individuals** — no feature-gating drift or provenance in Community.
2. **Charge for seats + namespaces** on Team — not per recall call (avoids perverse incentives).
3. **Charge for governance + uptime** on Enterprise — not for "more intelligence."
4. **No telemetry tax** — paid tiers add control plane; they do not phone home by default.

Concrete price points are intentionally deferred until C1 design-partner conversations.

---

## Success metrics

| Metric | Community | Team | Enterprise |
| --- | --- | --- | --- |
| Adoption | PyPI installs, GitHub stars, MCP configs | Paying namespaces | Contract renewals |
| Trust | Novelty gates green, drift eval scenarios | Audit export usage | Compliance pack delivery |
| Retention | Contributor return rate | Monthly active agent clients per namespace | SLA uptime |

---

## References

- [Roadmap](ROADMAP.md) — engineering + commercial tracks
- [Agent Goal](AGENT_GOAL.md) — current engineering milestone
- [Architecture](ARCHITECTURE.md) — where control-plane hooks attach
- [Release Automation](RELEASE_AUTOMATION.md) — shipping discipline for both tiers