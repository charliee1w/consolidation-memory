# Release Gates

Release gating is fail-closed: a release is allowed only when every mandatory
gate passes on complete, recent evidence. Missing or stale evidence is a failure,
a passing local run does not override failing CI gate evidence, and the
evaluator reports rather than degrades.

## Source Of Truth

- Gate evaluator: `src/consolidation_memory/release_gates.py`
- Gate CLI: `scripts/verify_release_gates.py`
- Publish workflow: `.github/workflows/publish.yml`
- Automated release trigger: `.github/workflows/release-on-main.yml`
- Criteria evaluator: `scripts/release_criteria.py`
- Automation reference: [RELEASE_AUTOMATION.md](RELEASE_AUTOMATION.md)
- Docs freshness guard: `scripts/check_release_docs.py` (`.github/workflows/release-docs-guard.yml`)

## Required Evidence

- Novelty evaluation JSON (`mode=full` for release).
- Scope alignment evidence (use-case string must exist in `docs/NOVELTY_WEDGE.md`).
- Gate report JSON produced by `scripts/verify_release_gates.py`.

## Mandatory Gates

`evaluate_release_gates()` in `src/consolidation_memory/release_gates.py` returns
four gates plus `overall_pass`; `overall_pass` is the conjunction of all four, so
any failure fails the release. Failures are appended to `errors` verbatim.

1. `scope_alignment_gate` — `--scope-use-case` must appear in
   `docs/NOVELTY_WEDGE.md` (case-insensitive, substring match). On a miss the
   CLI reports `use-case not found in wedge doc: '<value>'`.
2. `metric_threshold_gate` — `mode` must equal `full` and `overall_pass` must be
   `true` with every section `pass` true:
   `mode must be 'full' for release gating`,
   `metric threshold gate failed (overall_pass false or a section failed)`.
3. `evidence_completeness_gate` — the artifact must carry `benchmark`, `run_id`,
   `mode`, `generated_at`, `sections`, `overall_pass`; `run_id` must be a
   non-empty string; `generated_at` a valid ISO datetime; `sections` a non-empty
   object whose members each carry `aligned_metric_section`, `thresholds`,
   `measured`, `pass`:
   `missing top-level fields: …`, `sections must be a non-empty object`,
   `section '<name>' missing required fields: …`, `evidence completeness gate failed`.
4. `evidence_recency_gate` — `generated_at` must be no older than `max_age_days`
   (default 7, `--max-age-days` to override):
   `evidence recency gate failed (must be <= 7 days old)`.

The CLI prints the full report as JSON and exits non-zero when
`overall_pass` is false.

## Local Verification

Reproduce the gate evaluator against fresh evidence:

```bash
python -m benchmarks.novelty_eval --mode full --output benchmarks/results/novelty_eval_release_full.json
python scripts/verify_release_gates.py \
  --novelty-result benchmarks/results/novelty_eval_release_full.json \
  --scope-use-case "Drift-aware debugging memory" \
  --output benchmarks/results/release_gate_report.json
```

Reproduce the PR-CI quality set locally:

```bash
python -m pytest tests/ -q
python -m pytest tests/ -q -W error::ResourceWarning
python scripts/smoke_builder_base.py
python scripts/smoke_mcp_stdio.py
ruff check src/ tests/
mypy src/consolidation_memory/
python -m bandit -q -ll -r src scripts -s B608,B110
python scripts/generate_tool_reference.py --check
```

`scripts/release.py --bump patch --dry-run` is a **plan preview only**. It
validates the clean-tree / `main` / tag / PyPI state, computes the target
version and the changelog range, then prints steps `[1/9]`…`[9/9]` with the
mutating ones empty. It executes no quality gate, no release gate and no
artifact build, and it exits non-zero off `main`:

```text
[1/9] Validating git state...
  $ git branch --show-current
Release automation must run from 'main', found 'feat/mcp-typed-surface'.
```

Use it to check the target version, not to verify a release.

Without `--dry-run`, `release.py` runs the same publish-grade checks as tag
publish: clean `main`, `pip install -e ".[fastembed,rest,dev]"`, tests with
coverage, builder smoke, `ResourceWarning` gate, `ruff check src/ tests/`,
`mypy src/consolidation_memory/`, `bandit -q -ll -r src scripts -s B608,B110`,
full novelty evaluation plus `verify_release_gates.py`, then `python -m build` +
`twine check --strict`, rolling back `pyproject.toml` and `CHANGELOG.md` on any
failure. It bumps the version, commits, tags and pushes unless `--no-push`
(`--skip-pypi-check` skips the remote collision probe).

## CI Enforcement

- PR CI (`test.yml`) runs quick novelty checks: `novelty_eval --mode quick`,
  `coding_agent_eval --mode quick` and `real_world_eval --mode ci` on the
  fixture, each enforced on `overall_pass`.
- PR CI runs the test matrix on Python 3.10–3.13 × ubuntu/windows, and on
  3.13/ubuntu also the builder smoke, the **MCP stdio wire smoke**
  (`scripts/smoke_mcp_stdio.py`, 8 checks incl. negotiated `protocolVersion`,
  both `outputSchema` arms, and unknown-argument rejection as `isError`), the
  `ResourceWarning` gate, lint, type check and the bandit security scan.
- PR CI also validates wheel/sdist buildability and runs a dedicated
  `optional_surfaces` job with the `all` + `dev` extras (covers `rest`,
  `openai`, `dashboard`, `desktop` test suites).
- Main-branch automation (`release-on-main.yml`) evaluates release criteria and
  only creates a new release tag/version when eligible.
- Tag publish (`publish.yml`) requires the tagged commit to be on `origin/main`,
  runs release quality gates (tests with coverage, builder smoke,
  `ResourceWarning` gate, `ruff check src/ tests/`, `mypy`, `bandit`), then full
  novelty evaluation + gate enforcement, then build/`twine check`/publish.
  **The stdio wire smoke is not part of the publish gate** — it lives in
  3.13/ubuntu PR CI only, so a green publish does not imply wire coverage.
- Nightly (`novelty-full-nightly.yml`, cron `15 6 * * *`) refreshes full novelty
  + gate artifacts.
- `release-docs-guard.yml` fails when release-automation files change without
  matching updates in `docs/RELEASE_AUTOMATION.md` or `README.md`.
- `changelog-on-main.yml` refreshes the `Unreleased` changelog section and
  regenerates `docs/TOOLS.md` (both gated on the `RELEASE_AUTOMATION_PAT`
  secret being configured).
