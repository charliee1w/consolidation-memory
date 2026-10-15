# Release Automation

This repository supports automated stable releases from `main`.

## Workflow

- Changelog builder: `scripts/changelog_builder.py`
- Changelog updater: `scripts/update_changelog.py`
- Criteria evaluator: `scripts/release_criteria.py`
- Changelog workflow: `.github/workflows/changelog-on-main.yml`
- Release workflow: `.github/workflows/release-on-main.yml`
- Publish workflow: `.github/workflows/publish.yml` (tag-driven)
- Release script: `scripts/release.py`

Flow:

1. A push lands on `main`.
2. `changelog-on-main.yml` refreshes generated documentation and, when files changed, commits and pushes them with `[skip release]`.
   - Runs `python scripts/update_changelog.py` to rewrite the `## Unreleased` section in `CHANGELOG.md`, then `git add` / `git commit` / `git push` on the already-updated file (it does **not** re-run `--commit`, because the updater leaves `CHANGELOG.md` dirty).
   - Runs `python scripts/generate_tool_reference.py` to regenerate `docs/TOOLS.md` from the published tool schemas, then commits `chore(docs): refresh generated tool reference [skip release]` when it changed.
   - The suite's `tests/test_tool_reference_sync.py` fails when the committed `docs/TOOLS.md` is stale, and the release quality gates run that suite.
3. `release-on-main.yml` evaluates commits since the latest tag.
4. If eligible, it runs `scripts/release.py --bump <major|minor|patch>`.
5. The script bumps `pyproject.toml`, promotes the `## Unreleased` bullets into
   a versioned entry (falling back to the commit subjects since the previous tag
   when that section is empty), commits, tags (`vX.Y.Z`), and pushes.
6. The tag triggers `publish.yml`, which runs full gates and publishes release artifacts.

## Quick Setup Checklist

Do this once per repository:

1. Create a classic GitHub PAT with `repo` scope.
2. Add repo secret `RELEASE_AUTOMATION_PAT`.
3. Keep this PAT available to `changelog-on-main.yml` and `release-on-main.yml` (least privilege where possible).
4. Verify changelog automation:
   - Push a conventional commit (for example `fix: ...`) to `main`.
   - Confirm `Update Changelog On Main` commits an updated `## Unreleased` section when needed.
5. Verify release automation:
   - Push another releasable conventional commit and wait for the release job, or
   - `workflow_dispatch` `Automated Release On Main` with a forced bump (see [Manual Override](#manual-override)).

If both workflows run successfully, setup is complete.

## Local Changelog Preview

Refresh the unreleased section without releasing:

```bash
python scripts/update_changelog.py --dry-run
python scripts/update_changelog.py
```

Refresh the generated tool reference (`docs/TOOLS.md`):

```bash
python scripts/generate_tool_reference.py
python scripts/generate_tool_reference.py --check
```

`--check` exits non-zero when the committed file is stale — the same
condition `tests/test_tool_reference_sync.py` asserts inside the suite.

Commit locally when ready:

```bash
python scripts/update_changelog.py --commit
python scripts/update_changelog.py --commit --push
```

`--commit` allows a dirty tree when **only** `CHANGELOG.md` changed (for example after running the updater once, then committing in a second command). Other uncommitted files still block `--commit`.

## Changelog Entry Limit

`scripts/changelog_builder.py` keeps at most **200** bullets per release section
(`DEFAULT_SUBJECT_LIMIT`). Both entry points expose the same knob:

```bash
python scripts/update_changelog.py --limit 400
python scripts/release.py --bump minor --limit 400
```

200 is a runaway guard, not a routine filter. Releases run every few days and 200
bullets is only a few kilobytes of Markdown, so the cap has orders of magnitude of
headroom over any realistic release range. Recompute the widest range with
`git rev-list --count <previous-tag>..<tag>` across `git tag | sort -V`; if it
approaches the default, raise `DEFAULT_SUBJECT_LIMIT` in
`scripts/changelog_builder.py`.

### What is dropped first

When a range exceeds the limit, entries are ranked before the cut. Survivors
keep newest-first order; selection is:

- **Kept first** — `feat:`, `fix:`, `perf:`, `security:`, `revert:`, breaking
  subjects (`type(scope)!:`), unclassifiable subjects, and any dependency-scope
  change such as `chore(deps):` or `build(deps-dev):` (a raised dependency floor
  breaks downstream pins even when the subject reads like maintenance).
- **Dropped first** — `docs:`, `chore:`, `test:`, `ci:`, `style:`, `refactor:`,
  `build:` without a dependency scope.
- **Dropped last** — user-visible entries, oldest first, and only when they
  outnumber the limit on their own.

So a `docs:`/`chore:` wave can never displace a user-visible change. Entry
selection is independent of the semver bump: see [Criteria](#criteria) for what
a breaking subject does to the version.

### How entries are grouped

Bullets are rendered under `### Features`, `### Bug Fixes`, `### Performance`,
`### Security`, `### Dependencies`, `### Refactoring`, `### Documentation`,
`### Internal`, `### Other`, in that order. Dependency-scope changes get their own
`### Dependencies` section rather than `### Internal`, so a raised dependency
floor is visible to the reader it affects.

### The truncation warning

Truncation is never silent. The builder raises `ChangelogTruncationWarning` (a
`UserWarning`) from `collect_release_subjects`, and both scripts skip that path and
print the same report to **stderr** through `emit_selection_report`, which also
adds a `::warning::` annotation when running under GitHub Actions. Nothing is
written into `CHANGELOG.md`.

Verbatim report from an illustrative release range of 24 commits, 11 of them
releasable, selected with `--limit 10`, so 10 kept + 1 dropped = 11:

```text
[changelog] NOTE: kept 10 of 11 releasable commits; skipped 13 merge/release/skip-marked commit(s) and 0 duplicate subject(s).
[changelog] WARNING: Changelog truncated: kept 10 of 11 releasable commits (limit 10).
[changelog] WARNING: Dropped 1 entries: 1 internal (docs/chore/test/ci/refactor/style/build) and 0 user-visible (feat/fix/perf/security/breaking).
[changelog] WARNING: No user-visible change was lost; only maintainer-facing entries were dropped.
[changelog] WARNING: Re-run with a higher limit (for example --limit 200) to keep every entry.
[changelog] WARNING: Dropped entries:
[changelog] WARNING:   - chore(metrics): refresh published_metrics.json from 2026-06-16 full eval
```

Under GitHub Actions the same report is followed by a single annotation line:

```text
::warning::Changelog truncated to 10/11 commits (limit 10); dropped 0 user-visible and 1 internal entries. Raise the limit to at least 200.
```

The suggested limit is `max(2 × the current limit, the range size, 200)`, so a
small `--limit` never suggests going *below* the default.

When user-visible changes *are* dropped — `--limit 9` on the same range drops the
older `feat:` — the "no user-visible change was lost" line is replaced by
`User-visible changes were dropped (oldest first). Raise the limit before
releasing.` and the lost subjects are listed with them.

**Before releasing** with a warning in the log, re-run with the suggested
`--limit` value so no change ships undocumented. Raising the limit is the only
lever: `## Unreleased` is rewritten from commit headers on every push to `main`,
so text added there by hand does not survive. Notes an upgrading caller needs
belong in the commit body and the PR description.

## Versioning

The project stays on `0.x`. `0.x` is a public beta: the tool contract may break,
all surfaces are re-published together in one release, and no `1.x` line is
planned.

That decides what a contributor writes in a commit subject. Conventional Commits
reserves `!` and a `BREAKING CHANGE` body for a major bump, and
`scripts/release_criteria.py` below implements that literally — so either marker
anywhere in the release range produces `1.0.0` off a `0.x` version. **Neither
marker is used in this repository**, including for real contract breaks: an
ordinary `feat:` or `fix:` subject ships the break under a minor bump, which is
what [SECURITY.md](../SECURITY.md) describes for the supported line.

The guard is human. No workflow validates commit subjects, and `workflow_dispatch`
lets an operator force any bump; what actually prevents an outside contributor's
honest `!` from releasing `1.0.0` is review on the PR. A stray `!` that reaches
`main` is corrected with [Manual Override](#manual-override).

`CHANGELOG.md` carries no `### Breaking Changes` section. `## Unreleased` is
regenerated from commit headers by `update_changelog.py` on every push to `main`,
so hand-written prose inside it does not survive — see
[the changelog flow](#workflow). Upgrade-critical notes belong in the commit body
and the PR description.

## Criteria

`scripts/release_criteria.py` is deterministic and reads the commits since the
latest tag, newest first:

1. Head commit contains `[skip release]` or `[release skip]` → no release.
2. Head commit contains `[release major|minor|patch]` (or `[bump …]`) → forced bump.
3. Otherwise, the highest signal across the range wins:
   - breaking change (`!` in the conventional subject, or `BREAKING CHANGE` in
     the body) → `major`;
   - `feat:` → `minor`;
   - `fix:`, `perf:`, `refactor:`, `revert:`, `security:` → `patch`;
   - only `docs:`, `chore:`, `ci:`, `test:`, `build:`, `style:` (or
     unparseable subjects) → no release.

## Required Repository Secret

`RELEASE_AUTOMATION_PAT` must be able to push commits and tags to this
repository (`repo` scope for a classic PAT). A PAT-authorized tag push is what
triggers `publish.yml` reliably.

`changelog-on-main.yml` does **not** check out with the PAT. It checks out with
`persist-credentials: false`, so nothing in `.git/config` carries a push-capable
credential, and passes the PAT only to the two `git push` steps, through
`GIT_CONFIG_*` in that step's environment. That job installs the package and
imports it, which executes code resolved from PyPI; a push-capable token sitting
in `.git/config` for those steps is a much larger blast radius than the
automation needs. `release-on-main.yml` does check out with the PAT, because
`scripts/release.py` performs the tag push itself and needs the credential to
survive into that subprocess.

## Troubleshooting

### Release was skipped with missing PAT warning

Symptoms:

- `release_skipped_missing_pat` job runs, logging
  `Release criteria matched but RELEASE_AUTOMATION_PAT is not configured.`
- `release` job is skipped.

Fix:

1. Add or update `RELEASE_AUTOMATION_PAT` in repo secrets.
2. Re-run the latest `Automated Release On Main` workflow, or push a new commit to `main`.

### Criteria matched but no release happened

Check `decide` job output in `Automated Release On Main`:

- `should_release`
- `bump`
- `reason`
- `has_release_pat`

Expected for an actual release:

- `should_release=true`
- `has_release_pat=true`

### Changelog workflow failed with "Working tree is not clean"

Symptoms:

- `Update Changelog On Main` fails in the **Commit changelog update** step.
- Log shows `Working tree has uncommitted changes outside CHANGELOG.md. Commit or stash them before --commit.`

Cause:

- `--commit` ran against a tree carrying changes other than `CHANGELOG.md`.
  `changelog-on-main.yml` never calls it — the workflow runs the updater, then
  commits with `git add` / `git commit` / `git push` — so a local two-step
  (`update_changelog.py` then `update_changelog.py --commit`) is the usual
  trigger.

Fix:

- Stash or commit the other paths, then commit `CHANGELOG.md`. From a fully
  clean tree, `python scripts/update_changelog.py --commit --push` is a single
  step.

### Release docs guard failed in CI

Symptoms:

- `Release Docs Guard` fails with `Release automation files changed without docs updates`.
- Or it fails with `docs/RELEASE_AUTOMATION.md is missing required markers: …` /
  `README.md is missing the release automation documentation link.`

Fix:

- Update `docs/RELEASE_AUTOMATION.md` or `README.md` in the **same commit** whenever you change release automation scripts or workflows (`release-on-main.yml`, `changelog-on-main.yml`, `update_changelog.py`, `release.py`, `changelog_builder.py`, `release_criteria.py`, `generate_tool_reference.py`). The guard also requires the file `(docs/RELEASE_AUTOMATION.md)` link in `README.md` and a set of required markers in this document.

### Changelog truncated warning in the log

Symptoms:

- `Update Changelog On Main` or the release job logs `[changelog] WARNING: Changelog truncated: kept N of M releasable commits`.
- GitHub annotates the run with `Changelog truncated to N/M commits (limit L); dropped U user-visible and I internal entries.`

Cause:

- The range exceeded `DEFAULT_SUBJECT_LIMIT` (200), or a lower `--limit` was passed.

Fix:

1. Read the `Dropped entries:` list in the log.
2. Re-run with the suggested limit, for example `python scripts/update_changelog.py --limit 400` (or `python scripts/release.py --bump minor --limit 400`).
3. If the warning says user-visible changes were dropped, do not release until they appear in `## Unreleased` — re-run with the higher limit, or add the missing bullets by hand.

## Manual Override

`Automated Release On Main` takes a `workflow_dispatch` `bump` input —
`auto` (default), `patch`, `minor`, `major`. Anything but `auto` sets
`should_release=true` with that bump and skips criteria detection for that run
only. Requires `RELEASE_AUTOMATION_PAT`; without it the run lands in
`release_skipped_missing_pat`.

## Guardrails

- The automation no-ops when no releasable commits exist.
- The release commit/tag itself does not re-trigger a second release, because
  the tag commit carries no releasable subject and the release is scoped to the
  range since the previous tag.
- Stable release publishing remains gated by `publish.yml` quality + novelty
  checks.
- The MCP wire contract is gated separately, by `scripts/smoke_mcp_stdio.py` in
  the `Tests` workflow (3.13/ubuntu only): 8 checks — the negotiated protocol
  version, the published `outputSchema` arms, the published `inputSchema` set,
  the tool name/count cross-check against `docs/TOOLS.md`, a clean
  `memory_status` frame, a clean `memory_recall` frame, a non-ASCII round trip,
  and an unknown argument rejected with `isError: true`. `publish.yml` does
  **not** re-run it, so a release ships only what already passed `Tests` on the
  tagged commit.
- `scripts/release.py` refuses to start on a dirty tree, on a branch other than
  `main`, when the target tag already exists locally or on `origin`, or when the
  version is already on PyPI (`--skip-pypi-check` skips the last check). Any
  failing quality or release gate restores `pyproject.toml` and `CHANGELOG.md`
  and exits non-zero without committing, tagging or pushing. `--dry-run` runs the
  checks and prints the changelog selection report but writes nothing and runs no
  gates; `--no-push` commits and tags, then prints the push command to run by
  hand.
- `ruff` is pinned to `>=0.16.10,<0.17` with an explicit rule set in
  `pyproject.toml`. `Tests` and `Publish to PyPI` both lint `src/` and `tests/`.
  `scripts/` is linted by `scripts/pre_push_check.py` (and locally), so the
  release tooling ships clean without being a CI gate.

## Operational Notes

- A push to `main` runs `Update Changelog On Main`, `Automated Release On Main`,
  `Tests` and `Release Docs Guard`; a `v*` tag runs `Publish to PyPI`. The
  release bot skips the two `main` automation workflows to avoid loops.
- `Release Docs Guard` triggers when a release-automation path changes — the seven in
  `RELEASE_AUTOMATION_PATHS` plus the guard's own `scripts/check_release_docs.py` — or
  when `docs/RELEASE_AUTOMATION.md` / `README.md` change, on both push and pull request.
