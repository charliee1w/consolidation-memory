---
name: release
description: Evaluate and execute a release with version bump, changelog, tag, and GitHub release
argument-hint: "[version] or empty to auto-evaluate"
allowed-tools: Bash, Read, Write, Edit, Grep, Glob
---

# Release consolidation-memory

Releases are automated. A push to `main` runs
[docs/RELEASE_AUTOMATION.md](../../../docs/RELEASE_AUTOMATION.md), which evaluates the
commits since the latest tag and, when eligible, runs
`python scripts/release.py --bump <major|minor|patch>`. That script owns the version bump,
changelog promotion, commit, tag and push, and the resulting `v*` tag triggers
`.github/workflows/publish.yml` for the source guard, quality gates, release gates, build and
GitHub release.

Do not assemble a release by hand. Editing `pyproject.toml` or `CHANGELOG.md` by hand, running
tests as a release gate, tagging by hand or calling `gh release create` bypasses the criteria
and the gates, and a hand-pushed tag races the automation's own tag push into `publish.yml`.

[docs/RELEASE_GATES.md](../../../docs/RELEASE_GATES.md) documents what the gates require and
what a green release proves.

## If No Version Argument

`$ARGUMENTS` is empty: report whether a release is warranted, using the same deterministic
engine the workflow uses.

```bash
python scripts/release_criteria.py
```

It prints `should_release`, `bump`, `reason` and the `signals` counts for the commits since the
latest tag. The criteria, in order:

1. The **head** commit subject or body contains `[skip release]` or `[release skip]` → no release.
2. The head commit contains `[release major|minor|patch]` or `[bump …]` → that forced bump.
3. Otherwise the highest conventional-commit signal across the whole range wins:
   - `!` after the type/scope in the subject, or `BREAKING CHANGE` in the body → `major`
   - `feat:` → `minor`
   - `fix:`, `perf:`, `refactor:`, `revert:`, `security:` → `patch`
   - a range of only `docs:`, `chore:`, `ci:`, `test:`, `build:`, `style:` (or unparseable)
     subjects → no release

`refactor:` and `security:` are releasable. There is no "internal refactors only, test-only or
minor tooling" skip: a range skips only on a head-commit directive or on having no releasable
signal at all.

Report the verdict, the bump and the reason, then stop. Do not bump a version, write a
changelog entry, tag or open a release.

## If Version Provided

`$ARGUMENTS` is a version: do not execute it as a hand-rolled local release. Two supported
paths.

**Force the bump through the automation (default).** `Automated Release On Main` takes a
`workflow_dispatch` `bump` input — `auto`, `patch`, `minor`, `major` — and anything but `auto`
forces that bump and skips criteria detection for that run.

```bash
gh workflow run "Automated Release On Main" --ref main -f bump=patch
```

This needs the `RELEASE_AUTOMATION_PAT` repo secret; without it the run lands in
`release_skipped_missing_pat`. When an explicit `X.Y.Z` was requested, check it is the version
`main` is at plus the chosen bump, then dispatch that bump.

**Preview the target version locally.** `release.py --dry-run` is a plan preview only: it
validates the clean tree, the `main` branch requirement and the tag/PyPI collision probe,
computes the target version and the changelog range, then prints steps `[1/9]`…`[9/9]` with the
mutating ones empty. It runs no quality gate, no release gate and no build, and it exits
non-zero off `main`.

```bash
python scripts/release.py --bump patch --dry-run
```

Use it to check the target version, not to verify a release. Without `--dry-run`, `release.py`
must run from a clean `main` and installs `.[fastembed,rest,dev]`: tests with coverage, builder
smoke, the `ResourceWarning` gate, `ruff check src/ tests/`, `mypy src/consolidation_memory/`,
`bandit -q -ll -r src scripts -s B608,B110`, the full novelty evaluation plus
`scripts/verify_release_gates.py`, then `python -m build` + `twine check --strict` — rolling
back `pyproject.toml` and `CHANGELOG.md` on any failure before it bumps, commits, tags and
pushes.

## Changelog

`scripts/release.py` promotes the `## Unreleased` bullets into the versioned entry, falling back
to the commit subjects since the previous tag, and ranks entries before capping them at
`DEFAULT_SUBJECT_LIMIT` in `scripts/changelog_builder.py`. The release sections are
`CATEGORY_ORDER` in that module — read the list there instead of restating it, so a dependency
floor bump keeps its own `Dependencies` section. Raise `--limit` when a run logs
`[changelog] WARNING: Changelog truncated`, and do not release while the warning says
user-visible changes were dropped.

Never hand-write the version entry, and never put test counts or dated status statements in a
changelog section.
