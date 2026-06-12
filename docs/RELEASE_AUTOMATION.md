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
2. `changelog-on-main.yml` refreshes the `## Unreleased` section in `CHANGELOG.md` and, when the file changed, commits and pushes it with `[skip release]`.
   - Step 1 runs `python scripts/update_changelog.py` to rewrite `## Unreleased`.
   - Step 2 runs `git add` / `git commit` / `git push` on the already-updated file (it does **not** re-run `--commit`, because step 1 leaves `CHANGELOG.md` dirty).
3. `release-on-main.yml` evaluates commits since the latest tag.
4. If eligible, it runs `scripts/release.py --bump <major|minor|patch>`.
5. The script bumps `pyproject.toml`, promotes `## Unreleased` into a versioned entry (or falls back to git commits since the tag), commits, tags (`vX.Y.Z`), and pushes.
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
   - Trigger `workflow_dispatch` on `Automated Release On Main` with `patch`, `minor`, or `major`, or
   - Push another releasable conventional commit and wait for the release job.

If both workflows run successfully, setup is complete.

## Local Changelog Preview

Refresh the unreleased section without releasing:

```bash
python scripts/update_changelog.py --dry-run
python scripts/update_changelog.py
```

Commit locally when ready:

```bash
python scripts/update_changelog.py --commit
python scripts/update_changelog.py --commit --push
```

`--commit` allows a dirty tree when **only** `CHANGELOG.md` changed (for example after running the updater once, then committing in a second command). Other uncommitted files still block `--commit`.

## Criteria

The criteria engine is deterministic:

1. Head commit contains `[skip release]` -> no release.
2. Head commit contains `[release major|minor|patch]` -> forced bump.
3. Otherwise, scan commits since latest tag:
- Breaking change (`!` in conventional subject or `BREAKING CHANGE` in body) -> `major`.
- `feat:` -> `minor`.
- `fix:`, `perf:`, `refactor:`, `revert:`, `security:` -> `patch`.
- `docs:`, `chore:`, `ci:`, `test:`, `build:`, `style:` only -> no release.

## Required Repository Secret

Set repository secret:

- `RELEASE_AUTOMATION_PAT`

Use a PAT that can push commits and tags to this repository (`repo` scope for classic PAT).
This is required so tag pushes can trigger downstream workflows reliably.

## Troubleshooting

### Release was skipped with missing PAT warning

Symptoms:

- `release_skipped_missing_pat` job runs.
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
- Log shows `Working tree is not clean. Commit or stash changes before --commit.`

Cause:

- An older workflow reran `update_changelog.py --commit --push` after step 1 had already written `CHANGELOG.md`.

Fix:

- Ensure `changelog-on-main.yml` commits via `git add` / `git commit` / `git push` after the refresh step, or run a single local `python scripts/update_changelog.py --commit --push` from a clean tree.

### Release docs guard failed in CI

Symptoms:

- `Release Docs Guard` fails with `Release automation files changed without docs updates`.

Fix:

- Update `docs/RELEASE_AUTOMATION.md` or `README.md` in the **same commit** whenever you change release automation scripts or workflows (`release-on-main.yml`, `changelog-on-main.yml`, `update_changelog.py`, etc.).

### How to force one release now

Use `workflow_dispatch` on `Automated Release On Main` and select:

- `patch`
- `minor`
- `major`

This bypasses auto detection for that run only.

## Guardrails

- The automation no-ops when no releasable commits exist.
- The release commit/tag itself does not re-trigger a second release, because there are no commits past the new tag.
- Stable release publishing remains gated by `publish.yml` quality + novelty checks.

## Manual Override

`release-on-main.yml` also supports `workflow_dispatch` with optional forced bump:

- `patch`
- `minor`
- `major`

## Operational Notes

- Commit directive `[skip release]` on the head commit suppresses release.
- Commit directive `[release major|minor|patch]` on the head commit forces a bump.
- Release commit/tag pushes trigger downstream workflows:
  - `Tests` on `main`
  - `Publish to PyPI` on tag `v*`
