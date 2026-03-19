# Release Automation

This repository supports automated stable releases from `main`.

## Workflow

- Criteria evaluator: `scripts/release_criteria.py`
- Automation workflow: `.github/workflows/release-on-main.yml`
- Publish workflow: `.github/workflows/publish.yml` (tag-driven)
- Release script: `scripts/release.py`

Flow:

1. A push lands on `main`.
2. `release-on-main.yml` evaluates commits since the latest tag.
3. If eligible, it runs `scripts/release.py --bump <major|minor|patch>`.
4. The script updates version/changelog, commits, tags (`vX.Y.Z`), and pushes.
5. The tag triggers `publish.yml`, which runs full gates and publishes release artifacts.

## Quick Setup Checklist

Do this once per repository:

1. Create a classic GitHub PAT with `repo` scope.
2. Add repo secret `RELEASE_AUTOMATION_PAT`.
3. Keep this PAT available to `release-on-main.yml` only (least privilege where possible).
4. Verify with a dry run:
   - Push a conventional commit (for example `fix: ...`) to `main`, or
   - Trigger `workflow_dispatch` with `patch`, `minor`, or `major`.

If the `release` job appears and runs in `Automated Release On Main`, setup is complete.

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
