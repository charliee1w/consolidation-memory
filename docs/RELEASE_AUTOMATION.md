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

## Guardrails

- The automation no-ops when no releasable commits exist.
- The release commit/tag itself does not re-trigger a second release, because there are no commits past the new tag.
- Stable release publishing remains gated by `publish.yml` quality + novelty checks.

## Manual Override

`release-on-main.yml` also supports `workflow_dispatch` with optional forced bump:

- `patch`
- `minor`
- `major`
