---
name: release
description: Evaluate and execute a release with version bump, changelog, tag, and GitHub release
argument-hint: "[version] or empty to auto-evaluate"
allowed-tools: Bash, Read, Write, Edit, Grep, Glob
---

# Release consolidation-memory

## If No Version Argument

If `$ARGUMENTS` is empty, evaluate whether a release is warranted:

1. Find the latest git tag: `git describe --tags --abbrev=0`
2. List commits since that tag: `git log <tag>..HEAD --oneline`
3. Categorize changes (features, fixes, refactors, docs, tests)
4. Apply release criteria:
   - **Release**: new features, noteworthy bug fixes, pre-refactor checkpoint
   - **Skip**: internal refactors only, test-only, docs-only, minor tooling
5. If warranted, recommend a version number and ask for confirmation
6. If not warranted, explain why and stop

## If Version Provided

Execute release for version `$ARGUMENTS`:

1. Verify working tree is clean: `git status --porcelain`
2. Pull latest: `git pull --ff-only origin main`
3. Read current version from `pyproject.toml`
4. Bump version in `pyproject.toml` to `$ARGUMENTS`
5. Read `CHANGELOG.md` and the commits since last tag
6. Write a changelog entry for the new version matching the existing format:
   - Date: today's date (YYYY-MM-DD)
   - Sections as appropriate: Features, Bug Fixes, Refactoring, Performance, Security, Documentation, Internal
   - Include test count if it changed
7. Run tests: `python -m pytest tests/ -v`
8. Run linter: `python -m ruff check src/ tests/`
9. If tests/lint pass, commit: `git add pyproject.toml CHANGELOG.md && git commit -m "v$ARGUMENTS"`
10. Tag: `git tag v$ARGUMENTS`
11. Push: `git push origin main && git push origin v$ARGUMENTS`
12. Create GitHub release: `gh release create v$ARGUMENTS --title "v$ARGUMENTS" --notes-from-tag`

If tests or lint fail, stop and report the failures. Do NOT commit broken code.
