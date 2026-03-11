# Contributing

Thanks for contributing to `consolidation-memory`.

## Development Setup

1. Clone the repository.
2. Create and activate a virtual environment.
3. Install development dependencies:

```bash
pip install -r requirements-dev.txt
```

If you prefer not to use the helper file, the equivalent install is:

```bash
pip install -e ".[all,dev]"
```

## Local Validation

Run these checks before opening a pull request:

```bash
pytest -q
ruff check src tests
mypy src
bandit -q -r src scripts
```

## Pull Requests

1. Create a focused branch from `main`.
2. Keep changes scoped and include tests for behavior changes.
3. Update docs when user-visible behavior or release process changes.
4. Use the repository PR template and open a PR with:
   - Problem statement
   - Summary of changes
   - Test evidence (command output or equivalent)
   - Risk notes when trust, scope, or adapter parity changes

## Commit Style

- Use clear, imperative commit messages.
- Prefer small, reviewable commits.

## Reporting Bugs and Features

- Bug reports and feature requests: [GitHub Issues](https://github.com/charliee1w/consolidation-memory/issues)
- Usage questions and build showcases: [GitHub Discussions](https://github.com/charliee1w/consolidation-memory/discussions)
- Security issues: see [SECURITY.md](SECURITY.md)

## Code of Conduct

By participating, you agree to follow [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
