# Contributing

Thanks for helping improve SpiralFastLoop! This guide documents the local workflow, quality gates, and pull request flow used by the project.

## Local setup

1. Fork and clone the repository.
2. Create and activate a virtual environment (`python -m venv .venv && source .venv/bin/activate`).
3. Install the project in editable mode along with development tooling:

   ```bash
   pip install -e .[extras]
   pip install -r requirements-dev.txt  # if present
   pip install pre-commit
   ```

4. Install the Git hooks and validate the tree:

   ```bash
   pre-commit install
   pre-commit run --all-files
   ```

## Linting and tests

- **Formatting**: enforced via `black` and `isort` (triggered by pre-commit).
- **Static checks**: `flake8` (run automatically via pre-commit).
- **Unit tests**: run with `pytest` from the repository root.
- **Type checks** (optional): `mypy` or `pyright` depending on your setup.

You can run the full suite manually with:

```bash
pytest
pre-commit run --all-files
```

## Pull request flow

1. Create a feature branch from `main`. Use descriptive names such as `feature/trigger-logging` or `bugfix/loader-pin-memory`.
2. Keep commits focused and write clear commit messages that explain *why* a change exists.
3. Ensure pre-commit passes locally and attach any relevant test output in the PR description.
4. Open a PR targeting `main`, filling out the template with context, testing evidence, and any follow-up TODOs.
5. Address review feedback promptly. Squash or rebase as requested before merge.

## Branch strategy

- `main` is always deployable and should pass CI.
- Feature branches are short-lived and should be rebased regularly on top of `main` to avoid drift.
- Release branches (e.g., `release/0.2.x`) are cut when preparing tagged releases; hotfixes targeting production can branch from the latest release tag.

We appreciate your contributions—thank you for improving SpiralFastLoop!
