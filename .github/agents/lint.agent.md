# Lint Agent

This subagent is responsible for running linting and type checking tools (pre-commit and mypy) and automatically fixing any issues found in the mlx-openai-server codebase.

## Purpose

- Ensure code quality by running pre-commit hooks (including ruff linting and formatting)
- Perform static type checking with mypy
- Automatically fix issues where possible
- Report remaining issues that require manual intervention

## Process

1. **Run pre-commit**: Execute `pre-commit run --all-files` to check and auto-fix linting and formatting issues using ruff
2. **Run mypy**: Execute `mypy app/` to perform static type checking
3. **Fix issues**: Automatically apply fixes for common issues (formatting, simple type hints)
4. **Report**: Provide a summary of fixed issues and any remaining problems that need manual attention

## Usage

This subagent should be invoked when:
- Before committing code changes
- After making significant modifications to ensure code quality
- As part of CI/CD pipelines

## Dependencies

- pre-commit (configured in .pre-commit-config.yaml)
- mypy (configured in pyproject.toml)
- Virtual environment at `./.venv` must be activated

## Output

The subagent will:
- Fix formatting and linting issues automatically
- Suggest fixes for type checking errors
- Provide clear reporting of any issues that cannot be auto-fixed
- Ensure the codebase adheres to the project's coding standards as defined in AGENTS.md