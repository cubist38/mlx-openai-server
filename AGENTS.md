# AI Agent Development Guidelines

This document provides comprehensive guidelines for AI coding agents contributing to the mlx-openai-server project. Following these standards ensures consistency, maintainability, and quality across the codebase.

---

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [Code Quality & Style](#code-quality--style)
3. [Python Best Practices](#python-best-practices)
4. [Error Handling & Logging](#error-handling--logging)
5. [Version Control & Collaboration](#version-control--collaboration)
6. [Testing Requirements](#testing-requirements)
7. [Transparency & Communication](#transparency--communication)

---

## Environment Setup

### Virtual Environment

- **Repository-local environment**: Agents may create and use a virtual environment located at `./.venv` in the repository root.
- **Python interpreter**: All Python commands, scripts, and tests **must** use the interpreter at `./.venv/bin/python`. Never use the system Python or global environment.
- **Activation**: When running terminal commands, explicitly invoke `./.venv/bin/python` or `./.venv/bin/<tool>` rather than relying on shell activation.

### Package Management

- **Manifest-based installation**: Agents are permitted to install packages declared in repository manifests (e.g., `pyproject.toml`, `requirements.txt`) into the local `.venv` without requiring additional explicit approval for each installation.
- **Global environment prohibition**: Under no circumstances should packages be installed into the system or global Python environment. All dependencies must remain isolated within `./.venv`.
- **Dependency updates**: When adding new dependencies, update the appropriate manifest file (`pyproject.toml` preferred) and document the reason in commit messages or PR descriptions.

---

## Code Quality & Style

### Linting & Formatting Tools

This project uses the following tools to maintain consistent code quality:

- **ruff**: Automatic code formatter with options configured in `pyproject.toml`

### Formatting Workflow

Before committing code changes:

```bash
./.venv/bin/ruff check --fix <file_or_directory>
./.venv/bin/ruff format <file_or_directory>
```

Alternatively, format the entire project:

```bash
./.venv/bin/ruff check --fix app/ tests/
./.venv/bin/ruff format app/ tests/
```

---

## Python Best Practices

### Type Annotations

- **Mandatory typing**: Add type annotations to all function signatures, method signatures, and class attributes.
- **Return types**: Always specify return types, including `None` when applicable.
- **Forward references**: Use `from __future__ import annotations` to defer evaluation of type annotations, allowing forward references without string literals.
- **Python 3.11+ type hints**: Use built-in generic types instead of typing module equivalents (e.g., `dict[str, Any]` instead of `Dict[str, Any]`, `list[str]` instead of `List[str]`).

**Example:**

```python
from __future__ import annotations  # Modern approach: defers evaluation, no string literals needed
from typing import Any

def process_request(
    request_id: str, 
    data: dict[str, Any], 
    timeout: float | None = None
) -> list[Response]:
    """Process a request and return results."""
    ...

class Response:
    def __init__(self, status: str) -> None:
        self.status = status

# Example of self-referential type (forward reference)
class Node:
    def __init__(self, value: str, children: list[Node] | None = None) -> None:
        self.value = value
        self.children = children or []

# Alternative with string literals (for older Python or when not using __future__ import):
# class Node:
#     def __init__(self, value: str, children: list["Node"] | None = None) -> None:
#         self.value = value
#         self.children = children or []
```

### Documentation

- **Docstrings required**: All modules, classes, and methods must have descriptive docstrings.
- **Docstring format**: Use triple double-quotes. Method docstrings must follow NumPy style, with sections for Parameters, Returns, and Raises where applicable. The first line should be a concise summary; additional lines may include extended descriptions.
- **Update existing docstrings**: If modifying a function's behavior, update its docstring to reflect changes.

**Example:**

```python
def calculate_metrics(data: list[float], threshold: float) -> dict[str, float]:
    """Calculate statistical metrics for the provided data.
    
    Parameters
    ----------
    data : list[float]
        List of numerical values to analyze.
    threshold : float
        Minimum value threshold for filtering.
        
    Returns
    -------
    dict[str, float]
        Dictionary containing mean, median, and standard deviation.
        
    Raises
    ------
    ValueError
        If data list is empty or threshold is negative.
    """
    ...
```

### Code Organization

- **Import placement**: All import statements must appear at the top of the file, grouped as follows:
  1. Standard library imports
  2. Third-party library imports
  3. Local application imports
  
  Use blank lines to separate groups.

- **Preserve comments**: Retain all existing inline comments and block comments when editing code. They provide valuable context for future maintainers.

- **Modularity**: Break complex functions into smaller, testable units. Aim for single-responsibility principles.

---

## Error Handling & Logging

### Exception Handling

- **Specific exceptions**: Never catch bare `Exception` unless absolutely necessary. Always catch the most specific exception type(s) relevant to the operation.
- **Multiple exception types**: Use tuple syntax when catching multiple exceptions: `except (TypeError, ValueError) as e:`
- **Re-raise when appropriate**: If you catch an exception but cannot handle it meaningfully, re-raise it or raise a more descriptive custom exception.

**Example:**

```python
# ❌ Avoid this
try:
    result = risky_operation()
except Exception as e:
    logger.error(f"Something went wrong: {e}")

# ✅ Prefer this
try:
    result = risky_operation()
except (FileNotFoundError, PermissionError) as e:
    logger.error(f"File access error: {e}")
    raise
except json.JSONDecodeError as e:
    logger.warning(f"Invalid JSON response: {e}")
    return None
```

### Logging Practices

- **Use loguru**: 
  - The project uses `loguru` for logging. Import it as `from loguru import logger`.
  - Use f-string format for logging strings.
- **Appropriate log levels**:
  - `logger.debug()`: Detailed diagnostic information
  - `logger.info()`: General informational messages (startup, shutdown, major operations)
  - `logger.warning()`: Potentially problematic situations that don't prevent operation
  - `logger.error()`: Errors that prevent a specific operation from completing
  - `logger.critical()`: Severe errors that may cause the application to terminate
  
- **Contextual information**: Include relevant context in log messages (request IDs, file paths, configuration values, etc.).

**Example:**

```python
logger.info(f"Loading model from {model_path}")
try:
    model = load_model(model_path)
    logger.info(f"Model loaded successfully: {model.name}")
except ModelLoadError as e:
    logger.error(f"Failed to load model from {model_path}: {e}")
    raise
```

---

## Version Control & Collaboration

### Branch & PR Creation

- **Explicit user request required**: Only create new branches or open pull requests when the user explicitly asks for it **or** when the user includes the hashtag `#github-pull-request-agent` in their request.
- **Asynchronous agent handoff**: The `#github-pull-request-agent` hashtag signals that the task should be handed off to the asynchronous GitHub Copilot coding agent after all planning, analysis, and preparation are complete.
- **Default behavior**: By default, work directly on the current branch and commit changes locally without creating PRs.

### Commit Messages

- Use clear, descriptive commit messages that explain the "what" and "why" of changes.
- Follow conventional commit format when possible: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`

**Examples:**

```text
feat: add JIT model loading and idle auto-unload support
fix: resolve race condition in handler manager locking
docs: update README with new CLI options for JIT mode
test: add config validation tests for auto-unload settings
```

---

## Testing Requirements

### Test Coverage

- **New features**: All new functionality must include corresponding unit tests in the `tests/` directory.
- **Bug fixes**: When fixing a bug, add a regression test to prevent recurrence.
- **Run tests locally**: Before committing, run the test suite using:
  
  ```bash
  ./.venv/bin/python -m pytest tests/
  ```

- **Test organization**: Mirror the structure of `app/` in `tests/`, e.g., tests for `app/config.py` go in `tests/test_config.py`.

### Testing Best Practices

- Use `pytest` fixtures for setup/teardown and shared test data.
- Use descriptive test function names: `test_auto_unload_requires_jit_when_configured()`
- Mock external dependencies (network calls, file I/O) to keep tests fast and reliable.
- Verify both success paths and error conditions.
- **Parameterized tests**: Use parameterized tests to test multiple inputs efficiently, reducing code duplication.

---

## Transparency & Communication

### Deviation Disclosure

When an agent cannot or chooses not to follow one or more guidelines in this document, it **must** explicitly disclose:

1. **Which guideline(s)** are not being followed
2. **Concise reason** for the deviation

**Common reasons for deviations:**

- System or developer instruction conflicts
- Missing permissions or credentials
- Missing development dependencies (e.g., linters not installed)
- User-denied install or network access
- Safety policy restrictions

**Example disclosure:**

> **⚠️ Deviation Notice:**  
> The code was not formatted with ruff because the dev dependencies are not installed in the current environment. Run `./.venv/bin/pip install -e '.[dev]'` to enable linting/formatting tools.

### Communication Principles

- **Be explicit**: Clearly state what changes were made, which files were modified, and any side effects.
- **Provide context**: When suggesting alternative approaches, explain trade-offs and reasoning.
- **Highlight risks**: If a change introduces potential risks or requires manual verification, call it out.
- **Next steps**: After completing work, suggest logical next steps (e.g., running tests, reviewing logs, checking for regressions).

---

## Additional Guidelines

### Documentation Updates

- **README.md**: Update the README when adding significant features, changing CLI options, or modifying installation steps.
- **Inline documentation**: Maintain clear, up-to-date comments for complex algorithms, configuration settings, and architectural decisions.
- **API documentation**: If exposing new API endpoints, document request/response schemas and usage examples.

### Performance Considerations

- **Profiling**: When optimizing performance-critical code, measure before and after to quantify improvements.
- **Memory management**: Be mindful of memory usage, especially when loading large models. Use appropriate cleanup (`mx.clear_cache()`, `gc.collect()`) where needed.
- **Async patterns**: Leverage async/await for I/O-bound operations to maintain responsiveness.

### Security

- **Input validation**: Validate all user inputs, especially file paths and model parameters.
- **Secrets management**: Never hardcode API keys, tokens, or credentials. Use environment variables or secure configuration files.
- **Dependency auditing**: Periodically review dependencies for known vulnerabilities.

---

## Summary Checklist

Before finalizing any code contribution, verify:

- ✅ Virtual environment (`./.venv`) is used for all operations
- ✅ Code passes ruff linting and formatting
- ✅ Type annotations are present on all functions/methods
- ✅ Docstrings follow NumPy style conventions
- ✅ Specific exceptions are caught (not bare `Exception`)
- ✅ Appropriate logging is in place
- ✅ Existing comments are preserved
- ✅ Imports are organized at the top of files
- ✅ Tests are written and passing
- ✅ Documentation is updated (README, docstrings, etc.)
- ✅ No branches/PRs created unless explicitly requested
- ✅ Any deviations from guidelines are disclosed

---

**Questions or suggestions?** Open an issue or discuss in the project's communication channels.
