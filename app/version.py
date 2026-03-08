"""Version helpers for the MLX OpenAI Server package."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import tomllib
from typing import Final

PACKAGE_NAME: Final[str] = "mlx-openai-server"


def _read_version_from_pyproject() -> str:
    """Read the project version directly from ``pyproject.toml``.

    Returns
    -------
    str
        Version string defined in the project's ``[project]`` table.

    Raises
    ------
    FileNotFoundError
        If ``pyproject.toml`` cannot be found relative to this module.
    KeyError
        If the expected ``project.version`` key is missing.
    """

    pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
    with pyproject_path.open("rb") as pyproject_file:
        project_data = tomllib.load(pyproject_file)
    return project_data["project"]["version"]


def get_version() -> str:
    """Return the current package version.

    Returns
    -------
    str
        Installed package version when available, otherwise the version
        declared in the local ``pyproject.toml`` file.
    """

    try:
        return version(PACKAGE_NAME)
    except PackageNotFoundError:
        return _read_version_from_pyproject()


__version__: Final[str] = get_version()
