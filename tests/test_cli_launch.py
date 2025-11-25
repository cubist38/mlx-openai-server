"""Tests for the main CLI launch command."""

from __future__ import annotations

from click.testing import CliRunner

from app.cli import cli


def test_launch_requires_model_without_hub() -> None:
    """Launch command should fail when no model path is provided."""
    runner = CliRunner()
    result = runner.invoke(cli, ["launch"])  # Missing model-path
    assert result.exit_code != 0
    assert "--model-path" in result.output
