"""Tests for the CLI hub live status helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.cli import _print_hub_status, _print_watch_snapshot
from app.config import MLXServerConfig
from app.hub.config import MLXHubConfig


@pytest.fixture
def hub_config(tmp_path: Path) -> MLXHubConfig:
    """Provide a hub config with an isolated log directory."""

    log_dir = tmp_path / "logs"
    return MLXHubConfig(
        log_path=log_dir,
        models=[
            MLXServerConfig(model_path="/models/a", name="alpha", model_type="lm", group="tier")
        ],
    )


def test_print_hub_status_includes_live_state(
    capsys: pytest.CaptureFixture[str], hub_config: MLXHubConfig
) -> None:
    """_print_hub_status should surface live state and PIDs when provided."""

    live = {"models": [{"name": "alpha", "state": "running", "pid": 4321, "group": "tier"}]}
    _print_hub_status(hub_config, live_status=live)
    output = capsys.readouterr().out
    assert "running" in output
    assert "4321" in output
    assert "tier" in output


def test_print_watch_snapshot_handles_empty(capsys: pytest.CaptureFixture[str]) -> None:
    """_print_watch_snapshot should handle empty payloads gracefully."""

    _print_watch_snapshot({"timestamp": 0, "models": []})
    output = capsys.readouterr().out
    assert "no managed processes" in output


def test_print_watch_snapshot_sorts_models(capsys: pytest.CaptureFixture[str]) -> None:
    """Model entries should be sorted alphabetically for readability."""

    snapshot = {
        "timestamp": 0,
        "models": [
            {"name": "beta", "state": "running", "group": "tier", "pid": 2},
            {"name": "alpha", "state": "running", "group": None, "pid": 1},
        ],
    }
    _print_watch_snapshot(snapshot)
    output = capsys.readouterr().out
    first_line_index = output.index("alpha")
    second_line_index = output.index("beta")
    assert first_line_index < second_line_index
