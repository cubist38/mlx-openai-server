"""Tests for the CLI hub live status helpers."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from app.cli import _print_hub_status, _print_watch_snapshot
from app.config import MLXServerConfig
from app.hub.config import MLXHubConfig


def test_print_hub_status_includes_live_state(
    capsys: pytest.CaptureFixture[str],
    make_hub_config: Callable[..., MLXHubConfig],
) -> None:
    """_print_hub_status should surface live state and PIDs when provided."""
    # Simulate the Model objects returned by the hub_status API
    live = {
        "models": [
            {
                "id": "alpha",
                "object": "model",
                "created": 1234567890,
                "owned_by": "hub",
                "metadata": {
                    "status": "running",
                    "process_state": "running",
                    "memory_state": "loaded",
                    "group": "tier",
                    "default": False,
                    "model_type": "lm",
                    "model_path": "/models/a",
                    "pid": 4321,
                },
            },
        ],
    }
    cfg = make_hub_config(
        models=[
            MLXServerConfig(model_path="/models/a", name="alpha", model_type="lm", group="tier")
        ]
    )
    _print_hub_status(cfg, live_status=live)
    output = capsys.readouterr().out
    assert "running" in output
    assert "4321" in output
    assert "tier" in output


def test_print_watch_snapshot_handles_empty(capsys: pytest.CaptureFixture[str]) -> None:
    """_print_watch_snapshot should handle empty payloads gracefully."""
    _print_watch_snapshot({"timestamp": 0, "models": []})
    output = capsys.readouterr().out
    assert "no managed models" in output


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
    first_pos = output.find("alpha")
    second_pos = output.find("beta")
    assert first_pos >= 0, "Expected 'alpha' to appear in output"
    assert second_pos >= 0, "Expected 'beta' to appear in output"
    assert first_pos < second_pos
