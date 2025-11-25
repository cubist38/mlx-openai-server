"""Tests for hub status metadata aggregation."""

from __future__ import annotations

from pathlib import Path

from app.api.hub_routes import _build_models_from_config
from app.config import MLXServerConfig
from app.hub.config import MLXHubConfig


def _make_config(tmp_path: Path) -> MLXHubConfig:
    return MLXHubConfig(
        log_path=tmp_path / "logs",
        models=[MLXServerConfig(model_path="/models/foo", name="foo", model_type="lm")],
    )


def _live_snapshot(pid: int = 4321) -> dict[str, object]:
    return {
        "models": [
            {
                "name": "foo",
                "state": "running",
                "pid": pid,
                "log_path": "/tmp/foo.log",
                "started_at": 1,
            },
        ],
    }


def test_build_models_defaults_to_process_state(tmp_path: Path) -> None:
    """Models default to process state when runtime context is absent."""
    config = _make_config(tmp_path)

    models, counts = _build_models_from_config(config, _live_snapshot())
    metadata = models[0].metadata
    assert metadata is not None

    assert metadata["process_state"] == "running"
    assert metadata["memory_state"] is None
    assert metadata["status"] == "running"
    assert counts.started == 1
    assert counts.loaded == 1
