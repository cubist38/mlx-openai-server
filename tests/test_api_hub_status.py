"""Tests for hub status metadata aggregation."""

from __future__ import annotations

from pathlib import Path

from app.api.hub_routes import _build_models_from_config
from app.config import MLXServerConfig
from app.hub.config import MLXHubConfig
from app.hub.runtime import HubRuntime


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
            }
        ]
    }


def test_build_models_includes_runtime_metadata(tmp_path: Path) -> None:
    """Runtime-provided memory state should be merged into metadata."""

    config = _make_config(tmp_path)
    runtime = HubRuntime(config)
    runtime.mark_loading("foo")
    runtime.mark_loaded("foo")

    models, counts = _build_models_from_config(config, _live_snapshot(), runtime)
    metadata = models[0].metadata
    assert metadata is not None

    assert metadata["process_state"] == "running"
    assert metadata["memory_state"] == "loaded"
    assert metadata["status"] == "loaded"  # prefers memory state when available
    assert metadata["memory_last_transition_at"] is not None
    assert counts.started == 1
    assert counts.loaded == 1


def test_build_models_without_runtime_defaults_to_process(tmp_path: Path) -> None:
    """When runtime context is absent, fall back to process state."""

    config = _make_config(tmp_path)

    models, counts = _build_models_from_config(config, _live_snapshot(), runtime=None)
    metadata = models[0].metadata
    assert metadata is not None

    assert metadata["process_state"] == "running"
    assert metadata["memory_state"] is None
    assert metadata["status"] == "running"
    assert counts.started == 1
    assert counts.loaded == 1
