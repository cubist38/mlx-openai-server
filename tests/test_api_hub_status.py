"""Tests for hub status metadata aggregation."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from app.api.hub_routes import _build_models_from_config
from app.hub.config import MLXHubConfig


def test_build_models_defaults_to_process_state(
    make_config: Callable[[], MLXHubConfig],
    live_snapshot: Callable[[int], dict],
) -> None:
    """Models default to process state when runtime context is absent."""
    config = make_config()

    models, counts = _build_models_from_config(config, live_snapshot())
    metadata = models[0].metadata
    assert metadata is not None

    assert metadata["process_state"] == "running"
    assert metadata["memory_state"] is None
    assert metadata["status"] == "running"
    assert counts.started == 1
    assert counts.loaded == 1


@pytest.mark.parametrize(
    ("live_entry", "expected_status", "expected_memory_state", "expected_loaded"),
    [
        # No memory info; running process implies memory loaded
        ({"state": "running"}, "running", None, 1),
        # Explicit memory_state loaded
        ({"state": "running", "memory_state": "loaded"}, "running", "loaded", 1),
        # Explicit memory_state unloaded
        ({"state": "running", "memory_state": "unloaded"}, "running", "unloaded", 0),
        # Legacy boolean memory flag
        ({"state": "inactive", "memory_loaded": True}, "inactive", "loaded", 1),
        # Process stopped but VRAM still loaded
        ({"state": "stopped", "memory_state": "loaded"}, "stopped", "loaded", 1),
    ],
)
def test_build_models_memory_state_variants(
    make_config: Callable[[], MLXHubConfig],
    live_entry: dict,
    expected_status: str,
    expected_memory_state: str | None,
    expected_loaded: int,
) -> None:
    """Parameterize a set of live snapshots to ensure status and memory fields remain consistent."""
    config = make_config()

    snapshot = {"models": [{"name": "foo", **live_entry}]}
    models, counts = _build_models_from_config(config, snapshot)
    metadata = models[0].metadata
    assert metadata is not None

    # status reflects process state
    assert metadata["status"] == expected_status
    # memory_state reflects provided memory information (or None)
    assert metadata["memory_state"] == expected_memory_state
    # loaded count corresponds to whether memory is considered loaded
    assert counts.loaded == expected_loaded
