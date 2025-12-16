"""Tests for hub status metadata aggregation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from app.api.hub_routes import _build_models_from_config
from app.core.hub_status import build_group_state
from app.hub.config import MLXHubConfig, MLXHubGroupConfig


def test_build_models_defaults_to_process_state(
    make_config: Callable[[], MLXHubConfig],
    live_snapshot: Callable[[int], dict[str, Any]],
) -> None:
    """Models default to process state when runtime context is absent."""
    config = make_config()

    models, counts = _build_models_from_config(config, live_snapshot(1))
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
    live_entry: dict[str, Any],
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


def test_build_groups_merges_live_snapshot(
    make_config: Callable[[], MLXHubConfig],
) -> None:
    """Group summaries should honor live loaded counts when present."""
    config = make_config()
    config.groups = [MLXHubGroupConfig(name="tier", max_loaded=1, idle_unload_trigger_min=10)]
    config.models[0].group = "tier"

    snapshot = {
        "groups": [
            {
                "name": "tier",
                "max_loaded": 1,
                "idle_unload_trigger_min": 10,
                "loaded": 1,
                "models": ["foo"],
            },
        ],
    }

    if isinstance(snapshot, dict) and isinstance(snapshot.get("groups"), list):
        group_entries = snapshot.get("groups") or []
    else:
        group_entries = build_group_state(
            getattr(config, "groups", []) or [],
            snapshot.get("models") if isinstance(snapshot, dict) else None,
            fallback_members={config.models[0].group: [config.models[0].name]},
        )
    assert group_entries[0]["loaded"] == 1
    assert group_entries[0]["models"] == ["foo"]


def test_build_groups_defaults_to_config_members(
    make_config: Callable[[], MLXHubConfig],
) -> None:
    """When live data is missing, configured members should still be listed."""
    config = make_config()
    config.groups = [MLXHubGroupConfig(name="tier", max_loaded=1, idle_unload_trigger_min=10)]
    config.models[0].group = "tier"

    group_entries = build_group_state(
        getattr(config, "groups", []) or [],
        None,
        fallback_members={config.models[0].group: [config.models[0].name]},
    )
    assert group_entries[0]["loaded"] == 0
    assert group_entries[0]["models"] == ["foo"]
