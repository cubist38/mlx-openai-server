"""Unit tests validating the shared hub status formatting helper."""

from __future__ import annotations

from types import SimpleNamespace

from app.core.hub_status import build_group_state


def test_build_group_state_merges_snapshot_and_fallback_members() -> None:
    """`build_group_state` should merge live snapshots with fallback members."""
    groups = [
        SimpleNamespace(name="tier_one", max_loaded=1, idle_unload_trigger_min=5),
        SimpleNamespace(name="tier_two", max_loaded=2, idle_unload_trigger_min=None),
    ]
    snapshot = [
        {"name": "alpha", "group": "tier_one", "memory_loaded": True},
        {"name": "beta", "group": "tier_one", "memory_loaded": False},
        {"name": "gamma", "group": "tier_two", "memory_loaded": True},
    ]
    fallback = {"tier_two": ["gamma", "delta", " "]}

    result = build_group_state(groups, snapshot, fallback_members=fallback)

    assert result == [
        {
            "name": "tier_one",
            "max_loaded": 1,
            "idle_unload_trigger_min": 5,
            "loaded": 1,
            "models": ["alpha", "beta"],
        },
        {
            "name": "tier_two",
            "max_loaded": 2,
            "idle_unload_trigger_min": None,
            "loaded": 1,
            "models": ["delta", "gamma"],
        },
    ]
