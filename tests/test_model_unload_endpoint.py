"""Tests for explicit model unload control-plane behavior."""

from __future__ import annotations

from http import HTTPStatus
import types
from typing import Any

import pytest

from app.api import endpoints
from app.core.model_registry import ModelRegistry


class _FakeHandler:
    """Minimal handler exposing ``cleanup`` used by unload tests."""

    def __init__(self) -> None:
        self.cleanup_calls: int = 0

    async def cleanup(self) -> None:
        """Track cleanup invocations."""
        self.cleanup_calls += 1


def _build_request(registry: ModelRegistry) -> Any:
    """Build a minimal request object with app-state registry."""
    return types.SimpleNamespace(
        app=types.SimpleNamespace(state=types.SimpleNamespace(registry=registry))
    )


@pytest.mark.asyncio
async def test_unload_endpoint_unloads_active_on_demand_model() -> None:
    """Endpoint should unload a currently loaded on-demand model and run cleanup."""
    registry = ModelRegistry()
    await registry.register_on_demand_model(
        model_id="phi-4-reasoning-plus",
        model_cfg_dict={},
        model_type="lm",
        model_path="/tmp/models/phi-4",
        context_length=None,
        queue_config={},
        idle_timeout=60,
    )
    handler = _FakeHandler()
    registry._handlers["phi-4-reasoning-plus"] = handler
    registry._on_demand_loaded.add("phi-4-reasoning-plus")
    registry._on_demand_ref_count["phi-4-reasoning-plus"] = 0

    response = await endpoints.unload_model("phi-4-reasoning-plus", _build_request(registry))

    assert response.status_code == HTTPStatus.OK
    payload = response.json()
    assert payload == {"unloaded": True, "model": "phi-4-reasoning-plus"}
    assert handler.cleanup_calls == 1
    assert "phi-4-reasoning-plus" not in registry._handlers


@pytest.mark.asyncio
async def test_unload_endpoint_returns_false_when_not_loaded() -> None:
    """Endpoint should report a no-op when the model is not currently loaded."""
    registry = ModelRegistry()
    await registry.register_on_demand_model(
        model_id="phi-4-reasoning-plus",
        model_cfg_dict={},
        model_type="lm",
        model_path="/tmp/models/phi-4",
        context_length=None,
        queue_config={},
        idle_timeout=60,
    )

    response = await endpoints.unload_model("phi-4-reasoning-plus", _build_request(registry))

    assert response.status_code == HTTPStatus.OK
    payload = response.json()
    assert payload == {
        "unloaded": False,
        "model": "phi-4-reasoning-plus",
        "reason": "not currently loaded",
    }
    assert "phi-4-reasoning-plus" not in registry._handlers


@pytest.mark.asyncio
async def test_unload_endpoint_returns_404_for_unregistered_model() -> None:
    """Endpoint should return not found for unknown model names."""
    registry = ModelRegistry()

    response = await endpoints.unload_model("not-registered", _build_request(registry))

    assert response.status_code == HTTPStatus.NOT_FOUND
    payload = response.json()
    assert payload["error"]["code"] == HTTPStatus.NOT_FOUND.value
    assert payload["error"]["type"] == "model_not_found"


@pytest.mark.asyncio
async def test_unload_endpoint_does_not_unload_model_in_use() -> None:
    """Endpoint should keep a busy model mounted and return a reason."""
    registry = ModelRegistry()
    await registry.register_on_demand_model(
        model_id="phi-4-reasoning-plus",
        model_cfg_dict={},
        model_type="lm",
        model_path="/tmp/models/phi-4",
        context_length=None,
        queue_config={},
        idle_timeout=60,
    )
    handler = _FakeHandler()
    registry._handlers["phi-4-reasoning-plus"] = handler
    registry._on_demand_loaded.add("phi-4-reasoning-plus")
    registry._on_demand_ref_count["phi-4-reasoning-plus"] = 1

    response = await endpoints.unload_model("phi-4-reasoning-plus", _build_request(registry))

    assert response.status_code == HTTPStatus.OK
    payload = response.json()
    assert payload == {
        "unloaded": False,
        "model": "phi-4-reasoning-plus",
        "reason": "model currently in use",
    }
    assert handler.cleanup_calls == 0
    assert "phi-4-reasoning-plus" in registry._handlers
