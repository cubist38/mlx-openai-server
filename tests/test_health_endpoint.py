"""Tests for the /health endpoint behavior."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from fastapi.responses import JSONResponse
from mlx_openai_server.api import endpoints
from mlx_openai_server.schemas.openai import (
    HealthCheckResponse,
    HealthCheckStatus,
    HubStatusResponse,
)


class _DummyHandler:
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path


class _DummyHandlerManager:
    def __init__(self, handler: _DummyHandler | None) -> None:
        self.current_handler = handler


def _build_state(
    *,
    handler_manager: _DummyHandlerManager | None,
    handler: _DummyHandler | None,
    registry: object | None = None,
) -> SimpleNamespace:
    metadata_entry = {
        "id": "test-model",
        "object": "model",
        "created": 1,
        "owned_by": "local",
        "metadata": {},
    }
    return SimpleNamespace(
        handler_manager=handler_manager,
        handler=handler,
        server_config=SimpleNamespace(model_identifier="test-model"),
        model_metadata=[metadata_entry],
        registry=registry,
    )


def test_health_reports_unloaded_when_jit_handler_not_loaded() -> None:
    """Health endpoint returns OK with unloaded status under JIT."""

    state = _build_state(handler_manager=_DummyHandlerManager(handler=None), handler=None)
    request = SimpleNamespace(app=SimpleNamespace(state=state))

    response = asyncio.run(endpoints.health(request))
    assert isinstance(response, HealthCheckResponse)
    assert response.status == HealthCheckStatus.OK
    assert response.model_status == "unloaded"
    assert response.model_id == "test-model"


def test_health_returns_service_unavailable_without_manager() -> None:
    """Without a handler or manager, health should degrade to 503."""

    state = _build_state(handler_manager=None, handler=None)
    request = SimpleNamespace(app=SimpleNamespace(state=state))

    response = asyncio.run(endpoints.health(request))
    assert isinstance(response, JSONResponse)
    assert response.status_code == 503


def test_health_reports_initialized_when_handler_loaded() -> None:
    """When manager holds a handler, health reports initialized state."""

    handler = _DummyHandler("loaded-model")
    state = _build_state(handler_manager=_DummyHandlerManager(handler), handler=None)
    request = SimpleNamespace(app=SimpleNamespace(state=state))

    response = asyncio.run(endpoints.health(request))
    assert isinstance(response, HealthCheckResponse)
    assert response.model_status == "initialized"
    assert response.model_id == "loaded-model"


def test_health_reports_ok_when_controller_present() -> None:
    """Hub controller alone should mark the health endpoint as OK."""

    state = _build_state(handler_manager=None, handler=None)
    state.hub_controller = object()
    request = SimpleNamespace(app=SimpleNamespace(state=state))

    response = asyncio.run(endpoints.health(request))
    assert isinstance(response, HealthCheckResponse)
    assert response.status == HealthCheckStatus.OK
    assert response.model_status == "controller"


def test_hub_status_prefers_registry_snapshot() -> None:
    """Hub status should reflect registry data when available."""

    registry_payload = [
        {
            "id": "alpha",
            "object": "model",
            "created": 111,
            "owned_by": "local",
            "metadata": {"status": "initialized"},
        },
        {
            "id": "beta",
            "object": "model",
            "created": 222,
            "owned_by": "local",
            "metadata": {"status": "unloaded"},
        },
    ]

    class _Registry:
        def list_models(self) -> list[dict[str, object]]:
            return registry_payload

    state = _build_state(handler_manager=None, handler=None, registry=_Registry())
    state.hub_config_path = "/tmp/does-not-exist-hub.yaml"
    request = SimpleNamespace(app=SimpleNamespace(state=state))

    response = asyncio.run(endpoints.hub_status(request))
    assert isinstance(response, HubStatusResponse)
    assert response.counts.registered == 2
    assert response.counts.started == 1
    assert response.counts.loaded == 1
    assert response.warnings == []
    assert response.controller_available is False


def test_hub_status_falls_back_to_cached_metadata() -> None:
    """When registry missing, hub status should use cached metadata with warnings."""

    state = _build_state(handler_manager=None, handler=None, registry=None)
    state.hub_config_path = "/tmp/does-not-exist-hub.yaml"
    request = SimpleNamespace(app=SimpleNamespace(state=state))

    response = asyncio.run(endpoints.hub_status(request))
    assert isinstance(response, HubStatusResponse)
    assert response.counts.registered == 1
    assert response.counts.started == 0
    assert response.warnings  # warning present
    assert response.controller_available is False


def test_hub_status_marks_controller_available_when_present() -> None:
    """Controller flag reports availability when attached to app state."""

    state = _build_state(handler_manager=None, handler=None, registry=None)
    state.hub_config_path = "/tmp/does-not-exist-hub.yaml"
    state.hub_controller = object()
    request = SimpleNamespace(app=SimpleNamespace(state=state))

    response = asyncio.run(endpoints.hub_status(request))
    assert isinstance(response, HubStatusResponse)
    assert response.controller_available is True
