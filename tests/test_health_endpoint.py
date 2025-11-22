"""Tests for the /health endpoint behavior."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from fastapi.responses import JSONResponse
from mlx_openai_server.api import endpoints
from mlx_openai_server.schemas.openai import HealthCheckResponse, HealthCheckStatus


class _DummyHandler:
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path


class _DummyHandlerManager:
    def __init__(self, handler: _DummyHandler | None) -> None:
        self.current_handler = handler


def _build_state(
    *, handler_manager: _DummyHandlerManager | None, handler: _DummyHandler | None
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
