"""Tests for hub-aware handler acquisition in API endpoints."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from app.api.endpoints import _get_handler_or_error, _resolve_model_name
from app.hub.controller import HubControllerError


class _DummyController:
    """Test double that mimics HubController's acquire_handler API."""

    def __init__(self, handler: object | None = None) -> None:
        """Store the handler object that will be returned to callers."""

        self.handler = handler or object()
        self.calls: list[tuple[str, str]] = []
        self._error: Exception | None = None

    def set_error(self, exc: Exception) -> None:
        """Configure the controller to raise ``exc`` when invoked."""

        self._error = exc

    async def acquire_handler(self, name: str, reason: str) -> object:
        """Return the configured handler or raise the injected error."""

        if self._error is not None:
            raise self._error
        self.calls.append((name, reason))
        return self.handler


def _build_request(controller: _DummyController | None) -> SimpleNamespace:
    """Construct a shim request object that exposes ``app.state``."""

    state = SimpleNamespace(hub_controller=controller)
    app = SimpleNamespace(state=state)
    return SimpleNamespace(app=app)


def test_get_handler_uses_hub_controller() -> None:
    """Hub-aware helper should delegate to the controller."""

    controller = _DummyController()
    request = _build_request(controller)

    handler, error = asyncio.run(_get_handler_or_error(request, "unit-test", model_name=" alpha "))

    assert handler is controller.handler
    assert error is None
    assert controller.calls == [("alpha", "unit-test")]


def test_get_handler_surfaces_controller_errors() -> None:
    """Errors surfaced by the controller should return JSON responses."""

    controller = _DummyController()
    controller.set_error(HubControllerError("boom", status_code=429))
    request = _build_request(controller)

    handler, error = asyncio.run(_get_handler_or_error(request, "unit-test", model_name="beta"))

    assert handler is None
    assert error is not None
    assert error.status_code == 429


def test_get_handler_requires_model_name_for_hub() -> None:
    """Hub mode should require a model name to be provided."""

    controller = _DummyController()
    request = _build_request(controller)

    handler, error = asyncio.run(_get_handler_or_error(request, "unit-test", model_name="  "))

    assert handler is None
    assert error is not None
    assert error.status_code == 400


def test_resolve_model_name_requires_explicit_field_in_hub() -> None:
    """Hub deployments must receive an explicit ``model`` field."""

    controller = _DummyController()
    request = _build_request(controller)

    model, error = _resolve_model_name(
        request,
        "alpha",
        provided_explicitly=False,
    )

    assert model is None
    assert error is not None
    assert error.status_code == 400


def test_resolve_model_name_allows_defaults_when_not_hub() -> None:
    """Outside hub mode, defaults are still accepted for backwards compatibility."""

    request = _build_request(None)

    model, error = _resolve_model_name(
        request,
        "  beta  ",
        provided_explicitly=False,
    )

    assert model == "beta"
    assert error is None
