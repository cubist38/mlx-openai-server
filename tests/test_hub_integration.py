"""Integration-style tests that exercise FastAPI routes in hub mode."""

from __future__ import annotations

from http import HTTPStatus
from types import SimpleNamespace
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from app.api import endpoints
from app.hub.controller import HubControllerError
from app.server import configure_fastapi_app


class _StubLMHandler:
    """Minimal stand-in for MLXLMHandler used by hub routing tests."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.model_path = f"/fake/{name}"
        self.calls: list[str] = []

    async def generate_text_response(
        self, request: endpoints.ChatCompletionRequest
    ) -> dict[str, Any]:
        """Record the request and return a deterministic response payload."""

        self.calls.append(request.model)
        return {"response": f"{self.name}-reply", "usage": None}


class _StaticRegistry:
    """Registry double that returns pre-baked model entries."""

    def __init__(self, models: list[dict[str, Any]]) -> None:
        self._models = models

    def list_models(self) -> list[dict[str, Any]]:
        return self._models


class _FakeHubController:
    """Controller shim that tracks acquire_handler calls per model."""

    def __init__(self, handlers: dict[str, _StubLMHandler]) -> None:
        self.handlers = handlers
        self.calls: list[tuple[str, str]] = []
        self.load_calls: list[tuple[str, str]] = []
        self.unload_calls: list[tuple[str, str]] = []

    async def acquire_handler(self, name: str, *, reason: str) -> _StubLMHandler:
        self.calls.append((name, reason))
        try:
            return self.handlers[name]
        except KeyError as exc:  # pragma: no cover - defensive guard
            raise HubControllerError(
                f"Model '{name}' is not registered",
                status_code=HTTPStatus.NOT_FOUND,
            ) from exc

    async def load_model(self, name: str, *, reason: str) -> None:
        if name not in self.handlers:
            raise HubControllerError(
                f"Model '{name}' is not registered",
                status_code=HTTPStatus.NOT_FOUND,
            )
        self.load_calls.append((name, reason))

    async def unload_model(self, name: str, *, reason: str) -> None:
        if name not in self.handlers:
            raise HubControllerError(
                f"Model '{name}' is not registered",
                status_code=HTTPStatus.NOT_FOUND,
            )
        self.unload_calls.append((name, reason))


@pytest.fixture
def hub_test_client(monkeypatch: pytest.MonkeyPatch) -> tuple[TestClient, _FakeHubController]:
    """Return a TestClient configured with hub controller state."""

    app = FastAPI()
    configure_fastapi_app(app)

    handlers = {
        "alpha": _StubLMHandler("alpha"),
        "beta": _StubLMHandler("beta"),
    }
    controller = _FakeHubController(handlers)

    registry_payload = [
        {
            "id": name,
            "object": "model",
            "created": 1,
            "owned_by": "local",
            "metadata": {"status": "unloaded", "model_type": "lm", "model_path": f"/fake/{name}"},
        }
        for name in handlers
    ]

    app.state.hub_controller = controller
    app.state.registry = _StaticRegistry(registry_payload)
    app.state.server_config = SimpleNamespace(host="0.0.0.0", port=8000, enable_status_page=True)
    app.state.model_metadata = registry_payload

    monkeypatch.setattr(endpoints, "MLXLMHandler", _StubLMHandler)

    client = TestClient(app)
    try:
        yield client, controller
    finally:
        client.close()


def test_chat_completion_routes_by_model(
    hub_test_client: tuple[TestClient, _FakeHubController],
) -> None:
    """Each hub request should be routed to the requested model handler."""

    client, controller = hub_test_client

    payload = {
        "model": "alpha",
        "messages": [
            {"role": "user", "content": "ping"},
        ],
    }
    response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == HTTPStatus.OK
    body = response.json()
    assert body["choices"][0]["message"]["content"] == "alpha-reply"
    assert controller.calls == [("alpha", "chat_completions")]

    # Second request targets beta to ensure handlers stay independent
    payload["model"] = "beta"
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == HTTPStatus.OK
    body = response.json()
    assert body["choices"][0]["message"]["content"] == "beta-reply"
    assert controller.calls[-1] == ("beta", "chat_completions")


def test_chat_completion_requires_model_field_when_hub_enabled(
    hub_test_client: tuple[TestClient, _FakeHubController],
) -> None:
    """Hub deployments should reject requests that omit the model field."""

    client, _ = hub_test_client

    payload = {
        "messages": [
            {"role": "user", "content": "ping"},
        ],
    }
    response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == HTTPStatus.BAD_REQUEST
    assert "Model selection is required" in response.text


def test_chat_completion_surfaces_controller_errors(
    hub_test_client: tuple[TestClient, _FakeHubController],
) -> None:
    """Unknown hub model names should propagate the controller's error status."""

    client, _ = hub_test_client

    payload = {
        "model": "missing",
        "messages": [
            {"role": "user", "content": "ping"},
        ],
    }
    response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == HTTPStatus.NOT_FOUND
    assert "not registered" in response.text


def test_hub_html_status_page_enabled(
    hub_test_client: tuple[TestClient, _FakeHubController],
) -> None:
    """GET /hub should return HTML when the status page is enabled."""

    client, _ = hub_test_client
    response = client.get("/hub")

    assert response.status_code == HTTPStatus.OK
    assert "<title>MLX Hub Status" in response.text
    assert "Registered Models" in response.text


def test_hub_html_status_page_disabled(
    hub_test_client: tuple[TestClient, _FakeHubController],
) -> None:
    """Disabling the status page in config should return 404."""

    client, _ = hub_test_client
    client.app.state.server_config.enable_status_page = False

    response = client.get("/hub")

    assert response.status_code == HTTPStatus.NOT_FOUND


def test_models_endpoint_hides_unstarted_entries(
    hub_test_client: tuple[TestClient, _FakeHubController],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GET /v1/models should only list models reported as running."""

    client, _ = hub_test_client

    monkeypatch.setattr(endpoints, "get_running_hub_models", lambda _request: {"alpha"})

    response = client.get("/v1/models")
    assert response.status_code == HTTPStatus.OK
    data = response.json()["data"]
    assert [entry["id"] for entry in data] == ["alpha"]


def test_chat_completion_rejects_unstarted_models(
    hub_test_client: tuple[TestClient, _FakeHubController],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Requests referencing stopped models should return 404."""

    client, _ = hub_test_client
    monkeypatch.setattr(endpoints, "get_running_hub_models", lambda _request: {"alpha"})

    payload = {
        "model": "beta",
        "messages": [
            {"role": "user", "content": "ping"},
        ],
    }

    response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == HTTPStatus.NOT_FOUND
    assert "not started" in response.text
