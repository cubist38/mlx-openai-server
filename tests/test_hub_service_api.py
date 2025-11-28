"""Tests for hub service-backed FastAPI endpoints."""

from __future__ import annotations

from http import HTTPStatus
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
import pytest

from app.api.hub_routes import HubServiceError, hub_router
from app.core.model_registry import ModelRegistry
from app.hub.daemon import HubSupervisor
from app.server import configure_fastapi_app

# Stub classes `_StubServiceState` and `_StubController` are provided by `tests.conftest`

if TYPE_CHECKING:
    # Import test-only types for annotations without causing runtime imports
    from tests.conftest import _StubController, _StubServiceState


@pytest.fixture
def hub_service_app(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stub_service_state: _StubServiceState,
    stub_controller: _StubController,
) -> tuple[TestClient, object, object]:
    """Return a TestClient configured with a stubbed hub service backend."""
    config_dir = tmp_path / "hub-config"
    config_dir.mkdir()
    config_path = config_dir / "hub.yaml"
    config_path.write_text(
        """
host: 0.0.0.0
port: 8123
models:
  - name: alpha
    model_path: /models/alpha
    model_type: lm
    default: true
  - name: beta
    model_path: /models/beta
    model_type: lm
    group: test
""".strip(),
    )

    app = FastAPI()
    configure_fastapi_app(app)
    app.include_router(hub_router)
    app.state.server_config = SimpleNamespace(enable_status_page=True, host="0.0.0.0", port=8123)
    app.state.hub_config_path = config_path
    controller = stub_controller
    app.state.hub_controller = controller

    state = stub_service_state
    state.status_payload["models"] = [
        {
            "name": "alpha",
            "state": "running",
            "pid": 111,
            "group": "default",
            "log_path": str(tmp_path / "alpha.log"),
        },
    ]

    async def _stub_call(
        config: Any,
        method: str,
        path: str,
        *,
        json: Any | None = None,
        timeout: float = 5.0,
    ) -> dict[str, Any]:
        # Emulate the daemon HTTP surface used by the routes.
        if method == "GET" and path == "/health":
            if not state.available:
                raise HubServiceError("unavailable")
            return {"status": "ok"}
        if method == "POST" and path == "/hub/reload":
            if not state.available:
                raise HubServiceError("unavailable")
            state.reload_calls += 1
            return state.reload_result
        if method == "GET" and path == "/hub/status":
            if not state.available:
                raise HubServiceError("unavailable")
            return state.status_payload
        if method == "POST" and path.startswith("/hub/models/") and path.endswith("/start"):
            if not state.available:
                raise HubServiceError("unavailable")
            name = path.split("/")[-2]
            if name == "saturated":
                raise HubServiceError("group full", status_code=HTTPStatus.TOO_MANY_REQUESTS)
            state.start_calls.append(name)
            return {"message": "started"}
        if method == "POST" and path.startswith("/hub/models/") and path.endswith("/stop"):
            if not state.available:
                raise HubServiceError("unavailable")
            name = path.split("/")[-2]
            state.stop_calls.append(name)
            return {"message": "stopped"}
        if method == "POST" and path == "/hub/shutdown":
            if not state.available:
                raise HubServiceError("unavailable")
            state.shutdown_called = True
            return {"message": "shutdown"}
        raise HubServiceError("unhandled stub call")

    monkeypatch.setattr("app.api.hub_routes._call_daemon_api_async", _stub_call)

    def _fake_start_process(path: str, *, host: str | None = None, port: int | None = None) -> int:  # noqa: ARG001
        state.available = True
        return 4321

    monkeypatch.setattr("app.api.hub_routes.start_hub_service_process", _fake_start_process)

    def _fake_stop_controller(_config: Any) -> bool:  # noqa: ANN401
        state.controller_stop_calls += 1
        return True

    monkeypatch.setattr("app.api.hub_routes._stop_controller_process", _fake_stop_controller)

    client = TestClient(app)
    try:
        yield client, state, controller
    finally:
        client.close()


def test_hub_status_uses_service_snapshot(
    hub_service_app: tuple[TestClient, object, object],
) -> None:
    """Hub status endpoint should prefer hub service snapshots when available."""
    client, state, controller = hub_service_app
    state.available = True

    response = client.get("/hub/status")

    assert response.status_code == HTTPStatus.OK
    payload = response.json()
    assert payload["counts"] == {"registered": 2, "started": 1, "loaded": 1}
    assert payload["models"][0]["metadata"]["default"] is True
    assert payload["models"][0]["metadata"]["status"] == "running"
    assert controller.reload_count == 1


def test_hub_status_ok_when_service_unavailable_and_controller_available(
    hub_service_app: tuple[TestClient, object, object],
) -> None:
    """Hub status should be ok when controller is available, even if service is offline."""
    client, state, _controller = hub_service_app
    state.available = False

    response = client.get("/hub/status")
    assert response.status_code == HTTPStatus.OK
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["counts"]["registered"] == 2


def test_hub_model_start_calls_service_client(
    hub_service_app: tuple[TestClient, object, object],
) -> None:
    """Model start endpoint should call controller.start_model."""
    client, state, controller = hub_service_app
    state.available = True

    response = client.post("/hub/models/alpha/start", json={})

    assert response.status_code == HTTPStatus.OK
    assert controller.started == ["alpha"]
    assert state.reload_calls == 0


def test_hub_model_start_surfaces_capacity_errors(
    hub_service_app: tuple[TestClient, object, object],
) -> None:
    """Capacity errors from controller should translate to HTTP 429 responses."""
    client, state, _controller = hub_service_app
    state.available = True

    response = client.post("/hub/models/saturated/start", json={})

    assert response.status_code == HTTPStatus.TOO_MANY_REQUESTS
    body = response.json()
    assert body["error"]["message"] == "429: group full"


def test_hub_service_start_spawns_process_when_missing(
    hub_service_app: tuple[TestClient, object, object],
) -> None:
    """/hub/service/start should spawn the service when it is not running."""
    client, state, _controller = hub_service_app
    state.available = False

    response = client.post("/hub/service/start")

    assert response.status_code == HTTPStatus.OK
    assert state.reload_calls == 1  # reload invoked after start
    body = response.json()
    assert body["action"] == "start"
    assert body["details"]["pid"] == 4321


def test_hub_service_stop_handles_missing_manager(
    hub_service_app: tuple[TestClient, object, object],
) -> None:
    """Stop should still return success when the manager is offline.

    Parameters
    ----------
    hub_service_app : tuple[TestClient, _StubServiceState, _StubController]
        Fixture providing the test client and stubs.
    """
    client, state, _controller = hub_service_app
    state.available = False

    response = client.post("/hub/service/stop")

    assert response.status_code == HTTPStatus.OK
    assert state.shutdown_called is False
    assert state.controller_stop_calls == 1
    payload = response.json()
    assert payload["details"] == {"controller_stopped": True, "manager_shutdown": False}


def test_hub_service_stop_shuts_down_manager_when_available(
    hub_service_app: tuple[TestClient, object, object],
) -> None:
    """Stop should mirror CLI behavior by halting controller and manager.

    Parameters
    ----------
    hub_service_app : tuple[TestClient, _StubServiceState, _StubController]
        Fixture providing the test client and stubs.
    """
    client, state, _controller = hub_service_app
    state.available = True

    response = client.post("/hub/service/stop")

    assert response.status_code == HTTPStatus.OK
    assert state.shutdown_called is True
    assert state.reload_calls == 1
    assert state.controller_stop_calls == 1
    payload = response.json()
    assert payload["details"] == {"controller_stopped": True, "manager_shutdown": True}


def test_hub_service_reload_endpoint_returns_diff(
    hub_service_app: tuple[TestClient, object, object],
) -> None:
    """/hub/service/reload should surface the diff returned by the service."""
    client, state, controller = hub_service_app
    state.available = True

    response = client.post("/hub/service/reload")

    assert response.status_code == HTTPStatus.OK
    body = response.json()
    assert body["action"] == "reload"
    assert body["details"] == {
        "started": [],
        "stopped": ["old_model"],
        "unchanged": ["alpha", "beta"],
    }
    assert controller.reload_count == 1


def test_hub_load_model_invokes_controller(
    hub_service_app: tuple[TestClient, object, object],
) -> None:
    """/hub/models/{model}/load should call the controller."""
    client, _state, controller = hub_service_app

    response = client.post("/hub/models/alpha/load", json={"reason": "dashboard"})

    assert response.status_code == HTTPStatus.OK
    assert controller.loaded == ["alpha"]


def test_hub_memory_actions_surface_controller_errors(
    hub_service_app: tuple[TestClient, object, object],
) -> None:
    """Controller-originated failures should propagate to the client."""
    client, _state, controller = hub_service_app

    response = client.post("/hub/models/denied/load", json={})
    assert response.status_code == HTTPStatus.TOO_MANY_REQUESTS
    payload = response.json()
    assert "group busy" in payload["error"]["message"]

    response = client.post("/hub/models/missing/unload", json={})
    assert response.status_code == HTTPStatus.BAD_REQUEST
    assert controller.unloaded[-1] == "missing"


def test_vram_admin_endpoints_invoke_registry(
    hub_service_app: tuple[TestClient, object, object],
) -> None:
    """Admin VRAM endpoints should call the ModelRegistry on app.state."""
    client, _state, _controller = hub_service_app

    class _StubRegistry:
        def __init__(self) -> None:
            self.loaded: list[str] = []
            self.unloaded: list[str] = []

        async def request_vram_load(
            self,
            name: str,
            *,
            _force: bool = False,
            _timeout: float | None = None,
            **kwargs: Any,
        ) -> None:
            # Accept legacy keyword names for interface compatibility
            if "force" in kwargs:
                _force = kwargs.pop("force")
            if "timeout" in kwargs:
                _timeout = kwargs.pop("timeout")

            if name == "denied":
                # Simulate a validation error
                raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="denied")
            self.loaded.append(name)

        async def request_vram_unload(
            self, name: str, *, _timeout: float | None = None, **kwargs: Any
        ) -> None:
            # Accept legacy keyword name for interface compatibility
            if "timeout" in kwargs:
                _timeout = kwargs.pop("timeout")

            if name == "missing":
                raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="not loaded")
            self.unloaded.append(name)

    registry = _StubRegistry()
    client.app.state.model_registry = registry

    response = client.post("/hub/models/alpha/vram/load", json={})
    assert response.status_code == HTTPStatus.OK
    assert registry.loaded == ["alpha"]

    response = client.post("/hub/models/alpha/vram/unload", json={})
    assert response.status_code == HTTPStatus.OK
    assert registry.unloaded == ["alpha"]


def test_vram_admin_endpoints_surface_registry_errors(
    hub_service_app: tuple[TestClient, object, object],
) -> None:
    """Registry errors should propagate as HTTP responses from the VRAM endpoints."""
    client, _state, _controller = hub_service_app

    class _StubRegistryErr:
        async def request_vram_load(
            self,
            name: str,
            *,
            _force: bool = False,
            _timeout: float | None = None,
            **kwargs: Any,
        ) -> None:
            # Accept legacy keyword names for interface compatibility
            if "force" in kwargs:
                _force = kwargs.pop("force")
            if "timeout" in kwargs:
                _timeout = kwargs.pop("timeout")

            raise HTTPException(status_code=HTTPStatus.TOO_MANY_REQUESTS, detail="group busy")

        async def request_vram_unload(
            self, name: str, *, _timeout: float | None = None, **kwargs: Any
        ) -> None:
            # Accept legacy keyword name for interface compatibility
            if "timeout" in kwargs:
                _timeout = kwargs.pop("timeout")

            raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="not loaded")

    client.app.state.model_registry = _StubRegistryErr()

    response = client.post("/hub/models/denied/vram/load", json={})
    assert response.status_code == HTTPStatus.TOO_MANY_REQUESTS

    response = client.post("/hub/models/missing/vram/unload", json={})
    assert response.status_code == HTTPStatus.BAD_REQUEST


class FakeManager:
    """A minimal fake manager that simulates an unloaded handler."""

    async def unload(self, reason: str) -> bool:
        """Simulate unloading the handler.

        Returns False to indicate there was no loaded handler.
        """
        return False

    def is_vram_loaded(self) -> bool:
        """Return False to indicate VRAM is not loaded."""
        return False


@pytest.mark.asyncio
async def test_stop_model_removes_manager_and_updates_registry() -> None:
    """stop_model should clear the manager when unload() returns False.

    This ensures the supervisor no longer reports the model as running and
    that the registry is updated to reflect no attached handler.
    """
    hub_config = SimpleNamespace(
        models=[
            SimpleNamespace(
                name="m1",
                model_path="m1_path",
                is_default_model=False,
                jit_enabled=True,
                model_type="test",
                host="localhost",
                port=0,
                auto_unload_minutes=None,
            )
        ]
    )

    registry = ModelRegistry()
    # Register the model id used in the hub record so update_model_state can run
    registry.register_model("m1_path", handler=None, model_type="test")

    supervisor = HubSupervisor(hub_config, registry=registry)
    record = supervisor._models["m1"]
    # Attach a fake manager that will return False from unload()
    record.manager = FakeManager()

    result = await supervisor.stop_model("m1")

    assert result.get("status") == "stopped"
    assert record.manager is None
    # Registry handler should be None for the registered model id
    assert registry.get_handler("m1_path") is None
