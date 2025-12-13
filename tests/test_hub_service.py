"""Tests for hub service-backed FastAPI endpoints."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from http import HTTPStatus
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from app.api.hub_routes import _stop_controller_process, hub_router
from app.core.model_registry import ModelRegistry
from app.hub.config import load_hub_config
from app.hub.daemon import HubSupervisor
from app.server import _hub_sync_once

if TYPE_CHECKING:
    # Import test-only types for annotations without causing runtime imports
    from tests.conftest import _StubController, _StubServiceState


@pytest.fixture
def hub_service_app(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stub_service_state: _StubServiceState,
    stub_controller: _StubController,
) -> tuple[TestClient, Any, Any]:
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
    # Do not call `configure_fastapi_app` to avoid starting background
    # hub sync tasks during this focused unit test. We only need the
    # hub routes and minimal app state.
    app.include_router(hub_router)
    app.state.server_config = SimpleNamespace(enable_status_page=True, host="0.0.0.0", port=8123)
    app.state.hub_config_path = config_path
    controller = stub_controller
    # Load the config and set it on the controller so hub_status can use it
    controller.hub_config = load_hub_config(config_path)
    app.state.hub_controller = controller

    state = stub_service_state

    def _fake_stop_controller(_raw_request: Any, _background_tasks: Any) -> bool:  # noqa: ANN401
        state.controller_stop_calls += 1
        return True

    monkeypatch.setattr("app.api.hub_routes._stop_controller_process", _fake_stop_controller)

    client = TestClient(app)
    try:
        yield client, state, controller
    finally:
        client.close()


def test_hub_status_uses_service_snapshot(
    hub_service_app: tuple[TestClient, Any, Any],
) -> None:
    """Hub status endpoint should prefer hub service snapshots when available."""
    client, _state, controller = hub_service_app

    response = client.get("/hub/status")

    assert response.status_code == HTTPStatus.OK
    payload = response.json()
    assert payload["counts"] == {"registered": 2, "started": 1, "loaded": 1}
    assert payload["models"][0]["metadata"]["default"] is True
    assert payload["models"][0]["metadata"]["status"] == "running"
    assert controller.reload_count == 1


def test_hub_status_ok_when_service_unavailable_and_controller_available(
    hub_service_app: tuple[TestClient, Any, Any],
) -> None:
    """Hub status should be ok when controller is available, even if service is offline."""
    client, _state, controller = hub_service_app

    async def _raise() -> dict[str, Any]:  # pragma: no cover - test stub
        raise RuntimeError("status unavailable")

    controller.get_status = _raise  # type: ignore[assignment]

    response = client.get("/hub/status")
    assert response.status_code == HTTPStatus.OK
    payload = response.json()
    assert payload["status"] == "degraded"
    assert payload["counts"]["registered"] == 2


def test_hub_model_start_calls_service_client(
    hub_service_app: tuple[TestClient, Any, Any],
) -> None:
    """Model start endpoint should schedule controller VRAM loads."""
    client, _state, controller = hub_service_app

    response = client.post("/hub/models/alpha/start", json={})

    assert response.status_code == HTTPStatus.OK
    assert controller.scheduled_loads == ["alpha"]


def test_hub_model_start_surfaces_capacity_errors(
    hub_service_app: tuple[TestClient, Any, Any],
) -> None:
    """Capacity errors from controller should translate to HTTP 429 responses."""
    client, _state, _controller = hub_service_app

    response = client.post("/hub/models/saturated/start", json={})

    assert response.status_code == HTTPStatus.TOO_MANY_REQUESTS
    body = response.json()
    assert (
        body["error"]["message"]
        == "429: Group capacity exceeded. Unload another model or wait for auto-unload."
    )


def test_hub_model_start_blocked_when_registry_denies(
    hub_service_app: tuple[TestClient, Any, Any],
) -> None:
    """Start requests should fail fast when the registry rejects the model."""

    client, _state, controller = hub_service_app

    async def _seed_registry() -> None:
        registry = ModelRegistry()
        registry.register_model(
            model_id="/models/alpha",
            handler=None,
            model_type="lm",
            metadata_extras={"group": "shared"},
        )
        registry.register_model(
            model_id="/models/beta",
            handler=None,
            model_type="lm",
            metadata_extras={"group": "shared"},
        )
        await registry.update_model_state(
            "/models/beta",
            metadata_updates={"vram_loaded": True},
        )
        registry.set_group_policies({"shared": {"max_loaded": 1}})
        client.app.state.model_registry = registry

    asyncio.run(_seed_registry())

    response = client.post("/hub/models/alpha/start", json={})

    assert response.status_code == HTTPStatus.TOO_MANY_REQUESTS
    assert controller.scheduled_loads == []


def test_hub_vram_load_blocked_when_registry_denies(
    hub_service_app: tuple[TestClient, Any, Any],
) -> None:
    """VRAM load requests should reuse the cached availability guard."""

    client, _state, controller = hub_service_app

    async def _seed_registry() -> ModelRegistry:
        registry = ModelRegistry()
        registry.register_model(
            model_id="/models/alpha",
            handler=None,
            model_type="lm",
            metadata_extras={"group": "shared"},
        )
        registry.register_model(
            model_id="/models/beta",
            handler=None,
            model_type="lm",
            metadata_extras={"group": "shared"},
        )
        await registry.update_model_state(
            "/models/beta",
            metadata_updates={"vram_loaded": True},
        )
        registry.set_group_policies({"shared": {"max_loaded": 1}})
        return registry

    guarded_registry = asyncio.run(_seed_registry())
    client.app.state.model_registry = guarded_registry

    response = client.post("/hub/models/alpha/vram/load", json={})

    assert response.status_code == HTTPStatus.TOO_MANY_REQUESTS
    assert controller.scheduled_loads == []


def test_hub_service_start_returns_controller_snapshot(
    hub_service_app: tuple[TestClient, Any, Any],
) -> None:
    """/hub/service/start should report controller details without spawning processes."""
    client, _state, controller = hub_service_app

    response = client.post("/hub/service/start")

    assert response.status_code == HTTPStatus.OK
    body = response.json()
    assert body["action"] == "start"
    assert "models" in body["details"]
    assert controller.reload_count == 1


def test_hub_service_stop_handles_missing_manager(
    hub_service_app: tuple[TestClient, Any, Any],
) -> None:
    """Stop should return an error when no controller is available."""
    client, _state, _controller = hub_service_app
    client.app.state.hub_controller = None

    response = client.post("/hub/service/stop")

    assert response.status_code == HTTPStatus.SERVICE_UNAVAILABLE


def test_hub_service_stop_shuts_down_manager_when_available(
    hub_service_app: tuple[TestClient, Any, Any],
) -> None:
    """Stop should mirror CLI behavior by halting controller and manager.

    Parameters
    ----------
    hub_service_app : tuple[TestClient, _StubServiceState, _StubController]
        Fixture providing the test client and stubs.
    """
    client, state, _controller = hub_service_app

    response = client.post("/hub/service/stop")

    assert response.status_code == HTTPStatus.OK
    assert state.controller_stop_calls == 1
    payload = response.json()
    assert payload["details"] == {"controller_stopped": True}


def test_hub_service_reload_endpoint_returns_diff(
    hub_service_app: tuple[TestClient, Any, Any],
) -> None:
    """/hub/service/reload should surface the diff returned by the service."""
    client, _state, controller = hub_service_app

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
    hub_service_app: tuple[TestClient, Any, Any],
) -> None:
    """/hub/models/{model}/load should call the controller."""
    client, _state, controller = hub_service_app

    response = client.post("/hub/models/alpha/load", json={"reason": "dashboard"})

    assert response.status_code == HTTPStatus.OK
    payload = response.json()
    assert controller.scheduled_loads == ["alpha"]
    assert payload["action_id"] == "alpha-load-action"
    assert payload["state"] == "loading"


def test_hub_load_model_includes_registry_metadata(
    hub_service_app: tuple[TestClient, Any, Any],
) -> None:
    """Load responses should surface registry-provided action snapshots."""

    client, _state, controller = hub_service_app

    class _RegistryStub:
        def __init__(self) -> None:
            self.queries: list[tuple[str | None, str | None]] = []

        def get_vram_action_status(
            self,
            *,
            model_id: str | None = None,
            action_id: str | None = None,
        ) -> tuple[str, dict[str, Any]]:
            self.queries.append((model_id, action_id))
            return "/models/alpha", {
                "vram_action_id": action_id,
                "vram_action_state": "allocating",
                "vram_action_progress": 55.0,
                "vram_action_error": None,
                "worker_port": 9000,
                "vram_action_started_ts": 1,
                "vram_action_updated_ts": 2,
            }

    registry = _RegistryStub()
    client.app.state.model_registry = registry
    controller.force_action_id = "registry-action"

    response = client.post("/hub/models/alpha/load", json={})

    assert response.status_code == HTTPStatus.OK
    body = response.json()
    assert body["action_id"] == "registry-action"
    assert body["state"] == "allocating"
    assert body["progress"] == 55.0
    assert body["worker_port"] == 9000
    assert registry.queries == [(None, "registry-action")]


def test_hub_memory_actions_surface_controller_errors(
    hub_service_app: tuple[TestClient, Any, Any],
) -> None:
    """Controller-originated failures should propagate to the client."""
    client, _state, controller = hub_service_app

    response = client.post("/hub/models/denied/load", json={})
    assert response.status_code == HTTPStatus.TOO_MANY_REQUESTS
    payload = response.json()
    assert "group busy" in payload["error"]["message"]

    response = client.post("/hub/models/missing/unload", json={})
    assert response.status_code == HTTPStatus.BAD_REQUEST
    assert controller.scheduled_unloads[-1] == "missing"


def test_vram_admin_endpoints_schedule_controller(
    hub_service_app: tuple[TestClient, Any, Any],
) -> None:
    """Admin VRAM endpoints should enqueue controller actions."""

    client, _state, controller = hub_service_app

    response = client.post("/hub/models/alpha/vram/load", json={})
    assert response.status_code == HTTPStatus.OK
    assert controller.scheduled_loads[-1] == "alpha"

    response = client.post("/hub/models/alpha/vram/unload", json={})
    assert response.status_code == HTTPStatus.OK
    assert controller.scheduled_unloads[-1] == "alpha"


def test_vram_admin_endpoints_surface_controller_errors(
    hub_service_app: tuple[TestClient, Any, Any],
) -> None:
    """Scheduler errors should propagate through the VRAM endpoints."""

    client, _state, _controller = hub_service_app

    response = client.post("/hub/models/denied/vram/load", json={})
    assert response.status_code == HTTPStatus.TOO_MANY_REQUESTS

    response = client.post("/hub/models/missing/vram/unload", json={})
    assert response.status_code == HTTPStatus.BAD_REQUEST


def test_hub_service_stop_schedules_background_shutdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure `/hub/service/stop` schedules controller shutdown via BackgroundTasks.

    This test constructs a fresh FastAPI app (so it uses the real
    `_stop_controller_process`) and patches `asyncio.sleep` and `sys.exit`
    so the background shutdown runs immediately and does not terminate
    the test process.
    """
    # Create a minimal hub config on disk so `_load_hub_config_from_request`
    # can resolve a path.
    config_dir = tmp_path / "hub-config"
    config_dir.mkdir()
    config_path = config_dir / "hub.yaml"
    config_path.write_text(
        """
host: 0.0.0.0
port: 8123
models: []
""".strip()
    )

    # Build app without monkeypatching _stop_controller_process so the
    # real helper runs and registers BackgroundTasks. Avoid calling
    # `configure_fastapi_app` to prevent background loops from starting.
    app = FastAPI()
    app.include_router(hub_router)
    app.state.server_config = SimpleNamespace(enable_status_page=True, host="0.0.0.0", port=8123)
    app.state.hub_config_path = config_path

    # Fake controller that records whether shutdown_all was invoked.
    class FakeController:
        def __init__(self) -> None:
            self.shutdown_called = False

        async def shutdown_all(self) -> None:
            self.shutdown_called = True

    controller = FakeController()
    app.state.hub_controller = controller

    # Patch asyncio.sleep in the hub_routes module so the delayed exit runs immediately
    async def _no_sleep(_secs: float) -> None:  # pragma: no cover - test helper
        return None

    monkeypatch.setattr("app.api.hub_routes.asyncio.sleep", _no_sleep)

    exit_called = {"called": False}

    def _fake_exit(code: int = 0) -> None:
        exit_called["called"] = True

    monkeypatch.setattr("sys.exit", _fake_exit)

    # Call the helper directly with a dummy BackgroundTasks so we can run
    # scheduled tasks synchronously without involving ASGI/TestClient.

    class DummyBackgroundTasks:
        def __init__(self) -> None:
            self.tasks: list[tuple[Callable, tuple, dict]] = []

        def add_task(self, func: Callable, *args: object, **kwargs: object) -> None:
            self.tasks.append((func, args, kwargs))

        def run_all(self) -> None:
            for func, args, kwargs in list(self.tasks):
                res = func(*args, **kwargs)
                if asyncio.iscoroutine(res):
                    asyncio.run(res)

    background_tasks = DummyBackgroundTasks()
    fake_request = SimpleNamespace(app=app)

    # Execute stop helper directly
    result = _stop_controller_process(fake_request, background_tasks)
    assert result is True

    # Run scheduled background tasks (controller.shutdown_all, _shutdown_server)
    background_tasks.run_all()

    assert controller.shutdown_called is True
    assert exit_called["called"] is True


class FakeManager:
    """A minimal fake manager that simulates an unloaded handler."""

    async def unload(self, _reason: str) -> bool:
        """Simulate unloading the handler.

        Parameters
        ----------
        _reason : str
            Unused reason string (kept for compatibility with real managers).

        Returns
        -------
        bool
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


@pytest.mark.asyncio
async def test_hub_sync_once_updates_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `_hub_sync_once` reads hub snapshot and updates the registry."""
    # Prepare a fake snapshot returned by the hub daemon
    start_ts = 1764465807
    model_id = "mlx-community/Qwen3-30B-A3B-4bit-DWQ"
    snapshot = {
        "models": [
            {
                "name": "qwen3",
                "state": "running",
                "memory_loaded": True,
                "model_path": model_id,
                "started_at": start_ts,
                "active_requests": 2,
                "vram_load_error": None,
            }
        ]
    }

    def fake_load_cfg(req: object) -> SimpleNamespace:
        # Return a simple object with host/port attributes
        return SimpleNamespace(host="127.0.0.1", port=5005)

    # Patch helper imports in `app.server`.
    monkeypatch.setattr("app.server._load_hub_config_from_request", fake_load_cfg)

    app = FastAPI()
    registry = ModelRegistry()
    app.state.model_registry = registry

    class _FakeController:
        async def get_status(self) -> dict[str, Any]:
            return snapshot

    app.state.hub_controller = _FakeController()

    # Register the model so update_model_state will accept it
    registry.register_model(model_id, handler=None, model_type="lm")

    # Run one sync iteration
    await _hub_sync_once(app)

    # Validate registry was updated
    status = registry.get_vram_status(model_id)
    assert status["vram_loaded"] is True
    extra = registry._extra[model_id]
    assert extra["vram_last_load_ts"] == start_ts
    assert extra["active_requests"] == 2
    assert extra.get("vram_load_error") is None
    assert extra.get("status") == "loaded"
