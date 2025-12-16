"""Tests for FastAPI hub routes and lifecycle helper wiring."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from app.core.hub_lifecycle import HubServiceAdapter, get_hub_lifecycle_service
from app.server import configure_fastapi_app


class _FakeRegistry:
    """Minimal registry stub so availability guards always allow requests."""

    def has_model(self, _model_id: str) -> bool:
        return False

    def is_model_available(self, _model_id: str) -> bool:
        return True


class _StubHubController:
    """Controller stub that tracks lifecycle calls made by the hub routes."""

    def __init__(self) -> None:
        self.start_requests: list[str] = []
        self.memory_loads: list[str] = []
        self.memory_unloads: list[str] = []
        self.vram_load_requests: list[str] = []
        self.vram_unload_requests: list[str] = []
        self.reload_calls = 0
        self.shutdown_calls = 0

    async def start_model(self, name: str) -> dict[str, Any]:
        self.start_requests.append(name)
        return {"status": "started", "name": name, "state": "ready"}

    async def stop_model(self, name: str) -> dict[str, Any]:
        self.memory_unloads.append(name)
        return {"status": "stopped", "name": name}

    async def load_model(self, name: str) -> dict[str, Any]:
        self.memory_loads.append(name)
        return {"status": "loaded", "name": name}

    async def unload_model(self, name: str) -> dict[str, Any]:
        self.memory_unloads.append(name)
        return {"status": "unloaded", "name": name}

    async def get_status(self) -> dict[str, Any]:
        return {"timestamp": 1, "models": []}

    async def schedule_vram_load(
        self, name: str, *, settings: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        _ = settings
        self.vram_load_requests.append(name)
        return {
            "status": "accepted",
            "action": "vram_load",
            "model": name,
            "action_id": f"load-{name}",
            "state": "loading",
            "progress": 1.0,
        }

    async def schedule_vram_unload(self, name: str) -> dict[str, Any]:
        self.vram_unload_requests.append(name)
        return {
            "status": "accepted",
            "action": "vram_unload",
            "model": name,
            "action_id": f"unload-{name}",
            "state": "unloading",
            "progress": 0.0,
        }

    async def reload_config(self) -> dict[str, Any]:
        self.reload_calls += 1
        return {"started": [], "stopped": [], "unchanged": ["alpha"]}

    async def shutdown_all(self) -> None:
        self.shutdown_calls += 1


@dataclass(slots=True)
class HubAppHarness:
    """Bundle the configured FastAPI app, client, and stub controller."""

    app: FastAPI
    client: TestClient
    controller: _StubHubController


@pytest.fixture
def hub_app(tmp_path: Path) -> Iterator[HubAppHarness]:
    """Yield a configured FastAPI app with hub routes bound to a stub controller.

    Parameters
    ----------
    tmp_path : Path
        Pytest-provided temporary directory for writing a hub YAML file.

    Returns
    -------
    Iterator[HubAppHarness]
        Iterator yielding the harness so tests can access app, client, and controller.
    """
    cfg_path = tmp_path / "hub.yaml"
    cfg_path.write_text(
        """
host: 127.0.0.1
port: 8123
models:
  - name: alpha
    model_path: /models/alpha
    model_type: lm
""".strip(),
    )

    app = FastAPI()
    configure_fastapi_app(app, include_hub_routes=True)
    app.state.hub_config_path = cfg_path
    app.state.model_registry = _FakeRegistry()
    controller = _StubHubController()
    app.state.hub_controller = controller

    client = TestClient(app)
    harness = HubAppHarness(app=app, client=client, controller=controller)
    try:
        yield harness
    finally:
        client.close()


def test_hub_model_routes_delegate_to_controller(hub_app: HubAppHarness) -> None:
    """Verify model action routes exercise the controller start/load/unload logic.

    Parameters
    ----------
    hub_app : HubAppHarness
        Harness exposing the configured FastAPI test client and stub controller.

    Returns
    -------
    None
        This test asserts controller call sequencing via side effects.
    """
    client = hub_app.client
    controller = hub_app.controller

    response = client.post("/hub/models/alpha/start")
    assert response.status_code == 200
    assert controller.start_requests == ["alpha"]
    assert controller.vram_load_requests == ["alpha"], "start should trigger VRAM load"

    response = client.post("/hub/models/alpha/load")
    assert response.status_code == 200
    assert controller.vram_load_requests.count("alpha") == 2

    response = client.post("/hub/models/alpha/unload")
    assert response.status_code == 200
    assert controller.vram_unload_requests == ["alpha"]

    response = client.post("/hub/models/alpha/stop")
    assert response.status_code == 200
    assert controller.vram_unload_requests.count("alpha") == 2


def test_hub_reload_endpoint_calls_controller(hub_app: HubAppHarness) -> None:
    """Ensure POST /hub/reload delegates to the controller's reload_config.

    Parameters
    ----------
    hub_app : HubAppHarness
        Harness exposing the configured FastAPI test client and stub controller.

    Returns
    -------
    None
        Assertion-only test verifying controller invocation count.
    """
    client = hub_app.client
    controller = hub_app.controller

    response = client.post("/hub/reload")
    assert response.status_code == 200
    assert controller.reload_calls == 1


def test_hub_shutdown_runs_background_tasks(
    hub_app: HubAppHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Confirm POST /hub/shutdown schedules controller + process exit background tasks.

    Parameters
    ----------
    hub_app : HubAppHarness
        Harness exposing the configured FastAPI test client and stub controller.
    monkeypatch : pytest.MonkeyPatch
        Pytest helper used to bypass the real asyncio sleep and sys.exit calls.

    Returns
    -------
    None
        Assertions validate shutdown tasks run without exiting the test process.
    """
    controller = hub_app.controller
    controller.shutdown_all = AsyncMock()

    async def _immediate_sleep(_delay: float) -> None:
        return None

    exit_codes: list[int] = []

    def _fake_exit(code: int) -> None:
        exit_codes.append(code)

    monkeypatch.setattr("app.api.hub_routes.asyncio.sleep", _immediate_sleep)
    monkeypatch.setattr("app.api.hub_routes.sys.exit", _fake_exit)

    response = hub_app.client.post("/hub/shutdown?exit=1")
    assert response.status_code == 200
    assert controller.shutdown_all.await_count == 1
    assert exit_codes == [0]


@pytest.mark.asyncio
async def test_get_hub_lifecycle_service_prefers_state_controller() -> None:
    """Ensure get_hub_lifecycle_service resolves controllers attached to app.state.

    Returns
    -------
    None
        Test verifies the resolved service matches the injected controller.
    """
    controller = _StubHubController()
    container = SimpleNamespace(state=SimpleNamespace(hub_controller=controller))

    resolved = get_hub_lifecycle_service(container)

    assert resolved is not None
    await resolved.start_model("alpha")
    assert controller.start_requests == ["alpha"]


@pytest.mark.asyncio
async def test_hub_service_adapter_handles_sync_and_async_mix() -> None:
    """Adapter should normalize sync/async callables into awaited dict payloads.

    Returns
    -------
    None
        Assertions confirm adapters await async functions and pass through sync ones.
    """

    class _HybridBackend:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        def start_model(self, name: str) -> dict[str, Any]:
            self.calls.append(("start", name))
            return {"status": "ok"}

        async def stop_model(self, name: str) -> dict[str, Any]:
            self.calls.append(("stop", name))
            return {"status": "ok"}

        def load_model(self, name: str) -> dict[str, Any]:
            self.calls.append(("load", name))
            return {"status": "ok"}

        async def unload_model(self, name: str) -> dict[str, Any]:
            self.calls.append(("unload", name))
            return {"status": "ok"}

        async def get_status(self) -> dict[str, Any]:
            return {"models": [name for _, name in self.calls if _ == "start"]}

    backend = _HybridBackend()
    adapter = HubServiceAdapter(
        start_model_fn=backend.start_model,
        stop_model_fn=backend.stop_model,
        load_model_fn=backend.load_model,
        unload_model_fn=backend.unload_model,
        status_fn=backend.get_status,
    )

    await adapter.start_model("alpha")
    await adapter.stop_model("alpha")
    await adapter.load_model("beta")
    await adapter.unload_model("beta")
    status = await adapter.get_status()

    assert backend.calls == [
        ("start", "alpha"),
        ("stop", "alpha"),
        ("load", "beta"),
        ("unload", "beta"),
    ]
    assert status == {"models": ["alpha"]}
