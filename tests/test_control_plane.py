"""Tests for control-plane VRAM actions and data-plane gating."""

from __future__ import annotations

import asyncio
from http import HTTPStatus
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
import pytest

from app.api import endpoints
from app.api.hub_routes import hub_router
from app.core.model_registry import ModelRegistry


class _ControlStubController:
    """Stub controller implementing schedule_vram_load/unload hooks."""

    def __init__(self, registry: ModelRegistry) -> None:
        self.registry = registry
        self.load_calls: list[tuple[str, dict[str, Any] | None]] = []
        self.unload_calls: list[str] = []

    async def schedule_vram_load(
        self, model_id: str, *, settings: dict[str, Any] | None = None
    ) -> dict[str, Any]:  # noqa: B008
        self.load_calls.append((model_id, settings))
        action_id = "load-action"
        await self.registry.start_vram_action(model_id, action_id=action_id, state="loading")
        await self.registry.update_vram_action(
            model_id,
            action_id=action_id,
            state="ready",
            progress=55.0,
            worker_port=8125,
            error=None,
        )
        return {"action_id": action_id, "state": "loading", "progress": 0.0, "message": "scheduled"}

    async def schedule_vram_unload(self, model_id: str) -> dict[str, Any]:
        self.unload_calls.append(model_id)
        action_id = "unload-action"
        await self.registry.start_vram_action(model_id, action_id=action_id, state="loading")
        await self.registry.update_vram_action(
            model_id,
            action_id=action_id,
            state="ready",
            progress=100.0,
            error=None,
        )
        return {"action_id": action_id, "state": "loading", "progress": 0.0, "message": "scheduled"}


class _DeferredLoadController:
    """Stub controller that defers worker readiness until explicitly released."""

    def __init__(self, registry: ModelRegistry) -> None:
        self.registry = registry
        self._pending_actions: dict[str, str] = {}

    async def schedule_vram_load(
        self,
        model_id: str,
        *,
        settings: dict[str, Any] | None = None,
    ) -> dict[str, Any]:  # noqa: B008
        _ = settings
        action_id = f"pending-{model_id}"
        self._pending_actions[model_id] = action_id
        await self.registry.start_vram_action(model_id, action_id=action_id, state="loading")
        await self.registry.update_vram_action(
            model_id,
            action_id=action_id,
            state="loading",
            progress=0.0,
            worker_port=None,
        )
        return {"action_id": action_id, "state": "loading", "progress": 0.0}

    def mark_ready(self, model_id: str, *, worker_port: int = 8126) -> None:
        """Mark ``model_id`` as ready and assign a worker port."""

        action_id = self._pending_actions.get(model_id)
        if action_id is None:  # pragma: no cover - defensive
            raise AssertionError(f"No pending action recorded for {model_id}")
        asyncio.run(
            self.registry.update_vram_action(
                model_id,
                action_id=action_id,
                state="ready",
                progress=100.0,
                worker_port=worker_port,
                error=None,
            )
        )


class _LifecycleController:
    """Stub controller that simulates full load/unload cycles."""

    def __init__(self, registry: ModelRegistry, *, load_port: int = 9300) -> None:
        self.registry = registry
        self.load_port = load_port
        self.load_calls: list[str] = []
        self.unload_calls: list[str] = []

    async def schedule_vram_load(
        self, model_id: str, *, settings: dict[str, Any] | None = None
    ) -> dict[str, Any]:  # noqa: B008
        self.load_calls.append(model_id)
        _ = settings
        action_id = f"load-{model_id}"
        await self.registry.start_vram_action(model_id, action_id=action_id, state="loading")
        await self.registry.update_vram_action(
            model_id,
            action_id=action_id,
            state="ready",
            progress=100.0,
            worker_port=self.load_port,
            error=None,
        )
        return {"action_id": action_id, "state": "ready", "progress": 100.0}

    async def schedule_vram_unload(self, model_id: str) -> dict[str, Any]:
        self.unload_calls.append(model_id)
        action_id = f"unload-{model_id}"
        await self.registry.start_vram_action(model_id, action_id=action_id, state="loading")
        await self.registry.update_vram_action(
            model_id,
            action_id=action_id,
            state="ready",
            progress=100.0,
            worker_port=None,
            error=None,
        )
        return {"action_id": action_id, "state": "ready", "progress": 100.0}


def _build_control_app() -> tuple[TestClient, ModelRegistry, _ControlStubController]:
    app = FastAPI()
    app.include_router(hub_router)

    registry = ModelRegistry()
    registry.register_model("alpha", handler=None, model_type="lm")
    app.state.model_registry = registry

    controller = _ControlStubController(registry)
    app.state.hub_controller = controller

    client = TestClient(app)
    return client, registry, controller


def test_control_load_surfaces_registry_status() -> None:
    """Test that control load endpoint surfaces registry action status."""
    client, registry, controller = _build_control_app()
    try:
        response = client.post("/control/load", json={"model_id": "alpha"})

        assert response.status_code == HTTPStatus.OK
        payload = response.json()
        assert payload["action_id"] == "load-action"
        assert payload["state"] == "ready"
        assert payload["progress"] == 55.0
        assert payload["worker_port"] == 8125
        assert controller.load_calls == [("alpha", None)]

        status = client.get("/control/status", params={"action_id": "load-action"})
        assert status.status_code == HTTPStatus.OK
        status_body = status.json()
        assert status_body["model"] == "alpha"
        assert status_body["state"] == "ready"
        assert status_body["action_id"] == "load-action"
    finally:
        client.close()


def test_control_unload_surfaces_registry_status() -> None:
    """Test that control unload endpoint surfaces registry action status."""
    client, registry, controller = _build_control_app()
    try:
        response = client.post("/control/unload", json={"model_id": "alpha"})

        assert response.status_code == HTTPStatus.OK
        payload = response.json()
        assert payload["action"] == "vram_unload"
        assert payload["state"] == "ready"
        assert payload["progress"] == 100.0
        assert controller.unload_calls == ["alpha"]

        status = client.get("/control/status", params={"action_id": "unload-action"})
        assert status.status_code == HTTPStatus.OK
        assert status.json()["state"] == "ready"
    finally:
        client.close()


def test_control_status_handles_concurrent_polling() -> None:
    """Ensure control/status can poll while a VRAM load is still pending."""

    app = FastAPI()
    app.include_router(hub_router)

    registry = ModelRegistry()
    registry.register_model("alpha", handler=None, model_type="lm")
    app.state.model_registry = registry

    controller = _DeferredLoadController(registry)
    app.state.hub_controller = controller

    client = TestClient(app)
    try:
        response = client.post("/control/load", json={"model_id": "alpha"})
        assert response.status_code == HTTPStatus.OK
        payload = response.json()
        assert payload["state"] == "loading"
        action_id = payload["action_id"]

        inflight = client.get("/control/status", params={"action_id": action_id})
        assert inflight.status_code == HTTPStatus.OK
        assert inflight.json()["state"] == "loading"

        controller.mark_ready("alpha", worker_port=8129)

        ready = client.get("/control/status", params={"action_id": action_id})
        assert ready.status_code == HTTPStatus.OK
        ready_payload = ready.json()
        assert ready_payload["state"] == "ready"
        assert ready_payload["worker_port"] == 8129
    finally:
        client.close()


def test_data_plane_gates_on_vram_action_state() -> None:
    """Test that data-plane endpoints gate on VRAM action state and worker port."""
    app = FastAPI()
    app.include_router(endpoints.router)

    registry = ModelRegistry()
    registry.register_model("/models/foo", handler=None, model_type="lm")
    asyncio.run(registry.start_vram_action("/models/foo", state="loading"))
    app.state.model_registry = registry

    class _StubHandler:
        async def get_queue_stats(self) -> dict[str, Any]:
            return {"depth": 0}

    app.state.handler = _StubHandler()

    client = TestClient(app)
    try:
        loading = client.get("/v1/queue/stats", params={"model": "/models/foo"})
        assert loading.status_code == HTTPStatus.SERVICE_UNAVAILABLE
        assert "still loading" in loading.json()["error"]["message"]

        asyncio.run(
            registry.update_vram_action(
                "/models/foo",
                state="error",
                progress=0.0,
                error="sidecar failed",
            )
        )

        failing = client.get("/v1/queue/stats", params={"model": "/models/foo"})
        assert failing.status_code == HTTPStatus.SERVICE_UNAVAILABLE
        assert "sidecar failed" in failing.json()["error"]["message"]

        asyncio.run(
            registry.update_vram_action(
                "/models/foo",
                state="ready",
                progress=100.0,
                error=None,
                worker_port=None,
            )
        )

        missing_port = client.get("/v1/queue/stats", params={"model": "/models/foo"})
        assert missing_port.status_code == HTTPStatus.SERVICE_UNAVAILABLE
        assert "port" in missing_port.json()["error"]["message"].lower()

        asyncio.run(
            registry.update_vram_action(
                "/models/foo",
                state="ready",
                progress=100.0,
                error=None,
                worker_port=9000,
            )
        )

        ready = client.get("/v1/queue/stats", params={"model": "/models/foo"})
        assert ready.status_code == HTTPStatus.OK
        assert ready.json()["status"] == "ok"
    finally:
        client.close()


def test_chat_completions_rejects_when_worker_port_missing() -> None:
    """Ensure chat completions fail fast when worker ports are not ready."""

    app = FastAPI()
    app.include_router(endpoints.router)

    registry = ModelRegistry()
    registry.register_model("/models/foo", handler=None, model_type="lm")
    action_id = asyncio.run(registry.start_vram_action("/models/foo", state="ready"))
    asyncio.run(
        registry.update_vram_action(
            "/models/foo",
            action_id=action_id,
            state="ready",
            progress=100.0,
            worker_port=None,
        )
    )
    app.state.model_registry = registry
    app.state.hub_controller = object()

    client = TestClient(app)
    try:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "/models/foo",
                "messages": [{"role": "user", "content": "status"}],
            },
        )

        assert response.status_code == HTTPStatus.SERVICE_UNAVAILABLE
        assert "sidecar" in response.json()["error"]["message"].lower()
    finally:
        client.close()


def test_chat_completions_denied_when_group_capacity_reached() -> None:
    """Verify group capacity policies block new traffic and surface 429 errors."""

    class _LoadedHandler:
        def is_vram_loaded(self) -> bool:
            return True

    app = FastAPI()
    app.include_router(endpoints.router)

    registry = ModelRegistry()
    registry.register_model(
        "/models/hot",
        handler=None,
        model_type="lm",
        metadata_extras={"group": "shared"},
    )
    registry.register_model(
        "/models/cold",
        handler=None,
        model_type="lm",
        metadata_extras={"group": "shared"},
    )
    asyncio.run(registry.update_model_state("/models/hot", handler=_LoadedHandler()))
    registry.set_group_policies({"shared": {"max_loaded": 1}})
    app.state.model_registry = registry

    client = TestClient(app)
    try:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "/models/cold",
                "messages": [{"role": "user", "content": "ping"}],
            },
        )

        assert response.status_code == HTTPStatus.TOO_MANY_REQUESTS
        assert "429" in response.json()["error"]["message"]
    finally:
        client.close()


def test_chat_completions_proxy_routes_to_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure chat completions proxy to ready worker ports in hub mode."""

    app = FastAPI()
    app.include_router(endpoints.router)

    registry = ModelRegistry()
    registry.register_model("/models/foo", handler=None, model_type="lm")
    action_id = asyncio.run(registry.start_vram_action("/models/foo", state="loading"))
    asyncio.run(
        registry.update_vram_action(
            "/models/foo",
            action_id=action_id,
            state="ready",
            progress=100.0,
            worker_port=9100,
            error=None,
        )
    )
    app.state.model_registry = registry
    app.state.hub_controller = object()

    captured: dict[str, Any] = {}

    async def _fake_proxy(
        raw_request: Request, *, worker_port: int, model_id: str, body: bytes
    ) -> JSONResponse:
        captured["worker_port"] = worker_port
        captured["model_id"] = model_id
        captured["body"] = body
        return JSONResponse({"status": "proxied"}, status_code=202)

    monkeypatch.setattr(endpoints, "_proxy_request_to_worker", _fake_proxy)

    client = TestClient(app)
    try:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "/models/foo",
                "messages": [{"role": "user", "content": "ping"}],
            },
        )

        assert response.status_code == 202
        assert response.json() == {"status": "proxied"}
        assert captured["worker_port"] == 9100
        assert "ping" in captured["body"].decode()
    finally:
        client.close()


def test_control_unload_blocks_worker_proxy_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure unload requests tear down worker routing and block future proxies."""

    app = FastAPI()
    app.include_router(hub_router)
    app.include_router(endpoints.router)

    registry = ModelRegistry()
    registry.register_model("/models/alpha", handler=None, model_type="lm")
    app.state.model_registry = registry

    controller = _LifecycleController(registry, load_port=9400)
    app.state.hub_controller = controller

    captured: dict[str, Any] = {}

    async def _fake_proxy(
        raw_request: Request, *, worker_port: int, model_id: str, body: bytes
    ) -> JSONResponse:
        captured["worker_port"] = worker_port
        captured["model_id"] = model_id
        captured["body"] = body
        return JSONResponse({"status": "proxied"}, status_code=202)

    monkeypatch.setattr(endpoints, "_proxy_request_to_worker", _fake_proxy)

    client = TestClient(app)
    try:
        load_response = client.post("/control/load", json={"model_id": "/models/alpha"})
        assert load_response.status_code == HTTPStatus.OK

        proxied = client.post(
            "/v1/chat/completions",
            json={
                "model": "/models/alpha",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert proxied.status_code == 202
        assert captured["worker_port"] == 9400

        unload_response = client.post("/control/unload", json={"model_id": "/models/alpha"})
        assert unload_response.status_code == HTTPStatus.OK

        blocked = client.post(
            "/v1/chat/completions",
            json={
                "model": "/models/alpha",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert blocked.status_code == HTTPStatus.SERVICE_UNAVAILABLE
        assert "port" in blocked.json()["error"]["message"].lower()
    finally:
        client.close()
