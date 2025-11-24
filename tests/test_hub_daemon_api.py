"""Integration tests for the hub daemon HTTP API.

These tests exercise the HTTP surface exposed by `app.hub.daemon.create_app`
but stub out the `HubSupervisor` on `app.state.supervisor` so no real
processes are spawned.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
import httpx
from httpx import ASGITransport
from loguru import logger
import pytest

from app.hub.daemon import create_app


class _StubSupervisor:
    def __init__(self) -> None:
        self.shutdown_called = False
        self.started: dict[str, int] = {}

    def get_status(self) -> dict[str, Any]:
        return {"timestamp": 1, "models": [{"name": "alpha", "state": "stopped"}]}

    async def start_model(self, name: str) -> dict[str, Any]:
        self.started[name] = 1234
        return {"status": "started", "name": name, "pid": 1234}

    async def stop_model(self, name: str) -> dict[str, Any]:
        self.started.pop(name, None)
        return {"status": "stopped", "name": name}

    async def load_model(self, name: str) -> dict[str, Any]:
        return {"status": "memory_loaded", "name": name}

    async def unload_model(self, name: str) -> dict[str, Any]:
        return {"status": "memory_unloaded", "name": name}

    async def reload_config(self) -> dict[str, Any]:
        return {"started": [], "stopped": [], "unchanged": []}

    async def shutdown_all(self) -> None:
        # mark called so tests can assert the background task ran
        self.shutdown_called = True
        logger.info("Stub shutdown called")


@pytest.mark.asyncio
async def test_daemon_health_and_status(tmp_path: Path) -> None:
    """Health and status endpoints return expected shapes."""
    cfg = tmp_path / "hub.yaml"
    cfg.write_text(
        """
host: 127.0.0.1
port: 8123
models:
  - name: alpha
    model_path: /models/alpha
""".strip()
    )

    app = create_app(str(cfg))
    stub = _StubSupervisor()
    app.state.supervisor = stub

    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}

    r = client.get("/hub/status")
    assert r.status_code == 200
    payload = r.json()
    assert "models" in payload and isinstance(payload["models"], list)


@pytest.mark.asyncio
async def test_model_start_stop_and_memory_actions(tmp_path: Path) -> None:
    """Model lifecycle endpoints start/stop/load/unload behave as expected."""
    cfg = tmp_path / "hub.yaml"
    cfg.write_text(
        """
host: 127.0.0.1
port: 8123
models:
  - name: alpha
    model_path: /models/alpha
""".strip()
    )

    app = create_app(str(cfg))
    stub = _StubSupervisor()
    app.state.supervisor = stub

    client = TestClient(app)
    r = client.post("/hub/models/alpha/start")
    assert r.status_code == 200
    assert r.json()["status"] == "started"

    r = client.post("/hub/models/alpha/stop")
    assert r.status_code == 200
    assert r.json()["status"] == "stopped"

    r = client.post("/hub/models/alpha/load")
    assert r.status_code == 200
    assert r.json()["status"] == "memory_loaded"

    r = client.post("/hub/models/alpha/unload")
    assert r.status_code == 200
    assert r.json()["status"] == "memory_unloaded"


@pytest.mark.asyncio
async def test_shutdown_schedules_background_task(tmp_path: Path) -> None:
    """POST /hub/shutdown schedules the supervisor shutdown background task."""
    cfg = tmp_path / "hub.yaml"
    cfg.write_text(
        """
host: 127.0.0.1
port: 8123
models:
  - name: alpha
    model_path: /models/alpha
""".strip()
    )

    app = create_app(str(cfg))
    stub = _StubSupervisor()
    app.state.supervisor = stub

    async with httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        r = await client.post("/hub/shutdown")
        assert r.status_code == 200
        assert r.json() == {"status": "shutdown_scheduled"}

        # background tasks run after response; give the loop a moment
        await asyncio.sleep(0.1)
        assert stub.shutdown_called is True
