"""Tests for `/v1/models` availability filtering when supervisor reports memory state."""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.model_registry import ModelRegistry
from app.server import configure_fastapi_app


class StubSupervisor:
    """Simple stub of a hub supervisor exposing a `get_status` coroutine.

    This is intentionally minimal: tests inject a ready-made snapshot and the
    stub asynchronously returns it when queried.
    """

    def __init__(self, snapshot: dict) -> None:
        """Store the provided snapshot for later return from `get_status`."""
        self._snapshot = snapshot

    async def get_status(self) -> dict:
        """Asynchronously return the preconfigured status snapshot."""
        # Simulate asynchronous status retrieval
        await asyncio.sleep(0)
        return self._snapshot


def test_v1_models_filters_based_on_supervisor_memory_loaded() -> None:
    """Ensure `/v1/models` filters out models not reported loaded by supervisor."""
    app = FastAPI()
    configure_fastapi_app(app, include_hub_routes=True)

    # Install a model registry with three models in the same group
    registry = ModelRegistry()
    registry.register_model(
        "qwen3_4b", handler=None, model_type="lm", metadata_extras={"group": "shared"}
    )
    registry.register_model(
        "qwen3_30b", handler=None, model_type="lm", metadata_extras={"group": "shared"}
    )
    registry.register_model(
        "gpt-oss", handler=None, model_type="lm", metadata_extras={"group": "shared"}
    )

    # Configure group policy: allow only two loaded models concurrently
    registry.set_group_policies({"shared": {"max_loaded": 2}})

    # Supervisor reports qwen models are memory_loaded while gpt-oss is not
    snapshot = {
        "models": [
            {"model_path": "qwen3_4b", "memory_loaded": True, "state": "running"},
            {"model_path": "qwen3_30b", "memory_loaded": True, "state": "running"},
            {"model_path": "gpt-oss", "memory_loaded": False, "state": "running"},
        ]
    }

    app.state.model_registry = registry
    app.state.supervisor = StubSupervisor(snapshot)

    client = TestClient(app)
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    ids = {m["id"] for m in data["data"]}

    # qwen models should be listed; gpt-oss should not be present because
    # max_loaded=2 and it is not memory_loaded.
    assert "qwen3_4b" in ids
    assert "qwen3_30b" in ids
    assert "gpt-oss" not in ids


def test_v1_models_prefers_handler_when_manager_reports_loaded() -> None:
    """When running standalone and manager reports VRAM residency, prefer handler.get_models."""
    app = FastAPI()
    configure_fastapi_app(app, include_hub_routes=True)

    # Populate a cached snapshot that claims vram_loaded=False to ensure
    # the endpoint only reports True when the handler view is used.
    app.state.model_metadata = [
        {
            "id": "test-model",
            "object": "model",
            "created": 1,
            "owned_by": "local",
            "metadata": {"model_path": "test-model", "vram_loaded": False},
        }
    ]

    class FakeHandler:
        async def get_models(self) -> list[dict[str, Any]]:
            return [
                {
                    "id": "test-model",
                    "object": "model",
                    "created": 1,
                    "owned_by": "local",
                    "metadata": {"model_path": "test-model", "vram_loaded": True},
                }
            ]

    # Handler manager reports VRAM residency
    # Simulate standalone worker mode by removing the automatic registry
    app.state.model_registry = None

    app.state.handler_manager = type(
        "HM", (), {"is_vram_loaded": lambda self=None: True, "current_handler": FakeHandler()}
    )()

    client = TestClient(app)
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    md = data["data"][0]["metadata"]
    assert md["vram_loaded"] is True
