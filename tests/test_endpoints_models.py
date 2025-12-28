"""Tests for `/v1/models` availability filtering when supervisor reports memory state."""

from __future__ import annotations

import asyncio
import time
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
    """Ensure `/v1/models` hides unloaded models when max_loaded is reached without idle trigger."""
    app = FastAPI()
    configure_fastapi_app(app, include_hub_routes=True)

    # Install a model registry with three models in the same group
    registry = ModelRegistry()
    registry.register_model(
        "qwen3_4b",
        handler=None,
        model_type="lm",
        metadata_extras={"group": "shared", "started": True},
    )
    registry.register_model(
        "qwen3_30b",
        handler=None,
        model_type="lm",
        metadata_extras={"group": "shared", "started": True},
    )
    registry.register_model(
        "gpt-oss",
        handler=None,
        model_type="lm",
        metadata_extras={"group": "shared", "started": True},
    )

    # Configure group policy: allow only two loaded models concurrently
    # WITHOUT idle_unload_trigger_min, unloaded models should be hidden when at capacity
    registry.set_group_policies({"shared": {"max_loaded": 2}})

    # Mark qwen models as loaded, gpt-oss as started but not loaded
    # Supervisor snapshot is used to sync memory state into registry
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

    # When max_loaded is reached without idle_unload_trigger_min,
    # unloaded models should be hidden from /v1/models
    assert "qwen3_4b" in ids  # Loaded
    assert "qwen3_30b" in ids  # Loaded
    assert "gpt-oss" not in ids  # Unloaded and group at capacity


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


def test_v1_models_hides_stopped_models() -> None:
    """Ensure stopped models (started=False) are filtered out from /v1/models."""
    app = FastAPI()
    configure_fastapi_app(app, include_hub_routes=True)

    registry = ModelRegistry()
    # Register one started model and one stopped model
    registry.register_model(
        "started-model", handler=None, model_type="lm", metadata_extras={"started": True}
    )
    registry.register_model(
        "stopped-model", handler=None, model_type="lm", metadata_extras={"started": False}
    )

    app.state.model_registry = registry

    client = TestClient(app)
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    ids = {m["id"] for m in data["data"]}

    # Only the started model should appear
    assert "started-model" in ids
    assert "stopped-model" not in ids


def test_v1_models_shows_unloaded_when_idle_trigger_eligible() -> None:
    """When idle_unload_trigger_min is set and models are idle, unloaded models are visible."""
    app = FastAPI()
    configure_fastapi_app(app, include_hub_routes=True)

    registry = ModelRegistry()
    registry.register_model(
        "model-a",
        handler=None,
        model_type="lm",
        metadata_extras={"group": "shared", "started": True, "vram_last_request_ts": 1000},
    )
    registry.register_model(
        "model-b",
        handler=None,
        model_type="lm",
        metadata_extras={"group": "shared", "started": True, "vram_last_request_ts": 1000},
    )
    registry.register_model(
        "model-c",
        handler=None,
        model_type="lm",
        metadata_extras={"group": "shared", "started": True},
    )

    # Configure group with idle trigger: 5 minutes
    registry.set_group_policies({"shared": {"max_loaded": 2, "idle_unload_trigger_min": 5}})

    # Mark model-a and model-b as loaded (at capacity), model-c as unloaded
    # Both loaded models have been idle for > 5 minutes (last_activity was at ts=1000)
    snapshot = {
        "models": [
            {
                "model_path": "model-a",
                "memory_loaded": True,
                "state": "running",
                "last_activity_ts": 1000,
            },
            {
                "model_path": "model-b",
                "memory_loaded": True,
                "state": "running",
                "last_activity_ts": 1000,
            },
            {"model_path": "model-c", "memory_loaded": False, "state": "running"},
        ]
    }

    app.state.model_registry = registry
    app.state.supervisor = StubSupervisor(snapshot)

    client = TestClient(app)
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    ids = {m["id"] for m in data["data"]}

    # All started models should be visible when idle eligible models exist
    # (capacity is full but eviction is possible)
    assert "model-a" in ids
    assert "model-b" in ids
    assert "model-c" in ids  # Visible because loaded models are idle enough for eviction


def test_v1_models_hides_unloaded_when_no_idle_eligible() -> None:
    """When idle_unload_trigger_min is set but no models are idle enough, unloaded models hidden."""
    app = FastAPI()
    configure_fastapi_app(app, include_hub_routes=True)

    current_time = int(time.time())

    registry = ModelRegistry()
    registry.register_model(
        "model-a",
        handler=None,
        model_type="lm",
        metadata_extras={
            "group": "shared",
            "started": True,
            "vram_last_request_ts": current_time - 60,  # 1 minute ago
        },
    )
    registry.register_model(
        "model-b",
        handler=None,
        model_type="lm",
        metadata_extras={
            "group": "shared",
            "started": True,
            "vram_last_request_ts": current_time - 120,  # 2 minutes ago
        },
    )
    registry.register_model(
        "model-c",
        handler=None,
        model_type="lm",
        metadata_extras={"group": "shared", "started": True},
    )

    # Configure group with idle trigger: 10 minutes
    registry.set_group_policies({"shared": {"max_loaded": 2, "idle_unload_trigger_min": 10}})

    # Mark model-a and model-b as loaded (at capacity), model-c as unloaded
    # Both loaded models have been idle for < 10 minutes
    snapshot = {
        "models": [
            {
                "model_path": "model-a",
                "memory_loaded": True,
                "state": "running",
                "last_activity_ts": current_time - 60,
            },
            {
                "model_path": "model-b",
                "memory_loaded": True,
                "state": "running",
                "last_activity_ts": current_time - 120,
            },
            {"model_path": "model-c", "memory_loaded": False, "state": "running"},
        ]
    }

    app.state.model_registry = registry
    app.state.supervisor = StubSupervisor(snapshot)

    client = TestClient(app)
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    ids = {m["id"] for m in data["data"]}

    # Unloaded models should be hidden when at capacity and no idle eligible models
    assert "model-a" in ids
    assert "model-b" in ids
    assert "model-c" not in ids  # Hidden because capacity full and no eviction candidates


def test_v1_models_shows_all_when_below_capacity() -> None:
    """When loaded < max_loaded, all started models should be visible."""
    app = FastAPI()
    configure_fastapi_app(app, include_hub_routes=True)

    registry = ModelRegistry()
    registry.register_model(
        "model-a",
        handler=None,
        model_type="lm",
        metadata_extras={"group": "shared", "started": True},
    )
    registry.register_model(
        "model-b",
        handler=None,
        model_type="lm",
        metadata_extras={"group": "shared", "started": True},
    )
    registry.register_model(
        "model-c",
        handler=None,
        model_type="lm",
        metadata_extras={"group": "shared", "started": True},
    )

    # Configure group: max 3 loaded
    registry.set_group_policies({"shared": {"max_loaded": 3}})

    # Only one model is loaded (below capacity)
    snapshot = {
        "models": [
            {"model_path": "model-a", "memory_loaded": True, "state": "running"},
            {"model_path": "model-b", "memory_loaded": False, "state": "running"},
            {"model_path": "model-c", "memory_loaded": False, "state": "running"},
        ]
    }

    app.state.model_registry = registry
    app.state.supervisor = StubSupervisor(snapshot)

    client = TestClient(app)
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    ids = {m["id"] for m in data["data"]}

    # All started models should be visible when below capacity
    assert "model-a" in ids
    assert "model-b" in ids
    assert "model-c" in ids
