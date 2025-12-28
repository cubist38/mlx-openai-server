"""Tests for hub model lifecycle management.

These tests verify that models start/stop, load/unload correctly,
and that default models auto-start on hub initialization.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import HTTPException
from fastapi.testclient import TestClient
import pytest

from app.config import MLXServerConfig
from app.core.model_registry import ModelRegistry
from app.hub.config import MLXHubConfig, MLXHubGroupConfig
from app.hub.daemon import HubSupervisor, create_app
from app.server import LazyHandlerManager


class _MockHandlerManager:
    """Mock handler manager for testing."""

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self._handler: Any = None  # Can be MagicMock or None
        self._shutdown = False

    def is_vram_loaded(self) -> bool:
        return self._handler is not None

    async def ensure_loaded(self, reason: str = "request") -> Any:
        if not self._shutdown:
            self._handler = MagicMock()
            self._handler.model_path = self.model_path
        return self._handler

    async def unload(self, reason: str = "manual") -> bool:
        self._handler = None
        return True


class _TestHubSupervisor(HubSupervisor):
    """Test version of HubSupervisor with mocked handler creation."""

    def __init__(self, hub_config: MLXHubConfig) -> None:
        # Don't call super().__init__ to avoid registry setup
        self.hub_config = hub_config
        self.registry = None
        self.idle_controller = None
        self._models = {}
        self._lock = asyncio.Lock()
        self._model_locks = {}
        self._bg_tasks = []
        self._shutdown = False

        # Populate model records
        for model in getattr(hub_config, "models", []):
            name = getattr(model, "name", None) or str(model)
            record = MagicMock()
            record.name = name
            record.config = model
            record.group = getattr(model, "group", None)
            record.is_default = getattr(model, "is_default_model", False)
            record.model_path = getattr(model, "model_path", None)
            record.auto_unload_minutes = getattr(model, "auto_unload_minutes", None)
            record.manager = None
            record.worker = None
            record.started_at = None
            record.exit_code = None
            self._models[name] = record


@pytest.fixture
def hub_config_with_defaults(
    tmp_path: Path, make_model_mock: Callable[..., MagicMock]
) -> MLXHubConfig:
    """Create a hub config with default and non-default models."""
    config = MLXHubConfig(
        host="127.0.0.1",
        port=8123,
        model_starting_port=8000,
        log_level="INFO",
        log_path=tmp_path / "logs",
        enable_status_page=True,
        source_path=tmp_path / "hub.yaml",
    )

    # Mock model configs using the shared factory from tests.conftest
    default_model = make_model_mock(
        "default_model",
        "/models/default",
        "lm",
        is_default=True,
        group=None,
        auto_unload_minutes=120,
    )

    regular_model = make_model_mock(
        "regular_model",
        "/models/regular",
        "lm",
        is_default=False,
        group=None,
        auto_unload_minutes=120,
    )

    config.models = [default_model, regular_model]
    return config


@pytest.fixture
def mock_handler_manager() -> MagicMock:
    """Create a mock handler manager."""
    manager = MagicMock(spec=LazyHandlerManager)
    manager.is_vram_loaded.return_value = False
    manager.ensure_loaded = AsyncMock()
    manager.unload = AsyncMock(return_value=True)
    return manager


@pytest.mark.asyncio
async def test_model_start_sets_manager_and_memory_loaded(
    hub_config_with_defaults: MLXHubConfig,
) -> None:
    """Test that starting a model creates a handler manager and sets memory_loaded."""
    supervisor = _TestHubSupervisor(hub_config_with_defaults)

    # Mock the handler manager creation
    with (
        patch("app.hub.daemon.LazyHandlerManager") as mock_lhm,
        patch.object(
            HubSupervisor,
            "_start_worker",
            new_callable=AsyncMock,
        ) as mock_start_worker,
    ):
        mock_manager = MagicMock()
        manager_loaded = {"value": False}

        async def _ensure_loaded(_reason: str = "request") -> MagicMock:
            manager_loaded["value"] = True
            return MagicMock()

        def _is_loaded() -> bool:
            return manager_loaded["value"]

        mock_manager.ensure_loaded = AsyncMock(side_effect=_ensure_loaded)
        mock_manager.is_vram_loaded.side_effect = _is_loaded
        mock_manager.jit_enabled = False  # regular_model is not JIT-enabled
        mock_lhm.return_value = mock_manager
        mock_start_worker.return_value = MagicMock()

        result = await supervisor.start_model("regular_model")

        assert result["status"] == "loaded"
        assert result["name"] == "regular_model"

        # Check that the record was updated
        record = supervisor._models["regular_model"]
        assert record.manager is not None
        assert manager_loaded["value"] is True
        assert mock_manager.ensure_loaded.await_count == 1

        # Verify LazyHandlerManager was created with correct config
        mock_lhm.assert_called_once()
        call_args = mock_lhm.call_args[0][0]  # First positional arg
        assert call_args.model_path == "/models/regular"


@pytest.mark.asyncio
async def test_model_start_already_loaded_returns_early(
    hub_config_with_defaults: MLXHubConfig,
) -> None:
    """Test that starting an already loaded model returns early without recreation."""
    supervisor = _TestHubSupervisor(hub_config_with_defaults)
    record = supervisor._models["regular_model"]
    record.manager = MagicMock()
    record.manager.is_vram_loaded.return_value = True

    result = await supervisor.start_model("regular_model")

    with patch.object(
        HubSupervisor,
        "_start_worker",
        new_callable=AsyncMock,
    ) as mock_start_worker:
        result = await supervisor.start_model("regular_model")
        # When model is already loaded no worker start should be attempted
        mock_start_worker.assert_not_awaited()

    assert result["status"] == "already_loaded"
    assert result["name"] == "regular_model"


@pytest.mark.asyncio
async def test_start_with_existing_manager_starts_worker(
    hub_config_with_defaults: MLXHubConfig,
) -> None:
    """Starting a model when a manager exists but isn't loaded should start the worker."""
    supervisor = _TestHubSupervisor(hub_config_with_defaults)
    record = supervisor._models["regular_model"]
    # Simulate existing manager that isn't loaded
    mock_manager = MagicMock()
    mock_manager.is_vram_loaded.return_value = False
    record.manager = mock_manager

    with patch.object(
        HubSupervisor,
        "_start_worker",
        new_callable=AsyncMock,
    ) as mock_start_worker:
        mock_start_worker.return_value = MagicMock()
        result = await supervisor.start_model("regular_model")
        mock_start_worker.assert_awaited_once()
    assert result["status"] in {"loaded", "started"}


@pytest.mark.asyncio
async def test_model_stop_unloads_and_clears_state(hub_config_with_defaults: MLXHubConfig) -> None:
    """Test that stopping a model unloads it and clears the manager."""
    supervisor = _TestHubSupervisor(hub_config_with_defaults)
    record = supervisor._models["regular_model"]
    mock_manager = MagicMock()
    mock_manager.unload = AsyncMock(return_value=True)
    record.manager = mock_manager

    result = await supervisor.stop_model("regular_model")

    assert result["status"] == "stopped"
    assert result["name"] == "regular_model"

    # Verify unload was called and manager was cleared
    mock_manager.unload.assert_called_once_with("stop")
    assert record.manager is None


@pytest.mark.asyncio
async def test_model_load_and_unload(hub_config_with_defaults: MLXHubConfig) -> None:
    """Test load_model and unload_model methods."""
    supervisor = _TestHubSupervisor(hub_config_with_defaults)

    # First start the model to initialize the manager
    with (
        patch("app.hub.daemon.LazyHandlerManager") as mock_lhm,
        patch.object(
            HubSupervisor,
            "_start_worker",
            new_callable=AsyncMock,
        ) as mock_start_worker,
    ):
        mock_manager = MagicMock()
        manager_loaded = {"value": False}

        async def _ensure_loaded(_reason: str = "request") -> MagicMock:
            manager_loaded["value"] = True
            return MagicMock()

        def _is_loaded() -> bool:
            return manager_loaded["value"]

        async def _unload(_reason: str = "manual") -> bool:
            manager_loaded["value"] = False
            return True

        mock_manager.ensure_loaded = AsyncMock(side_effect=_ensure_loaded)
        mock_manager.is_vram_loaded.side_effect = _is_loaded
        mock_manager.jit_enabled = False  # regular_model is not JIT-enabled
        mock_manager.unload = AsyncMock(side_effect=_unload)
        mock_lhm.return_value = mock_manager
        mock_start_worker.return_value = MagicMock()

        start_result = await supervisor.start_model("regular_model")
        assert start_result["status"] == "loaded"

    # Now test load - model is already loaded from start_model
    result = await supervisor.load_model("regular_model")
    assert result["status"] == "already_loaded"  # Model is already loaded from start_model
    assert result["name"] == "regular_model"

    # Test unload - model is still loaded
    result = await supervisor.unload_model("regular_model")
    assert result["status"] == "unloaded"
    assert result["name"] == "regular_model"

    # Verify unload was called
    mock_manager.unload.assert_called_once_with("unload")


@pytest.mark.asyncio
async def test_get_handler_waits_for_model_load(tmp_path: Path) -> None:
    """Supervisor.get_handler should trigger a load when VRAM is empty."""
    config = MLXHubConfig(log_path=tmp_path / "logs")
    config.models = [
        MLXServerConfig(
            name="jit_model",
            model_path="/models/jit",
            model_type="lm",
            jit_enabled=True,
        )
    ]
    supervisor = HubSupervisor(config)

    record = supervisor._models["jit_model"]
    manager = MagicMock(spec=LazyHandlerManager)
    handler = MagicMock()
    manager.ensure_loaded = AsyncMock(return_value=handler)

    manager_state = {"loaded": False}

    def _is_loaded() -> bool:
        return manager_state["loaded"]

    manager.is_vram_loaded.side_effect = _is_loaded
    record.manager = manager

    async def fake_load_model(name: str) -> dict[str, Any]:
        assert name == "jit_model"
        manager_state["loaded"] = True
        return {"status": "loaded", "name": name}

    supervisor.load_model = AsyncMock(side_effect=fake_load_model)  # type: ignore[method-assign]

    acquired = await supervisor.get_handler("jit_model")

    assert acquired is handler
    supervisor.load_model.assert_awaited_once_with("jit_model")
    manager.ensure_loaded.assert_awaited_once()


@pytest.mark.asyncio
async def test_group_idle_trigger_unloads_longest_idle_model(tmp_path: Path) -> None:
    """Longest-idle group member should be unloaded to free capacity."""
    config = MLXHubConfig(log_path=tmp_path / "logs")
    config.groups = [MLXHubGroupConfig(name="tier", max_loaded=1, idle_unload_trigger_min=5)]
    config.models = [
        MLXServerConfig(name="resident_a", model_path="/models/a", model_type="lm", group="tier"),
        MLXServerConfig(name="resident_b", model_path="/models/b", model_type="lm", group="tier"),
        MLXServerConfig(name="incoming", model_path="/models/c", model_type="lm", group="tier"),
    ]
    supervisor = HubSupervisor(config)

    resident_a = supervisor._models["resident_a"]
    resident_b = supervisor._models["resident_b"]
    incoming = supervisor._models["incoming"]

    manager_a = MagicMock(spec=LazyHandlerManager)
    manager_a.is_vram_loaded.return_value = True
    manager_a.seconds_since_last_activity.return_value = 600  # 10 minutes
    manager_a.unload = AsyncMock(return_value=True)

    manager_b = MagicMock(spec=LazyHandlerManager)
    manager_b.is_vram_loaded.return_value = True
    manager_b.seconds_since_last_activity.return_value = 1200  # 20 minutes
    manager_b.unload = AsyncMock(return_value=True)

    incoming_manager = MagicMock(spec=LazyHandlerManager)
    incoming_manager.is_vram_loaded.return_value = False

    resident_a.manager = manager_a
    resident_b.manager = manager_b
    incoming.manager = incoming_manager

    await supervisor._ensure_group_capacity("incoming")

    manager_b.unload.assert_awaited_once_with("group-capacity")
    manager_a.unload.assert_not_called()


@pytest.mark.asyncio
async def test_group_idle_trigger_raises_when_no_candidate(tmp_path: Path) -> None:
    """An error is raised when no group members meet the idle threshold."""
    config = MLXHubConfig(log_path=tmp_path / "logs")
    config.groups = [MLXHubGroupConfig(name="tier", max_loaded=1, idle_unload_trigger_min=15)]
    config.models = [
        MLXServerConfig(name="resident", model_path="/models/a", model_type="lm", group="tier"),
        MLXServerConfig(name="incoming", model_path="/models/b", model_type="lm", group="tier"),
    ]
    supervisor = HubSupervisor(config)

    resident = supervisor._models["resident"]
    resident_manager = MagicMock(spec=LazyHandlerManager)
    resident_manager.is_vram_loaded.return_value = True
    resident_manager.seconds_since_last_activity.return_value = 300  # 5 minutes < threshold
    resident_manager.unload = AsyncMock(return_value=True)
    resident.manager = resident_manager

    incoming = supervisor._models["incoming"]
    incoming_manager = MagicMock(spec=LazyHandlerManager)
    incoming_manager.is_vram_loaded.return_value = False
    incoming.manager = incoming_manager

    with pytest.raises(HTTPException) as exc:
        await supervisor._ensure_group_capacity("incoming")

    assert exc.value.status_code == 409
    resident_manager.unload.assert_not_called()


@pytest.mark.asyncio
async def test_cannot_load_second_model_when_group_max_loaded_one_and_no_idle_trigger(
    tmp_path: Path,
) -> None:
    """Ensure loading a second model in a group with max_loaded=1 and no.

    idle_unload_trigger_min is rejected with HTTP 409.
    """
    config = MLXHubConfig(log_path=tmp_path / "logs")
    # Group has max_loaded=1 but no idle_unload_trigger_min set -> immediate violation
    config.groups = [MLXHubGroupConfig(name="only_one", max_loaded=1)]
    config.models = [
        MLXServerConfig(
            name="first", model_path="/models/first", model_type="lm", group="only_one"
        ),
        MLXServerConfig(
            name="second", model_path="/models/second", model_type="lm", group="only_one"
        ),
    ]

    supervisor = HubSupervisor(config)

    # Simulate the first model already loaded
    first = supervisor._models["first"]
    first_manager = MagicMock(spec=LazyHandlerManager)
    first_manager.is_vram_loaded.return_value = True
    first.manager = first_manager

    # Prepare the second model with an initialized manager that is not loaded
    second = supervisor._models["second"]
    second.manager = MagicMock(spec=LazyHandlerManager)
    second.manager.is_vram_loaded.return_value = False

    with pytest.raises(HTTPException) as exc:
        await supervisor.load_model("second")

    assert exc.value.status_code == 409


@pytest.mark.asyncio
async def test_get_status_returns_correct_state(hub_config_with_defaults: MLXHubConfig) -> None:
    """Test that get_status returns correct running/stopped states."""
    supervisor = _TestHubSupervisor(hub_config_with_defaults)

    # Set up one loaded model
    record = supervisor._models["regular_model"]
    mock_manager = MagicMock()
    mock_manager.is_vram_loaded.return_value = True
    record.manager = mock_manager

    status = await supervisor.get_status()

    assert "models" in status
    assert len(status["models"]) == 2

    @pytest.mark.asyncio
    async def test_start_worker_updates_registry(hub_config_with_defaults: MLXHubConfig) -> None:
        """Ensure that starting the worker updates the registry metadata with worker_port."""
        # Create a real registry and supervisor so we exercise update_model_state
        registry = ModelRegistry()
        # Register the test model id similar to app startup
        cfg = hub_config_with_defaults
        model_cfg = cfg.models[1]  # regular_model entry
        model_id = getattr(model_cfg, "model_path")
        registry.register_model(model_id=model_id, handler=None, model_type=model_cfg.model_type)

        supervisor = HubSupervisor(cfg, registry)

        record = supervisor._models[model_cfg.name]
        # Inject a fake worker instance to avoid spawning a real process
        fake_worker = MagicMock()
        fake_worker.start = AsyncMock()
        fake_worker.port = 12345
        fake_worker.pid = 999
        record.worker = fake_worker

        # Call _start_worker which should attempt to persist worker_port
        await supervisor._start_worker(name=model_cfg.name, record=record, model_id=model_id)

        models = registry.list_models()
        found = next((m for m in models if m.get("id") == model_id), None)
        assert found is not None
        assert found.get("metadata", {}).get("worker_port") == 12345

    # Find the models in the status
    regular_model_status = next(m for m in status["models"] if m["name"] == "regular_model")
    default_model_status = next(m for m in status["models"] if m["name"] == "default_model")

    assert regular_model_status["state"] == "running"
    assert default_model_status["state"] == "stopped"


@pytest.mark.asyncio
async def test_worker_probe_unload_timestamp_does_not_override_expected(tmp_path: Path) -> None:
    """Ensure a worker probe's vram_last_unload_ts (past value) does not override.

    The supervisor's computed expected unload_timestamp (which should be future).
    """
    cfg = MLXHubConfig(log_path=tmp_path / "logs")
    cfg.models = [
        MLXServerConfig(
            name="m",
            model_path="/models/m",
            model_type="lm",
            auto_unload_minutes=60,
            jit_enabled=True,
        )
    ]
    sup = HubSupervisor(cfg)

    # Simulate a loaded model with a future expected unload timestamp
    record = sup._models["m"]
    manager = MagicMock(spec=LazyHandlerManager)
    manager.is_vram_loaded.return_value = True
    record.manager = manager
    # Replace idle_controller to return a future timestamp
    future_ts = time.time() + 3600
    sup.idle_controller = MagicMock()
    sup.idle_controller.get_expected_unload_timestamp.return_value = future_ts

    # Patch the probe helper to return a past unload timestamp in extras
    async def fake_probe(_rec: Any, idx: int) -> tuple[int, bool | None, dict[str, object] | None]:
        return (
            idx,
            True,
            {"last_activity_ts": time.time() - 10, "unload_timestamp": int(time.time())},
        )

    with patch("app.hub.daemon._probe_worker_for_memory", new=AsyncMock(side_effect=fake_probe)):
        status = await sup.get_status()

    m_entry = next(m for m in status["models"] if m["name"] == "m")
    # The expected unload timestamp from idle_controller should be preserved
    assert m_entry.get("unload_timestamp") == future_ts


def test_hub_status_unload_timestamp_after_real_load(
    write_hub_yaml: Callable[[str, str], Path],
) -> None:
    """Integration test: start and load a model and verify /hub/status shows expected unload timestamp."""
    cfg = write_hub_yaml(
        """
host: 127.0.0.1
port: 8123
models:
  - name: m
    model_path: /models/m
    model_type: lm
    jit_enabled: true
    auto_unload_minutes: 1
""",
        "hub-integration.yaml",
    )

    app = create_app(str(cfg))

    # Prevent actual sidecar processes from being started
    # Prevent actual sidecar processes from being started and stub manager behavior
    with (
        patch.object(HubSupervisor, "_start_worker", new_callable=AsyncMock),
        patch("app.hub.daemon.LazyHandlerManager") as mock_lhm,
    ):
        mock_manager = MagicMock()
        mock_manager.ensure_loaded = AsyncMock(return_value=MagicMock())
        mock_manager.is_vram_loaded.return_value = True
        mock_manager.seconds_since_last_activity.return_value = 0
        # Ensure manager advertises auto-unload timeout so the controller
        # can compute an expected unload timestamp.
        mock_manager.auto_unload_minutes = 1
        mock_manager.jit_enabled = True
        mock_lhm.return_value = mock_manager

        with TestClient(app) as client:
            # Start the model (makes manager and worker available)
            r = client.post("/hub/models/m/start")
            assert r.status_code == 200

            # Now load the model into VRAM
            r = client.post("/hub/models/m/load")
            assert r.status_code == 200
            # Ensure the load is fully executed (run synchronously in-test)
            asyncio.run(app.state.supervisor.load_model("m"))

            # Optionally check the HTTP endpoint too (sanity) then inspect supervisor snapshot
            r = client.get("/hub/status")
            assert r.status_code == 200

            # Inspect supervisor directly to avoid any HTTP-side rendering differences
            status_snapshot = asyncio.run(app.state.supervisor.get_status())
            m_entry = next(
                (m for m in status_snapshot.get("models", []) if m.get("name") == "m"), None
            )
            assert m_entry is not None
            # Sanity-check registry/handler state to help debug failures
            st = app.state.model_registry.get_vram_status("/models/m")
            assert st.get("vram_loaded") is True
            handler = app.state.model_registry.get_handler("/models/m")
            assert handler is not None
            assert getattr(handler, "auto_unload_minutes", None) == 1
            unload_ts = m_entry.get("unload_timestamp")
            assert unload_ts is not None
            now = time.time()
            # auto_unload_minutes=1 => approx +60s; allow small drift
            assert unload_ts > now + 50 and unload_ts < now + 70


@pytest.mark.asyncio
async def test_reload_config_preserves_loaded_models(
    hub_config_with_defaults: MLXHubConfig,
) -> None:
    """Test that reload_config preserves loaded model state for unchanged models."""
    supervisor = _TestHubSupervisor(hub_config_with_defaults)

    # Load a model before reload
    record = supervisor._models["regular_model"]
    mock_manager = MagicMock()
    mock_manager.is_vram_loaded.return_value = True
    record.manager = mock_manager
    record.started_at = 12345

    # Mock load_hub_config to return the same config
    with patch(
        "app.hub.daemon.load_hub_config",
        return_value=hub_config_with_defaults,
    ) as mock_load:
        # Reload config (should preserve the loaded state)
        result = await supervisor.reload_config()

        assert "started" in result
        assert "stopped" in result
        assert "unchanged" in result

        # Verify the loaded model was preserved
        record_after = supervisor._models["regular_model"]
        assert record_after.manager is mock_manager
        assert record_after.manager.is_vram_loaded() is True
        assert record_after.started_at == 12345

        mock_load.assert_called_once_with(hub_config_with_defaults.source_path)


@pytest.mark.asyncio
async def test_reload_config_rejects_model_path_change_for_started_models(
    hub_config_with_defaults: MLXHubConfig,
) -> None:
    """Test that reload_config rejects model_path changes for started models."""
    supervisor = _TestHubSupervisor(hub_config_with_defaults)

    # Load a model before reload
    record = supervisor._models["regular_model"]
    mock_manager = MagicMock()
    record.manager = mock_manager
    original_path = record.model_path
    assert original_path is not None  # Should be set from config

    # Create a new config with changed model_path for the started model
    new_config = MLXHubConfig(
        source_path=hub_config_with_defaults.source_path,
        host=hub_config_with_defaults.host,
        port=hub_config_with_defaults.port,
        models=[
            MLXServerConfig(
                name="regular_model",
                model_path="/new/path",  # Changed path
                model_type="lm",
                group="test",
                jit_enabled=True,  # Required for auto_unload_minutes
                auto_unload_minutes=30,
            )
        ],
    )

    with patch(
        "app.hub.daemon.load_hub_config",
        return_value=new_config,
    ):
        # Reload config should raise HTTPException
        with pytest.raises(HTTPException) as exc_info:
            await supervisor.reload_config()

        assert exc_info.value.status_code == 400
        assert "Cannot change model_path" in str(exc_info.value.detail)
        assert original_path in str(exc_info.value.detail)
        assert "/new/path" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_start_and_stop_update_registry_started_flag(tmp_path: Path) -> None:
    """Ensure start_model marks models as started and stop_model clears the flag."""

    cfg = MLXHubConfig(log_path=tmp_path / "logs")
    cfg.models = [
        MLXServerConfig(
            name="tracked",
            model_path="/models/tracked",
            model_type="lm",
            jit_enabled=True,
        )
    ]

    registry = ModelRegistry()
    registry.register_model(
        model_id="/models/tracked",
        handler=None,
        model_type="lm",
    )

    supervisor = HubSupervisor(cfg, registry)

    with (
        patch("app.hub.daemon.LazyHandlerManager") as mock_lhm,
        patch.object(HubSupervisor, "_start_worker", new_callable=AsyncMock) as mock_start_worker,
        patch.object(HubSupervisor, "_stop_worker", new_callable=AsyncMock) as mock_stop_worker,
    ):
        mock_manager = MagicMock()
        mock_manager.is_vram_loaded.return_value = False
        mock_manager.jit_enabled = True
        mock_manager.unload = AsyncMock(return_value=True)
        mock_lhm.return_value = mock_manager

        await supervisor.start_model("tracked")
        mock_start_worker.assert_awaited()

        metadata = registry.list_models()[0]["metadata"]
        assert metadata.get("started") is True

        await supervisor.stop_model("tracked")
        mock_stop_worker.assert_awaited()

        metadata = registry.list_models()[0]["metadata"]
        assert metadata.get("started") is False


@pytest.mark.asyncio
async def test_default_models_auto_start_during_lifespan(
    write_hub_yaml: Callable[[str, str], Path],
) -> None:
    """Test that default models are automatically started during daemon lifespan."""
    cfg = write_hub_yaml(
        """
host: 127.0.0.1
port: 8123
models:
  - name: default_model
    model_path: /models/default
    model_type: lm
    default: true
  - name: regular_model
    model_path: /models/regular
    model_type: lm
""",
        "hub.yaml",
    )

    app = create_app(str(cfg))

    # Mock the supervisor to track start calls
    original_supervisor = app.state.supervisor
    start_calls = []

    async def mock_start_model(name: str) -> dict[str, Any]:
        start_calls.append(name)
        return {"status": "loaded", "name": name}

    # Replace the supervisor's start_model method
    original_supervisor.start_model = mock_start_model

    # Use TestClient as context manager to trigger the lifespan
    with TestClient(app):
        # The lifespan should have auto-started the default model
        pass

    # Restore the original supervisor
    app.state.supervisor = original_supervisor

    # Assert that the default model was auto-started
    assert "default_model" in start_calls
    assert "regular_model" not in start_calls


@pytest.mark.asyncio
async def test_hub_api_endpoints_call_supervisor_methods(
    write_hub_yaml: Callable[[str, str], Path],
) -> None:
    """Test that hub API endpoints properly call supervisor methods."""
    cfg = write_hub_yaml(
        """
host: 127.0.0.1
port: 8123
models:
  - name: test_model
    model_path: /models/test
    model_type: lm
""",
        "hub.yaml",
    )

    app = create_app(str(cfg))

    # Replace the supervisor with a mock to avoid side-effects (handler/worker startup)
    supervisor = MagicMock()
    supervisor.start_model = AsyncMock(return_value={"status": "loaded", "name": "test_model"})
    app.state.supervisor = supervisor
    app.state.hub_controller = supervisor
    load_response = {
        "status": "accepted",
        "action": "vram_load",
        "model": "test_model",
        "action_id": "action-1",
        "state": "loading",
        "progress": 0.0,
    }
    unload_response = {
        "status": "accepted",
        "action": "vram_unload",
        "model": "test_model",
        "action_id": "action-2",
        "state": "loading",
        "progress": 0.0,
    }

    # Delay binding of schedule_vram_load until we hit the explicit /load
    # endpoint to avoid the start endpoint attempting to call it.
    supervisor.schedule_vram_unload = AsyncMock(return_value=unload_response)

    # Prevent real sidecar processes from being started during the test
    with patch.object(HubSupervisor, "_start_worker", new_callable=AsyncMock):
        client = TestClient(app)

        # Test start -> should call supervisor.start_model
        response = client.post("/hub/models/test_model/start")
        assert response.status_code == 200
        supervisor.start_model.assert_awaited_with("test_model")

        # Test stop
        response = client.post("/hub/models/test_model/stop")
        assert response.status_code == 200
        supervisor.schedule_vram_unload.assert_awaited_with("test_model")

        # Bind the load method now and test the /load endpoint
        supervisor.schedule_vram_load = AsyncMock(return_value=load_response)
        response = client.post("/hub/models/test_model/load")
        assert response.status_code == 200
        assert supervisor.schedule_vram_load.await_count == 1

        # Test unload
        response = client.post("/hub/models/test_model/unload")
        assert response.status_code == 200
        assert supervisor.schedule_vram_unload.await_count == 2
