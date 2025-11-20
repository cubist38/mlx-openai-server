"""Tests for the HubController orchestration logic."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path

import pytest

from app.config import MLXServerConfig
from app.core.model_registry import ModelRegistry
from app.hub.config import MLXHubConfig, MLXHubGroupConfig
from app.hub.controller import HubController, HubControllerError
from app.hub.runtime import HubRuntime


class _FakeHandler:
    """Minimal handler that mimics the interface used by tests."""

    def __init__(self, name: str) -> None:
        """Store basic handler metadata for registry propagation.

        Parameters
        ----------
        name : str
            Identifier used to derive ``model_path``.
        """
        self.name = name
        self.model_path = f"/fake/{name}"

    async def cleanup(self) -> None:  # pragma: no cover - simple stub
        """Pretend to release resources after unloading.

        Returns
        -------
        None
            Awaitable placeholder for interface parity.
        """
        await asyncio.sleep(0)


class _FakeHandlerManager:
    """Minimal stand-in for LazyHandlerManager used in tests."""

    def __init__(
        self, config: MLXServerConfig, *, on_change: HandlerCallback | None = None
    ) -> None:
        """Store config for assertions and invoke the change callback.

        Parameters
        ----------
        config : MLXServerConfig
            Model configuration used to seed the handler.
        on_change : HandlerCallback | None, optional
            Callback triggered whenever the handler reference mutates.
        """
        self.config_args = config
        self._on_change = on_change
        self._handler = _FakeHandler(config.name or config.model_path)
        self.loaded = False
        self.recorded_activity = False
        if self._on_change:
            self._on_change(None)

    @property
    def current_handler(self) -> _FakeHandler | None:
        """Return the handler when loaded, else ``None``."""

        return self._handler if self.loaded else None

    def record_activity(self) -> None:
        """Track that the handler was used (no-op for tests)."""

        self.recorded_activity = True

    async def ensure_loaded(self, _reason: str = "manual") -> _FakeHandler:
        """Mark the handler as loaded and fire callbacks.

        Parameters
        ----------
        _reason : str, default="manual"
            Human-readable context for debugging.

        Returns
        -------
        _FakeHandler
            Loaded handler instance.
        """
        self.loaded = True
        if self._on_change:
            self._on_change(self._handler)
        return self._handler

    async def unload(self, _reason: str = "manual") -> bool:
        """Mark the handler as unloaded and fire callbacks.

        Parameters
        ----------
        _reason : str, default="manual"
            Human-readable context for debugging.

        Returns
        -------
        bool
            ``True`` when an active handler transitioned to unloaded.
        """
        if not self.loaded:
            return False
        self.loaded = False
        if self._on_change:
            self._on_change(None)
        return True

    async def shutdown(self) -> None:
        """Convenience wrapper used by controller shutdown.

        Returns
        -------
        None
            Completes after delegating to ``unload``.
        """
        await self.unload("shutdown")


HandlerCallback = Callable[[_FakeHandler | None], None]


def _factory(config: MLXServerConfig, on_change: HandlerCallback) -> _FakeHandlerManager:
    """Return a fake handler manager for dependency injection.

    Parameters
    ----------
    config : MLXServerConfig
        Model configuration that would normally seed a real handler.
    on_change : HandlerCallback
        Callback relayed to the fake manager.

    Returns
    -------
    _FakeHandlerManager
        The constructed fake manager.
    """
    return _FakeHandlerManager(config, on_change=on_change)


def test_controller_bootstrap_does_not_load_by_default(tmp_path: Path) -> None:
    """Controller startup should not auto-load models even if marked default.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory injected by pytest.
    """
    asyncio.run(_run_bootstrap_test(tmp_path))


async def _run_bootstrap_test(tmp_path: Path) -> None:
    config = MLXHubConfig(
        log_path=tmp_path / "logs",
        models=[
            MLXServerConfig(model_path="/models/a", name="alpha", is_default_model=True),
            MLXServerConfig(model_path="/models/b", name="beta"),
        ],
    )
    runtime = HubRuntime(config)
    registry = ModelRegistry()

    controller = HubController(runtime, registry, handler_factory=_factory)

    await controller.start()
    await controller.wait_for_registry_idle()

    statuses = {entry["name"]: entry["status"] for entry in runtime.describe_models(None)}
    assert statuses["alpha"] == "unloaded"
    assert statuses["beta"] == "unloaded"

    registry_data = {model["id"]: model for model in registry.list_models()}
    assert registry_data["alpha"]["metadata"]["status"] == "unloaded"

    await controller.shutdown()


def test_controller_enforces_group_limits(tmp_path: Path) -> None:
    """Group caps should prevent loading more than the configured maximum.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory injected by pytest.
    """
    asyncio.run(_run_group_limit_test(tmp_path))


async def _run_group_limit_test(tmp_path: Path) -> None:
    config = MLXHubConfig(
        log_path=tmp_path / "logs",
        models=[
            MLXServerConfig(model_path="/models/a", name="alpha", group="g", is_default_model=True),
            MLXServerConfig(model_path="/models/b", name="beta", group="g", is_default_model=True),
        ],
        groups=[MLXHubGroupConfig(name="g", max_loaded=1)],
    )
    runtime = HubRuntime(config)
    registry = ModelRegistry()
    controller = HubController(runtime, registry, handler_factory=_factory)

    await controller.start()
    await controller.wait_for_registry_idle()
    await controller.load_model("alpha")
    await controller.wait_for_registry_idle()

    statuses = {entry["name"]: entry["status"] for entry in runtime.describe_models(None)}
    assert statuses["alpha"] == "loaded"
    assert statuses["beta"] == "unloaded"

    with pytest.raises(HubControllerError):
        await controller.load_model("beta")

    await controller.unload_model("alpha")
    await controller.load_model("beta")
    await controller.wait_for_registry_idle()
    statuses = {entry["name"]: entry["status"] for entry in runtime.describe_models(None)}
    assert statuses["beta"] == "loaded"

    await controller.shutdown()


def test_controller_unloads_and_updates_registry(tmp_path: Path) -> None:
    """Unloading should propagate to the registry metadata.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory injected by pytest.
    """
    asyncio.run(_run_unload_test(tmp_path))


async def _run_unload_test(tmp_path: Path) -> None:
    config = MLXHubConfig(
        log_path=tmp_path / "logs",
        models=[MLXServerConfig(model_path="/models/a", name="alpha")],
    )
    runtime = HubRuntime(config)
    registry = ModelRegistry()
    controller = HubController(runtime, registry, handler_factory=_factory)

    await controller.start()
    await controller.wait_for_registry_idle()
    await controller.load_model("alpha")
    await controller.unload_model("alpha")
    await controller.wait_for_registry_idle()

    models = {model["id"]: model for model in registry.list_models()}
    assert models["alpha"]["metadata"]["status"] == "unloaded"

    await controller.shutdown()


def test_controller_acquire_handler_loads_on_demand(tmp_path: Path) -> None:
    """acquire_handler should load models and record activity."""

    asyncio.run(_run_acquire_handler_test(tmp_path))


async def _run_acquire_handler_test(tmp_path: Path) -> None:
    config = MLXHubConfig(
        log_path=tmp_path / "logs",
        models=[MLXServerConfig(model_path="/models/a", name="alpha")],
    )
    runtime = HubRuntime(config)
    registry = ModelRegistry()
    controller = HubController(runtime, registry, handler_factory=_factory)

    await controller.start()
    await controller.wait_for_registry_idle()

    handler = await controller.acquire_handler("alpha", reason="unit-test")
    assert handler.name == "alpha"

    manager = controller.get_handler_manager("alpha")
    assert manager.loaded is True
    assert manager.recorded_activity is True

    await controller.shutdown()
