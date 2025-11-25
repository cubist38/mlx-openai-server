"""Regression tests: ensure ModelRegistry stays in sync with HubSupervisor.

These tests verify that starting, stopping, loading and unloading models
update the registry handler attachment/status as expected.
"""

import asyncio

from app.config import MLXServerConfig
from app.core.model_registry import ModelRegistry
from app.hub.config import MLXHubConfig
from app.hub.daemon import HubSupervisor


class StubManager:
    """Lightweight stub implementing the minimal manager protocol used by tests.

    The stub provides `is_vram_loaded`, `ensure_loaded`, `unload`, and
    `remove_log_sink` so it can be attached to `HubSupervisor` records
    without initializing real MLX handlers.
    """

    def __init__(self) -> None:
        self._loaded = False

    def is_vram_loaded(self) -> bool:
        """Return whether the stub is currently considered loaded."""
        return self._loaded

    async def ensure_loaded(self, reason: str = "test") -> object:
        """Mark the stub as loaded and return a dummy handler object."""
        self._loaded = True
        return object()

    async def unload(self, reason: str = "test") -> bool:
        """Unload if currently loaded and return whether an unload occurred."""
        if self._loaded:
            self._loaded = False
            return True
        return False

    def remove_log_sink(self) -> None:
        """No-op for removing per-model log sinks in the stub."""
        # Explicitly returning None is unnecessary.


def test_registry_updates_on_start_stop_load_unload() -> None:
    """Ensure registry handler attachment is updated after lifecycle actions."""

    async def scenario() -> None:
        cfg = MLXServerConfig(model_path="modelA", name="modelA", jit_enabled=True)
        hub_cfg = MLXHubConfig(models=[cfg])
        registry = ModelRegistry()
        # Register model in registry using the model_path as identifier
        registry.register_model(cfg.model_path, None, cfg.model_type)

        supervisor = HubSupervisor(hub_cfg, registry)

        # Start model should attach a manager in JIT mode
        await supervisor.start_model(cfg.name)
        assert registry.get_handler(cfg.model_path) is not None

        # Stop model should clear the manager and detach from registry
        await supervisor.stop_model(cfg.name)
        assert registry.get_handler(cfg.model_path) is None

        # Attach a stub manager and load the model via supervisor
        stub = StubManager()
        supervisor._models[cfg.name].manager = stub
        await supervisor.load_model(cfg.name)
        assert registry.get_handler(cfg.model_path) is stub

        # Unload should remove the manager and update registry
        await supervisor.unload_model(cfg.name)
        assert registry.get_handler(cfg.model_path) is None

    asyncio.run(scenario())
