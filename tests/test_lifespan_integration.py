"""Integration tests for FastAPI lifespan/startup behavior.

These tests patch `instantiate_handler` to a lightweight fake and verify
that `create_lifespan` honors the `jit_enabled` setting (loads handler
immediately when JIT is disabled and defers when JIT is enabled).
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

from fastapi import FastAPI
import pytest

from app.config import MLXServerConfig
from app.server import create_lifespan


def test_lifespan_respects_jit_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """When JIT is disabled the handler is loaded during lifespan startup."""
    called = {"count": 0}

    async def fake_instantiate(cfg: MLXServerConfig) -> object:
        called["count"] += 1
        # Return a simple handler-like object
        return SimpleNamespace(model_path=cfg.model_path, model_created=int(time.time()))

    monkeypatch.setattr("app.server.instantiate_handler", fake_instantiate)

    config = MLXServerConfig(model_path="/models/test", jit_enabled=False, auto_unload_minutes=None)
    app = FastAPI()
    lifespan = create_lifespan(config)

    async def _run() -> None:
        async with lifespan(app):
            # Handler manager should have loaded the handler on startup
            hm = app.state.handler_manager
            assert hm.current_handler is not None
            assert called["count"] == 1

    asyncio.run(_run())


def test_lifespan_respects_jit_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """When JIT is enabled the handler is not loaded at startup."""
    called = {"count": 0}

    async def fake_instantiate(cfg: MLXServerConfig) -> object:
        called["count"] += 1
        return SimpleNamespace(model_path=cfg.model_path, model_created=int(time.time()))

    monkeypatch.setattr("app.server.instantiate_handler", fake_instantiate)

    config = MLXServerConfig(model_path="/models/test", jit_enabled=True, auto_unload_minutes=None)
    app = FastAPI()
    # Populate the model metadata cache the way `setup_server` normally does
    app.state.model_metadata = [
        {
            "id": config.model_identifier,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "local",
            "metadata": {"model_path": config.model_identifier},
        }
    ]
    lifespan = create_lifespan(config)

    async def _run() -> None:
        async with lifespan(app):
            hm = app.state.handler_manager
            # No handler should be loaded at startup when JIT is enabled
            assert hm.current_handler is None
            assert called["count"] == 0

    asyncio.run(_run())


def test_lifespan_with_fake_handler_ensure_vram_and_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Integration test that uses a featureful fake handler to verify.

    `ensure_vram_loaded` is invoked via `handler_session` and `cleanup` is called
    on shutdown when JIT is disabled (handler loaded at startup).
    """
    calls = {"ensure": 0, "cleanup": 0}

    class FakeHandler:
        def __init__(self, cfg: MLXServerConfig) -> None:
            self.model_path = cfg.model_path

        async def initialize(self, _cfg: MLXServerConfig) -> None:  # pragma: no cover - trivial
            return None

        async def ensure_vram_loaded(
            self,
            *,
            _force: bool = False,
            _timeout: float | None = None,
        ) -> None:
            await asyncio.sleep(0)
            calls["ensure"] += 1

        async def cleanup(self) -> None:
            await asyncio.sleep(0)
            calls["cleanup"] += 1

    async def fake_instantiate(cfg: MLXServerConfig) -> object:
        return FakeHandler(cfg)

    monkeypatch.setattr("app.server.instantiate_handler", fake_instantiate)

    config = MLXServerConfig(model_path="/models/test", jit_enabled=False, auto_unload_minutes=None)
    app = FastAPI()
    lifespan = create_lifespan(config)

    async def _run() -> None:
        async with lifespan(app):
            registry = app.state.model_registry
            registry_model_id = next(m["id"] for m in registry.list_models())

            # Wait until the registry has attached the handler (update happens
            # asynchronously from the handler_manager on_change callback).
            for _ in range(20):
                try:
                    h = registry.get_handler(registry_model_id)
                except KeyError:
                    h = None
                if h is not None:
                    break
                await asyncio.sleep(0.01)

            # Enter a handler_session which should call ensure_vram_loaded on the handler
            async with registry.handler_session(registry_model_id):
                # ensure_vram_loaded should have been invoked by the session
                assert calls["ensure"] >= 1

        # After exiting lifespan, cleanup should have been called

    asyncio.run(_run())
    assert calls["cleanup"] >= 1


def test_jit_triggers_handler_load_on_request(monkeypatch: pytest.MonkeyPatch) -> None:
    """When JIT is enabled, handler should be loaded on first request via handler_manager.ensure_loaded."""
    called = {"count": 0}

    async def fake_instantiate(cfg: MLXServerConfig) -> object:
        called["count"] += 1
        return SimpleNamespace(model_path=cfg.model_path, model_created=int(time.time()))

    monkeypatch.setattr("app.server.instantiate_handler", fake_instantiate)

    config = MLXServerConfig(model_path="/models/test", jit_enabled=True, auto_unload_minutes=None)
    app = FastAPI()
    lifespan = create_lifespan(config)

    async def _run() -> None:
        async with lifespan(app):
            hm = app.state.handler_manager
            # Initially not loaded under JIT
            assert hm.current_handler is None

            # Trigger load via ensure_loaded (simulates first request)
            handler = await hm.ensure_loaded("test-request")
            assert handler is not None
            assert called["count"] == 1

    asyncio.run(_run())


def test_cached_metadata_reflects_manager_vram(monkeypatch: pytest.MonkeyPatch) -> None:
    """When a handler is loaded via JIT the cached metadata should report vram_loaded."""

    async def fake_instantiate(cfg: MLXServerConfig) -> object:
        # Return a simple handler-like object
        return SimpleNamespace(model_path=cfg.model_path, model_created=int(time.time()))

    monkeypatch.setattr("app.server.instantiate_handler", fake_instantiate)

    config = MLXServerConfig(model_path="/models/test", jit_enabled=True, auto_unload_minutes=None)
    app = FastAPI()
    lifespan = create_lifespan(config)

    async def _run() -> None:
        async with lifespan(app):
            # Ensure the model metadata cache exists (populated by setup_server in normal runs)
            if getattr(app.state, "model_metadata", None) is None:
                app.state.model_metadata = [
                    {
                        "id": config.model_identifier,
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "local",
                        "metadata": {"model_path": config.model_identifier},
                    }
                ]

            hm = app.state.handler_manager

            # Trigger load via ensure_loaded (simulates first request)
            handler = await hm.ensure_loaded("test-request")
            assert handler is not None

            # The manager should report VRAM residency
            assert hm.is_vram_loaded()

            # Wait for the on_change callback to update the cached metadata
            for _ in range(50):
                md = getattr(app.state, "model_metadata", None)
                if md and md[0].get("metadata", {}).get("vram_loaded"):
                    break
                await asyncio.sleep(0.01)

            md = app.state.model_metadata
            assert md and md[0]["metadata"]["vram_loaded"] is True

    asyncio.run(_run())
