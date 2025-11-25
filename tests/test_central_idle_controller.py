"""Tests for the CentralIdleAutoUnloadController behavior.

This module contains both higher-level integration-style tests that use the
real `ModelRegistry` and lightweight unit-style tests that mock registry
responses. They were merged to keep controller behavior tests in one place.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
import time

import pytest

from app.core.model_registry import ModelRegistry
from app.server import CentralIdleAutoUnloadController


async def _wait_for_condition(
    condition_func: Callable[[], bool],
    timeout: float = 1.0,
    poll_interval: float = 0.01,
) -> bool:
    """Wait for a condition to become true with a bounded timeout."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        if condition_func():
            return True
        await asyncio.sleep(poll_interval)
    return False


class FakeManager:  # type: ignore[override]
    """Simple fake manager that tracks unload calls and VRAM state."""

    def __init__(self) -> None:
        """Initialize as loaded with zero unloads."""
        self._loaded = True
        self.unload_calls = 0

    def is_vram_loaded(self) -> bool:
        """Return whether VRAM is currently loaded."""
        return self._loaded

    async def ensure_vram_loaded(
        self,
        *,
        force: bool = False,
        _timeout: float | None = None,
    ) -> None:
        """Simulate ensuring VRAM is loaded (idempotent)."""
        await asyncio.sleep(0)
        if not self._loaded or force:
            self._loaded = True

    async def release_vram(self, *, _timeout: float | None = None) -> None:
        """Simulate releasing VRAM with a tiny sleep and record the call."""
        await asyncio.sleep(0)
        if self._loaded:
            self.unload_calls += 1
            self._loaded = False


def test_central_controller_unloads_idle_model() -> None:
    """Controller should unload a model marked with immediate auto-unload."""

    async def _test() -> None:
        registry = ModelRegistry()
        mgr = FakeManager()

        # Register model with metadata that indicates immediate auto-unload
        registry.register_model(  # type: ignore[arg-type]
            model_id="m1",
            handler=mgr,
            model_type="lm",
            metadata_extras={"auto_unload_minutes": 0},
        )

        controller = CentralIdleAutoUnloadController(registry)
        registry.register_activity_notifier(controller.notify_activity)
        controller.start()

        # Wait for the controller to unload the model
        unloaded = await _wait_for_condition(
            lambda: not registry.get_vram_status("m1")["vram_loaded"],
        )
        assert unloaded, "Model should have been unloaded by the controller"

        status = registry.get_vram_status("m1")
        assert status["vram_loaded"] is False
        assert mgr.unload_calls >= 1

        await controller.stop()

    asyncio.run(_test())


def test_activity_resets_idle_timer() -> None:
    """Activity during a session prevents the central controller from unloading."""

    async def _test() -> None:
        registry = ModelRegistry()
        mgr = FakeManager()

        registry.register_model(  # type: ignore[arg-type]
            model_id="m2",
            handler=mgr,
            model_type="lm",
            metadata_extras={"auto_unload_minutes": 0},
        )

        controller = CentralIdleAutoUnloadController(registry)
        registry.register_activity_notifier(controller.notify_activity)
        controller.start()

        # While a session is active, the controller must not unload the model
        async with registry.handler_session("m2") as _m:
            # Give a moment for any potential unload to occur (it shouldn't)
            await asyncio.sleep(0.05)
            assert mgr.is_vram_loaded() is True

        # After session ends, controller should unload promptly
        unloaded = await _wait_for_condition(
            lambda: not registry.get_vram_status("m2")["vram_loaded"],
        )
        assert unloaded, "Model should have been unloaded after session ended"

        status = registry.get_vram_status("m2")
        assert status["vram_loaded"] is False

        await controller.stop()

    asyncio.run(_test())


def test_unload_failure_triggers_backoff() -> None:
    """If unload fails the controller applies a backoff and does not retry immediately."""

    async def _test() -> None:
        calls = {"count": 0}

        class FailManager(FakeManager):
            async def release_vram(self, *, timeout: float | None = None) -> None:
                calls["count"] += 1
                raise RuntimeError("simulated unload failure")

        registry = ModelRegistry()
        mgr = FailManager()

        registry.register_model(  # type: ignore[arg-type]
            model_id="m3",
            handler=mgr,
            model_type="lm",
            metadata_extras={"auto_unload_minutes": 0},
        )

        controller = CentralIdleAutoUnloadController(registry)
        registry.register_activity_notifier(controller.notify_activity)
        controller.start()

        # Wait for the controller to attempt unload once (should fail)
        first_attempt = await _wait_for_condition(lambda: calls["count"] >= 1)
        assert first_attempt, "Controller should have attempted to unload the model"
        assert calls["count"] == 1

        # Short wait to ensure no immediate retry (backoff applied)
        # Wait a bit longer than the backoff period to ensure no retry
        await asyncio.sleep(0.2)
        assert calls["count"] == 1, "No retry should occur during backoff period"

        await controller.stop()

    asyncio.run(_test())


@pytest.mark.asyncio
async def test_central_controller_unloads_idle_model_with_fake_registry() -> None:
    """Central controller should call registry.request_vram_unload for idle models."""
    unloaded = asyncio.Event()

    class FakeRegistry:
        def list_models(self) -> list[dict]:
            return [{"id": "m1", "metadata": {"auto_unload_minutes": 0}}]

        def get_vram_status(self, _model_id: str) -> dict:
            return {
                "vram_loaded": True,
                "vram_last_load_ts": time.time() - 3600,
                "vram_last_unload_ts": 0,
                "active_requests": 0,
            }

        def get_handler(self, _model_id: str) -> None:
            return None

        async def request_vram_unload(self, _model_id: str) -> None:
            unloaded.set()

    registry = FakeRegistry()
    controller = CentralIdleAutoUnloadController(registry)
    try:
        controller.start()
        # Expect unload to be requested within a short timeout
        await asyncio.wait_for(unloaded.wait(), timeout=2)
    finally:
        await controller.stop()


@pytest.mark.asyncio
async def test_notify_activity_prevents_unload_with_fake_registry() -> None:
    """Calling notify_activity should prevent immediate unload when activity is recent."""
    unloaded = asyncio.Event()

    class FakeHandler:
        def __init__(self) -> None:
            self._seconds = 120.0

        def seconds_since_last_activity(self) -> float:
            return self._seconds

        def set_recent(self) -> None:
            self._seconds = 0.0

    handler = FakeHandler()

    class FakeRegistry:
        def list_models(self) -> list[dict]:
            return [{"id": "m1", "metadata": {"auto_unload_minutes": 1}}]

        def get_vram_status(self, _model_id: str) -> dict:
            return {
                "vram_loaded": True,
                "vram_last_load_ts": time.time() - 3600,
                "vram_last_unload_ts": 0,
                "active_requests": 0,
            }

        def get_handler(self, _model_id: str) -> FakeHandler:
            return handler

        async def request_vram_unload(self, _model_id: str) -> None:
            unloaded.set()

    registry = FakeRegistry()
    controller = CentralIdleAutoUnloadController(registry)
    try:
        controller.start()

        # Simulate recent activity before controller attempts unload
        handler.set_recent()
        controller.notify_activity("m1")

        # Give the controller a short window to act; it should NOT call unload
        try:
            await asyncio.wait_for(unloaded.wait(), timeout=1)
            pytest.fail("Unload was triggered despite recent activity")
        except TimeoutError:
            # Expected: no unload call
            pass
    finally:
        await controller.stop()
