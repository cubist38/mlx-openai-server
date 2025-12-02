"""Unit tests for ModelRegistry behavior."""

from __future__ import annotations

import asyncio
from contextlib import AbstractAsyncContextManager
import time
from types import TracebackType
from typing import Any

from app.core.model_registry import ModelRegistry


def test_registry_tracks_handlers_and_metadata() -> None:
    """Ensure the registry records status, handler references, and metadata."""
    asyncio.run(_exercise_registry())


async def _exercise_registry() -> None:
    registry = ModelRegistry()
    registry.register_model(
        model_id="foo",
        handler=None,
        model_type="lm",
        context_length=2048,
        metadata_extras={"group": "default"},
    )

    models = registry.list_models()
    assert len(models) == 1
    first = models[0]
    assert first["id"] == "foo"
    assert first["metadata"]["status"] == "unloaded"
    assert first["metadata"]["group"] == "default"

    handler = object()
    await registry.update_model_state(
        "foo",
        handler=handler,
        status="initialized",
        metadata_updates={"model_path": "/models/foo"},
    )

    assert registry.get_handler("foo") is handler
    updated = registry.list_models()[0]
    assert updated["metadata"]["status"] == "initialized"
    assert updated["metadata"]["model_path"] == "/models/foo"

    await registry.update_model_state("foo", handler=None, status="unloaded")
    assert registry.get_handler("foo") is None

    await registry.unregister_model("foo")
    assert registry.get_model_count() == 0


class DummyManager:
    """A simple manager mock that records calls and simulates VRAM state."""

    def __init__(self) -> None:
        self._loaded = False
        self.load_calls = 0
        self.unload_calls = 0
        self.ensure_lock = asyncio.Lock()
        self._active_requests = 0

    def is_vram_loaded(self) -> bool:
        """Return whether VRAM is currently loaded for this manager."""
        return self._loaded

    async def ensure_vram_loaded(
        self,
        *,
        force: bool = False,
        timeout: float | None = None,
    ) -> None:
        """Ensure VRAM is loaded; simulate a delay for loading."""
        async with self.ensure_lock:
            # Simulate expensive load
            if not self._loaded or force:
                self.load_calls += 1
                await asyncio.sleep(0.01)
                self._loaded = True

    async def release_vram(self, *, timeout: float | None = None) -> None:
        """Release VRAM for this manager, simulating an unload delay."""
        async with self.ensure_lock:
            if self._loaded:
                self.unload_calls += 1
                await asyncio.sleep(0.005)
                self._loaded = False

    def request_session(
        self,
        *,
        ensure_vram: bool = True,
        ensure_timeout: float | None = None,
    ) -> AbstractAsyncContextManager[Any]:
        """Return an async context manager for request sessions."""
        return DummySession(self, ensure_vram, ensure_timeout)


class DummySession:
    """Simple async context manager for DummyManager sessions."""

    def __init__(
        self,
        manager: DummyManager,
        ensure_vram: bool,
        ensure_timeout: float | None,
    ) -> None:
        self.manager = manager
        self.ensure_vram = ensure_vram
        self.ensure_timeout = ensure_timeout

    async def __aenter__(self) -> DummyManager:
        """Enter the async context manager."""
        self.manager._active_requests += 1
        if self.ensure_vram:
            await self.manager.ensure_vram_loaded(timeout=self.ensure_timeout)
        return self.manager

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit the async context manager."""
        self.manager._active_requests -= 1


def test_get_or_attach_manager_concurrent() -> None:
    """Concurrent callers should share a single loader invocation (stress test).

    This test uses the same registry implementation but spawns multiple
    concurrent callers to `get_or_attach_manager` to ensure only one
    loader runs.
    """

    async def _test() -> None:
        registry = ModelRegistry()
        model_id = "concurrent-model"
        registry.register_model(model_id, handler=None, model_type="lm")

        loader_call_count = 0

        async def loader(mid: str) -> DummyManager:
            nonlocal loader_call_count
            loader_call_count += 1
            # Simulate async init delay
            await asyncio.sleep(0.02)
            return DummyManager()

        # Spawn multiple concurrent callers that should share one loader task.
        tasks = [
            asyncio.create_task(registry.get_or_attach_manager(model_id, loader)) for _ in range(8)
        ]
        results = await asyncio.gather(*tasks)

        # All returned managers should be the same instance
        assert all(r is results[0] for r in results)
        assert loader_call_count == 1
        assert registry.get_handler(model_id) is results[0]

    asyncio.run(_test())


def test_request_vram_load_unload_idempotent_concurrent() -> None:
    """Concurrent load/unload calls should behave idempotently."""

    async def _test() -> None:
        registry = ModelRegistry()
        model_id = "vram-model"
        registry.register_model(model_id, handler=None, model_type="lm")

        # Attach a DummyManager directly via update_model_state (simulate an attached manager)
        manager = DummyManager()
        await registry.update_model_state(model_id, handler=manager)

        # Concurrently request loads
        load_tasks = [asyncio.create_task(registry.request_vram_load(model_id)) for _ in range(6)]
        await asyncio.gather(*load_tasks)

        # Manager must report loaded and registry should reflect it
        status = registry.get_vram_status(model_id)
        assert status["vram_loaded"] is True
        assert status["vram_last_load_ts"] is not None
        assert manager.load_calls == 1

        # Concurrently request unloads
        unload_tasks = [
            asyncio.create_task(registry.request_vram_unload(model_id)) for _ in range(4)
        ]
        await asyncio.gather(*unload_tasks)

        status2 = registry.get_vram_status(model_id)
        assert status2["vram_loaded"] is False
        assert status2["vram_last_unload_ts"] is not None
        # After unload, a load then another load (force) should increment
        await registry.request_vram_load(model_id)
        assert manager.load_calls >= 2

    asyncio.run(_test())


def test_handler_session_updates_active_requests_and_notifies() -> None:
    """handler_session should update active_requests and call notifier (stress test)."""

    async def _test() -> None:
        registry = ModelRegistry()
        model_id = "session-model"
        registry.register_model(model_id, handler=None, model_type="lm")

        manager = DummyManager()
        await registry.update_model_state(model_id, handler=manager)

        notified: list[str] = []

        def notifier(mid: str) -> None:
            notified.append(mid)

        registry.register_activity_notifier(notifier)

        async with registry.handler_session(model_id):
            # inside session active_requests should be 1
            status = registry.get_vram_status(model_id)
            assert status["active_requests"] == 1

        # after exiting, active_requests should be 0 and notifier called
        status2 = registry.get_vram_status(model_id)
        assert status2["active_requests"] == 0
        assert model_id in notified

    asyncio.run(_test())


def test_request_vram_idempotency() -> None:
    """VRAM load/unload requests should be idempotent and respect `force`."""

    async def _test() -> None:
        registry = ModelRegistry()
        manager = DummyManager()
        registry.register_model(model_id="idm", handler=manager, model_type="lm")

        # First load
        await registry.request_vram_load("idm")
        assert manager.is_vram_loaded()
        assert manager.load_calls == 1

        # Second load without force should be no-op (manager may be idempotent)
        await registry.request_vram_load("idm")
        assert manager.load_calls == 1

        # Force reload triggers another load
        await registry.request_vram_load("idm", force=True)
        assert manager.load_calls == 2

        # Unload
        await registry.request_vram_unload("idm")
        assert not manager.is_vram_loaded()
        assert manager.unload_calls == 1

        # Double unload is a no-op
        await registry.request_vram_unload("idm")
        assert manager.unload_calls == 1

    asyncio.run(_test())


def test_handler_session_counts_and_ensure() -> None:
    """handler_session should increment active_requests and ensure VRAM loaded."""

    async def _test() -> None:
        registry = ModelRegistry()
        manager = DummyManager()
        registry.register_model(model_id="sess", handler=manager, model_type="lm")

        # Before any sessions
        status0 = registry.get_vram_status("sess")
        assert status0["active_requests"] == 0

        # Enter a session; active_requests increments and ensure_vram called
        async with registry.handler_session("sess") as m:
            assert m is manager
            status1 = registry.get_vram_status("sess")
            assert status1["active_requests"] == 1
            # ensure_vram should have loaded the model
            assert manager.is_vram_loaded()

        # After exit, active_requests returns to zero
        status2 = registry.get_vram_status("sess")
        assert status2["active_requests"] == 0

    asyncio.run(_test())


def test_group_availability_limits_and_idle_override() -> None:
    """Group policies should gate availability and honor idle thresholds."""

    async def _test() -> None:
        registry = ModelRegistry()
        for name in ("alpha", "beta"):
            registry.register_model(
                model_id=name,
                handler=None,
                model_type="lm",
                metadata_extras={"group": "shared"},
            )

        now = time.time()
        await registry.update_model_state(
            "alpha",
            metadata_updates={"vram_loaded": True, "vram_last_request_ts": int(now)},
        )
        await registry.update_model_state("beta", metadata_updates={"vram_loaded": False})

        registry.set_group_policies({"shared": {"max_loaded": 1}})
        assert registry.is_model_available("alpha") is True
        assert registry.is_model_available("beta") is False
        snapshots = registry.get_group_snapshots()
        assert snapshots["shared"]["mode"] == "loaded-only"

        registry.set_group_policies({"shared": {"max_loaded": 1, "idle_unload_trigger_min": 1}})
        await registry.update_model_state(
            "alpha",
            metadata_updates={"vram_loaded": True, "vram_last_request_ts": int(now - 120)},
        )
        assert registry.is_model_available("beta") is True
        refreshed = registry.get_group_snapshots()["shared"]
        assert "alpha" in refreshed["idle_eligible"]

    asyncio.run(_test())


def test_available_model_ids_return_cached_snapshot_copy() -> None:
    """`get_available_model_ids` should return a copy of the cached set."""

    async def _test() -> None:
        registry = ModelRegistry()
        registry.register_model(
            model_id="alpha",
            handler=None,
            model_type="lm",
            metadata_extras={"group": "shared"},
        )
        registry.register_model(
            model_id="beta",
            handler=None,
            model_type="lm",
            metadata_extras={"group": "shared"},
        )
        await registry.update_model_state("alpha", metadata_updates={"vram_loaded": True})

        registry.set_group_policies({"shared": {"max_loaded": 1}})

        snapshot = registry.get_available_model_ids()
        assert snapshot == {"alpha"}

        snapshot.add("beta")
        assert registry.is_model_available("beta") is False

    asyncio.run(_test())
