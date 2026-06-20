"""Regression tests for handler subprocess startup timeout configuration."""

from __future__ import annotations

import queue
from typing import Any

import pytest

from app.config import ModelEntryConfig
from app.core.handler_process import HandlerProcessProxy


class _FakeProcess:
    """Minimal process stub used by ``HandlerProcessProxy.start``."""

    pid: int = 12345
    exitcode: int | None = None

    def start(self) -> None:
        """Pretend to start a subprocess."""

    def is_alive(self) -> bool:
        """Return True while startup is being tested."""

        return True


class _FakeSpawnContext:
    """Minimal spawn context that avoids creating a real child process."""

    def Queue(self) -> queue.Queue[dict[str, Any]]:
        """Return an in-process queue compatible with the reader thread."""

        return queue.Queue()

    def Process(self, **_kwargs: Any) -> _FakeProcess:
        """Return a fake process object."""

        return _FakeProcess()


@pytest.mark.asyncio
async def test_start_uses_configured_startup_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Proxy startup should wait using ``startup_timeout``, not request timeout."""

    model_cfg = ModelEntryConfig(
        model_path="dummy-model",
        model_type="lm",
        served_model_name="dummy-model",
        startup_timeout=3600,
        queue_timeout=30,
    )
    proxy = HandlerProcessProxy(
        model_cfg_dict=model_cfg.__dict__.copy(),
        model_type=model_cfg.model_type,
        model_path=model_cfg.model_path,
        served_model_name=model_cfg.served_model_name,
    )
    proxy._ctx = _FakeSpawnContext()  # type: ignore[assignment]
    observed_timeout: list[float] = []

    async def _fake_wait_for_ready(
        _ready_queue: queue.Queue[dict[str, Any]],
        timeout: float = 300,
    ) -> dict[str, Any]:
        observed_timeout.append(timeout)
        return {"type": "ready", "success": True}

    monkeypatch.setattr(proxy, "_wait_for_ready", _fake_wait_for_ready)

    try:
        await proxy.start(
            {
                "startup_timeout": model_cfg.startup_timeout,
                "timeout": model_cfg.queue_timeout,
                "queue_size": model_cfg.queue_size,
            }
        )
    finally:
        proxy._running = False
        if proxy._reader_thread is not None:
            proxy._reader_thread.join(timeout=2)

    assert observed_timeout == [3600.0]
    assert proxy._rpc_timeout == 30.0
