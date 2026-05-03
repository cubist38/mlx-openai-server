"""Tests for :class:`app.core.inference_worker.InferenceWorker`."""

from __future__ import annotations

from contextlib import contextmanager
import threading
import types
from typing import Any

import pytest

from app.core.inference_worker import InferenceWorker


@pytest.mark.asyncio
async def test_submit_runs_inside_worker_thread_local_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Submitted work should execute inside the worker-owned MLX stream context."""
    fake_mlx = types.ModuleType("mlx")
    fake_mx = types.ModuleType("mlx.core")
    local = threading.local()
    stream_obj = object()

    fake_mx.default_device = lambda: "gpu"
    fake_mx.new_thread_local_stream = lambda device: stream_obj

    @contextmanager
    def fake_stream(stream: object) -> Any:
        local.active_stream = stream
        try:
            yield
        finally:
            local.active_stream = None

    fake_mx.stream = fake_stream
    fake_mlx.core = fake_mx
    monkeypatch.setitem(__import__("sys").modules, "mlx", fake_mlx)
    monkeypatch.setitem(__import__("sys").modules, "mlx.core", fake_mx)

    worker = InferenceWorker()
    worker.start()
    try:
        result = await worker.submit(
            lambda: local.active_stream is stream_obj and worker._stream is stream_obj
        )
    finally:
        worker.stop()

    assert result is True
    assert worker._stream is None
