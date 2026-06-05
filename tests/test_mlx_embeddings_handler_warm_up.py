"""Tests that ``MLXEmbeddingsHandler.initialize()`` warms the model on the caller thread.

Same root cause as the LM warm-ups in ``BatchScheduler._warm_up_model`` and
``MLXLMHandler._warm_up_model`` (see those docstrings). Embedding requests
route through ``InferenceWorker``, which owns its own per-thread MLX stream
from #295; without a loader-thread warm-up the model's deferred state binds
to the worker thread on first use and later cross-thread evaluations raise
``RuntimeError: There is no Stream(gpu, N) in current thread``. The
embedding fix is preventive — there is no specific report against this
path yet, but it shares the failure mode #290/#295 covered for LM.
"""

from __future__ import annotations

import threading
from typing import Any

import pytest


def _install_stub_handler_deps(monkeypatch: pytest.MonkeyPatch, captured: list[int]) -> Any:
    """Stub the embeddings model wrapper so we can construct the handler
    without loading a real MLX model. Returns the handler module.
    """

    class _StubEmbeddingsModel:
        # Sentinel attribute so the warm-up can detect a real model vs a
        # bare ``object()``. Mirrors the ``.layers`` check used in the LM
        # path; an embedding model exposes ``__call__`` so we use the same
        # callable-based skip rule.
        layers = (object(),)

        def __init__(self, _model_path: str) -> None:
            pass

        def __call__(self, texts: list[str], max_length: int = 512) -> list[list[float]]:
            captured.append(threading.get_ident())
            return [[0.0] * 4 for _ in texts]

        def cleanup(self) -> None:
            return None

    import app.models.mlx_embeddings as model_module

    monkeypatch.setattr(model_module, "MLX_Embeddings", _StubEmbeddingsModel)

    import app.handler.mlx_embeddings as handler_module

    return handler_module


@pytest.mark.asyncio
async def test_initialize_warms_up_embedding_model_on_caller_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``initialize()`` must invoke the embedding model on the caller thread before the worker spawns."""
    main_tid = threading.get_ident()
    call_threads: list[int] = []
    handler_module = _install_stub_handler_deps(monkeypatch, call_threads)

    handler = handler_module.MLXEmbeddingsHandler(model_path="stub/path")
    try:
        await handler.initialize({})
    finally:
        if hasattr(handler, "inference_worker"):
            handler.inference_worker.stop()

    assert main_tid in call_threads, (
        "MLXEmbeddingsHandler.initialize() must run a warm-up embed on the "
        f"caller thread (id={main_tid}); observed call threads={call_threads}"
    )
