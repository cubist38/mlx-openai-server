"""Tests that ``MLXLMHandler.initialize()`` warms the model on the caller thread.

Same root cause as ``BatchScheduler._warm_up_model`` (see that docstring).
The May-14 follow-up comment on #290 reports Gemma 4 still failing with
``--disable-batching`` after #295 — that path routes through
``InferenceWorker``, which owns its own thread-local stream. Without a
loader-thread warm-up the model's deferred MLX state binds to the worker
thread on first use and later cross-thread evaluations raise
``RuntimeError: There is no Stream(gpu, N) in current thread``.
"""

from __future__ import annotations

import threading
from typing import Any

import pytest


def _install_stub_handler_deps(monkeypatch: pytest.MonkeyPatch, captured: list[int]) -> Any:
    """Stub out heavy handler dependencies so we can construct the handler
    in-process without loading a real MLX model. Returns the imported
    handler module (so the test can read its class).
    """

    class _StubRawModel:
        # Presence of ``.layers`` distinguishes a real model from unit-test
        # mocks; the warm-up must skip when absent.
        layers = (object(),)

        def __call__(self, _ids: Any, cache: Any = None) -> Any:
            captured.append(threading.get_ident())
            return object()  # opaque logits

    class _StubTokenizer:
        bos_token_id = 1
        pad_token_id = 0
        eos_token_ids: list[int] = [1]
        eos_token_id = 1
        vocab_size = 32000

    class _StubMLX_LM:
        def __init__(self, model_path: str, **_kwargs: Any) -> None:
            self.model = _StubRawModel()
            self.tokenizer = _StubTokenizer()
            self.draft_model = None
            self.draft_tokenizer = None
            self.model_type = "stub"
            self.pad_token_id = 0
            self.bos_token = "<s>"

        def get_model_type(self) -> str:
            return self.model_type

    # Patch the wrapper before the handler module imports it.
    import app.models.mlx_lm as model_module

    monkeypatch.setattr(model_module, "MLX_LM", _StubMLX_LM)

    # ``test_batch_scheduler.test_request_seed_does_not_disable_batching``
    # (and others in that module) pop ``app.handler.mlx_lm`` from
    # ``sys.modules`` and re-import it against a fake ``app.core`` that
    # replaces ``InferenceWorker`` with a no-arg stub. The pop is not
    # undone by ``monkeypatch``, so the handler module we just imported
    # may carry a stale ``_FakeInferenceWorker``. Re-bind to the real
    # class defensively.
    from app.core.inference_worker import InferenceWorker as _RealInferenceWorker
    import app.handler.mlx_lm as handler_module

    monkeypatch.setattr(handler_module, "InferenceWorker", _RealInferenceWorker)

    # Bypass parser loading (returns ``None`` is safe; handler tolerates it).
    monkeypatch.setattr(
        handler_module.MessageConverterManager,
        "create_converter",
        staticmethod(lambda **_kwargs: None),
    )

    # Stub the warm-up's MLX-side calls so it runs end-to-end without real MLX.
    import mlx_lm.models.cache as cache_mod

    monkeypatch.setattr(cache_mod, "make_prompt_cache", lambda _model: [])
    import mlx.core as mx

    monkeypatch.setattr(mx, "array", lambda _data: object())
    monkeypatch.setattr(mx, "eval", lambda *_args, **_kwargs: None)

    return handler_module


@pytest.mark.asyncio
async def test_initialize_warms_up_model_on_caller_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``initialize()`` must invoke the model on the caller thread before the worker spawns.

    This is the ``--disable-batching`` path — the one the May-14 follow-up
    comment on #290 reports is still broken on Gemma 4 even after #295.
    """
    main_tid = threading.get_ident()
    call_threads: list[int] = []
    handler_module = _install_stub_handler_deps(monkeypatch, call_threads)

    handler = handler_module.MLXLMHandler(model_path="stub/path")
    try:
        await handler.initialize({})
    finally:
        # Best-effort cleanup; the worker thread is harmless either way.
        if hasattr(handler, "inference_worker"):
            handler.inference_worker.stop()

    assert main_tid in call_threads, (
        "MLXLMHandler.initialize() must run a warm-up forward pass on the "
        f"caller thread (id={main_tid}); observed call threads={call_threads}"
    )
