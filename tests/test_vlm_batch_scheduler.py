"""Unit tests for :class:`app.core.vlm_batch_scheduler.VLMBatchScheduler`."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import sys
import types
from typing import Any

import pytest


@contextmanager
def _fake_stream_cm(_stream: Any):
    yield


def _fake_device(_device: Any = None) -> object:
    """Return a placeholder MLX device object for scheduler tests."""
    return object()


def _zero() -> int:
    """Return zero for fake memory counters."""
    return 0


@dataclass
class _FakeResponse:
    """Small response object matching the fields used by the scheduler."""

    uid: int
    token: int
    finish_reason: str | None = None


@dataclass
class _FakeScript:
    """Pre-programmed output for one fake sequence."""

    tokens: list[int]
    finish_reason: str = "length"


class _FakeStats:
    """Fake batch statistics."""

    prompt_tps = 123.0


class FakeVLMBatchGenerator:
    """In-memory stand-in for ``mlx_vlm.generate.BatchGenerator``."""

    script_queue: list[_FakeScript] = []
    init_kwargs: dict[str, Any] | None = None

    def __init__(self, model: Any, processor: Any, **kwargs: Any) -> None:
        self.model = model
        self.processor = processor
        self.__class__.init_kwargs = kwargs
        self._uid = 0
        self._pending: list[tuple[int, _FakeScript, int]] = []
        self._unprocessed_sequences: list[Any] = []
        self.removed: list[int] = []
        self.closed = False

    def insert(
        self,
        prompts: list[list[int]],
        max_tokens: int | list[int] | None = None,
        prompt_kwargs: list[dict[str, Any]] | None = None,
        logits_processors: list[Any] | None = None,
    ) -> list[int]:
        uids: list[int] = []
        for _prompt in prompts:
            script = (
                self.script_queue.pop(0)
                if self.script_queue
                else _FakeScript(tokens=[0], finish_reason="length")
            )
            uid = self._uid
            self._uid += 1
            self._pending.append((uid, script, 0))
            uids.append(uid)
        return uids

    def next(self) -> tuple[list[Any], list[Any]]:
        responses: list[Any] = []
        pending: list[tuple[int, _FakeScript, int]] = []
        for uid, script, index in self._pending:
            is_last = index == len(script.tokens) - 1
            responses.append(
                _FakeResponse(
                    uid=uid,
                    token=script.tokens[index],
                    finish_reason=script.finish_reason if is_last else None,
                )
            )
            if not is_last:
                pending.append((uid, script, index + 1))
        self._pending = pending
        return [], responses

    def remove(self, uid: int) -> bool:
        self.removed.append(uid)
        self._pending = [item for item in self._pending if item[0] != uid]
        return True

    @property
    def unprocessed_prompts(self) -> list[Any]:
        return self._unprocessed_sequences

    @property
    def has_pending_prompts(self) -> bool:
        return bool(self._unprocessed_sequences)

    def stats(self) -> _FakeStats:
        return _FakeStats()

    def close(self) -> None:
        self.closed = True


class FakeTokenizer:
    """Minimal tokenizer surface used by the scheduler."""

    def decode(self, tokens: list[int]) -> str:
        """Decode fake token ids into visible text."""
        return "".join(f"<{token}>" for token in tokens)


class FakeProcessor:
    """Minimal processor with a tokenizer."""

    tokenizer = FakeTokenizer()


class _FakeEmbeddingOutput:
    """Fake embedding result returned by ``get_input_embeddings``."""

    def to_dict(self) -> dict[str, Any]:
        """Return fake embedding kwargs."""
        return {"inputs_embeds": [[1, 2, 3]]}


class FakeModel:
    """Minimal VLM wrapper model used by tests."""

    language_model = object()

    def get_input_embeddings(self, *args: Any, **kwargs: Any) -> _FakeEmbeddingOutput:
        """Return fake embeddings for the scheduler to pass to BatchGenerator."""
        return _FakeEmbeddingOutput()


@pytest.fixture
def patched_vlm_scheduler(monkeypatch):
    """Patch MLX stream calls and VLM BatchGenerator."""
    from app.core import vlm_batch_scheduler as module

    monkeypatch.setattr(module, "BatchGenerator", FakeVLMBatchGenerator)
    monkeypatch.setattr(module.mx, "stream", _fake_stream_cm)
    monkeypatch.setattr(module.mx, "new_stream", _fake_device)
    monkeypatch.setattr(module.mx, "new_thread_local_stream", _fake_device, raising=False)
    monkeypatch.setattr(module.mx, "default_device", _fake_device)
    monkeypatch.setattr(module.mx, "get_peak_memory", _zero)
    FakeVLMBatchGenerator.script_queue = []
    FakeVLMBatchGenerator.init_kwargs = None
    return module


@pytest.mark.asyncio
async def test_vlm_scheduler_streams_chunks(patched_vlm_scheduler):
    """The VLM scheduler should stream per-request chunks and final stats."""
    FakeVLMBatchGenerator.script_queue = [_FakeScript(tokens=[11, 12])]
    scheduler = patched_vlm_scheduler.VLMBatchScheduler(
        FakeModel(),
        FakeProcessor(),
        completion_batch_size=4,
        prefill_batch_size=1,
        prefill_step_size=16,
        kv_bits=4,
        queue_size=10,
    )
    scheduler.start()
    try:
        stream = scheduler.submit_stream(
            {"input_ids": [[1, 2, 3]], "pixel_values": "image"},
            prompt_tokens=3,
            max_tokens=2,
        )
        chunks = [chunk async for chunk in stream]
    finally:
        scheduler.stop()

    assert [chunk.text for chunk in chunks] == ["<11>", "<12>"]
    assert chunks[-1].finish_reason == "length"
    assert chunks[-1].prompt_tokens == 3
    assert chunks[-1].generation_tokens == 2
    assert FakeVLMBatchGenerator.init_kwargs["completion_batch_size"] == 4
    assert FakeVLMBatchGenerator.init_kwargs["prefill_batch_size"] == 1
    assert FakeVLMBatchGenerator.init_kwargs["prefill_step_size"] == 16
    assert FakeVLMBatchGenerator.init_kwargs["kv_bits"] == 4


def test_vlm_handler_batchability_flag(monkeypatch):
    """``--disable-batching`` should route VLM requests to the worker path."""
    fake_mlx_vlm_module = types.ModuleType("mlx_vlm")
    fake_mlx_vlm_module.load = lambda *args, **kwargs: (None, None)
    fake_mlx_vlm_module.stream_generate = lambda *args, **kwargs: iter(())
    fake_sample_utils = types.ModuleType("mlx_vlm.sample_utils")
    fake_sample_utils.top_p_sampling = lambda logprobs, *_args: logprobs
    fake_utils = types.ModuleType("mlx_vlm.utils")
    fake_utils.process_inputs_with_fallback = lambda *args, **kwargs: {}
    fake_video = types.ModuleType("mlx_vlm.video_generate")
    fake_video.process_vision_info = lambda _messages: ([], [])
    fake_outlines = types.ModuleType("outlines")
    fake_processors = types.ModuleType("outlines.processors")

    class _FakeJSONLogitsProcessor:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.args = args
            self.kwargs = kwargs

    fake_processors.JSONLogitsProcessor = _FakeJSONLogitsProcessor
    fake_outlines_tokenizer = types.ModuleType("app.utils.outlines_transformer_tokenizer")

    class _FakeOutlinesTransformerTokenizer:
        def __init__(self, tokenizer: Any) -> None:
            self.tokenizer = tokenizer

    fake_outlines_tokenizer.OutlinesTransformerTokenizer = _FakeOutlinesTransformerTokenizer
    monkeypatch.setitem(sys.modules, "mlx_vlm", fake_mlx_vlm_module)
    monkeypatch.setitem(sys.modules, "mlx_vlm.sample_utils", fake_sample_utils)
    monkeypatch.setitem(sys.modules, "mlx_vlm.utils", fake_utils)
    monkeypatch.setitem(sys.modules, "mlx_vlm.video_generate", fake_video)
    monkeypatch.setitem(sys.modules, "outlines", fake_outlines)
    monkeypatch.setitem(sys.modules, "outlines.processors", fake_processors)
    monkeypatch.setitem(
        sys.modules,
        "app.utils.outlines_transformer_tokenizer",
        fake_outlines_tokenizer,
    )
    monkeypatch.delitem(sys.modules, "app.models.mlx_vlm", raising=False)
    monkeypatch.delitem(sys.modules, "app.handler.mlx_vlm", raising=False)

    from app.handler import mlx_vlm as module

    handler = object.__new__(module.MLXVLMHandler)
    handler._disable_batching = False
    monkeypatch.setattr(module, "VLM_BATCHING_AVAILABLE", True)
    assert handler._is_request_batchable(object()) is True

    handler._disable_batching = True
    assert handler._is_request_batchable(object()) is False
