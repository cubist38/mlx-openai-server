"""Unit tests for the VLM continuous batch scheduler."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import importlib
import sys
import types
from typing import Any
from unittest.mock import Mock

import pytest


@contextmanager
def _fake_stream_cm(_stream: Any):
    yield


def _fake_device(_device: Any = None) -> object:
    """Return a placeholder MLX device object for scheduler tests."""
    return object()


class _FakeArray:
    """Tiny array stand-in with the surface used by the scheduler."""

    ndim = 2

    def __init__(self, values: list[list[int]]) -> None:
        self._values = values

    def tolist(self) -> list[list[int]]:
        """Return nested token IDs."""
        return self._values


class _FakeDetokenizer:
    """Incremental detokenizer used by scheduler tests."""

    def __init__(self) -> None:
        self.text = ""
        self.offset = 0

    def reset(self) -> None:
        """Reset accumulated text."""
        self.text = ""
        self.offset = 0

    def add_token(self, token: int) -> None:
        """Append a visible token segment."""
        self.text += f"<{token}>"

    def finalize(self) -> None:
        """Finalize the fake detokenizer."""
        return

    @property
    def last_segment(self) -> str:
        """Return text since the previous read."""
        segment = self.text[self.offset :]
        self.offset = len(self.text)
        return segment


@dataclass
class _FakeResponse:
    """Shape of ``mlx_vlm.generate.GenerationBatch.Response``."""

    uid: int
    token: int
    finish_reason: str | None = None


class _FakeVLMBatchGenerator:
    """In-memory stand-in for ``mlx_vlm.generate.BatchGenerator``."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.uid_count = 0
        self.pending: list[tuple[int, int]] = []
        self.closed = False

    def insert(
        self,
        prompts: list[list[int]],
        max_tokens: list[int],
        prompt_kwargs: list[dict[str, Any]],
        logits_processors: list[list[Any] | None],
    ) -> list[int]:
        """Insert prompts and return generated UIDs."""
        assert prompt_kwargs[0]["inputs_embeds"] == "embeds"
        assert logits_processors == [[]]
        uids: list[int] = []
        for _prompt in prompts:
            uid = self.uid_count
            self.uid_count += 1
            self.pending.append((uid, 0))
            uids.append(uid)
        return uids

    @property
    def has_work(self) -> bool:
        """Return whether any sequences remain."""
        return bool(self.pending)

    def next(self) -> tuple[list[Any], list[_FakeResponse]]:
        """Emit two tokens per request."""
        responses: list[_FakeResponse] = []
        updated: list[tuple[int, int]] = []
        for uid, index in self.pending:
            token = 10 + index
            finish = "length" if index == 1 else None
            responses.append(_FakeResponse(uid=uid, token=token, finish_reason=finish))
            if finish is None:
                updated.append((uid, index + 1))
        self.pending = updated
        return [], responses

    def remove(self, uid: int) -> bool:
        """Remove a pending sequence."""
        self.pending = [entry for entry in self.pending if entry[0] != uid]
        return True

    def close(self) -> None:
        """Mark the fake generator closed."""
        self.closed = True


@pytest.fixture
def vlm_scheduler_module(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Import the VLM scheduler with MLX and mlx-vlm faked out."""
    fake_mx = types.ModuleType("mlx.core")
    fake_mx.stream = _fake_stream_cm
    fake_mx.new_stream = _fake_device
    fake_mx.new_thread_local_stream = _fake_device
    fake_mx.default_device = _fake_device
    fake_mx.get_peak_memory = lambda: 0

    fake_mlx = types.ModuleType("mlx")
    fake_mlx.core = fake_mx

    fake_generate = types.ModuleType("mlx_vlm.generate")
    fake_generate.BatchGenerator = _FakeVLMBatchGenerator
    fake_vlm = types.ModuleType("mlx_vlm")
    fake_vlm.generate = fake_generate

    monkeypatch.setitem(sys.modules, "mlx", fake_mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)
    monkeypatch.setitem(sys.modules, "mlx_vlm", fake_vlm)
    monkeypatch.setitem(sys.modules, "mlx_vlm.generate", fake_generate)
    monkeypatch.delitem(sys.modules, "app.core.vlm_batch_scheduler", raising=False)

    return importlib.import_module("app.core.vlm_batch_scheduler")


@pytest.mark.asyncio
async def test_vlm_batch_scheduler_streams_tokens(vlm_scheduler_module: Any) -> None:
    """A VLM batch request should stream decoded chunks and a final chunk."""

    class _FakeEmbedding:
        def to_dict(self) -> dict[str, str]:
            return {"inputs_embeds": "embeds"}

    class _FakeInnerModel:
        language_model = object()

        def get_input_embeddings(self, *_args: Any, **_kwargs: Any) -> _FakeEmbedding:
            return _FakeEmbedding()

    class _FakeWrapper:
        model = _FakeInnerModel()
        processor = types.SimpleNamespace(detokenizer=_FakeDetokenizer())
        kv_bits = None
        kv_group_size = 64
        quantized_kv_start = 0

        def create_batch_prompt_kwargs(
            self, _model_inputs: dict[str, Any]
        ) -> tuple[list[int], dict[str, Any]]:
            return [1, 2, 3], {"inputs_embeds": "embeds"}

    scheduler = vlm_scheduler_module.VLMBatchScheduler(_FakeWrapper())
    scheduler.start()
    try:
        stream = scheduler.submit_stream(
            model_inputs={"input_ids": _FakeArray([[1, 2, 3]])},
            max_tokens=2,
            logits_processors=[],
        )
        chunks = [chunk async for chunk in stream]
    finally:
        scheduler.stop()

    assert [chunk.text for chunk in chunks] == ["<10>", "<11>"]
    assert chunks[-1].finish_reason == "length"
    assert chunks[-1].prompt_tokens == 3


def test_vlm_handler_does_not_batch_audio_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Audio-derived model inputs should stay on the single-request VLM path."""
    fake_scheduler = types.ModuleType("app.core.vlm_batch_scheduler")
    fake_scheduler.VLM_BATCHING_AVAILABLE = True
    fake_scheduler.VLMBatchScheduler = object
    monkeypatch.setitem(sys.modules, "app.core.vlm_batch_scheduler", fake_scheduler)

    fake_core = types.ModuleType("app.core")
    fake_core.AudioProcessor = object
    fake_core.ImageProcessor = object
    fake_core.InferenceWorker = object
    fake_core.VideoProcessor = object
    monkeypatch.setitem(sys.modules, "app.core", fake_core)

    fake_model_module = types.ModuleType("app.models.mlx_vlm")
    fake_model_module.DEFAULT_TEMPERATURE = 0.0
    fake_model_module.DEFAULT_TOP_P = 1.0
    fake_model_module.CompletionResponse = object
    fake_model_module.MLX_VLM = object
    monkeypatch.setitem(sys.modules, "app.models.mlx_vlm", fake_model_module)
    monkeypatch.delitem(sys.modules, "app.handler.mlx_vlm", raising=False)

    handler_module = importlib.import_module("app.handler.mlx_vlm")
    handler = handler_module.MLXVLMHandler.__new__(handler_module.MLXVLMHandler)

    request = Mock(seed=0)
    model_params = {
        "temperature": 0.0,
        "top_p": 1.0,
        "model_inputs": {
            "input_features": object(),
            "feature_attention_mask": object(),
        },
    }

    assert handler._is_request_batchable(request, model_params) is False
