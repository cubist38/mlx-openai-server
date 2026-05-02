"""Unit tests for the VLM continuous batch scheduler."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from dataclasses import dataclass
import importlib
import sys
import types
from typing import Any, Self
from unittest.mock import Mock

import pytest


@contextmanager
def _fake_stream_cm(_stream: Any):
    yield


def _fake_device(_device: Any = None) -> object:
    """Return a placeholder MLX device object for scheduler tests."""
    return object()


async def _collect_stream(stream: Any) -> list[Any]:
    """Collect an async stream into a list."""
    return [chunk async for chunk in stream]


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

    last_kwargs: dict[str, Any] = {}
    script_tokens: list[int] = [10, 11]
    script_finish_reason: str = "length"
    last_instance: _FakeVLMBatchGenerator | None = None

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        type(self).last_kwargs = dict(_kwargs)
        type(self).last_instance = self
        self.uid_count = 0
        self.pending: list[tuple[int, int]] = []
        self.removed: list[int] = []
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
            token = type(self).script_tokens[index]
            is_last = index == len(type(self).script_tokens) - 1
            finish = type(self).script_finish_reason if is_last else None
            responses.append(_FakeResponse(uid=uid, token=token, finish_reason=finish))
            if finish is None:
                updated.append((uid, index + 1))
        self.pending = updated
        return [], responses

    def remove(self, uid: int) -> bool:
        """Remove a pending sequence."""
        self.removed.append(uid)
        self.pending = [entry for entry in self.pending if entry[0] != uid]
        return True

    def close(self) -> None:
        """Mark the fake generator closed."""
        self.closed = True


@pytest.fixture
def vlm_scheduler_module(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Import the VLM scheduler with MLX and mlx-vlm faked out."""
    _FakeVLMBatchGenerator.last_kwargs = {}
    _FakeVLMBatchGenerator.script_tokens = [10, 11]
    _FakeVLMBatchGenerator.script_finish_reason = "length"
    _FakeVLMBatchGenerator.last_instance = None
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

    class _FakeProcessor:
        tokenizer = types.SimpleNamespace(
            eos_token_id=2,
            decode=lambda tokens: "".join(f"<{token}>" for token in tokens),
        )

        @property
        def detokenizer(self) -> _FakeDetokenizer:
            return _FakeDetokenizer()

    class _FakeWrapper:
        model = _FakeInnerModel()
        processor = _FakeProcessor()
        config = types.SimpleNamespace(eos_token_id=2)
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
    assert _FakeVLMBatchGenerator.last_kwargs["stop_tokens"] == {2}


@pytest.mark.asyncio
async def test_vlm_generation_lock_is_released_between_decode_steps(
    vlm_scheduler_module: Any,
) -> None:
    """Seeded fallback requests need lock opportunities while a VLM batch is active."""

    class _FakeWrapper:
        model = types.SimpleNamespace(language_model=object())
        processor = types.SimpleNamespace(
            tokenizer=types.SimpleNamespace(
                eos_token_id=2,
                decode=lambda tokens: "".join(f"<{token}>" for token in tokens),
            )
        )
        config = types.SimpleNamespace(eos_token_id=2)
        kv_bits = None
        kv_group_size = 64
        quantized_kv_start = 0

        def create_batch_prompt_kwargs(
            self, _model_inputs: dict[str, Any]
        ) -> tuple[list[int], dict[str, Any]]:
            return [1, 2, 3], {"inputs_embeds": "embeds"}

    class _CountingLock:
        def __init__(self) -> None:
            self.enters = 0

        def __enter__(self) -> Self:
            self.enters += 1
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    lock = _CountingLock()
    scheduler = vlm_scheduler_module.VLMBatchScheduler(
        _FakeWrapper(),
        generation_lock=lock,
        idle_poll_timeout=0.01,
    )
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

    assert [chunk.token for chunk in chunks] == [10, 11]
    assert lock.enters >= 2


@pytest.mark.asyncio
async def test_vlm_batch_scheduler_forces_stop_on_eos_token(
    vlm_scheduler_module: Any,
) -> None:
    """Scheduler should stop even if upstream VLM batcher misses EOS."""

    class _FakeWrapper:
        model = types.SimpleNamespace(language_model=object())
        processor = types.SimpleNamespace(
            tokenizer=types.SimpleNamespace(
                eos_token_id=2,
                decode=lambda tokens: "".join(f"<{token}>" for token in tokens),
            )
        )
        config = types.SimpleNamespace(eos_token_id=2)
        kv_bits = None
        kv_group_size = 64
        quantized_kv_start = 0

        def create_batch_prompt_kwargs(
            self, _model_inputs: dict[str, Any]
        ) -> tuple[list[int], dict[str, Any]]:
            return [1, 2, 3], {"inputs_embeds": "embeds"}

    _FakeVLMBatchGenerator.script_tokens = [2, 99]
    scheduler = vlm_scheduler_module.VLMBatchScheduler(_FakeWrapper(), idle_poll_timeout=0.01)
    scheduler.start()
    try:
        stream = scheduler.submit_stream(
            model_inputs={"input_ids": _FakeArray([[1, 2, 3]])},
            max_tokens=8,
            logits_processors=[],
        )
        chunks = [chunk async for chunk in stream]
        fake = _FakeVLMBatchGenerator.last_instance
    finally:
        scheduler.stop()

    assert len(chunks) == 1
    assert chunks[0].token == 2
    assert chunks[0].text == ""
    assert chunks[0].finish_reason == "stop"
    assert fake is not None
    assert fake.removed == [0]


@pytest.mark.asyncio
async def test_vlm_batch_scheduler_uses_decode_when_detokenizer_is_shared(
    vlm_scheduler_module: Any,
) -> None:
    """A shared processor detokenizer should not be reused across active requests."""

    class _FakeTokenizer:
        eos_token_id = 2

        def decode(self, tokens: list[int]) -> str:
            """Decode tokens into visible fake text."""
            return "".join(f"<{token}>" for token in tokens)

    class _FakeWrapper:
        model = types.SimpleNamespace(language_model=object())
        processor = types.SimpleNamespace(
            tokenizer=_FakeTokenizer(),
            detokenizer=_FakeDetokenizer(),
        )
        config = types.SimpleNamespace(eos_token_id=2)
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
        first = scheduler.submit_stream(
            model_inputs={"input_ids": _FakeArray([[1, 2, 3]])},
            max_tokens=2,
            logits_processors=[],
        )
        second = scheduler.submit_stream(
            model_inputs={"input_ids": _FakeArray([[1, 2, 3]])},
            max_tokens=2,
            logits_processors=[],
        )
        first_chunks, second_chunks = await asyncio.gather(
            _collect_stream(first),
            _collect_stream(second),
        )
    finally:
        scheduler.stop()

    assert [chunk.text for chunk in first_chunks] == ["<10>", "<11>"]
    assert [chunk.text for chunk in second_chunks] == ["<10>", "<11>"]


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


def test_vlm_handler_positive_seed_disables_batching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Positive seeded requests should use the single-request VLM path."""
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
    model_params = {"temperature": 0.0, "top_p": 1.0, "model_inputs": {}}

    assert handler._is_request_batchable(Mock(seed=None), model_params) is True
    assert handler._is_request_batchable(Mock(seed=0), model_params) is True
    assert handler._is_request_batchable(Mock(seed=-1), model_params) is True
    assert handler._is_request_batchable(Mock(seed=42), model_params) is False


def test_vlm_batch_scheduler_stop_is_bounded(vlm_scheduler_module: Any) -> None:
    """Scheduler stop should not wait for the parent process shutdown timeout."""

    class _StuckThread:
        def __init__(self) -> None:
            self.join_timeout: float | None = None

        def join(self, timeout: float | None = None) -> None:
            """Record the requested join timeout without exiting."""
            self.join_timeout = timeout

        def is_alive(self) -> bool:
            """Pretend the thread is still exiting."""
            return True

    scheduler = vlm_scheduler_module.VLMBatchScheduler(types.SimpleNamespace())
    stuck_thread = _StuckThread()
    scheduler._running = True
    scheduler._thread = stuck_thread

    scheduler.stop()

    assert scheduler._running is False
    assert stuck_thread.join_timeout == 2.0


def test_vlm_batch_scheduler_accepts_scalar_eos_token_ids(vlm_scheduler_module: Any) -> None:
    """Tokenizers may expose ``eos_token_ids`` as a scalar int."""

    wrapper = types.SimpleNamespace(
        processor=types.SimpleNamespace(
            tokenizer=types.SimpleNamespace(eos_token_ids=7, eos_token_id=8)
        ),
        config=types.SimpleNamespace(eos_token_id=[2, None]),
    )
    scheduler = vlm_scheduler_module.VLMBatchScheduler(wrapper)

    assert scheduler._resolve_stop_tokens() == {2, 7, 8}


def test_vlm_batch_scheduler_resets_orphaned_upstream_work(
    vlm_scheduler_module: Any,
) -> None:
    """Stale upstream batch work should not keep taking the model lock."""

    scheduler = vlm_scheduler_module.VLMBatchScheduler(
        types.SimpleNamespace(
            model=types.SimpleNamespace(language_model=object()),
            processor=types.SimpleNamespace(),
        )
    )
    old_batch = _FakeVLMBatchGenerator()
    old_batch.pending.append((0, 0))
    scheduler._batch_generator = old_batch
    scheduler._batch_generator_kwargs = {}

    scheduler._reset_orphaned_batch_generator()

    assert old_batch.closed is True
    assert scheduler._batch_generator is not old_batch
    assert scheduler._batch_generator.has_work is False


def test_vlm_batch_scheduler_encodes_textual_eos_token(vlm_scheduler_module: Any) -> None:
    """Textual EOS markers should be encoded into stop token ids."""

    class _MockProcessor:
        tokenizer = types.SimpleNamespace(eos_token="[EOS]")

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            """Encode fake EOS text into a deterministic token id."""
            if "[EOS]" in text and add_special_tokens is False:
                return [32008]
            return [1]

    wrapper = types.SimpleNamespace(
        processor=_MockProcessor(),
        config=types.SimpleNamespace(eos_token_id=[2, 32000, 32007]),
    )
    scheduler = vlm_scheduler_module.VLMBatchScheduler(wrapper)

    assert scheduler._resolve_stop_tokens() == {2, 32000, 32007, 32008}
