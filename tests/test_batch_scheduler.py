"""Unit tests for :class:`app.core.batch_scheduler.BatchScheduler`.

The tests stub out ``mlx_lm.generate.BatchGenerator`` so the scheduler logic
(request admission, dispatch of generation responses, cancellation, and
final-chunk stats) can be exercised without loading a real MLX model.
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from dataclasses import dataclass, field
import importlib
import sys
import threading
import types
from typing import Any

import pytest


@contextmanager
def _fake_stream_cm(_stream: Any):
    yield


def _fake_device(_device: Any = None) -> object:
    """Return a placeholder MLX device object for sandboxed tests."""
    return object()


def _zero() -> int:
    """Return zero for fake memory counters."""
    return 0


class _FakeGenerationResponse:
    """Shape of ``mlx_lm.generate.GenerationBatch.Response`` used by the scheduler."""

    __slots__ = (
        "uid",
        "token",
        "logprobs",
        "finish_reason",
        "current_state",
        "match_sequence",
        "prompt_cache",
        "all_tokens",
    )

    def __init__(
        self,
        uid: int,
        token: int,
        finish_reason: str | None = None,
        all_tokens: list[int] | None = None,
    ) -> None:
        self.uid = uid
        self.token = token
        self.logprobs = None
        self.finish_reason = finish_reason
        self.current_state = None
        self.match_sequence = None
        self.prompt_cache = [] if finish_reason is not None else None
        self.all_tokens = all_tokens if finish_reason is not None else None


@dataclass
class _FakeScript:
    """Pre-programmed per-sequence output for :class:`FakeBatchGenerator`."""

    tokens: list[int]
    finish_reason: str = "length"


class FakeBatchGenerator:
    """In-memory stand-in for ``mlx_lm.generate.BatchGenerator``.

    Uses a shared ``FakeBatchGenerator.script_queue`` so tests can pre-load
    the outputs each subsequent ``insert`` call will emit. ``step_delay`` lets
    tests slow generation down so cancellation races can be exercised
    reliably.
    """

    script_queue: list[_FakeScript] = []
    step_delay: float = 0.0

    def __init__(self, model: Any, **_kwargs: Any) -> None:
        self._uid_counter = 0
        self._pending: list[tuple[int, _FakeScript, int]] = []
        self.removed: list[int] = []
        self.closed = False
        # Overridable by tests so the scheduler's admission-time reclaim
        # has a non-zero ``active`` figure to subtract against the LRU cap.
        self.prompt_cache_nbytes = 0

    def insert(
        self,
        prompts: list[list[int]],
        max_tokens: list[int] | None = None,
        caches: Any = None,
        all_tokens: Any = None,
        samplers: Any = None,
        logits_processors: Any = None,
        state_machines: Any = None,
    ) -> list[int]:
        uids: list[int] = []
        for _ in prompts:
            script = (
                self.script_queue.pop(0)
                if self.script_queue
                else _FakeScript(tokens=[0], finish_reason="length")
            )
            uid = self._uid_counter
            self._uid_counter += 1
            self._pending.append((uid, script, 0))
            uids.append(uid)
        return uids

    def insert_segments(
        self,
        segments: list[list[list[int]]],
        max_tokens: list[int] | None = None,
        caches: Any = None,
        all_tokens: Any = None,
        samplers: Any = None,
        logits_processors: Any = None,
        state_machines: Any = None,
    ) -> list[int]:
        # Flatten segments back into a single prompt per sequence; the fake
        # does not model segment-boundary prefill events.
        flattened = [[tok for seg in seqs for tok in seg] for seqs in segments]
        return self.insert(
            flattened,
            max_tokens=max_tokens,
            caches=caches,
            all_tokens=all_tokens,
            samplers=samplers,
            logits_processors=logits_processors,
            state_machines=state_machines,
        )

    def extract_cache(self, uids: list[int]) -> dict[int, tuple[Any, list[int]]]:
        # Return empty cache payloads so the scheduler's extract-on-segment
        # path is exercised without the fake needing to track real state.
        return {uid: ([], []) for uid in uids}

    def next(self) -> tuple[list[Any], list[Any]]:
        if self.step_delay > 0:
            import time as _time

            _time.sleep(self.step_delay)
        gen_responses: list[Any] = []
        updated: list[tuple[int, _FakeScript, int]] = []
        for uid, script, idx in self._pending:
            if idx >= len(script.tokens):
                continue
            tok = script.tokens[idx]
            is_last = idx == len(script.tokens) - 1
            gen_responses.append(
                _FakeGenerationResponse(
                    uid=uid,
                    token=tok,
                    finish_reason=script.finish_reason if is_last else None,
                    all_tokens=list(script.tokens) if is_last else None,
                )
            )
            if not is_last:
                updated.append((uid, script, idx + 1))
        self._pending = updated
        return [], gen_responses

    def remove(self, uids: list[int]) -> None:
        self.removed.extend(uids)
        self._pending = [p for p in self._pending if p[0] not in uids]

    def close(self) -> None:
        self.closed = True


@dataclass
class FakeTokenizer:
    """Minimal tokenizer surface used by :class:`BatchScheduler`.

    ``detokenizer`` returns a per-access stateful object so each active
    request gets its own incremental decoder.
    """

    eos_token_ids: list[int] = field(default_factory=lambda: [1])

    @property
    def detokenizer(self) -> _FakeDetokenizer:
        return _FakeDetokenizer()

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Encode fake stop words into deterministic token ids."""
        return [ord(ch) for ch in text]


class _FakeDetokenizer:
    def __init__(self) -> None:
        self.text = ""
        self._offset = 0
        self.tokens: list[int] = []

    def reset(self) -> None:
        self.text = ""
        self._offset = 0
        self.tokens = []

    def add_token(self, token: int) -> None:
        self.tokens.append(token)
        self.text += f"<{token}>"

    def finalize(self) -> None:
        return None

    @property
    def last_segment(self) -> str:
        segment = self.text[self._offset :]
        self._offset = len(self.text)
        return segment


class _FakeSequenceStateMachine:
    """Stub for the scheduler's ``SequenceStateMachine`` import."""

    last_transitions: dict[str, Any] | None = None
    last_initial: str | None = None

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.args = _args
        self.kwargs = _kwargs
        self.__class__.last_transitions = _kwargs.get("transitions")
        if self.__class__.last_transitions is None and _args:
            self.__class__.last_transitions = _args[0]
        self.__class__.last_initial = _kwargs.get("initial")


@pytest.fixture
def patched_scheduler(monkeypatch):
    """Load ``batch_scheduler`` with ``BatchGenerator`` + mlx stream stubbed."""

    from app.core import batch_scheduler as bsm

    monkeypatch.setattr(bsm, "BatchGenerator", FakeBatchGenerator)
    monkeypatch.setattr(bsm, "SequenceStateMachine", _FakeSequenceStateMachine)
    monkeypatch.setattr(bsm.mx, "stream", _fake_stream_cm)
    monkeypatch.setattr(bsm.mx, "new_stream", _fake_device)
    monkeypatch.setattr(bsm.mx, "new_thread_local_stream", _fake_device, raising=False)
    monkeypatch.setattr(bsm.mx, "default_device", _fake_device)
    monkeypatch.setattr(bsm.mx, "get_peak_memory", _zero)
    # Reset shared state each test.
    FakeBatchGenerator.script_queue = []
    FakeBatchGenerator.step_delay = 0.0
    bsm.pytest_monkeypatch = monkeypatch
    return bsm


@pytest.mark.asyncio
async def test_single_request_streams_all_tokens_and_final_chunk(patched_scheduler):
    """A single submit should yield one chunk per token plus a finish chunk."""
    FakeBatchGenerator.script_queue = [_FakeScript(tokens=[10, 11, 12], finish_reason="length")]

    scheduler = patched_scheduler.BatchScheduler(
        model=object(),
        tokenizer=FakeTokenizer(),
        idle_poll_timeout=0.01,
    )
    scheduler.start()
    try:
        stream = scheduler.submit_stream(input_ids=[7, 8], max_tokens=16)
        chunks = [chunk async for chunk in stream]
    finally:
        scheduler.stop()

    assert [c.token for c in chunks] == [10, 11, 12]
    assert [c.text for c in chunks] == ["<10>", "<11>", "<12>"]
    finals = [c for c in chunks if c.finish_reason is not None]
    assert len(finals) == 1
    assert finals[0].finish_reason == "length"
    assert finals[0].generation_tokens == 3


@pytest.mark.asyncio
async def test_concurrent_requests_are_routed_by_uid(patched_scheduler):
    """Two concurrent submits should each receive only their own tokens."""
    FakeBatchGenerator.script_queue = [
        _FakeScript(tokens=[100, 101], finish_reason="stop"),
        _FakeScript(tokens=[200, 201, 202], finish_reason="length"),
    ]

    scheduler = patched_scheduler.BatchScheduler(
        model=object(),
        tokenizer=FakeTokenizer(),
        idle_poll_timeout=0.01,
    )
    scheduler.start()
    try:
        s1 = scheduler.submit_stream(input_ids=[1], max_tokens=8)
        s2 = scheduler.submit_stream(input_ids=[2], max_tokens=8)

        async def _collect(stream):
            return [c async for c in stream]

        c1, c2 = await asyncio.gather(_collect(s1), _collect(s2))
    finally:
        scheduler.stop()

    assert [c.token for c in c1] == [100, 101]
    assert [c.token for c in c2] == [200, 201, 202]
    assert c1[-1].finish_reason == "stop"
    assert c2[-1].finish_reason == "length"


@pytest.mark.asyncio
async def test_cancellation_removes_sequence_from_batch(patched_scheduler):
    """Closing the stream early should propagate a ``remove`` call."""
    FakeBatchGenerator.script_queue = [
        _FakeScript(tokens=list(range(50, 100)), finish_reason="length")
    ]
    # Slow each generation step so the scheduler can't burn through all 50
    # tokens before the test has a chance to cancel.
    FakeBatchGenerator.step_delay = 0.01

    scheduler = patched_scheduler.BatchScheduler(
        model=object(),
        tokenizer=FakeTokenizer(),
        idle_poll_timeout=0.01,
    )
    scheduler.start()
    try:
        stream = scheduler.submit_stream(input_ids=[3], max_tokens=1000)
        first = await stream.__anext__()
        assert first.token == 50
        # Close the generator early — this must trigger cancel + remove().
        await stream.aclose()

        # Wait for the scheduler thread to observe the cancel event.
        fake: FakeBatchGenerator | None = None
        for _ in range(200):
            fake = getattr(scheduler, "_batch_generator", None)
            if isinstance(fake, FakeBatchGenerator) and fake.removed:
                break
            await asyncio.sleep(0.01)
        assert isinstance(fake, FakeBatchGenerator)
        assert fake.removed, "expected scheduler to call remove() after cancellation"
    finally:
        scheduler.stop()


@pytest.mark.asyncio
async def test_submit_before_start_raises(patched_scheduler):
    scheduler = patched_scheduler.BatchScheduler(
        model=object(),
        tokenizer=FakeTokenizer(),
    )
    with pytest.raises(RuntimeError, match="not running"):
        scheduler.submit_stream(input_ids=[1], max_tokens=4)


@pytest.mark.asyncio
async def test_stop_closes_batch_generator(patched_scheduler):
    scheduler = patched_scheduler.BatchScheduler(
        model=object(),
        tokenizer=FakeTokenizer(),
        idle_poll_timeout=0.01,
    )
    scheduler.start()
    # Give the worker thread a chance to construct the generator.
    for _ in range(50):
        if isinstance(getattr(scheduler, "_batch_generator", None), FakeBatchGenerator):
            break
        await asyncio.sleep(0.01)
    fake = scheduler._batch_generator
    scheduler.stop()
    assert isinstance(fake, FakeBatchGenerator)
    assert fake.closed is True


def test_default_state_machine_builds_with_eos(patched_scheduler):
    """EOS tokens from the tokenizer should be wired into the default state machine."""
    tok = FakeTokenizer(eos_token_ids=[2, 3])
    # Should not raise and should construct an instance of the (stubbed) state machine.
    sm = patched_scheduler.BatchScheduler._build_default_state_machine(tok)
    assert isinstance(sm, _FakeSequenceStateMachine)


def test_state_machine_includes_request_stop_words(patched_scheduler):
    """Per-request stop strings should be encoded into the batched state machine."""
    tok = FakeTokenizer(eos_token_ids=[2])

    sm = patched_scheduler.BatchScheduler.build_state_machine(tok, stop_words=["END"])

    assert isinstance(sm, _FakeSequenceStateMachine)
    transitions = _FakeSequenceStateMachine.last_transitions
    assert transitions is not None
    assert transitions["normal"] == [([2], None), ([69, 78, 68], None)]
    assert _FakeSequenceStateMachine.last_initial == "normal"


def test_admission_queue_accepts_before_start(patched_scheduler):
    """Constructing the scheduler without start() should not spin a thread."""
    scheduler = patched_scheduler.BatchScheduler(model=object(), tokenizer=FakeTokenizer())
    assert scheduler.is_running is False
    assert isinstance(scheduler._admission_queue.qsize(), int)
    assert scheduler._thread is None
    assert isinstance(threading.current_thread(), threading.Thread)  # sanity


# ---------------------------------------------------------------------------
# Regression: per-request seeds must NOT disable batching. Operators who need
# positive request seeds honored must opt into the single-request path with
# --disable-batching.
# ---------------------------------------------------------------------------


def _load_handler_module_for_batchability(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Import the LM handler with MLX-backed core modules stubbed."""
    fake_core = types.ModuleType("app.core")
    fake_batch_scheduler = types.ModuleType("app.core.batch_scheduler")

    class _FakeBatchScheduler:
        pass

    class _FakeInferenceWorker:
        pass

    fake_core.BatchScheduler = _FakeBatchScheduler
    fake_core.InferenceWorker = _FakeInferenceWorker
    fake_batch_scheduler.BATCHING_AVAILABLE = True

    monkeypatch.setitem(sys.modules, "app.core", fake_core)
    monkeypatch.setitem(sys.modules, "app.core.batch_scheduler", fake_batch_scheduler)
    sys.modules.pop("app.handler.mlx_lm", None)

    return importlib.import_module("app.handler.mlx_lm")


def test_request_seed_does_not_disable_batching(monkeypatch):
    handler_module = _load_handler_module_for_batchability(monkeypatch)

    class _FakeModel:
        has_draft_model = False
        cache_is_batchable = True

    class _Req:
        def __init__(self, seed):
            self.seed = seed

    handler = handler_module.MLXLMHandler.__new__(handler_module.MLXLMHandler)
    handler.model = _FakeModel()
    handler._disable_batching = False

    assert handler._is_request_batchable(_Req(seed=None)) is True
    assert handler._is_request_batchable(_Req(seed=0)) is True
    assert handler._is_request_batchable(_Req(seed=-1)) is True
    assert handler._is_request_batchable(_Req(seed=42)) is True


def test_disable_batching_routes_to_single_request_path(monkeypatch):
    handler_module = _load_handler_module_for_batchability(monkeypatch)

    class _FakeModel:
        has_draft_model = False
        cache_is_batchable = True

    class _Req:
        seed = None

    handler = handler_module.MLXLMHandler.__new__(handler_module.MLXLMHandler)
    handler.model = _FakeModel()
    handler._disable_batching = True

    assert handler._is_request_batchable(_Req()) is False


def test_non_mergeable_cache_disables_batching(monkeypatch):
    """Models whose prompt caches don't expose ``merge`` must fall back."""
    handler_module = _load_handler_module_for_batchability(monkeypatch)

    class _FakeModel:
        has_draft_model = False
        cache_is_batchable = False

    class _Req:
        seed = None

    handler = handler_module.MLXLMHandler.__new__(handler_module.MLXLMHandler)
    handler.model = _FakeModel()
    handler._disable_batching = False

    assert handler._is_request_batchable(_Req()) is False


@pytest.mark.asyncio
async def test_admission_reclaims_lru_based_on_live_batch(patched_scheduler):
    """LRU trim_to must subtract the live batch's prompt cache bytes
    from the LRU cap at admission time — mirroring mlx_lm.server's
    ``total - active`` reclaim so the batch and the LRU share a budget.
    """
    FakeBatchGenerator.script_queue = [_FakeScript(tokens=[42], finish_reason="length")]

    class _FakeLRU:
        max_bytes = 1_000
        trim_calls: list[int] = []

        def fetch_nearest_cache(self, _tokens):
            return None, list(_tokens)

        def trim_to(self, *, n_bytes):
            self.trim_calls.append(n_bytes)

        def insert_cache(self, *_args, **_kwargs):
            pass

    lru = _FakeLRU()
    scheduler = patched_scheduler.BatchScheduler(
        model=object(),
        tokenizer=FakeTokenizer(),
        prompt_cache=lru,
        idle_poll_timeout=0.01,
    )
    scheduler.start()
    try:
        # Pretend the batch is already holding 400 bytes of live KV cache;
        # after admission the LRU should be trimmed to (1000 - 400) = 600.
        for _ in range(100):
            fake = getattr(scheduler, "_batch_generator", None)
            if isinstance(fake, FakeBatchGenerator):
                fake.prompt_cache_nbytes = 400
                break
            await asyncio.sleep(0.01)
        assert isinstance(fake, FakeBatchGenerator)

        stream = scheduler.submit_stream(input_ids=[1, 2, 3], max_tokens=4)
        _ = [c async for c in stream]
    finally:
        scheduler.stop()

    assert 600 in lru.trim_calls, (
        f"expected trim_to(n_bytes=600) after admission, got {lru.trim_calls}"
    )


@pytest.mark.asyncio
async def test_stop_drains_pending_admission_queue(patched_scheduler):
    """Requests submitted just before stop() must receive a terminal error
    instead of hanging forever on out_queue.get().
    """

    scheduler = patched_scheduler.BatchScheduler(
        model=object(),
        tokenizer=FakeTokenizer(),
        idle_poll_timeout=0.01,
    )
    scheduler.start()

    # Submit a batch of requests, then stop immediately. Some of them will
    # race past ``_admit_pending`` before the worker notices ``_running``
    # flipped; the rest must be drained by the finally handler.
    streams = [scheduler.submit_stream(input_ids=[i], max_tokens=8) for i in range(8)]
    scheduler.stop()

    async def _collect(stream):
        try:
            async for _ in stream:
                pass
        except RuntimeError:
            return "errored"
        return "ended"

    # Must complete quickly — no stream may be left hanging.
    results = await asyncio.wait_for(
        asyncio.gather(*[_collect(s) for s in streams]),
        timeout=2.0,
    )
    # Every stream terminated one way or another (clean end or explicit error).
    assert all(r in {"errored", "ended"} for r in results)


@pytest.mark.asyncio
async def test_exact_cache_hit_is_backed_off_before_kickoff_token(patched_scheduler):
    """Exact cache hits must not reuse a full-prompt cache with a shorter prefix."""
    FakeBatchGenerator.script_queue = [_FakeScript(tokens=[99], finish_reason="length")]

    class _FakeLayer:
        def __init__(self, nbytes: int = 0) -> None:
            self.nbytes = nbytes

    class _FakeLRU:
        def fetch_nearest_cache(self, tokens):
            if tokens == [1, 2, 3]:
                return [_FakeLayer()], []
            raise AssertionError(f"unexpected fetch for tokens={tokens}")

        def insert_cache(self, *_args, **_kwargs):
            pass

        def trim_to(self, **_kwargs):
            pass

    trimmed: list[int] = []
    monkeypatch = (
        patched_scheduler.pytest_monkeypatch
        if hasattr(patched_scheduler, "pytest_monkeypatch")
        else None
    )
    if monkeypatch is None:
        pytest.skip("patched scheduler fixture does not expose monkeypatch")

    monkeypatch.setattr(patched_scheduler, "can_trim_prompt_cache", lambda _cache: True)
    monkeypatch.setattr(
        patched_scheduler,
        "trim_prompt_cache",
        lambda _cache, n: trimmed.append(n),
    )

    scheduler = patched_scheduler.BatchScheduler(
        model=object(),
        tokenizer=FakeTokenizer(),
        prompt_cache=_FakeLRU(),
        idle_poll_timeout=0.01,
    )
    scheduler.start()
    try:
        stream = scheduler.submit_stream(input_ids=[1, 2, 3], max_tokens=4)
        chunks = [chunk async for chunk in stream]
    finally:
        fake = scheduler._batch_generator
        scheduler.stop()

    assert trimmed == [1]
    assert isinstance(fake, FakeBatchGenerator)
    assert chunks[-1].cached_prompt_tokens == 2


@pytest.mark.asyncio
async def test_exact_non_trimmable_cache_hit_falls_back_to_reprefill(patched_scheduler):
    """Non-trimmable exact hits without a shorter prefix must be discarded safely."""
    FakeBatchGenerator.script_queue = [_FakeScript(tokens=[77], finish_reason="length")]

    class _FakeLayer:
        nbytes = 0

    class _FakeLRU:
        def fetch_nearest_cache(self, tokens):
            if tokens == [1, 2, 3]:
                return [_FakeLayer()], []
            if tokens == [1, 2]:
                return None, [1, 2]
            raise AssertionError(f"unexpected fetch for tokens={tokens}")

        def insert_cache(self, *_args, **_kwargs):
            pass

        def trim_to(self, **_kwargs):
            pass

    monkeypatch = (
        patched_scheduler.pytest_monkeypatch
        if hasattr(patched_scheduler, "pytest_monkeypatch")
        else None
    )
    if monkeypatch is None:
        pytest.skip("patched scheduler fixture does not expose monkeypatch")

    monkeypatch.setattr(patched_scheduler, "can_trim_prompt_cache", lambda _cache: False)

    scheduler = patched_scheduler.BatchScheduler(
        model=object(),
        tokenizer=FakeTokenizer(),
        prompt_cache=_FakeLRU(),
        idle_poll_timeout=0.01,
    )
    scheduler.start()
    try:
        stream = scheduler.submit_stream(input_ids=[1, 2, 3], max_tokens=4)
        chunks = [chunk async for chunk in stream]
    finally:
        scheduler.stop()

    assert chunks[-1].cached_prompt_tokens == 0


@pytest.mark.asyncio
async def test_exact_non_trimmable_cache_hit_logs_info(patched_scheduler):
    """Discarded exact hits should be logged so latency regressions are visible."""
    FakeBatchGenerator.script_queue = [_FakeScript(tokens=[77], finish_reason="length")]
    info_messages: list[str] = []

    def _capture_info(message: str, *args: object) -> None:
        """Capture info logs while tolerating Loguru-style argument binding."""
        if args:
            try:
                message = message.format(*args)
            except (IndexError, KeyError, ValueError):
                message = " ".join([message, *(str(arg) for arg in args)])
        info_messages.append(message)

    class _FakeLayer:
        nbytes = 0

    class _FakeLRU:
        def fetch_nearest_cache(self, tokens):
            if tokens == [1, 2, 3]:
                return [_FakeLayer()], []
            if tokens == [1, 2]:
                return None, [1, 2]
            raise AssertionError(f"unexpected fetch for tokens={tokens}")

        def insert_cache(self, *_args, **_kwargs):
            pass

        def trim_to(self, **_kwargs):
            pass

    monkeypatch = (
        patched_scheduler.pytest_monkeypatch
        if hasattr(patched_scheduler, "pytest_monkeypatch")
        else None
    )
    if monkeypatch is None:
        pytest.skip("patched scheduler fixture does not expose monkeypatch")

    monkeypatch.setattr(patched_scheduler, "can_trim_prompt_cache", lambda _cache: False)
    monkeypatch.setattr(patched_scheduler.logger, "info", _capture_info)

    scheduler = patched_scheduler.BatchScheduler(
        model=object(),
        tokenizer=FakeTokenizer(),
        prompt_cache=_FakeLRU(),
        idle_poll_timeout=0.01,
    )
    scheduler.start()
    try:
        stream = scheduler.submit_stream(input_ids=[1, 2, 3], max_tokens=4)
        _chunks = [chunk async for chunk in stream]
    finally:
        scheduler.stop()

    assert any(
        "Discarding exact prompt-cache hit because it cannot be safely backed off by one token"
        in message
        and "prompt_tokens=3" in message
        for message in info_messages
    )


@pytest.mark.asyncio
async def test_submit_stream_raises_queue_full_when_admission_queue_is_saturated(patched_scheduler):
    """The scheduler must preserve bounded admission and propagate QueueFull."""
    scheduler = patched_scheduler.BatchScheduler(
        model=object(),
        tokenizer=FakeTokenizer(),
        queue_size=1,
    )
    scheduler._running = True
    scheduler._admission_queue.put_nowait(
        patched_scheduler._PendingRequest(
            input_ids=[1],
            prompt_cache=None,
            cached_prefix_len=0,
            max_tokens=4,
            sampler=None,
            logits_processors=None,
            state_machine=object(),
            loop=asyncio.get_running_loop(),
            out_queue=asyncio.Queue(),
            cancel_event=threading.Event(),
        )
    )

    with pytest.raises(asyncio.QueueFull):
        scheduler.submit_stream(input_ids=[2], max_tokens=4)
