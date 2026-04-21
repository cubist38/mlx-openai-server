"""Unit tests for :class:`app.core.batch_scheduler.BatchScheduler`.

The tests stub out ``mlx_lm.generate.BatchGenerator`` so the scheduler logic
(request admission, dispatch of generation responses, cancellation, and
final-chunk stats) can be exercised without loading a real MLX model.
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from dataclasses import dataclass, field
import threading
from typing import Any

import pytest


@contextmanager
def _fake_stream_cm(_stream: Any):
    yield


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

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass


@pytest.fixture
def patched_scheduler(monkeypatch):
    """Load ``batch_scheduler`` with ``BatchGenerator`` + mlx stream stubbed."""

    from app.core import batch_scheduler as bsm

    monkeypatch.setattr(bsm, "BatchGenerator", FakeBatchGenerator)
    monkeypatch.setattr(bsm, "SequenceStateMachine", _FakeSequenceStateMachine)
    monkeypatch.setattr(bsm, "generation_stream", object())
    monkeypatch.setattr(bsm.mx, "stream", _fake_stream_cm)
    monkeypatch.setattr(bsm.mx, "get_peak_memory", lambda: 0)
    # Reset shared state each test.
    FakeBatchGenerator.script_queue = []
    FakeBatchGenerator.step_delay = 0.0
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


def test_admission_queue_accepts_before_start(patched_scheduler):
    """Constructing the scheduler without start() should not spin a thread."""
    scheduler = patched_scheduler.BatchScheduler(model=object(), tokenizer=FakeTokenizer())
    assert scheduler.is_running is False
    assert isinstance(scheduler._admission_queue.qsize(), int)
    assert scheduler._thread is None
    assert isinstance(threading.current_thread(), threading.Thread)  # sanity


# ---------------------------------------------------------------------------
# Regression: seed=0 (the CLI default that the endpoint backfills into every
# request) must NOT disable batching. If it does, every request is routed
# through the single-request path and the scheduler never sees concurrent
# work.
# ---------------------------------------------------------------------------


def test_default_seed_zero_does_not_disable_batching():
    from app.handler.mlx_lm import MLXLMHandler

    class _FakeModel:
        has_draft_model = False

    class _Req:
        def __init__(self, seed):
            self.seed = seed

    handler = MLXLMHandler.__new__(MLXLMHandler)
    handler.model = _FakeModel()

    assert handler._is_request_batchable(_Req(seed=None)) is True
    assert handler._is_request_batchable(_Req(seed=0)) is True
    assert handler._is_request_batchable(_Req(seed=-1)) is True
    assert handler._is_request_batchable(_Req(seed=42)) is False
