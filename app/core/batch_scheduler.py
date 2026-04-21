"""Continuous-batch request scheduler backed by ``mlx_lm.generate.BatchGenerator``.

The scheduler runs a single background thread that owns a
:class:`mlx_lm.generate.BatchGenerator`. Async callers submit prompts via
:meth:`BatchScheduler.submit_stream`, which returns an async generator yielding
per-request chunks as soon as they are produced. New requests are admitted
between decode steps so requests do not need to wait for the current batch to
drain — this matches the continuous-batching behavior used by ``mlx_lm``'s own
HTTP server (``mlx_lm/server.py``) and produces the same throughput benefits
that vLLM-style servers achieve.

Per-request settings (sampler, logits processors, max tokens, stop token
sequences, and pre-computed prompt cache) are forwarded to ``BatchGenerator``
through its ``samplers`` / ``logits_processors`` / ``state_machines`` lanes so
features such as repetition penalty, logit bias, and JSON-schema constrained
decoding continue to work while batching.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass, field
import queue
import threading
import time
from typing import TYPE_CHECKING, Any

from loguru import logger
import mlx.core as mx

if TYPE_CHECKING:
    from mlx_lm.tokenizer_utils import TokenizerWrapper


try:
    from mlx_lm.generate import BatchGenerator, SequenceStateMachine

    BATCHING_AVAILABLE = True
except ImportError:  # pragma: no cover — only exercised on older mlx-lm pins
    BatchGenerator = None  # type: ignore[assignment,misc]
    SequenceStateMachine = None  # type: ignore[assignment,misc]
    BATCHING_AVAILABLE = False
    logger.warning(
        "mlx_lm.generate.BatchGenerator is unavailable — continuous batching is"
        " disabled. Upgrade mlx-lm (git main or the next PyPI release after"
        " 0.31.2) to enable the batch scheduler."
    )


@dataclass
class BatchChunk:
    """Per-request chunk emitted by the scheduler.

    Mirrors the fields the handler already reads off of
    :class:`mlx_lm.generate.GenerationResponse` so the existing streaming and
    non-streaming paths work unchanged.

    Parameters
    ----------
    text : str
        Incremental decoded text segment since the previous chunk.
    token : int
        The sampled token id for this step.
    finish_reason : str | None
        ``None`` for in-progress chunks, ``"stop"`` when a stop sequence matched,
        ``"length"`` when ``max_tokens`` was reached, ``"cancelled"`` when the
        request was removed from the batch.
    generation_tokens : int
        Number of tokens generated for this request so far.
    generation_tps : float
        Average generation tokens-per-second for this request (only populated
        on the final chunk).
    prompt_tokens : int
        Number of prompt tokens that were actually processed (i.e. not served
        from a pre-computed cache).
    prompt_tps : float
        Prompt processing tokens-per-second (only populated on the final
        chunk; 0.0 otherwise).
    peak_memory : float
        Peak MLX memory in GB (only populated on the final chunk).
    """

    text: str
    token: int
    finish_reason: str | None = None
    generation_tokens: int = 0
    generation_tps: float = 0.0
    prompt_tokens: int = 0
    prompt_tps: float = 0.0
    peak_memory: float = 0.0


@dataclass
class _PendingRequest:
    """Queued request waiting to be inserted into the batch."""

    input_ids: list[int]
    prompt_cache: list[Any] | None
    cached_prefix_len: int
    max_tokens: int
    sampler: Callable[[mx.array], mx.array] | None
    logits_processors: list[Callable[[mx.array, mx.array], mx.array]] | None
    state_machine: Any
    loop: asyncio.AbstractEventLoop
    out_queue: asyncio.Queue[Any]
    cancel_event: threading.Event
    admission_event: threading.Event
    uid_slot: dict[str, int]


@dataclass
class _ActiveRequest:
    """Per-request state tracked while the sequence is live in the batch."""

    loop: asyncio.AbstractEventLoop
    out_queue: asyncio.Queue[Any]
    detokenizer: Any
    cancel_event: threading.Event
    start_time: float
    first_token_time: float | None = None
    generation_tokens: int = 0
    prompt_tokens: int = 0
    all_tokens: list[int] = field(default_factory=list)


_STREAM_SENTINEL: Any = object()


class BatchScheduler:
    """Continuous-batch scheduler on top of :class:`BatchGenerator`.

    Parameters
    ----------
    model : Any
        The MLX language model.
    tokenizer : TokenizerWrapper
        The tokenizer used to build per-request detokenizers and stop tokens.
    completion_batch_size : int, optional
        Maximum number of concurrent sequences in the generation batch.
    prefill_batch_size : int, optional
        Maximum number of sequences that can be prefilled simultaneously.
    prefill_step_size : int, optional
        Maximum tokens processed per prefill step.
    max_kv_size : int | None, optional
        Optional rotating-KV-cache size; ``None`` keeps full history.
    idle_poll_timeout : float, optional
        Seconds to wait for a new request when the batch is empty before
        looping again; defaults to 0.1.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: TokenizerWrapper,
        *,
        prompt_cache: Any = None,
        completion_batch_size: int = 32,
        prefill_batch_size: int = 8,
        prefill_step_size: int = 2048,
        max_kv_size: int | None = None,
        idle_poll_timeout: float = 0.1,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        # Optional :class:`~app.utils.prompt_cache.LRUPromptCache`. When
        # provided, the scheduler fetches and inserts cache entries on its
        # *own* thread — the same single-thread discipline ``mlx_lm.server``
        # uses — to avoid cross-thread Metal command-buffer races.
        self._prompt_cache = prompt_cache
        self._completion_batch_size = completion_batch_size
        self._prefill_batch_size = prefill_batch_size
        self._prefill_step_size = prefill_step_size
        self._max_kv_size = max_kv_size
        self._idle_poll_timeout = idle_poll_timeout

        self._admission_queue: queue.Queue[_PendingRequest] = queue.Queue()
        self._batch_generator: BatchGenerator | None = None
        self._active: dict[int, _ActiveRequest] = {}
        self._default_state_machine = self._build_default_state_machine(tokenizer)

        self._thread: threading.Thread | None = None
        self._running = False
        self._state_lock = threading.Lock()

    @staticmethod
    def _build_default_state_machine(tokenizer: TokenizerWrapper) -> Any:
        """Build a state machine that stops on any EOS token.

        The mlx-lm ``BatchGenerator`` uses a state machine per-sequence to
        detect stop sequences; we pass EOS tokens so finished sequences are
        removed from the batch as soon as they complete.
        """
        eos_token_ids = getattr(tokenizer, "eos_token_ids", None) or []
        if not eos_token_ids:
            eos_id = getattr(tokenizer, "eos_token_id", None)
            if eos_id is not None:
                eos_token_ids = [eos_id]
        stop_sequences = [[int(t)] for t in eos_token_ids]
        return SequenceStateMachine(
            {"normal": [(seq, None) for seq in stop_sequences]} if stop_sequences else {},
            initial="normal",
        )

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        """Start the background generation thread (idempotent).

        The underlying :class:`BatchGenerator` is constructed *inside* the
        worker thread (see :meth:`_run`) so every Metal call — init, forward
        pass, and close — happens on a single thread. This matches the
        pattern used by mlx-lm's own HTTP server and avoids
        ``-[_MTLCommandBuffer addCompletedHandler:]: Completed handler
        provided after commit call`` assertion failures when MLX command
        buffers are touched from more than one thread.
        """
        with self._state_lock:
            if self._running:
                return
            if BatchGenerator is None:
                raise RuntimeError(
                    "mlx_lm.generate.BatchGenerator is not available in the"
                    " installed mlx-lm; upgrade mlx-lm to enable batching."
                )
            self._running = True
            self._ready_event = threading.Event()
            self._start_error: BaseException | None = None
            self._thread = threading.Thread(
                target=self._run, daemon=True, name="mlx-batch-scheduler"
            )
            self._thread.start()
        # Wait briefly for the worker thread to finish constructing the
        # BatchGenerator so start() surfaces init errors synchronously rather
        # than deferring them to the first submit.
        self._ready_event.wait(timeout=30.0)
        if self._start_error is not None:
            raise self._start_error
        logger.info(
            "BatchScheduler started (completion={}, prefill={}, prefill_step={}, max_kv={})",
            self._completion_batch_size,
            self._prefill_batch_size,
            self._prefill_step_size,
            self._max_kv_size,
        )

    def stop(self) -> None:
        """Signal the scheduler thread to stop and wait for it to join.

        The worker thread is solely responsible for closing the
        :class:`BatchGenerator` in its own ``finally`` block, keeping every
        Metal call (init, forward passes, close) on one thread — see
        :meth:`start` for rationale.
        """
        with self._state_lock:
            if not self._running:
                return
            self._running = False
            thread = self._thread
            self._thread = None
        if thread is not None:
            thread.join(timeout=10.0)
        logger.info("BatchScheduler stopped")

    def submit_stream(
        self,
        input_ids: list[int],
        *,
        prompt_cache: list[Any] | None = None,
        cached_prefix_len: int = 0,
        max_tokens: int = 1024,
        sampler: Callable[[mx.array], mx.array] | None = None,
        logits_processors: list[Callable[[mx.array, mx.array], mx.array]] | None = None,
        state_machine: Any | None = None,
    ) -> AsyncGenerator[BatchChunk, None]:
        """Submit a request and return an async generator of :class:`BatchChunk`.

        Parameters
        ----------
        input_ids : list[int]
            Tokens that still need to be processed — i.e. the suffix of the
            full prompt that is not already covered by ``prompt_cache``.
        prompt_cache : list[Any] | None, optional
            A pre-computed prompt cache whose contents already cover the first
            ``cached_prefix_len`` tokens of the full prompt. Pass ``None`` to
            let the scheduler allocate a fresh cache.
        cached_prefix_len : int, optional
            Number of tokens already in ``prompt_cache``. Used to track usage
            stats.
        max_tokens : int, optional
            Maximum tokens to generate.
        sampler : Callable, optional
            Per-request sampler; falls back to greedy argmax when ``None``.
        logits_processors : list[Callable], optional
            Per-request logits processors.
        state_machine : SequenceStateMachine, optional
            Overrides the default EOS-only stop state machine.

        Returns
        -------
        AsyncGenerator[BatchChunk, None]
            An async generator yielding incremental chunks; the final chunk
            has ``finish_reason`` set.
        """
        if not input_ids:
            raise ValueError("input_ids must contain at least one token for batch generation")

        if not self._running:
            raise RuntimeError("BatchScheduler is not running; call start() first")

        loop = asyncio.get_running_loop()
        out_queue: asyncio.Queue[Any] = asyncio.Queue()
        cancel_event = threading.Event()
        admission_event = threading.Event()
        uid_slot: dict[str, int] = {}

        request = _PendingRequest(
            input_ids=list(input_ids),
            prompt_cache=prompt_cache,
            cached_prefix_len=cached_prefix_len,
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
            state_machine=state_machine or self._default_state_machine,
            loop=loop,
            out_queue=out_queue,
            cancel_event=cancel_event,
            admission_event=admission_event,
            uid_slot=uid_slot,
        )
        self._admission_queue.put(request)

        async def _stream() -> AsyncGenerator[BatchChunk, None]:
            try:
                while True:
                    item = await out_queue.get()
                    if item is _STREAM_SENTINEL:
                        return
                    if isinstance(item, BaseException):
                        raise item
                    yield item
            finally:
                cancel_event.set()

        return _stream()

    def _run(self) -> None:
        """Main worker loop: construct the generator, admit requests, dispatch.

        Metal command buffers are bound to the thread that creates them and
        MLX asserts ``Completed handler provided after commit call`` if they
        are touched from more than one thread. Constructing and closing the
        :class:`BatchGenerator` here, *and* running every ``next()`` here,
        keeps all Metal work pinned to this single OS thread.
        """
        try:
            self._batch_generator = BatchGenerator(
                self._model,
                completion_batch_size=self._completion_batch_size,
                prefill_batch_size=self._prefill_batch_size,
                prefill_step_size=self._prefill_step_size,
                max_kv_size=self._max_kv_size,
            )
        except BaseException as exc:  # noqa: BLE001 — surface to start()
            self._start_error = exc
            self._ready_event.set()
            self._running = False
            return
        self._ready_event.set()

        try:
            while self._running:
                self._admit_pending(block_if_empty=len(self._active) == 0)

                if not self._active:
                    continue

                try:
                    # ``BatchGenerator.next()`` already wraps itself in
                    # ``mx.stream(generation_stream)``; don't double-wrap.
                    prompt_responses, gen_responses = self._batch_generator.next()
                except Exception as exc:  # noqa: BLE001 — propagate per-request
                    logger.exception(f"BatchGenerator.next() raised: {exc!s}")
                    self._fail_all_active(exc)
                    continue

                for resp in gen_responses:
                    self._handle_generation_response(resp)

                # Prompt-phase responses are advisory (progress); we don't
                # surface them to clients in this first cut, but we clean up
                # cancelled prompts below.
                del prompt_responses

                self._process_cancellations()
        finally:
            try:
                if self._batch_generator is not None:
                    self._batch_generator.close()
            except Exception as exc:  # noqa: BLE001 — teardown best-effort
                logger.warning(f"Error closing BatchGenerator: {exc!s}")
            self._batch_generator = None
            self._fail_all_active(RuntimeError("BatchScheduler stopped"))

    def _generation_stream(self) -> Any:
        """Return the mlx generation stream used by ``BatchGenerator`` internally."""
        from mlx_lm.generate import generation_stream

        return generation_stream

    def _admit_pending(self, *, block_if_empty: bool) -> None:
        """Drain the admission queue into the batch generator."""
        first_wait = block_if_empty
        while True:
            try:
                if first_wait:
                    request = self._admission_queue.get(timeout=self._idle_poll_timeout)
                    first_wait = False
                else:
                    request = self._admission_queue.get_nowait()
            except queue.Empty:
                return

            if not self._running:
                self._fail_request(request, RuntimeError("BatchScheduler stopped"))
                return

            if request.cancel_event.is_set():
                self._send(request.loop, request.out_queue, _STREAM_SENTINEL)
                continue

            # Fetch a matching cache from the LRU on *this* thread (matching
            # mlx_lm.server's architecture). If a caller already supplied a
            # cache via prompt_cache=..., use it directly.
            cache_from_lru: list[Any] | None = request.prompt_cache
            rest_input_ids = request.input_ids
            cached_prefix_len = 0
            if cache_from_lru is None and self._prompt_cache is not None:
                try:
                    fetched_cache, fetched_rest = self._prompt_cache.fetch_nearest_cache(
                        request.input_ids
                    )
                except Exception as exc:  # noqa: BLE001 — fallback to fresh cache
                    logger.warning(f"prompt_cache.fetch_nearest_cache failed: {exc!s}")
                    fetched_cache, fetched_rest = None, request.input_ids
                if fetched_cache is not None:
                    cache_from_lru = fetched_cache
                    rest_input_ids = fetched_rest
                    cached_prefix_len = len(request.input_ids) - len(fetched_rest)

            if not rest_input_ids:
                # The cache already covers the entire prompt; keep one token
                # so BatchGenerator has a kickoff input.
                rest_input_ids = request.input_ids[-1:]
                cached_prefix_len = len(request.input_ids) - 1

            try:
                uids = self._batch_generator.insert(
                    prompts=[rest_input_ids],
                    max_tokens=[request.max_tokens],
                    caches=[cache_from_lru] if cache_from_lru is not None else None,
                    all_tokens=[request.input_ids[:cached_prefix_len]],
                    samplers=[request.sampler] if request.sampler is not None else None,
                    logits_processors=(
                        [request.logits_processors]
                        if request.logits_processors is not None
                        else None
                    ),
                    state_machines=[request.state_machine],
                )
            except Exception as exc:  # noqa: BLE001 — report per-request
                self._fail_request(request, exc)
                continue

            (uid,) = uids
            request.uid_slot["uid"] = uid
            request.admission_event.set()

            detokenizer = self._tokenizer.detokenizer
            detokenizer.reset()
            self._active[uid] = _ActiveRequest(
                loop=request.loop,
                out_queue=request.out_queue,
                detokenizer=detokenizer,
                cancel_event=request.cancel_event,
                start_time=time.perf_counter(),
                prompt_tokens=max(
                    0, len(request.input_ids) - 0
                ),  # cached prefix is already in the cache
            )
            logger.info(
                f"BatchScheduler admitted uid={uid} "
                f"(active={len(self._active)}, prompt_tokens={len(request.input_ids)})"
            )

    def _handle_generation_response(self, resp: Any) -> None:
        """Forward a single generation-batch response to the owning request."""
        state = self._active.get(resp.uid)
        if state is None:
            # Sequence was cancelled and removed from active; ignore.
            return

        if state.cancel_event.is_set():
            # Will be removed in _process_cancellations on this iteration.
            return

        if state.first_token_time is None:
            state.first_token_time = time.perf_counter()

        state.detokenizer.add_token(resp.token)
        segment = state.detokenizer.last_segment
        state.generation_tokens += 1

        chunk_finish = resp.finish_reason
        is_final = chunk_finish is not None

        if is_final:
            # Flush any remaining text pieces before reporting the final chunk.
            try:
                state.detokenizer.finalize()
                segment += state.detokenizer.last_segment
            except Exception:  # noqa: BLE001 — finalize is best-effort
                pass

        chunk = BatchChunk(
            text=segment,
            token=int(resp.token),
            finish_reason=chunk_finish,
            generation_tokens=state.generation_tokens,
            generation_tps=self._compute_tps(state) if is_final else 0.0,
            prompt_tokens=state.prompt_tokens,
            prompt_tps=0.0,
            peak_memory=(mx.get_peak_memory() / 1e9) if is_final else 0.0,
        )
        self._send(state.loop, state.out_queue, chunk)

        if is_final:
            # Persist the final KV cache back into the LRU from *this* thread.
            # Matches ``mlx_lm.server`` (which also does ``insert_cache`` on
            # its generation thread) and keeps all cache mutations off the
            # FastAPI event-loop thread.
            if (
                self._prompt_cache is not None
                and resp.prompt_cache is not None
                and resp.all_tokens
            ):
                try:
                    self._prompt_cache.insert_cache(
                        list(resp.all_tokens),
                        resp.prompt_cache,
                        cache_type="assistant",
                    )
                except Exception as exc:  # noqa: BLE001 — cache save is best-effort
                    logger.warning(f"prompt_cache.insert_cache failed for uid={resp.uid}: {exc!s}")
            self._send(state.loop, state.out_queue, _STREAM_SENTINEL)
            self._active.pop(resp.uid, None)

    @staticmethod
    def _compute_tps(state: _ActiveRequest) -> float:
        if state.first_token_time is None or state.generation_tokens == 0:
            return 0.0
        elapsed = time.perf_counter() - state.first_token_time
        if elapsed <= 0:
            return 0.0
        return state.generation_tokens / elapsed

    def _process_cancellations(self) -> None:
        """Remove any active sequences whose client has cancelled."""
        cancelled_uids = [uid for uid, state in self._active.items() if state.cancel_event.is_set()]
        if not cancelled_uids:
            return
        try:
            with mx.stream(self._generation_stream()):
                self._batch_generator.remove(cancelled_uids)
        except Exception as exc:  # noqa: BLE001 — removal errors are non-fatal
            logger.warning(f"BatchGenerator.remove failed for {cancelled_uids}: {exc!s}")
        for uid in cancelled_uids:
            state = self._active.pop(uid, None)
            if state is None:
                continue
            # Emit a terminal cancelled chunk so downstream ``async for`` loops
            # observe a finish_reason instead of just a silent close.
            chunk = BatchChunk(
                text="",
                token=0,
                finish_reason="cancelled",
                generation_tokens=state.generation_tokens,
                generation_tps=self._compute_tps(state),
                prompt_tokens=state.prompt_tokens,
                peak_memory=mx.get_peak_memory() / 1e9,
            )
            self._send(state.loop, state.out_queue, chunk)
            self._send(state.loop, state.out_queue, _STREAM_SENTINEL)

    def _fail_all_active(self, exc: BaseException) -> None:
        for uid, state in list(self._active.items()):
            self._send(state.loop, state.out_queue, exc)
            self._send(state.loop, state.out_queue, _STREAM_SENTINEL)
            self._active.pop(uid, None)

    def _fail_request(self, request: _PendingRequest, exc: BaseException) -> None:
        self._send(request.loop, request.out_queue, exc)
        self._send(request.loop, request.out_queue, _STREAM_SENTINEL)

    @staticmethod
    def _send(loop: asyncio.AbstractEventLoop, q: asyncio.Queue[Any], item: Any) -> None:
        """Thread-safe put of ``item`` onto an asyncio.Queue owned by ``loop``."""
        try:
            loop.call_soon_threadsafe(q.put_nowait, item)
        except RuntimeError:
            # Event loop is closed; nothing we can do.
            logger.debug("Event loop closed before scheduler response could be delivered")
