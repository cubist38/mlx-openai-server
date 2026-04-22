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
from dataclasses import dataclass
import inspect
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
    # ``BatchGenerator`` gained a ``stream=`` kwarg in mlx-lm PR #1090.
    # We feature-detect it so the scheduler can forward its own thread-local
    # stream automatically once the installed mlx-lm ships that change —
    # older versions fall back to BatchGenerator's module-level stream.
    _BATCH_GENERATOR_ACCEPTS_STREAM = (
        "stream" in inspect.signature(BatchGenerator.__init__).parameters
    )
except ImportError:  # pragma: no cover — only exercised on older mlx-lm pins
    BatchGenerator = None  # type: ignore[assignment,misc]
    SequenceStateMachine = None  # type: ignore[assignment,misc]
    BATCHING_AVAILABLE = False
    _BATCH_GENERATOR_ACCEPTS_STREAM = False
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
    cached_prompt_tokens: int = 0
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
    # Optional prompt-level segments for non-trimmable caches. ``segments``
    # partitions ``input_ids`` — e.g. ``[system_and_history, current_turn]``
    # — and ``segment_types`` labels each one (``"system"`` / ``"user"`` /
    # ``"assistant"``) so the scheduler can save an LRU checkpoint at each
    # segment boundary (matching ``mlx_lm.server``'s approach).
    segments: list[list[int]] | None = None
    segment_types: list[str] | None = None


@dataclass
class _ActiveRequest:
    """Per-request state tracked while the sequence is live in the batch."""

    loop: asyncio.AbstractEventLoop
    out_queue: asyncio.Queue[Any]
    detokenizer: Any
    cancel_event: threading.Event
    prompt_tokens: int
    cached_prompt_tokens: int
    # Reversed ``segment_types`` list; pop()'d each time a non-terminal
    # segment boundary fires during prefill so the popped label becomes the
    # ``cache_type`` for the extracted checkpoint.
    pending_segment_types: list[str]
    first_token_time: float | None = None
    generation_tokens: int = 0


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
        self._ready_event = threading.Event()
        self._start_error: BaseException | None = None
        # MLX stream owned by the scheduler thread. Allocated in ``_run`` so
        # the stream is created on the same OS thread that will use it —
        # mlx ≥ 0.31.2 binds streams to their creating thread (see mlx
        # PR #3355 / mlx-lm PR #1090) and touching a cross-thread stream
        # raises "no Stream(gpu, 0) in current thread".
        self._stream: Any | None = None

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
            self._ready_event.clear()
            self._start_error = None
            self._running = True
            self._thread = threading.Thread(
                target=self._run, daemon=True, name="mlx-batch-scheduler"
            )
            self._thread.start()
        # Wait for the worker thread to finish constructing the BatchGenerator
        # so start() surfaces init errors synchronously rather than deferring
        # them to the first submit.
        if not self._ready_event.wait(timeout=60.0):
            # Timed out waiting for init — the worker may be stuck inside
            # ``mx.set_wired_limit`` or model plumbing. Roll back to a clean
            # state so subsequent calls to start() can retry.
            with self._state_lock:
                self._running = False
            raise RuntimeError(
                "BatchScheduler did not finish initializing within 60s;"
                " check the scheduler thread for a hang in BatchGenerator init"
            )
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
        segments: list[list[int]] | None = None,
        segment_types: list[str] | None = None,
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
        segments : list[list[int]], optional
            Partition of ``input_ids`` into segments. When provided, the
            scheduler uses ``BatchGenerator.insert_segments`` and extracts a
            cache checkpoint at each non-terminal segment boundary. Required
            to get cache reuse on models with non-trimmable (hybrid / SSM)
            caches — the default ``fetch_nearest_cache`` "longer match"
            trimming path is blocked on those.
        segment_types : list[str], optional
            Parallel labels for ``segments`` (``"system"``, ``"user"``,
            ``"assistant"``). Used as ``cache_type`` when each segment's
            checkpoint is saved into the LRU. Must be the same length as
            ``segments``.

        Returns
        -------
        AsyncGenerator[BatchChunk, None]
            An async generator yielding incremental chunks; the final chunk
            has ``finish_reason`` set.
        """
        if segments is not None and segment_types is not None and len(segments) != len(
            segment_types
        ):
            raise ValueError(
                f"segments / segment_types length mismatch: {len(segments)} vs {len(segment_types)}"
            )
        if not input_ids:
            raise ValueError("input_ids must contain at least one token for batch generation")

        if not self._running:
            raise RuntimeError("BatchScheduler is not running; call start() first")

        loop = asyncio.get_running_loop()
        out_queue: asyncio.Queue[Any] = asyncio.Queue()
        cancel_event = threading.Event()

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
            segments=[list(s) for s in segments] if segments is not None else None,
            segment_types=list(segment_types) if segment_types is not None else None,
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
            # Allocate the scheduler's MLX stream on *this* thread so that
            # every Metal command buffer submitted by the BatchGenerator is
            # bound to the thread that will actually poll it. Prefer
            # ``new_thread_local_stream`` (mlx ≥ 0.31.2 / PR #3355) when
            # available — it refuses cross-thread use at the MLX layer
            # instead of silently failing at submit time. Fall back to a
            # plain ``new_stream`` on older mlx.
            #
            # On mlx-lm versions that accept ``stream=`` (post PR #1090) we
            # also hand the stream into BatchGenerator so its internal
            # ``mx.stream(...)`` wraps run against *our* stream rather than
            # the module-level one; older versions silently ignore this
            # branch and keep using ``generation_stream``.
            new_thread_local_stream = getattr(mx, "new_thread_local_stream", None)
            if new_thread_local_stream is not None:
                self._stream = new_thread_local_stream(mx.default_device())
            else:
                self._stream = mx.new_stream(mx.default_device())
            batch_generator_kwargs: dict[str, Any] = {
                "completion_batch_size": self._completion_batch_size,
                "prefill_batch_size": self._prefill_batch_size,
                "prefill_step_size": self._prefill_step_size,
                "max_kv_size": self._max_kv_size,
            }
            if _BATCH_GENERATOR_ACCEPTS_STREAM:
                batch_generator_kwargs["stream"] = self._stream
            self._batch_generator = BatchGenerator(self._model, **batch_generator_kwargs)
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
                    # ``BatchGenerator.next()`` already wraps itself in its
                    # own ``mx.stream(...)`` context, so we don't double-wrap
                    # here. (On mlx-lm ≤ 0.31.3 that stream is the module-
                    # level ``generation_stream``; after PR #1090 it becomes
                    # whatever was passed via ``BatchGenerator(stream=...)``.)
                    prompt_responses, gen_responses = self._batch_generator.next()
                except Exception as exc:  # noqa: BLE001 — propagate per-request
                    logger.exception(f"BatchGenerator.next() raised: {exc!s}")
                    self._fail_all_active(exc)
                    continue

                self._handle_prompt_responses(prompt_responses)

                for resp in gen_responses:
                    self._handle_generation_response(resp)

                self._process_cancellations()
        finally:
            try:
                if self._batch_generator is not None:
                    self._batch_generator.close()
            except Exception as exc:  # noqa: BLE001 — teardown best-effort
                logger.warning(f"Error closing BatchGenerator: {exc!s}")
            self._batch_generator = None
            # Release the stream reference on the thread that owns it so the
            # next ``start()`` gets a fresh stream rather than reusing a
            # handle that may now be invalid.
            self._stream = None
            self._fail_all_active(RuntimeError("BatchScheduler stopped"))
            # Drain any requests that were enqueued during shutdown (or that
            # raced past ``submit_stream``'s ``is_running`` check after
            # ``stop()`` flipped the flag). Without this, callers blocked on
            # ``out_queue.get()`` would hang forever — the main loop has
            # already exited and nothing else will deliver a sentinel.
            self._drain_admission_queue(RuntimeError("BatchScheduler stopped"))

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

            # Resolve the prompt cache on *this* thread (matching
            # ``mlx_lm.server``'s architecture). Priority: caller-supplied
            # cache → LRU lookup → fresh cache created by BatchGenerator.
            if request.prompt_cache is not None:
                cache_from_lru: list[Any] | None = request.prompt_cache
                cached_prefix_len = request.cached_prefix_len
            elif self._prompt_cache is not None:
                try:
                    fetched_cache, fetched_rest = self._prompt_cache.fetch_nearest_cache(
                        request.input_ids
                    )
                except Exception as exc:  # noqa: BLE001 — fallback to fresh cache
                    logger.warning(f"prompt_cache.fetch_nearest_cache failed: {exc!s}")
                    fetched_cache, fetched_rest = None, request.input_ids
                cache_from_lru = fetched_cache
                cached_prefix_len = len(request.input_ids) - len(fetched_rest)
            else:
                cache_from_lru = None
                cached_prefix_len = 0

            # Resolve segments + trim against cached prefix. If the caller
            # didn't supply segments, treat the whole prompt as one
            # "assistant" segment (matches ``mlx_lm.server``'s default for
            # non-chat or assistant-terminated prompts).
            segments: list[list[int]]
            segment_types: list[str]
            if request.segments is not None and request.segment_types is not None:
                segments = [list(s) for s in request.segments]
                segment_types = list(request.segment_types)
            else:
                segments = [list(request.input_ids)]
                segment_types = ["assistant"]

            # Drop segments fully covered by the cache; trim the first
            # partially-covered one (mirrors mlx_lm.server ``while N > 0``).
            remaining = cached_prefix_len
            while remaining > 0 and segments:
                if remaining >= len(segments[0]):
                    remaining -= len(segments.pop(0))
                    segment_types.pop(0)
                else:
                    segments[0] = segments[0][remaining:]
                    remaining = 0

            if not segments or not any(segments):
                # Cache covered the whole prompt; keep a single-token
                # kickoff segment so BatchGenerator has something to
                # process. Non-trimmable caches get here when the trie
                # returned a "longer" match via insert_segments fallback.
                segments = [request.input_ids[-1:]]
                segment_types = ["assistant"]
                cached_prefix_len = len(request.input_ids) - 1

            try:
                uids = self._batch_generator.insert_segments(
                    segments=[segments],
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

            detokenizer = self._tokenizer.detokenizer
            detokenizer.reset()
            # Reversed so we can ``pop()`` each label as its segment
            # boundary fires during prefill.
            pending_segment_types = list(reversed(segment_types))
            self._active[uid] = _ActiveRequest(
                loop=request.loop,
                out_queue=request.out_queue,
                detokenizer=detokenizer,
                cancel_event=request.cancel_event,
                prompt_tokens=len(request.input_ids),
                cached_prompt_tokens=cached_prefix_len,
                pending_segment_types=pending_segment_types,
            )
            logger.info(
                f"BatchScheduler admitted uid={uid} "
                f"(active={len(self._active)}, prompt_tokens={len(request.input_ids)}, "
                f"cached_prefix={cached_prefix_len}, segments={len(segments)})"
            )

            self._reclaim_prompt_cache()

    def _reclaim_prompt_cache(self) -> None:
        """Shrink the LRU so the live batch's caches stay within budget.

        Mirrors ``mlx_lm.server``: after each admission, subtract the bytes
        currently held by ``BatchGenerator`` (live, in-flight caches) from
        the LRU's byte cap and trim the LRU to whatever remains. Without
        this, a saturated batch can push total KV usage past the cap
        because the LRU only accounts for its own cached entries.

        No-op when the LRU has no byte cap configured or when the installed
        mlx-lm does not expose ``prompt_cache_nbytes`` (older versions).
        """
        if self._prompt_cache is None or self._batch_generator is None:
            return
        max_bytes = getattr(self._prompt_cache, "max_bytes", None)
        if not max_bytes or max_bytes >= (1 << 62):
            # Either unbounded (default) or effectively so — nothing useful
            # to reclaim; avoid the cost of computing ``prompt_cache_nbytes``.
            return
        active = getattr(self._batch_generator, "prompt_cache_nbytes", None)
        if active is None:
            return
        try:
            self._prompt_cache.trim_to(n_bytes=max_bytes - active)
        except Exception as exc:  # noqa: BLE001 — reclaim is best-effort
            logger.warning(f"prompt_cache.trim_to failed: {exc!s}")

    def _handle_prompt_responses(self, prompt_responses: list[Any]) -> None:
        """Save cache checkpoints at non-terminal segment boundaries.

        ``BatchGenerator`` reports ``end_of_segment=True`` after it finishes
        prefilling each segment of a multi-segment insert. For every such
        event that is *not* the end of the whole prompt, we extract the cache
        state and save it into the LRU under the matching ``cache_type``
        label — same approach as ``mlx_lm.server``. This is what gives
        non-trimmable caches (hybrid / SSM / recurrent) a reusable prefix:
        the "system" or "user" checkpoint sits in the trie so future
        requests with the same prefix find it via the normal ``shorter``
        match path.
        """
        if not prompt_responses or self._prompt_cache is None:
            return
        eos_uids: list[int] = []
        for r in prompt_responses:
            if not getattr(r, "end_of_segment", False):
                continue
            if getattr(r, "end_of_prompt", False):
                # Final segment's checkpoint is redundant with the post-
                # generation save in _handle_generation_response; skip to
                # avoid a duplicate insert.
                continue
            state = self._active.get(r.uid)
            if state is None or not state.pending_segment_types:
                continue
            eos_uids.append(r.uid)
        if not eos_uids:
            return
        try:
            caches = self._batch_generator.extract_cache(eos_uids)
        except Exception as exc:  # noqa: BLE001 — extraction failures are non-fatal
            logger.warning(f"BatchGenerator.extract_cache failed for {eos_uids}: {exc!s}")
            return
        for uid, (cache, cache_key) in caches.items():
            state = self._active.get(uid)
            if state is None or not state.pending_segment_types:
                continue
            cache_type = state.pending_segment_types.pop()
            try:
                self._prompt_cache.insert_cache(
                    list(cache_key),
                    cache,
                    cache_type=cache_type,
                )
                logger.debug(
                    f"BatchScheduler saved {cache_type} checkpoint for uid={uid}"
                    f" (key_len={len(cache_key)})"
                )
            except Exception as exc:  # noqa: BLE001 — cache save is best-effort
                logger.warning(
                    f"prompt_cache.insert_cache ({cache_type}) failed for uid={uid}: {exc!s}"
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
            cached_prompt_tokens=state.cached_prompt_tokens,
            prompt_tps=0.0,
            peak_memory=(mx.get_peak_memory() / 1e9) if is_final else 0.0,
        )
        self._send(state.loop, state.out_queue, chunk)

        if is_final:
            # Persist the final KV cache back into the LRU from *this* thread.
            # Matches ``mlx_lm.server`` (which also does ``insert_cache`` on
            # its generation thread) and keeps all cache mutations off the
            # FastAPI event-loop thread.
            saved = False
            key_len = 0
            if self._prompt_cache is not None:
                if resp.prompt_cache is None or not resp.all_tokens:
                    logger.debug(
                        f"Skipping cache save for uid={resp.uid}: "
                        f"prompt_cache={'present' if resp.prompt_cache else 'missing'}, "
                        f"all_tokens_len={len(resp.all_tokens) if resp.all_tokens else 0}"
                    )
                else:
                    key_len = len(resp.all_tokens)
                    try:
                        self._prompt_cache.insert_cache(
                            list(resp.all_tokens),
                            resp.prompt_cache,
                            cache_type="assistant",
                        )
                        saved = True
                    except Exception as exc:  # noqa: BLE001 — cache save is best-effort
                        logger.warning(
                            f"prompt_cache.insert_cache failed for uid={resp.uid}: {exc!s}"
                        )
            logger.info(
                f"BatchScheduler uid={resp.uid} finished "
                f"(finish_reason={chunk_finish}, "
                f"generated={state.generation_tokens}, "
                f"cache_saved={saved}, cache_key_len={key_len})"
            )
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
            with mx.stream(self._stream):
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
                cached_prompt_tokens=state.cached_prompt_tokens,
                peak_memory=mx.get_peak_memory() / 1e9,
            )
            self._send(state.loop, state.out_queue, chunk)
            self._send(state.loop, state.out_queue, _STREAM_SENTINEL)

    def _fail_all_active(self, exc: BaseException) -> None:
        for uid, state in list(self._active.items()):
            self._send(state.loop, state.out_queue, exc)
            self._send(state.loop, state.out_queue, _STREAM_SENTINEL)
            self._active.pop(uid, None)

    def _drain_admission_queue(self, exc: BaseException) -> None:
        """Fail every request still sitting in ``_admission_queue``.

        Called from the scheduler thread's ``finally`` block so no caller
        blocked on ``submit_stream`` can be left waiting for a sentinel that
        will never arrive. Uses ``get_nowait`` in a loop to tolerate
        additional requests being put on the queue by a concurrent
        ``submit_stream`` that raced past the ``is_running`` check.
        """
        while True:
            try:
                request = self._admission_queue.get_nowait()
            except queue.Empty:
                return
            self._fail_request(request, exc)

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
