"""Continuous-batch scheduler for ``mlx-vlm`` generation."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import nullcontext
from dataclasses import dataclass
import inspect
import queue
import threading
import time
from typing import Any

from loguru import logger
import mlx.core as mx

try:
    from mlx_vlm.generate import BatchGenerator

    VLM_BATCHING_AVAILABLE = True
    _VLM_BATCH_GENERATOR_ACCEPTS_STREAM = (
        "stream" in inspect.signature(BatchGenerator.__init__).parameters
    )
except (ImportError, RuntimeError) as exc:  # pragma: no cover - version/environment fallback
    BatchGenerator = None  # type: ignore[assignment,misc]
    VLM_BATCHING_AVAILABLE = False
    _VLM_BATCH_GENERATOR_ACCEPTS_STREAM = False
    logger.warning(
        "mlx_vlm.generate.BatchGenerator is unavailable; multimodal continuous "
        f"batching is disabled: {exc!s}"
    )


@dataclass
class VLMBatchChunk:
    """Per-request generation chunk emitted by the VLM batch scheduler."""

    text: str
    token: int
    finish_reason: str | None = None
    generation_tokens: int = 0
    generation_tps: float = 0.0
    prompt_tokens: int = 0
    prompt_tps: float = 0.0
    peak_memory: float = 0.0


@dataclass
class _PendingVLMRequest:
    """Queued multimodal request waiting to enter the VLM batcher."""

    model_inputs: dict[str, Any]
    max_tokens: int
    logits_processors: list[Any] | None
    loop: asyncio.AbstractEventLoop
    out_queue: asyncio.Queue[Any]
    cancel_event: threading.Event


@dataclass
class _ActiveVLMRequest:
    """Per-request state tracked during VLM batched decoding."""

    loop: asyncio.AbstractEventLoop
    out_queue: asyncio.Queue[Any]
    detokenizer: Any | None
    cancel_event: threading.Event
    prompt_tokens: int
    prompt_start_time: float
    first_token_time: float | None = None
    generation_tokens: int = 0
    tokens: list[int] | None = None
    previous_text: str = ""


_STREAM_SENTINEL: Any = object()


class VLMBatchScheduler:
    """Continuous scheduler backed by ``mlx_vlm.generate.BatchGenerator``.

    The installed ``mlx-vlm`` batcher accepts precomputed multimodal
    embeddings and a single sampler for the whole scheduler. The handler only
    routes greedy/no-seed requests here; sampled requests stay on the existing
    single-request worker path so per-request sampling semantics are preserved.
    """

    def __init__(
        self,
        model_wrapper: Any,
        *,
        completion_batch_size: int = 32,
        prefill_batch_size: int = 8,
        prefill_step_size: int = 2048,
        generation_lock: threading.RLock | None = None,
        queue_size: int = 100,
        idle_poll_timeout: float = 0.1,
    ) -> None:
        self._model_wrapper = model_wrapper
        self._completion_batch_size = completion_batch_size
        self._prefill_batch_size = prefill_batch_size
        self._prefill_step_size = prefill_step_size
        self._generation_lock = generation_lock
        self._queue_size = queue_size
        self._idle_poll_timeout = idle_poll_timeout

        self._admission_queue: queue.Queue[_PendingVLMRequest | None] = queue.Queue(
            maxsize=queue_size
        )
        self._batch_generator: BatchGenerator | None = None
        self._active: dict[int, _ActiveVLMRequest] = {}
        self._thread: threading.Thread | None = None
        self._running = False
        self._state_lock = threading.Lock()
        self._ready_event = threading.Event()
        self._start_error: BaseException | None = None
        self._stream: Any | None = None
        self._stop_tokens: set[int] = set()
        self._batch_generator_kwargs: dict[str, Any] = {}

    @property
    def is_running(self) -> bool:
        """Return whether the scheduler thread is accepting work."""
        return self._running

    def start(self) -> None:
        """Start the scheduler thread and construct ``BatchGenerator`` there."""
        with self._state_lock:
            if self._running:
                return
            if BatchGenerator is None:
                raise RuntimeError(
                    "mlx_vlm.generate.BatchGenerator is not available in the "
                    "installed mlx-vlm; upgrade mlx-vlm to enable VLM batching."
                )
            self._ready_event.clear()
            self._start_error = None
            self._running = True
            self._thread = threading.Thread(
                target=self._run, daemon=True, name="mlx-vlm-batch-scheduler"
            )
            self._thread.start()
        if not self._ready_event.wait(timeout=60.0):
            with self._state_lock:
                self._running = False
            raise RuntimeError("VLMBatchScheduler did not initialize within 60s")
        if self._start_error is not None:
            raise self._start_error
        logger.info(
            "VLMBatchScheduler started (completion={}, prefill={}, prefill_step={})",
            self._completion_batch_size,
            self._prefill_batch_size,
            self._prefill_step_size,
        )

    def stop(self, timeout: float = 2.0) -> None:
        """Stop the scheduler thread and close the VLM batcher."""
        with self._state_lock:
            if not self._running:
                return
            self._running = False
            thread = self._thread
            self._thread = None
        try:
            self._admission_queue.put_nowait(None)
        except queue.Full:
            pass
        if thread is not None:
            thread.join(timeout=timeout)
            if thread.is_alive():
                logger.warning(
                    f"VLMBatchScheduler did not stop within {timeout:.1f} s; "
                    "continuing shutdown with daemon thread still exiting"
                )
        logger.info("VLMBatchScheduler stopped")

    def submit_stream(
        self,
        *,
        model_inputs: dict[str, Any],
        max_tokens: int,
        logits_processors: list[Any] | None = None,
    ) -> AsyncGenerator[VLMBatchChunk, None]:
        """Submit a prepared multimodal request and stream decoded chunks."""
        if not self._running:
            raise RuntimeError("VLMBatchScheduler is not running; call start() first")

        loop = asyncio.get_running_loop()
        out_queue: asyncio.Queue[Any] = asyncio.Queue()
        cancel_event = threading.Event()
        request = _PendingVLMRequest(
            model_inputs=model_inputs,
            max_tokens=max_tokens,
            logits_processors=logits_processors,
            loop=loop,
            out_queue=out_queue,
            cancel_event=cancel_event,
        )
        try:
            self._admission_queue.put_nowait(request)
        except queue.Full as exc:
            raise asyncio.QueueFull("VLMBatchScheduler admission queue is full") from exc

        async def _stream() -> AsyncGenerator[VLMBatchChunk, None]:
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
        try:
            new_thread_local_stream = getattr(mx, "new_thread_local_stream", None)
            if new_thread_local_stream is not None:
                self._stream = new_thread_local_stream(mx.default_device())
            else:
                self._stream = mx.new_stream(mx.default_device())

            self._stop_tokens = self._resolve_stop_tokens()
            self._batch_generator_kwargs = {
                "completion_batch_size": self._completion_batch_size,
                "prefill_batch_size": self._prefill_batch_size,
                "prefill_step_size": self._prefill_step_size,
                "stop_tokens": self._stop_tokens,
                "kv_bits": self._model_wrapper.kv_bits,
                "kv_group_size": self._model_wrapper.kv_group_size,
                "quantized_kv_start": self._model_wrapper.quantized_kv_start,
                "compute_logprobs": False,
            }
            if _VLM_BATCH_GENERATOR_ACCEPTS_STREAM:
                self._batch_generator_kwargs["stream"] = self._stream
            self._batch_generator = self._new_batch_generator()
        except BaseException as exc:  # noqa: BLE001 - surface to start()
            self._start_error = exc
            self._ready_event.set()
            self._running = False
            return

        self._ready_event.set()
        try:
            while self._running:
                lock_context = (
                    self._generation_lock if self._generation_lock is not None else nullcontext()
                )
                did_work = False
                with lock_context:
                    self._admit_pending(block_if_empty=len(self._active) == 0)
                    if (
                        self._running
                        and not self._active
                        and self._batch_generator is not None
                        and self._batch_generator.has_work
                    ):
                        self._reset_orphaned_batch_generator()
                    if self._running and self._active and self._batch_generator.has_work:
                        try:
                            # Run one decode step per lock hold. Seeded or sampled
                            # requests use the fallback worker and need a chance to
                            # acquire the shared generation lock between batch steps.
                            _prompt_responses, gen_responses = self._batch_generator.next()
                        except Exception as exc:  # noqa: BLE001 - propagate per request
                            logger.exception(f"VLM BatchGenerator.next() raised: {exc!s}")
                            self._fail_all_active(exc)
                            continue
                        for response in gen_responses:
                            self._handle_generation_response(response)
                        self._process_cancellations()
                        self._admit_pending(block_if_empty=False)
                        did_work = True
                if did_work:
                    time.sleep(0)
        finally:
            try:
                if self._batch_generator is not None:
                    self._batch_generator.close()
            except Exception as exc:  # noqa: BLE001 - teardown best-effort
                logger.warning(f"Error closing VLM BatchGenerator: {exc!s}")
            self._batch_generator = None
            self._stream = None
            self._batch_generator_kwargs = {}
            self._fail_all_active(RuntimeError("VLMBatchScheduler stopped"))
            self._drain_admission_queue(RuntimeError("VLMBatchScheduler stopped"))

    def _new_batch_generator(self) -> Any:
        """Create a fresh upstream VLM batch generator."""
        return BatchGenerator(
            self._model_wrapper.model.language_model,
            self._model_wrapper.processor,
            **self._batch_generator_kwargs,
        )

    def _reset_orphaned_batch_generator(self) -> None:
        """Reset upstream work that no longer maps to an active server request."""
        logger.warning(
            "VLMBatchScheduler detected upstream BatchGenerator work with no active "
            "server requests; resetting stale VLM batch state"
        )
        try:
            self._batch_generator.close()
        except Exception as exc:  # noqa: BLE001 - stale cleanup is best-effort
            logger.warning(f"Error closing stale VLM BatchGenerator: {exc!s}")
        self._batch_generator = self._new_batch_generator()

    def _admit_pending(self, *, block_if_empty: bool) -> None:
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
                if request is not None:
                    self._fail_request(request, RuntimeError("VLMBatchScheduler stopped"))
                return
            if request is None:
                return
            if request.cancel_event.is_set():
                self._send(request.loop, request.out_queue, _STREAM_SENTINEL)
                return

            try:
                input_ids, prompt_kwargs = self._model_wrapper.create_batch_prompt_kwargs(
                    request.model_inputs
                )
                uids = self._batch_generator.insert(
                    [input_ids],
                    max_tokens=[request.max_tokens],
                    prompt_kwargs=[prompt_kwargs],
                    logits_processors=[request.logits_processors],
                )
            except Exception as exc:  # noqa: BLE001 - report per request
                self._fail_request(request, exc)
                continue

            (uid,) = uids
            self._active[uid] = _ActiveVLMRequest(
                loop=request.loop,
                out_queue=request.out_queue,
                detokenizer=None,
                cancel_event=request.cancel_event,
                prompt_tokens=len(input_ids),
                prompt_start_time=time.perf_counter(),
                tokens=[],
            )
            logger.info(
                f"VLMBatchScheduler admitted uid={uid} "
                f"(active={len(self._active)}, prompt_tokens={len(input_ids)})"
            )
            # Each request's multimodal embeddings are prepared independently.
            # The upstream VLM batcher can prefill multiple prompts together
            # only when ``inputs_embeds`` was produced as one batched tensor, so
            # admit at most one new prompt per scheduler step. Decode batches
            # still grow continuously as prefills finish.
            return

    def _handle_generation_response(self, resp: Any) -> None:
        state = self._active.get(resp.uid)
        if state is None or state.cancel_event.is_set():
            return
        if state.first_token_time is None:
            state.first_token_time = time.perf_counter()

        token = int(resp.token)
        finish_reason = resp.finish_reason
        if finish_reason is None and token in self._stop_tokens:
            finish_reason = "stop"
            self._remove_uid(resp.uid)
            logger.debug(f"VLMBatchScheduler forced stop for uid={resp.uid} on EOS token={token}")
        is_final = finish_reason is not None
        state.generation_tokens += 1

        if finish_reason == "stop":
            segment = self._finalize_detokenizer(state)
        else:
            segment = self._decode_token(state, token)
            if is_final:
                segment += self._finalize_detokenizer(state)

        chunk = VLMBatchChunk(
            text=segment,
            token=token,
            finish_reason=finish_reason,
            generation_tokens=state.generation_tokens,
            generation_tps=self._compute_generation_tps(state) if is_final else 0.0,
            prompt_tokens=state.prompt_tokens,
            prompt_tps=self._compute_prompt_tps(state) if is_final else 0.0,
            peak_memory=(mx.get_peak_memory() / 1e9) if is_final else 0.0,
        )
        self._send(state.loop, state.out_queue, chunk)

        if is_final:
            logger.info(
                f"VLMBatchScheduler uid={resp.uid} finished "
                f"(finish_reason={finish_reason}, generated={state.generation_tokens})"
            )
            self._send(state.loop, state.out_queue, _STREAM_SENTINEL)
            self._active.pop(resp.uid, None)

    def _remove_uid(self, uid: int) -> None:
        """Remove a uid from the underlying batch generator."""
        try:
            self._batch_generator.remove(uid)
        except Exception as exc:  # noqa: BLE001 - best-effort forced stop
            logger.warning(f"VLM BatchGenerator.remove failed for uid={uid}: {exc!s}")

    def _resolve_stop_tokens(self) -> set[int]:
        """Resolve EOS token ids that should stop VLM batched generation."""
        stop_tokens: set[int] = set()
        config = getattr(self._model_wrapper, "config", None) or getattr(
            getattr(self._model_wrapper, "model", None), "config", None
        )
        stop_tokens.update(self._coerce_token_ids(getattr(config, "eos_token_id", None)))

        tokenizer = self._get_tokenizer()
        stop_tokens.update(self._coerce_token_ids(getattr(tokenizer, "eos_token_ids", None)))
        stop_tokens.update(self._coerce_token_ids(getattr(tokenizer, "eos_token_id", None)))
        stop_tokens.update(self._encode_stop_text(getattr(tokenizer, "eos_token", None)))
        return stop_tokens

    @staticmethod
    def _coerce_token_ids(value: Any) -> set[int]:
        """Normalize scalar or iterable token ids into a set of ints."""
        if value is None:
            return set()
        if isinstance(value, int):
            return {value}
        return {int(token) for token in value if token is not None}

    def _encode_stop_text(self, text: Any) -> set[int]:
        """Encode textual EOS markers like upstream ``mlx-vlm``."""
        if not isinstance(text, str) or not text:
            return set()
        for encoder in (self._model_wrapper.processor, self._get_tokenizer()):
            encode = getattr(encoder, "encode", None)
            if encode is None:
                continue
            try:
                token_ids = encode(f" {text}", add_special_tokens=False)
            except TypeError:
                token_ids = encode(f" {text}")
            if token_ids:
                return {int(token_ids[-1])}
        return set()

    def _get_tokenizer(self) -> Any:
        """Return the tokenizer-like object used for VLM text decoding."""
        processor = self._model_wrapper.processor
        return getattr(processor, "tokenizer", processor)

    def _make_detokenizer(self) -> Any | None:
        """Create an isolated incremental detokenizer when the tokenizer exposes one."""
        for source in (self._get_tokenizer(), self._model_wrapper.processor):
            if not hasattr(source, "detokenizer"):
                continue
            first = source.detokenizer
            second = source.detokenizer
            if first is not second:
                return first
        return None

    def _decode_token(self, state: _ActiveVLMRequest, token: int) -> str:
        """Decode one token using upstream ``mlx-vlm`` server's token-list method."""
        if state.tokens is None:
            state.tokens = []
        state.tokens.append(token)
        tokenizer = self._get_tokenizer()
        decode = getattr(tokenizer, "decode", None)
        if decode is not None:
            current_text = decode(state.tokens)
            segment = current_text[len(state.previous_text) :]
            state.previous_text = current_text
            return segment

        if state.detokenizer is None:
            state.detokenizer = self._make_detokenizer()
            if state.detokenizer is not None:
                state.detokenizer.reset()
        if state.detokenizer is None:
            return ""
        state.detokenizer.add_token(token)
        current_text = state.detokenizer.text
        segment = current_text[len(state.previous_text) :]
        state.previous_text = current_text
        return segment

    @staticmethod
    def _finalize_detokenizer(state: _ActiveVLMRequest) -> str:
        """Flush pending text from an incremental detokenizer."""
        if state.detokenizer is None:
            return ""
        try:
            state.detokenizer.finalize()
            return state.detokenizer.last_segment
        except Exception:  # noqa: BLE001 - finalize is best-effort
            return ""

    @staticmethod
    def _compute_generation_tps(state: _ActiveVLMRequest) -> float:
        if state.first_token_time is None or state.generation_tokens == 0:
            return 0.0
        elapsed = time.perf_counter() - state.first_token_time
        return state.generation_tokens / elapsed if elapsed > 0 else 0.0

    @staticmethod
    def _compute_prompt_tps(state: _ActiveVLMRequest) -> float:
        elapsed = (state.first_token_time or time.perf_counter()) - state.prompt_start_time
        return state.prompt_tokens / elapsed if elapsed > 0 else 0.0

    def _process_cancellations(self) -> None:
        cancelled = [uid for uid, state in self._active.items() if state.cancel_event.is_set()]
        if not cancelled:
            return
        for uid in cancelled:
            try:
                self._batch_generator.remove(uid)
            except Exception as exc:  # noqa: BLE001 - cancellation is best-effort
                logger.warning(f"VLM BatchGenerator.remove failed for uid={uid}: {exc!s}")
            state = self._active.pop(uid, None)
            if state is None:
                continue
            chunk = VLMBatchChunk(
                text="",
                token=0,
                finish_reason="cancelled",
                generation_tokens=state.generation_tokens,
                generation_tps=self._compute_generation_tps(state),
                prompt_tokens=state.prompt_tokens,
                prompt_tps=self._compute_prompt_tps(state),
                peak_memory=mx.get_peak_memory() / 1e9,
            )
            self._send(state.loop, state.out_queue, chunk)
            self._send(state.loop, state.out_queue, _STREAM_SENTINEL)

    def _fail_all_active(self, exc: BaseException) -> None:
        for _uid, state in list(self._active.items()):
            self._send(state.loop, state.out_queue, exc)
            self._send(state.loop, state.out_queue, _STREAM_SENTINEL)
        self._active.clear()

    def _drain_admission_queue(self, exc: BaseException) -> None:
        while True:
            try:
                request = self._admission_queue.get_nowait()
            except queue.Empty:
                return
            if request is None:
                continue
            self._fail_request(request, exc)

    def _fail_request(self, request: _PendingVLMRequest, exc: BaseException) -> None:
        self._send(request.loop, request.out_queue, exc)
        self._send(request.loop, request.out_queue, _STREAM_SENTINEL)

    def get_stats(self) -> dict[str, Any]:
        """Return current scheduler queue and active-request stats."""
        return {
            "running": self._running,
            "queue_size": self._admission_queue.qsize(),
            "max_queue_size": self._queue_size,
            "active_requests": len(self._active),
        }

    @staticmethod
    def _send(loop: asyncio.AbstractEventLoop, q: asyncio.Queue[Any], item: Any) -> None:
        try:
            loop.call_soon_threadsafe(q.put_nowait, item)
        except RuntimeError:
            logger.debug("Event loop closed before VLM scheduler response could be delivered")
