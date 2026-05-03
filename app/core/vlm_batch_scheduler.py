"""Continuous-batch scheduler for ``mlx-vlm`` multimodal generation."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass
import queue
import threading
import time
from typing import Any

from loguru import logger
import mlx.core as mx

try:
    from mlx_vlm.generate import (
        DEFAULT_KV_GROUP_SIZE,
        DEFAULT_KV_QUANT_SCHEME,
        DEFAULT_QUANTIZED_KV_START,
        BatchGenerator,
    )

    VLM_BATCHING_AVAILABLE = True
except (ImportError, RuntimeError) as exc:  # pragma: no cover - optional backend
    DEFAULT_KV_GROUP_SIZE = 64
    DEFAULT_KV_QUANT_SCHEME = "uniform"
    DEFAULT_QUANTIZED_KV_START = 0
    BatchGenerator = None  # type: ignore[assignment,misc]
    VLM_BATCHING_AVAILABLE = False
    logger.warning(
        "mlx_vlm.generate.BatchGenerator is unavailable; multimodal continuous "
        f"batching is disabled. Upgrade mlx-vlm to enable VLM batching: {exc!s}"
    )


@dataclass
class VLMBatchChunk:
    """Per-request chunk emitted by :class:`VLMBatchScheduler`.

    Parameters
    ----------
    text : str
        Incremental decoded text segment since the previous chunk.
    token : int
        Sampled token id for this step.
    finish_reason : str | None
        ``None`` while generation is in progress, otherwise the terminal
        reason reported by ``mlx-vlm``.
    generation_tokens : int
        Number of generated tokens for this request.
    generation_tps : float
        Average generation tokens per second, populated on terminal chunks.
    prompt_tokens : int
        Number of prompt tokens processed for the request.
    prompt_tps : float
        Prompt processing tokens per second, populated from ``BatchGenerator``
        aggregate stats on terminal chunks when available.
    peak_memory : float
        Peak MLX memory in GB, populated on terminal chunks.
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
class _PendingVLMRequest:
    """Queued multimodal request waiting to enter the VLM batch."""

    raw_inputs: dict[str, Any]
    prompt_tokens: int
    max_tokens: int
    sampler: Callable[[mx.array], mx.array] | None
    logits_processors: list[Callable[[mx.array, mx.array], mx.array]] | None
    loop: asyncio.AbstractEventLoop
    out_queue: asyncio.Queue[Any]
    cancel_event: threading.Event


@dataclass
class _ActiveVLMRequest:
    """State tracked for a live multimodal sequence."""

    loop: asyncio.AbstractEventLoop
    out_queue: asyncio.Queue[Any]
    cancel_event: threading.Event
    prompt_tokens: int
    tokens: list[int]
    previous_text: str = ""
    first_token_time: float | None = None


_STREAM_SENTINEL: Any = object()


class VLMBatchScheduler:
    """Continuous-batch scheduler backed by ``mlx_vlm.generate.BatchGenerator``.

    The full multimodal model is used only to build input embeddings on this
    scheduler thread. The underlying ``BatchGenerator`` then batches the
    model's language model exactly as the upstream ``mlx-vlm`` server does.
    Prompt-prefix caching is intentionally not implemented because mlx-vlm's
    cache support is immature and cross-thread cache reuse is fragile.
    """

    def __init__(
        self,
        model: Any,
        processor: Any,
        *,
        completion_batch_size: int = 32,
        prefill_batch_size: int = 8,
        prefill_step_size: int = 2048,
        kv_bits: int | None = None,
        kv_group_size: int = DEFAULT_KV_GROUP_SIZE,
        kv_quant_scheme: str = DEFAULT_KV_QUANT_SCHEME,
        quantized_kv_start: int = DEFAULT_QUANTIZED_KV_START,
        queue_size: int = 100,
        idle_poll_timeout: float = 0.1,
    ) -> None:
        self._model = model
        self._language_model = getattr(model, "language_model", model)
        self._processor = processor
        self._tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
        self._completion_batch_size = completion_batch_size
        self._prefill_batch_size = prefill_batch_size
        self._prefill_step_size = prefill_step_size
        self._kv_bits = kv_bits
        self._kv_group_size = kv_group_size
        self._kv_quant_scheme = kv_quant_scheme
        self._quantized_kv_start = quantized_kv_start
        self._queue_size = queue_size
        self._idle_poll_timeout = idle_poll_timeout

        self._admission_queue: queue.Queue[_PendingVLMRequest] = queue.Queue(maxsize=queue_size)
        self._batch_generator: Any | None = None
        self._active: dict[int, _ActiveVLMRequest] = {}
        self._stream: Any | None = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._state_lock = threading.Lock()
        self._ready_event = threading.Event()
        self._start_error: BaseException | None = None
        self._stop_tokens = self._resolve_stop_tokens()

    @property
    def is_running(self) -> bool:
        """Return whether the scheduler worker thread is active."""
        return self._running

    def start(self) -> None:
        """Start the VLM batch scheduler thread."""
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

    def stop(self) -> None:
        """Stop the scheduler and fail pending requests."""
        with self._state_lock:
            if not self._running:
                return
            self._running = False
            thread = self._thread
            self._thread = None
        if thread is not None:
            thread.join(timeout=10.0)
        logger.info("VLMBatchScheduler stopped")

    def submit_stream(
        self,
        raw_inputs: dict[str, Any],
        *,
        prompt_tokens: int,
        max_tokens: int,
        sampler: Callable[[mx.array], mx.array] | None = None,
        logits_processors: list[Callable[[mx.array, mx.array], mx.array]] | None = None,
    ) -> AsyncGenerator[VLMBatchChunk, None]:
        """Submit one multimodal request to the continuous batcher."""
        if not self._running:
            raise RuntimeError("VLMBatchScheduler is not running; call start() first")

        loop = asyncio.get_running_loop()
        out_queue: asyncio.Queue[Any] = asyncio.Queue()
        cancel_event = threading.Event()
        request = _PendingVLMRequest(
            raw_inputs=dict(raw_inputs),
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            sampler=sampler,
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
        """Own the MLX stream, VLM embedding, and ``BatchGenerator`` loop."""
        try:
            new_thread_local_stream = getattr(mx, "new_thread_local_stream", None)
            if new_thread_local_stream is not None:
                self._stream = new_thread_local_stream(mx.default_device())
            else:
                self._stream = mx.new_stream(mx.default_device())
            self._ready_event.set()
        except BaseException as exc:  # noqa: BLE001 - surface to start()
            self._start_error = exc
            self._ready_event.set()
            self._running = False
            return

        try:
            while self._running:
                self._admit_pending(block_if_empty=not self._active)
                self._process_cancellations()

                if not self._active or self._batch_generator is None:
                    continue

                self._step()
                self._process_cancellations()
        finally:
            try:
                if self._batch_generator is not None:
                    self._batch_generator.close()
            except Exception as exc:  # noqa: BLE001 - teardown best effort
                logger.warning(f"Error closing VLM BatchGenerator: {exc!s}")
            self._batch_generator = None
            self._stream = None
            self._fail_all_active(RuntimeError("VLMBatchScheduler stopped"))
            self._drain_admission_queue(RuntimeError("VLMBatchScheduler stopped"))

    def _admit_pending(self, *, block_if_empty: bool) -> None:
        """Drain queued requests into ``BatchGenerator``."""
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

            if request.cancel_event.is_set():
                self._send(request.loop, request.out_queue, _STREAM_SENTINEL)
                continue

            if self._batch_generator is None:
                self._batch_generator = self._create_batch_generator(request.sampler)

            try:
                input_ids, prompt_kwargs = self._build_prompt_kwargs(request.raw_inputs)
                if prompt_kwargs.get("inputs_embeds") is not None and getattr(
                    self._batch_generator, "unprocessed_prompts", None
                ):
                    self._flush_pending_prompts()

                (uid,) = self._batch_generator.insert(
                    [self._as_token_list(input_ids)],
                    max_tokens=request.max_tokens,
                    prompt_kwargs=[prompt_kwargs],
                    logits_processors=(
                        [request.logits_processors]
                        if request.logits_processors is not None
                        else None
                    ),
                )
            except Exception as exc:  # noqa: BLE001 - report per request
                self._fail_request(request, exc)
                continue

            self._active[uid] = _ActiveVLMRequest(
                loop=request.loop,
                out_queue=request.out_queue,
                cancel_event=request.cancel_event,
                prompt_tokens=request.prompt_tokens,
                tokens=[],
            )
            logger.info(
                f"VLMBatchScheduler admitted uid={uid} "
                f"(active={len(self._active)}, prompt_tokens={request.prompt_tokens})"
            )

            # VLM/image requests need their prefill started immediately so
            # their embedding kwargs do not get mixed with later text-only work.
            self._step()

    def _create_batch_generator(
        self,
        sampler: Callable[[mx.array], mx.array] | None,
    ) -> Any:
        """Create an ``mlx-vlm`` batch generator on the scheduler thread."""
        return BatchGenerator(
            self._language_model,
            self._processor,
            stop_tokens=self._stop_tokens,
            sampler=sampler,
            completion_batch_size=self._completion_batch_size,
            prefill_batch_size=self._prefill_batch_size,
            prefill_step_size=self._prefill_step_size,
            kv_bits=self._kv_bits,
            kv_group_size=self._kv_group_size,
            kv_quant_scheme=self._kv_quant_scheme,
            quantized_kv_start=self._quantized_kv_start,
            stream=self._stream,
        )

    def _build_prompt_kwargs(self, raw_inputs: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        """Run VLM embedding on the scheduler thread and return prompt kwargs."""
        input_ids = raw_inputs.get("input_ids")
        pixel_values = raw_inputs.get("pixel_values")
        mask = raw_inputs.get("attention_mask")
        if mask is None:
            mask = raw_inputs.get("mask")
        data_kwargs = {
            key: value
            for key, value in raw_inputs.items()
            if key not in {"input_ids", "pixel_values", "attention_mask", "mask"}
        }

        embedding_output = self._model.get_input_embeddings(
            input_ids,
            pixel_values,
            mask=mask,
            **data_kwargs,
        )
        return input_ids, {**data_kwargs, **embedding_output.to_dict()}

    @staticmethod
    def _as_token_list(input_ids: Any) -> list[int]:
        """Convert a 1-row token tensor/array/list into a plain token list."""
        if hasattr(input_ids, "squeeze"):
            input_ids = input_ids.squeeze(0)
        if hasattr(input_ids, "tolist"):
            input_ids = input_ids.tolist()
        if input_ids and isinstance(input_ids[0], list):
            input_ids = input_ids[0]
        return [int(token) for token in input_ids]

    def _step(self) -> None:
        """Run one VLM batch step and forward responses."""
        try:
            _, responses = self._batch_generator.next()
        except Exception as exc:  # noqa: BLE001 - propagate to active requests
            logger.exception(f"VLM BatchGenerator.next() raised: {exc!s}")
            self._fail_all_active(exc)
            return

        for response in responses:
            self._handle_generation_response(response)

    def _flush_pending_prompts(self) -> None:
        """Drain queued prompt prefill work before inserting another image prompt."""
        while getattr(self._batch_generator, "has_pending_prompts", False):
            self._step()

    def _handle_generation_response(self, response: Any) -> None:
        """Forward one ``BatchGenerator`` response to its request queue."""
        state = self._active.get(response.uid)
        if state is None or state.cancel_event.is_set():
            return

        if state.first_token_time is None:
            state.first_token_time = time.perf_counter()

        token = response.token.item() if hasattr(response.token, "item") else response.token
        finish_reason = response.finish_reason
        state.tokens.append(int(token))

        if finish_reason == "stop":
            text = ""
        else:
            current_text = self._tokenizer.decode(state.tokens)
            text = current_text[len(state.previous_text) :]
            state.previous_text = current_text

        generation_tokens = len(state.tokens)
        is_final = finish_reason is not None
        stats = self._batch_generator.stats() if is_final else None
        chunk = VLMBatchChunk(
            text=text,
            token=int(token),
            finish_reason=finish_reason,
            generation_tokens=generation_tokens,
            generation_tps=self._compute_tps(state) if is_final else 0.0,
            prompt_tokens=state.prompt_tokens,
            prompt_tps=getattr(stats, "prompt_tps", 0.0) if stats is not None else 0.0,
            peak_memory=mx.get_peak_memory() / 1e9 if is_final else 0.0,
        )
        self._send(state.loop, state.out_queue, chunk)

        if is_final:
            logger.info(
                f"VLMBatchScheduler uid={response.uid} finished "
                f"(finish_reason={finish_reason}, generated={generation_tokens})"
            )
            self._send(state.loop, state.out_queue, _STREAM_SENTINEL)
            self._active.pop(response.uid, None)

    def _process_cancellations(self) -> None:
        """Remove cancelled sequences from ``BatchGenerator``."""
        cancelled_uids = [uid for uid, state in self._active.items() if state.cancel_event.is_set()]
        if not cancelled_uids or self._batch_generator is None:
            return
        for uid in cancelled_uids:
            try:
                self._batch_generator.remove(uid)
            except Exception as exc:  # noqa: BLE001 - cancellation is best effort
                logger.warning(f"VLM BatchGenerator.remove failed for uid={uid}: {exc!s}")
            state = self._active.pop(uid, None)
            if state is not None:
                chunk = VLMBatchChunk(
                    text="",
                    token=0,
                    finish_reason="cancelled",
                    generation_tokens=len(state.tokens),
                    generation_tps=self._compute_tps(state),
                    prompt_tokens=state.prompt_tokens,
                    peak_memory=mx.get_peak_memory() / 1e9,
                )
                self._send(state.loop, state.out_queue, chunk)
                self._send(state.loop, state.out_queue, _STREAM_SENTINEL)

    def _resolve_stop_tokens(self) -> set[int]:
        """Collect EOS token ids from model config."""
        stop_tokens: set[int] = set()
        eos_token_id = getattr(getattr(self._model, "config", None), "eos_token_id", None)
        if isinstance(eos_token_id, list):
            stop_tokens.update(int(token) for token in eos_token_id)
        elif eos_token_id is not None:
            stop_tokens.add(int(eos_token_id))
        return stop_tokens

    def get_stats(self) -> dict[str, Any]:
        """Current scheduler queue and active-request stats."""
        return {
            "running": self._running,
            "queue_size": self._admission_queue.qsize(),
            "max_queue_size": self._queue_size,
            "active_requests": len(self._active),
        }

    @staticmethod
    def _compute_tps(state: _ActiveVLMRequest) -> float:
        if state.first_token_time is None or not state.tokens:
            return 0.0
        elapsed = time.perf_counter() - state.first_token_time
        if elapsed <= 0:
            return 0.0
        return len(state.tokens) / elapsed

    def _fail_all_active(self, exc: BaseException) -> None:
        for uid, state in list(self._active.items()):
            self._send(state.loop, state.out_queue, exc)
            self._send(state.loop, state.out_queue, _STREAM_SENTINEL)
            self._active.pop(uid, None)

    def _drain_admission_queue(self, exc: BaseException) -> None:
        while True:
            try:
                request = self._admission_queue.get_nowait()
            except queue.Empty:
                return
            self._fail_request(request, exc)

    def _fail_request(self, request: _PendingVLMRequest, exc: BaseException) -> None:
        self._send(request.loop, request.out_queue, exc)
        self._send(request.loop, request.out_queue, _STREAM_SENTINEL)

    @staticmethod
    def _send(loop: asyncio.AbstractEventLoop, out_queue: asyncio.Queue[Any], item: Any) -> None:
        """Thread-safe put of ``item`` onto an asyncio queue."""
        try:
            loop.call_soon_threadsafe(out_queue.put_nowait, item)
        except RuntimeError:
            logger.debug("Event loop closed before VLM scheduler response could be delivered")
