"""Dedicated inference thread for non-blocking MLX model execution.

A **single** background thread owns all model computation while async
handlers submit work via thread-safe queues, keeping the event loop
responsive.  Includes built-in request queuing with backpressure,
timeout handling, and statistics — replacing the need for a separate
``RequestQueue`` layer.
"""

from __future__ import annotations

import asyncio
import gc
import queue
import threading
from collections.abc import AsyncGenerator, Generator
from threading import Thread
from typing import Any, Callable

from loguru import logger

# Sentinel object used to signal the end of a stream.
_SENTINEL = object()


def _safe_set_result(future: asyncio.Future[Any], result: Any) -> None:
    """Set *result* on *future*, silently ignoring if already done.

    This avoids ``InvalidStateError`` when the caller has timed out
    (and the future was cancelled) but the work closure completes later.
    """
    if not future.done():
        try:
            future.set_result(result)
        except asyncio.InvalidStateError:
            pass


def _safe_set_exception(future: asyncio.Future[Any], exc: BaseException) -> None:
    """Set *exc* on *future*, silently ignoring if already done."""
    if not future.done():
        try:
            future.set_exception(exc)
        except asyncio.InvalidStateError:
            pass


class InferenceWorker:
    """Single dedicated thread for running blocking MLX model inference.

    All heavy model work (prompt processing, token generation, embedding
    computation, image generation, audio transcription, etc.) is executed
    on one background thread so that the asyncio event loop is never
    blocked.

    Combines the responsibilities of the old inference thread *and* the
    ``RequestQueue`` into a single unified component with built-in
    backpressure, timeout handling, and statistics.

    Usage
    -----
    >>> worker = InferenceWorker(queue_size=100, timeout=300.0)
    >>> worker.start()
    >>> # non-streaming
    >>> result = await worker.submit(model, input_ids=ids, stream=False)
    >>> # streaming
    >>> async for chunk in worker.submit_stream(model, input_ids=ids, stream=True):
    ...     process(chunk)
    >>> worker.stop()
    """

    def __init__(
        self,
        queue_size: int = 100,
        timeout: float = 300.0,
    ) -> None:
        """Initialize the inference worker.

        Parameters
        ----------
        queue_size : int
            Maximum number of pending work items.  When full, ``submit``
            and ``submit_stream`` raise ``asyncio.QueueFull``.
        timeout : float
            Default timeout in seconds for awaiting non-streaming
            results via ``submit``.
        """
        self._queue_size = queue_size
        self._timeout = timeout
        self._work_queue: queue.Queue[Callable[[], None]] = queue.Queue(
            maxsize=queue_size
        )
        self._thread: Thread | None = None
        self._running = False

        # Stats — protected by ``_stats_lock`` for thread-safe access.
        self._stats_lock = threading.Lock()
        self._active = False
        self._completed_count = 0
        self._failed_count = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the inference worker thread.

        Safe to call multiple times; subsequent calls are no-ops.
        """
        if self._running:
            return
        self._running = True
        self._thread = Thread(
            target=self._run, daemon=True, name="inference-worker"
        )
        self._thread.start()
        logger.info("Inference worker thread started")

    def stop(self) -> None:
        """Stop the inference worker thread.

        Blocks until the thread finishes (up to 5 s timeout).
        """
        if not self._running:
            return
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.info("Inference worker thread stopped")

    # ------------------------------------------------------------------
    # Internal thread loop
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """Background thread loop — execute work items sequentially."""
        while self._running:
            try:
                work = self._work_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            with self._stats_lock:
                self._active = True
            try:
                work()
            except Exception:
                # Exceptions are captured inside each work closure and
                # forwarded to the event loop via futures / queues.
                # This outer catch is a safety net to keep the thread alive.
                pass
            finally:
                with self._stats_lock:
                    self._active = False

    # ------------------------------------------------------------------
    # Stats helpers
    # ------------------------------------------------------------------

    def _inc_completed(self) -> None:
        """Record a successful work-item completion."""
        with self._stats_lock:
            self._completed_count += 1

    def _inc_failed(self) -> None:
        """Record a failed work-item execution."""
        with self._stats_lock:
            self._failed_count += 1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def submit(
        self, func: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        """Submit a blocking callable and ``await`` its result.

        The callable is executed on the dedicated inference thread.
        The caller's coroutine is suspended (not blocking the event
        loop) until the result is available or the timeout fires.

        Parameters
        ----------
        func : Callable
            The blocking function to execute on the inference thread.
        *args : Any
            Positional arguments forwarded to *func*.
        **kwargs : Any
            Keyword arguments forwarded to *func*.

        Returns
        -------
        Any
            The return value of *func*.

        Raises
        ------
        asyncio.QueueFull
            If the work queue has reached its capacity.
        TimeoutError
            If the result is not available within the configured timeout.
        Exception
            Any exception raised by *func* is re-raised in the caller.
        """
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()

        def _work() -> None:
            try:
                result = func(*args, **kwargs)
                loop.call_soon_threadsafe(_safe_set_result, future, result)
                self._inc_completed()
            except BaseException as exc:
                loop.call_soon_threadsafe(_safe_set_exception, future, exc)
                self._inc_failed()
            finally:
                gc.collect()

        try:
            self._work_queue.put_nowait(_work)
        except queue.Full:
            raise asyncio.QueueFull("Inference queue is full")

        try:
            return await asyncio.wait_for(future, timeout=self._timeout)
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"Inference timed out after {self._timeout}s"
            )

    def submit_stream(
        self,
        func: Callable[..., Generator[Any, None, None]],
        *args: Any,
        **kwargs: Any,
    ) -> AsyncGenerator[Any, None]:
        """Submit a callable that returns a sync iterable, yielding items asynchronously.

        The callable is executed on the inference thread.  Each item
        produced by the returned iterable is forwarded to the async
        generator through an ``asyncio.Queue``, keeping the event loop
        free between items.

        Parameters
        ----------
        func : Callable
            Function that returns a synchronous iterable / generator.
        *args : Any
            Positional arguments forwarded to *func*.
        **kwargs : Any
            Keyword arguments forwarded to *func*.

        Returns
        -------
        AsyncGenerator
            Async generator yielding items from the sync iterable.

        Raises
        ------
        asyncio.QueueFull
            If the work queue has reached its capacity.
        """
        loop = asyncio.get_running_loop()
        token_queue: asyncio.Queue[Any] = asyncio.Queue()

        def _work() -> None:
            try:
                gen = func(*args, **kwargs)
                for item in gen:
                    loop.call_soon_threadsafe(token_queue.put_nowait, item)
                loop.call_soon_threadsafe(token_queue.put_nowait, _SENTINEL)
                self._inc_completed()
            except BaseException as exc:
                loop.call_soon_threadsafe(token_queue.put_nowait, exc)
                self._inc_failed()
            finally:
                gc.collect()

        try:
            self._work_queue.put_nowait(_work)
        except queue.Full:
            raise asyncio.QueueFull("Inference queue is full")

        return self._read_stream(token_queue)

    def get_stats(self) -> dict[str, Any]:
        """Get current worker and queue statistics.

        Returns
        -------
        dict[str, Any]
            Dictionary with the following keys:

            - ``running`` – whether the worker thread is alive.
            - ``queue_size`` – number of items waiting in the queue.
            - ``max_queue_size`` – maximum queue capacity.
            - ``active_requests`` – ``1`` if currently processing, else ``0``.
            - ``completed_requests`` – total successful completions.
            - ``failed_requests`` – total failed work items.
        """
        with self._stats_lock:
            return {
                "running": self._running,
                "queue_size": self._work_queue.qsize(),
                "max_queue_size": self._queue_size,
                "active_requests": 1 if self._active else 0,
                "completed_requests": self._completed_count,
                "failed_requests": self._failed_count,
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    async def _read_stream(
        token_queue: asyncio.Queue[Any],
    ) -> AsyncGenerator[Any, None]:
        """Consume items from *token_queue* as an async generator.

        Yields
        ------
        Any
            Items produced by the inference thread, one at a time.

        Raises
        ------
        BaseException
            If the inference thread forwarded an exception.
        """
        while True:
            item = await token_queue.get()
            if item is _SENTINEL:
                break
            if isinstance(item, BaseException):
                raise item
            yield item
