"""Dedicated inference thread for non-blocking MLX model execution.

Mirrors the pattern from upstream ``mlx-lm`` ``server.py``
(``ResponseGenerator``), adapted for asyncio.  A **single** background
thread owns all model computation while async handlers submit work via
thread-safe queues, keeping the event loop responsive.
"""

from __future__ import annotations

import asyncio
import queue
from collections.abc import AsyncGenerator, Generator
from threading import Thread
from typing import Any, Callable

from loguru import logger

# Sentinel object used to signal the end of a stream.
_SENTINEL = object()


class InferenceWorker:
    """Single dedicated thread for running blocking MLX model inference.

    All heavy model work (prompt processing, token generation, embedding
    computation, image generation, etc.) is executed on one background
    thread so that the asyncio event loop is never blocked.

    Usage
    -----
    >>> worker = InferenceWorker()
    >>> worker.start()
    >>> # non-streaming
    >>> result = await worker.submit(model, input_ids=ids, stream=False)
    >>> # streaming
    >>> async for chunk in worker.submit_stream(model, input_ids=ids, stream=True):
    ...     process(chunk)
    >>> worker.stop()
    """

    def __init__(self) -> None:
        self._work_queue: queue.Queue[Callable[[], None]] = queue.Queue()
        self._thread: Thread | None = None
        self._running = False

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
            try:
                work()
            except Exception:
                # Exceptions are captured inside each work closure and
                # forwarded to the event loop via futures / queues.
                # This outer catch is a safety net to keep the thread alive.
                pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def submit(
        self, func: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        """Submit a blocking callable and ``await`` its result.

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
        Exception
            Any exception raised by *func* is re-raised in the caller.
        """
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()

        def _work() -> None:
            try:
                result = func(*args, **kwargs)
                loop.call_soon_threadsafe(future.set_result, result)
            except BaseException as exc:
                loop.call_soon_threadsafe(future.set_exception, exc)

        self._work_queue.put(_work)
        return await future

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
        """
        loop = asyncio.get_running_loop()
        token_queue: asyncio.Queue[Any] = asyncio.Queue()

        def _work() -> None:
            try:
                gen = func(*args, **kwargs)
                for item in gen:
                    loop.call_soon_threadsafe(token_queue.put_nowait, item)
                loop.call_soon_threadsafe(token_queue.put_nowait, _SENTINEL)
            except BaseException as exc:
                loop.call_soon_threadsafe(token_queue.put_nowait, exc)

        self._work_queue.put(_work)
        return self._read_stream(token_queue)

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
