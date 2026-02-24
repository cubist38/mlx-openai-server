"""Single background thread for blocking MLX inference; async handlers submit work via a queue."""

from __future__ import annotations

import asyncio
import queue
import threading
from collections.abc import AsyncGenerator, Generator
from threading import Thread
from typing import Any, Callable

from loguru import logger

_SENTINEL = object()


def _resolve_future(
    future: asyncio.Future[Any],
    result: Any = None,
    exc: BaseException | None = None,
) -> None:
    """Set result or exception on future from another thread; no-op if future already done."""
    if future.done():
        return
    try:
        if exc is not None:
            future.set_exception(exc)
        else:
            future.set_result(result)
    except asyncio.InvalidStateError:
        pass


class InferenceWorker:
    """Runs blocking inference on one thread; submit() and submit_stream() from async code."""

    def __init__(self, queue_size: int = 100, timeout: float = 300.0) -> None:
        self._queue_size = queue_size
        self._timeout = timeout
        self._work_queue: queue.Queue[Callable[[], None]] = queue.Queue(maxsize=queue_size)
        self._thread: Thread | None = None
        self._running = False
        self._lock = threading.Lock()
        self._active = False
        self._completed = 0
        self._failed = 0

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = Thread(target=self._run, daemon=True, name="inference-worker")
        self._thread.start()
        logger.info("Inference worker thread started")

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.info("Inference worker thread stopped")

    def _run(self) -> None:
        while self._running:
            try:
                work = self._work_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            with self._lock:
                self._active = True
            try:
                work()
            except Exception:
                logger.exception("Inference worker: exception escaped work closure")
            finally:
                with self._lock:
                    self._active = False

    def _record(self, success: bool) -> None:
        with self._lock:
            if success:
                self._completed += 1
            else:
                self._failed += 1

    async def submit(
        self, func: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        """Run func on the worker thread; await its result. Raises QueueFull, TimeoutError, or func's exception."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()

        def _work() -> None:
            try:
                out = func(*args, **kwargs)
                loop.call_soon_threadsafe(_resolve_future, future, out, None)
                self._record(True)
            except BaseException as e:
                loop.call_soon_threadsafe(_resolve_future, future, None, e)
                self._record(False)

        try:
            self._work_queue.put_nowait(_work)
        except queue.Full:
            raise asyncio.QueueFull("Inference queue is full")
        try:
            return await asyncio.wait_for(future, timeout=self._timeout)
        except asyncio.TimeoutError:
            raise TimeoutError(f"Inference timed out after {self._timeout}s")

    def submit_stream(
        self,
        func: Callable[..., Generator[Any, None, None]],
        *args: Any,
        **kwargs: Any,
    ) -> AsyncGenerator[Any, None]:
        """Run func on the worker thread; yield its generator items asynchronously. Raises QueueFull or func's exception."""
        loop = asyncio.get_running_loop()
        token_queue: asyncio.Queue[Any] = asyncio.Queue()

        def _work() -> None:
            try:
                gen = func(*args, **kwargs)
                for item in gen:
                    loop.call_soon_threadsafe(token_queue.put_nowait, item)
                loop.call_soon_threadsafe(token_queue.put_nowait, _SENTINEL)
                self._record(True)
            except BaseException as e:
                loop.call_soon_threadsafe(token_queue.put_nowait, e)
                self._record(False)

        try:
            self._work_queue.put_nowait(_work)
        except queue.Full:
            raise asyncio.QueueFull("Inference queue is full")
        return self._read_stream(token_queue)

    @staticmethod
    async def _read_stream(q: asyncio.Queue[Any]) -> AsyncGenerator[Any, None]:
        while True:
            item = await q.get()
            if item is _SENTINEL:
                break
            if isinstance(item, BaseException):
                raise item
            yield item

    def get_stats(self) -> dict[str, Any]:
        """Current queue and worker stats."""
        with self._lock:
            return {
                "running": self._running,
                "queue_size": self._work_queue.qsize(),
                "max_queue_size": self._queue_size,
                "active_requests": 1 if self._active else 0,
                "completed_requests": self._completed,
                "failed_requests": self._failed,
            }
