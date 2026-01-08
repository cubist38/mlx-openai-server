"""Asynchronous request queue with concurrency control."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
import contextlib
import gc
import time
from typing import Any, Generic, TypeVar

from loguru import logger

T = TypeVar("T")


class RequestItem(Generic[T]):
    """Represents a single request in the queue."""

    def __init__(self, request_id: str, data: T) -> None:
        self.request_id = request_id
        self.data = data
        self.created_at = time.time()
        self.future: asyncio.Future[T] = asyncio.Future()

    def set_result(self, result: T) -> None:
        """
        Set the result for this request.

        Parameters
        ----------
        result : T
            The result value to set on the request's future.

        Returns
        -------
        None
        """
        if not self.future.done():
            self.future.set_result(result)

    def set_exception(self, exc: Exception) -> None:
        """
        Set an exception for this request.

        Parameters
        ----------
        exc : Exception
            Exception to set on the request's future.

        Returns
        -------
        None
        """
        if not self.future.done():
            self.future.set_exception(exc)

    async def get_result(self) -> T:
        """
        Wait for and return the result of this request.

        Returns
        -------
        T
            The result of the request once available.
        """
        return await self.future


class RequestQueue:
    """A simple asynchronous request queue with configurable concurrency."""

    def __init__(
        self, max_concurrency: int = 2, timeout: float = 300.0, queue_size: int = 100
    ) -> None:
        """
        Initialize the request queue.

        Parameters
        ----------
        max_concurrency : int, optional
            Maximum number of concurrent requests to process. Default is 2.
        timeout : float, optional
            Timeout in seconds for request processing. Default is 300.0.
        queue_size : int, optional
            Maximum queue size. Default is 100.

        Returns
        -------
        None
        """
        self.max_concurrency = max_concurrency
        self.timeout = timeout
        self.queue_size = queue_size
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.queue: asyncio.Queue[RequestItem[Any]] = asyncio.Queue(maxsize=queue_size)
        self.active_requests: dict[str, RequestItem[Any]] = {}
        self._worker_task: asyncio.Task[None] | None = None
        self._running = False
        self._tasks: set[asyncio.Task[None]] = set()

    async def start(self, processor: Callable[[Any], Awaitable[Any]]) -> None:
        """
        Start the queue worker.

        Parameters
        ----------
        processor : Callable[[Any], Awaitable[Any]]
            Async function that processes queue items.

        Returns
        -------
        None
        """
        if self._running:
            return

        self._running = True
        self._worker_task = asyncio.create_task(self._worker_loop(processor))
        logger.info(f"Started request queue with max concurrency: {self.max_concurrency}")

    async def stop(self) -> None:
        """
        Stop the queue worker and cancel active tasks.

        Returns
        -------
        None
        """
        if not self._running:
            return

        self._running = False

        # Cancel the worker task
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker_task

        # Cancel all in-flight tasks
        tasks_snapshot = list(self._tasks)
        for task in tasks_snapshot:
            if not task.done():
                task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.gather(*tasks_snapshot, return_exceptions=True)
        self._tasks.clear()

        # Cancel all pending requests
        pending_requests = list(self.active_requests.values())
        for request in pending_requests:
            if not request.future.done():
                request.future.cancel()
            # Clean up request data
            try:
                if hasattr(request, "data"):
                    del request.data
            except Exception as e:
                logger.opt(exception=e).debug("Failed to remove request.data")

        self.active_requests.clear()

        # Clear the queue
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        # Force garbage collection after cleanup
        gc.collect()
        logger.info("Stopped request queue")

    async def _worker_loop(self, processor: Callable[[Any], Awaitable[Any]]) -> None:
        """
        Process queue items in the main worker loop.

        Parameters
        ----------
        processor : Callable[[Any], Awaitable[Any]]
            Async function that processes queue items.

        Returns
        -------
        None
        """
        while self._running:
            try:
                # Get the next item from the queue
                request = await self.queue.get()

                # Process the request with concurrency control
                task = asyncio.create_task(self._process_request(request, processor))
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in worker loop. {type(e).__name__}: {e}")

    async def _process_request(
        self, request: RequestItem[Any], processor: Callable[[Any], Awaitable[Any]]
    ) -> None:
        """
        Process a single request with timeout and error handling.

        Parameters
        ----------
        request : RequestItem[Any]
            The request to process.
        processor : Callable[[Any], Awaitable[Any]]
            Async function that processes the request.

        Returns
        -------
        None

        Raises
        ------
        TimeoutError
            If processing exceeds the configured timeout.
        Exception
            For other processing errors; these are set on the request and logged.
        """
        # Use semaphore to limit concurrency
        async with self.semaphore:
            try:
                # Process with timeout
                processing_start = time.time()
                result = await asyncio.wait_for(processor(request.data), timeout=self.timeout)
                processing_time = time.time() - processing_start

                # Set the result
                request.set_result(result)
                logger.info(f"Request {request.request_id} processed in {processing_time:.2f}s")

            except TimeoutError:
                request.set_exception(
                    TimeoutError(f"Request processing timed out after {self.timeout}s")
                )
                logger.warning(f"Request {request.request_id} timed out after {self.timeout}s")
            except asyncio.CancelledError as e:
                # Propagate cancellation but ensure the future is not left hanging
                if not request.future.done():
                    request.future.set_exception(e)
                logger.info(f"Request {request.request_id} was cancelled")
                raise
            except Exception as e:
                request.set_exception(e)
                logger.error(
                    f"Error processing request {request.request_id}. {type(e).__name__}: {e}"
                )

            finally:
                # Always remove from active requests, even if an error occurred
                removed_request = self.active_requests.pop(request.request_id, None)
                if removed_request:
                    # Clean up the request object
                    try:
                        if hasattr(removed_request, "data"):
                            del removed_request.data
                    except Exception as e:
                        logger.opt(exception=e).debug("Failed to remove request.data")
                # Force garbage collection periodically to prevent memory buildup
                if len(self.active_requests) % 10 == 0:  # Every 10 requests
                    gc.collect()

    async def enqueue(self, request_id: str, data: Any) -> RequestItem[Any]:
        """
        Add a request to the queue.

        Parameters
        ----------
        request_id : str
            Unique ID for the request.
        data : Any
            The request data to process.

        Returns
        -------
        RequestItem[Any]
            The queued request item.

        Raises
        ------
        asyncio.QueueFull
            If the queue is full or cannot accept new requests.
        """
        if not self._running:
            raise RuntimeError("Queue is not running")

        # Create request item
        request = RequestItem(request_id, data)

        # Add to active requests and queue
        self.active_requests[request_id] = request

        try:
            # This will raise QueueFull if the queue is full
            await asyncio.wait_for(
                self.queue.put(request),
                timeout=1.0,  # Short timeout for queue put
            )
        except TimeoutError:
            self.active_requests.pop(request_id, None)
            raise asyncio.QueueFull(
                "Request queue is full and timed out waiting for space"
            ) from None
        else:
            queue_time = time.time() - request.created_at
            logger.info(f"Request {request_id} queued (wait: {queue_time:.2f}s)")
            return request

    async def submit(self, request_id: str, data: Any) -> Any:
        """
        Submit a request and wait for its result.

        Parameters
        ----------
        request_id : str
            Unique ID for the request.
        data : Any
            The request data to process.

        Returns
        -------
        Any
            The result of processing the request.

        Raises
        ------
        Exception
            Various exceptions that may occur during processing are propagated.
        """
        request = await self.enqueue(request_id, data)
        return await request.get_result()

    def get_queue_stats(self) -> dict[str, Any]:
        """
        Get queue statistics.

        Returns
        -------
        dict[str, Any]
            Dictionary with queue statistics keys: 'running', 'queue_size',
            'max_queue_size', 'active_requests', and 'max_concurrency'.
        """
        return {
            "running": self._running,
            "queue_size": self.queue.qsize(),
            "max_queue_size": self.queue_size,
            "active_requests": len(self.active_requests),
            "max_concurrency": self.max_concurrency,
        }

    # Alias for the async stop method to maintain consistency in cleanup interfaces
    async def stop_async(self) -> None:
        """
        Alias for stop - stops the queue worker asynchronously.

        Returns
        -------
        None
        """
        await self.stop()
