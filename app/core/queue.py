import asyncio
import time
from typing import Any, Dict, Optional, Callable, Awaitable, TypeVar, Generic
import gc
from loguru import logger

T = TypeVar('T')

class RequestItem(Generic[T]):
    """
    Represents a single request in the queue.
    """
    def __init__(self, request_id: str, data: Any):
        self.request_id = request_id
        self.data = data
        self.created_at = time.time()
        self.future = asyncio.Future()
        
    def set_result(self, result: T) -> None:
        """Set the result for this request."""
        if not self.future.done():
            self.future.set_result(result)
            
    def set_exception(self, exc: Exception) -> None:
        """Set an exception for this request."""
        if not self.future.done():
            self.future.set_exception(exc)
            
    async def get_result(self) -> T:
        """Wait for and return the result of this request."""
        return await self.future

class RequestQueue:
    """
    A simple asynchronous request queue with configurable concurrency.
    """
    def __init__(self, max_concurrency: int = 2, timeout: float = 300.0, queue_size: int = 100):
        """
        Initialize the request queue.
        
        Args:
            max_concurrency (int): Maximum number of concurrent requests to process.
            timeout (float): Timeout in seconds for request processing.
            queue_size (int): Maximum queue size.
        """
        self.max_concurrency = max_concurrency
        self.timeout = timeout
        self.queue_size = queue_size
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.queue = asyncio.Queue(maxsize=queue_size)
        self.active_requests: Dict[str, RequestItem] = {}
        self._worker_task = None
        self._running = False
        
    async def start(self, processor: Callable[[Any], Awaitable[Any]]):
        """
        Start the queue worker.
        
        Args:
            processor: Async function that processes queue items.
        """
        if self._running:
            return
            
        self._running = True
        self._worker_task = asyncio.create_task(self._worker_loop(processor))
        logger.info(f"Started request queue with max concurrency: {self.max_concurrency}")
        
    async def stop(self):
        """Stop the queue worker."""
        if not self._running:
            return
            
        self._running = False
        
        # Cancel the worker task
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            
        # Cancel all pending requests
        pending_requests = list(self.active_requests.values())
        for request in pending_requests:
            if not request.future.done():
                request.future.cancel()
            # Clean up request data
            try:
                if hasattr(request, 'data'):
                    del request.data
            except:
                pass
                
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
        
    async def _worker_loop(self, processor: Callable[[Any], Awaitable[Any]]):
        """
        Main worker loop that processes queue items.
        
        Args:
            processor: Async function that processes queue items.
        """
        while self._running:
            try:
                # Get the next item from the queue
                request = await self.queue.get()
                
                # Process the request with concurrency control
                asyncio.create_task(self._process_request(request, processor))
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in worker loop: {str(e)}")
                
    async def _process_request(self, request: RequestItem, processor: Callable[[Any], Awaitable[Any]]):
        """
        Process a single request with timeout and error handling.
        
        Args:
            request: The request to process.
            processor: Async function that processes the request.
        """
        # Use semaphore to limit concurrency
        async with self.semaphore:
            try:
                # Process with timeout
                processing_start = time.time()
                result = await asyncio.wait_for(
                    processor(request.data),
                    timeout=self.timeout
                )
                processing_time = time.time() - processing_start
                
                # Set the result
                request.set_result(result)
                logger.info(f"Request {request.request_id} processed in {processing_time:.2f}s")
                
            except asyncio.TimeoutError:
                request.set_exception(TimeoutError(f"Request processing timed out after {self.timeout}s"))
                logger.warning(f"Request {request.request_id} timed out after {self.timeout}s")
                
            except Exception as e:
                request.set_exception(e)
                logger.error(f"Error processing request {request.request_id}: {str(e)}")
                
            finally:
                # Always remove from active requests, even if an error occurred
                removed_request = self.active_requests.pop(request.request_id, None)
                if removed_request:
                    # Clean up the request object
                    try:
                        if hasattr(removed_request, 'data'):
                            del removed_request.data
                    except:
                        pass
                # Force garbage collection periodically to prevent memory buildup
                if len(self.active_requests) % 10 == 0:  # Every 10 requests
                    gc.collect()
                
    async def enqueue(self, request_id: str, data: Any) -> RequestItem:
        """
        Add a request to the queue.
        
        Args:
            request_id: Unique ID for the request.
            data: The request data to process.
            
        Returns:
            RequestItem: The queued request item.
            
        Raises:
            asyncio.QueueFull: If the queue is full.
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
                timeout=1.0  # Short timeout for queue put
            )
            queue_time = time.time() - request.created_at
            logger.info(f"Request {request_id} queued (wait: {queue_time:.2f}s)")
            return request
            
        except asyncio.TimeoutError:
            self.active_requests.pop(request_id, None)
            raise asyncio.QueueFull("Request queue is full and timed out waiting for space")
            
    async def submit(self, request_id: str, data: Any) -> Any:
        """
        Submit a request and wait for its result.
        
        Args:
            request_id: Unique ID for the request.
            data: The request data to process.
            
        Returns:
            The result of processing the request.
            
        Raises:
            Various exceptions that may occur during processing.
        """
        request = await self.enqueue(request_id, data)
        return await request.get_result()
        
    def get_queue_stats(self) -> Dict[str, Any]:
        """
        Get queue statistics.
        
        Returns:
            Dict with queue statistics.
        """
        return {
            "running": self._running,
            "queue_size": self.queue.qsize(),
            "max_queue_size": self.queue_size,
            "active_requests": len(self.active_requests),
            "max_concurrency": self.max_concurrency
        }

    # Alias for the async stop method to maintain consistency in cleanup interfaces
    async def stop_async(self):
        """Alias for stop - stops the queue worker asynchronously."""
        await self.stop() 