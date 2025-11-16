"""Request tracking middleware for correlation IDs and request logging."""

import time
import uuid
from typing import Callable

from fastapi import Request, Response
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware


class RequestTrackingMiddleware(BaseHTTPMiddleware):
    """
    Middleware that adds request correlation IDs and logging.

    Features:
    - Generates unique request ID (UUID4) for each request
    - Accepts X-Request-ID header from client if provided
    - Stores request ID in request.state for handler access
    - Adds X-Request-ID to response headers
    - Logs request start/end with timing information
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process each request with correlation ID tracking.

        Args:
            request: Incoming FastAPI request
            call_next: Next middleware/handler in chain

        Returns:
            Response with X-Request-ID header added
        """
        # Get or generate request ID
        request_id = request.headers.get("X-Request-ID")
        if not request_id:
            request_id = str(uuid.uuid4())

        # Store in request state for handler access
        request.state.request_id = request_id

        # Log request start
        start_time = time.time()
        logger.info(
            f"Request started: {request.method} {request.url.path} "
            f"[request_id={request_id}]"
        )

        try:
            # Process request
            response = await call_next(request)

            # Calculate duration
            duration = time.time() - start_time

            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id

            # Log request completion
            logger.info(
                f"Request completed: {request.method} {request.url.path} "
                f"status={response.status_code} duration={duration:.3f}s "
                f"[request_id={request_id}]"
            )

            return response

        except Exception as e:
            # Log error with request ID
            duration = time.time() - start_time
            logger.error(
                f"Request failed: {request.method} {request.url.path} "
                f"error={str(e)} duration={duration:.3f}s "
                f"[request_id={request_id}]",
                exc_info=True
            )
            raise
