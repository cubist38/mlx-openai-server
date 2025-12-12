"""Request tracking middleware for correlation IDs and request logging."""

import asyncio
from collections.abc import Awaitable, Callable
import time
import uuid

from fastapi import Request, Response
from fastapi.responses import JSONResponse
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

    # Paths to log at DEBUG level for successful requests
    DEBUG_PATHS = frozenset(["/health", "/hub", "/favicon.ico", "/hub/status"])

    # Paths to skip logging entirely (logged elsewhere, e.g., model-specific logs)
    NO_LOG_PATHS = frozenset(["/v1/chat/completions"])

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """
        Process each request with correlation ID tracking.

        Args
        ----
            request: Incoming FastAPI request
            call_next: Next middleware/handler in chain

        Returns
        -------
            Response with X-Request-ID header added
        """
        # Get or generate request ID
        request_id = request.headers.get("X-Request-ID")
        if not request_id:
            request_id = str(uuid.uuid4())

        # Store in request state for handler access
        request.state.request_id = request_id

        # Log request start (skip for paths logged elsewhere)
        start_time = time.time()
        is_debug_path = request.url.path in self.DEBUG_PATHS
        is_no_log_path = request.url.path in self.NO_LOG_PATHS
        if not is_no_log_path:
            log_level = logger.debug if is_debug_path else logger.info
            log_level(
                f"Request started: {request.method} {request.url.path} [request_id={request_id}]",
            )

        try:
            # Process request
            try:
                response: Response | None = await call_next(request)
            except asyncio.CancelledError:
                # Client disconnected / request cancelled. Log and return a
                # clear status to aid diagnostics. In many cases the client
                # will have closed the socket and the response won't be
                # delivered, but returning a JSONResponse here ensures the
                # server-side logs and tests observe a consistent outcome.
                duration = time.time() - start_time
                if not is_no_log_path:
                    logger.warning(
                        f"Request cancelled by client: {request.method} {request.url.path} "
                        f"duration={duration:.3f}s [request_id={request_id}]",
                    )

                return JSONResponse(
                    content={"error": "request cancelled by client"},
                    status_code=499,
                    headers={"X-Request-ID": request_id},
                )
            except RuntimeError as e:
                # Starlette raises RuntimeError("No response returned.") when
                # the ASGI app did not produce a response (often due to an
                # EndOfStream / client disconnect inside the receive stream).
                # Log the cause and return a clear JSON error so the server
                # surface is easier to diagnose.
                duration = time.time() - start_time
                cause = getattr(e, "__cause__", None)
                try:
                    headers = dict(request.headers)
                except Exception:
                    headers = {}
                headers_preview = dict(list(headers.items())[:10])
                if not is_no_log_path:
                    logger.exception(
                        "No response returned for request: %s %s duration=%.3fs [request_id=%s] headers=%s cause=%s",
                        request.method,
                        request.url.path,
                        duration,
                        request_id,
                        headers_preview,
                        repr(cause),
                    )

                return JSONResponse(
                    content={
                        "error": "internal server error: no response returned",
                        "detail": str(e),
                        "cause": repr(cause),
                    },
                    status_code=500,
                    headers={"X-Request-ID": request_id},
                )

            # Defensive: some ASGI apps/middlewares may return None in
            # unusual error cases. Convert that into a 500 JSON response
            # and log context to make debugging easier than the generic
            # "No response returned." runtime error originating in
            # Starlette internals.
            if response is None:
                duration = time.time() - start_time
                try:
                    headers = dict(request.headers)
                except Exception:
                    headers = {}
                headers_preview = dict(list(headers.items())[:10])
                logger.error(
                    f"call_next returned None for request: {request.method} {request.url.path} "
                    f"duration={duration:.3f}s [request_id={request_id}] headers={headers_preview}"
                )

                return JSONResponse(
                    content={"error": "internal server error: no response returned"},
                    status_code=500,
                    headers={"X-Request-ID": request_id},
                )

            # Calculate duration
            duration = time.time() - start_time

            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id

            # Log request completion
            is_debug_completion = is_debug_path and response.status_code == 200
            if not is_no_log_path and not is_debug_completion:
                logger.info(
                    f"Request completed: {request.method} {request.url.path} "
                    f"status={response.status_code} duration={duration:.3f}s "
                    f"[request_id={request_id}]",
                )
            elif not is_no_log_path and is_debug_completion:
                logger.debug(
                    f"Request completed: {request.method} {request.url.path} "
                    f"status={response.status_code} duration={duration:.3f}s "
                    f"[request_id={request_id}]",
                )

        except Exception:
            # Log error with request ID
            duration = time.time() - start_time
            if not is_no_log_path:
                logger.exception(
                    f"Request failed: {request.method} {request.url.path} "
                    f"duration={duration:.3f}s [request_id={request_id}]",
                )
            raise
        return response
