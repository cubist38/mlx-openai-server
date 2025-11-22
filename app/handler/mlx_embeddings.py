"""MLX embeddings model handler for text embeddings."""

from __future__ import annotations

import asyncio
import gc
from http import HTTPStatus
import time
from typing import Any
import uuid

from fastapi import HTTPException
from loguru import logger

from ..core.queue import RequestQueue
from ..models.mlx_embeddings import MLX_Embeddings
from ..schemas.openai import EmbeddingRequest
from ..utils.errors import create_error_response


class MLXEmbeddingsHandler:
    """
    Handler class for making requests to the underlying MLX embeddings model service.

    Provides request queuing, metrics tracking, and robust error handling with memory management.
    """

    def __init__(self, model_path: str, max_concurrency: int = 1) -> None:
        """
        Initialize the handler with the specified model path.

        Args:
            model_path (str): Path to the embeddings model to load.
            max_concurrency (int): Maximum number of concurrent model inference tasks.
        """
        self.model_path = model_path
        self.model: MLX_Embeddings = MLX_Embeddings(model_path)
        self.model_created = int(time.time())  # Store creation time when model is loaded

        # Initialize request queue for embedding tasks
        self.request_queue = RequestQueue(max_concurrency=max_concurrency)

        logger.info(f"Initialized MLXEmbeddingsHandler with model path: {model_path}")

    async def get_models(self) -> list[dict[str, Any]]:
        """Get list of available models with their metadata."""
        try:
            return [
                {
                    "id": self.model_path,
                    "object": "model",
                    "created": self.model_created,
                    "owned_by": "local",
                }
            ]
        except Exception as e:
            logger.error(f"Error getting models. {type(e).__name__}: {e}")
            return []

    async def initialize(self, _config: dict[str, Any]) -> None:
        """
        Initialize the request queue with configuration.

        Args:
            _config: Dictionary containing queue configuration (Not currently used).
        """
        # TODO: Wire relevant keys from config through to RequestQueue (or rebuild the queue with the configured values)
        await self.request_queue.start(self._process_request)

    async def generate_embeddings_response(self, request: EmbeddingRequest) -> list[list[float]]:
        """
        Generate embeddings for a given text input.

        Args:
            request: EmbeddingRequest object containing the text input.

        Returns
        -------
            list[list[float]]: Embeddings for the input text.
        """
        try:
            # Create a unique request ID
            request_id = f"embeddings-{uuid.uuid4()}"
            if isinstance(request.input, str):
                request.input = [request.input]
            request_data = {
                "type": "embeddings",
                "input": request.input,
                "max_length": getattr(request, "max_length", 512),
            }

            # Submit to the request queue
            response: list[list[float]] = await self.request_queue.submit(request_id, request_data)

        except asyncio.QueueFull:
            logger.error("Too many requests. Service is at capacity.")
            content = create_error_response(
                "Too many requests. Service is at capacity.",
                "rate_limit_exceeded",
                HTTPStatus.TOO_MANY_REQUESTS,
            )
            raise HTTPException(status_code=HTTPStatus.TOO_MANY_REQUESTS, detail=content) from None
        except Exception as e:
            logger.error(f"Error in embeddings generation. {type(e).__name__}: {e}")
            content = create_error_response(
                f"Failed to generate embeddings: {e}",
                "server_error",
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail=content) from e
        else:
            return response

    async def _process_request(self, request_data: dict[str, Any]) -> list[list[float]]:
        """
        Process an embeddings request. This is the worker function for the request queue.

        Args:
            request_data: Dictionary containing the request data.

        Returns
        -------
            list[list[float]]: The embeddings for the input texts.
        """
        try:
            # Check if the request is for embeddings
            if request_data.get("type") == "embeddings":
                result = self.model(
                    texts=request_data["input"], max_length=request_data.get("max_length", 512)
                )
                # Force garbage collection after embeddings
                gc.collect()
                return result

            raise ValueError(f"Unknown request type: {request_data.get('type')}")

        except Exception as e:
            logger.error(f"Error processing embeddings request. {type(e).__name__}: {e}")
            # Clean up on error
            gc.collect()
            raise

    async def get_queue_stats(self) -> dict[str, Any]:
        """
        Get statistics from the request queue.

        Returns
        -------
            Dict with queue statistics.
        """
        return self.request_queue.get_queue_stats()

    async def cleanup(self) -> None:
        """
        Cleanup resources and stop the request queue before shutdown.

        This method ensures all pending requests are properly cancelled
        and resources are released.
        """
        try:
            logger.info("Cleaning up MLXEmbeddingsHandler resources")
            if hasattr(self, "request_queue"):
                await self.request_queue.stop()
            if hasattr(self, "model"):
                self.model.cleanup()
            logger.info("MLXEmbeddingsHandler cleanup completed successfully")
        except Exception as e:
            logger.error(f"Error during MLXEmbeddingsHandler cleanup. {type(e).__name__}: {e}")
