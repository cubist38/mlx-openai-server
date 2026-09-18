import gc
from http import HTTPStatus
import time
from typing import Any

from fastapi import HTTPException
from loguru import logger

from ..core import InferenceWorker
from ..models.mlx_embeddings import MLX_Embeddings
from ..schemas.openai import EmbeddingRequest
from ..utils.errors import create_error_response


class MLXEmbeddingsHandler:
    """
    Handler class for making requests to the underlying MLX embeddings model service.
    Provides request queuing, metrics tracking, and robust error handling with memory management.
    """

    handler_type: str = "embeddings"

    def __init__(self, model_path: str):
        """
        Initialize the handler with the specified model path.

        Args:
            model_path (str): Path to the embeddings model to load.
        """
        self.model_path = model_path
        self.model = MLX_Embeddings(model_path)
        self.model_created = int(time.time())  # Store creation time when model is loaded

        # Dedicated inference thread — keeps the event loop free during
        # blocking MLX model computation.
        self.inference_worker = InferenceWorker()

        logger.info(f"Initialized MLXEmbeddingsHandler with model path: {model_path}")

    async def get_models(self) -> list[dict[str, Any]]:
        """
        Get list of available models with their metadata.
        """
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
            logger.error(f"Error getting models: {e!s}")
            return []

    async def initialize(self, config: dict[str, Any]) -> None:
        """Initialize the handler and start the inference worker.

        Parameters
        ----------
        config : dict[str, Any]
            Dictionary with ``queue_size`` and ``timeout`` keys used
            to configure the inference worker's internal queue.
        """
        self.inference_worker = InferenceWorker(
            queue_size=config.get("queue_size", 100),
            timeout=config.get("timeout", 300),
        )
        self.inference_worker.start()
        logger.info("Initialized MLXEmbeddingsHandler and started inference worker")

    def _normalize_input(self, value: str | list[Any]) -> list[str]:
        """Reduce any accepted ``input`` shape to a list of strings.

        The OpenAI embeddings API accepts pre-tokenized input, so token ids are
        decoded back to text before the model tokenizes them again.

        Parameters
        ----------
        value : str | list[Any]
            A string, a list of strings, a token id array, or a list of token
            id arrays.

        Returns
        -------
        list[str]
            Texts ready for the embedding model.

        Raises
        ------
        ValueError
            If the input mixes shapes or holds unsupported item types.
        """
        if isinstance(value, str):
            return [value]
        if not value:
            return []

        if all(isinstance(item, str) for item in value):
            return list(value)

        tokenizer = getattr(self.model.tokenizer, "_tokenizer", self.model.tokenizer)
        if all(isinstance(item, int) for item in value):
            return [tokenizer.decode(list(value))]
        if all(
            isinstance(item, list) and all(isinstance(token, int) for token in item)
            for item in value
        ):
            return [tokenizer.decode(item) for item in value]

        raise ValueError(
            "Embedding input must be a string, a list of strings, a token id array, "
            "or a list of token id arrays."
        )

    async def generate_embeddings_response(self, request: EmbeddingRequest):
        """
        Generate embeddings for a given text input.

        Args:
            request: EmbeddingRequest object containing the text input.

        Returns:
            List[float]: Embeddings for the input text.
        """
        try:
            request.input = self._normalize_input(request.input)
            # Submit directly to the inference thread
            return await self.inference_worker.submit(
                self.model,
                texts=request.input,
                max_length=getattr(request, "max_length", 512),
            )

        except Exception as e:
            logger.error(f"Error in embeddings generation: {e!s}")
            content = create_error_response(
                f"Failed to generate embeddings: {e!s}",
                "server_error",
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            raise HTTPException(status_code=500, detail=content)

    async def get_queue_stats(self) -> dict[str, Any]:
        """Get statistics from the inference worker.

        Returns
        -------
        dict[str, Any]
            Dictionary with ``queue_stats`` sub-dictionary.
        """
        return {
            "queue_stats": self.inference_worker.get_stats(),
        }

    async def cleanup(self) -> None:
        """Cleanup resources and stop the inference worker before shutdown.

        This method ensures all pending requests are properly completed
        and resources are released.
        """
        try:
            logger.info("Cleaning up MLXEmbeddingsHandler resources")
            if hasattr(self, "inference_worker"):
                self.inference_worker.stop()
            if hasattr(self, "model"):
                self.model.cleanup()
            # Force garbage collection
            gc.collect()
            logger.info("MLXEmbeddingsHandler cleanup completed successfully")
        except Exception as e:
            logger.error(f"Error during MLXEmbeddingsHandler cleanup: {e!s}")
            raise

    def __del__(self):
        """
        Destructor to ensure cleanup on object deletion.
        Note: Async cleanup cannot be reliably performed in __del__.
        Please use 'await cleanup()' explicitly.
        """
        if hasattr(self, "_cleaned") and self._cleaned:
            return
        # Set flag to prevent multiple cleanup attempts
        self._cleaned = True
