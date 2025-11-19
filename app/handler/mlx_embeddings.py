import gc
import time
import uuid
from http import HTTPStatus
from typing import Any, Dict, List

from fastapi import HTTPException
from loguru import logger

from ..core.queue import RequestQueue
from ..schemas.openai import EmbeddingRequest
from ..utils.errors import create_error_response
from ..models.mlx_embeddings import MLX_Embeddings

class MLXEmbeddingsHandler:
    """
    Handler class for making requests to the underlying MLX embeddings model service.
    Provides request queuing, metrics tracking, and robust error handling with memory management.
    """

    def __init__(self, model_path: str, max_concurrency: int = 1):
        """
        Initialize the handler with the specified model path.
        
        Args:
            model_path (str): Path to the embeddings model to load.
            max_concurrency (int): Maximum number of concurrent model inference tasks.
        """
        self.model_path = model_path
        self.model = MLX_Embeddings(model_path)
        self.model_created = int(time.time())  # Store creation time when model is loaded
        
        # Initialize request queue for embedding tasks
        self.request_queue = RequestQueue(max_concurrency=max_concurrency)

        logger.info(f"Initialized MLXEmbeddingsHandler with model path: {model_path}")
    
    async def get_models(self) -> List[Dict[str, Any]]:
        """
        Get list of available models with their metadata.
        """
        try:
            return [{
                "id": self.model_path,
                "object": "model",
                "created": self.model_created,
                "owned_by": "local" 
            }]
        except Exception as e:
            logger.error(f"Error getting models: {str(e)}")
            return []

    async def initialize(self, config: Dict[str, Any]):
        """
        Initialize the request queue with configuration.
        
        Args:
            config: Dictionary containing queue configuration.
        """
        await self.request_queue.start(self._process_request)

    async def generate_embeddings_response(self, request: EmbeddingRequest):
        """
        Generate embeddings for a given text input.
        
        Args:
            request: EmbeddingRequest object containing the text input.
        
        Returns:
            List[float]: Embeddings for the input text.
        """
        try:
            # Create a unique request ID
            request_id = f"embeddings-{uuid.uuid4()}"
            if isinstance(request.input, str):
                request.input = [request.input]
            request_data = {
                "type": "embeddings",
                "input": request.input,
                "max_length": getattr(request, 'max_length', 512)
            }

            # Submit to the request queue
            response = await self.request_queue.submit(request_id, request_data)
            
            return response

        except Exception as e:
            logger.error(f"Error in embeddings generation: {str(e)}")
            content = create_error_response(f"Failed to generate embeddings: {str(e)}", "server_error", HTTPStatus.INTERNAL_SERVER_ERROR)
            raise HTTPException(status_code=500, detail=content)

    async def _process_request(self, request_data: Dict[str, Any]) -> List[List[float]]:
        """
        Process an embeddings request. This is the worker function for the request queue.
        
        Args:
            request_data: Dictionary containing the request data.
            
        Returns:
            List[List[float]]: The embeddings for the input texts.
        """
        try:
            # Check if the request is for embeddings
            if request_data.get("type") == "embeddings":
                result = self.model(
                    texts=request_data["input"],
                    max_length=request_data.get("max_length", 512)
                )
                # Force garbage collection after embeddings
                gc.collect()
                return result
            
            raise ValueError(f"Unknown request type: {request_data.get('type')}")
            
        except Exception as e:
            logger.error(f"Error processing embeddings request: {str(e)}")
            # Clean up on error
            gc.collect()
            raise

    async def get_queue_stats(self) -> Dict[str, Any]:
        """
        Get statistics from the request queue and performance metrics.
        
        Returns:
            Dict with queue and performance statistics.
        """
        queue_stats = self.request_queue.get_queue_stats()
        
        return {
            "queue_stats": queue_stats,
        }
        
    async def cleanup(self):
        """
        Cleanup resources and stop the request queue before shutdown.
        
        This method ensures all pending requests are properly cancelled
        and resources are released.
        """
        try:
            logger.info("Cleaning up MLXEmbeddingsHandler resources")
            if hasattr(self, 'request_queue'):
                await self.request_queue.stop()
            if hasattr(self, 'model'):
                self.model.cleanup()
            logger.info("MLXEmbeddingsHandler cleanup completed successfully")
        except Exception as e:
            logger.error(f"Error during MLXEmbeddingsHandler cleanup: {str(e)}")
            raise

    def __del__(self):
        """
        Destructor to ensure cleanup on object deletion.
        Note: Async cleanup cannot be reliably performed in __del__.
        Please use 'await cleanup()' explicitly.
        """
        if hasattr(self, '_cleaned') and self._cleaned:
            return
        # Set flag to prevent multiple cleanup attempts
        self._cleaned = True 