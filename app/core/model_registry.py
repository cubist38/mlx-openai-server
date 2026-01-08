"""Model registry for managing multiple model handlers."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from loguru import logger

from app.schemas.model import ModelMetadata


class ModelRegistry:
    """
    Registry for managing model handlers.

    Maintains a thread-safe registry of loaded models and their handlers.
    In Phase 1, this wraps the existing single-model flow. Future phases
    will extend this to support multi-model loading and hot-swapping.

    Attributes
    ----------
        _handlers: Dict mapping model_id to handler instance
        _metadata: Dict mapping model_id to ModelMetadata
        _lock: Async lock for thread-safe operations
    """

    def __init__(self) -> None:
        """
        Initialize empty model registry.

        Returns
        -------
        None
        """
        self._handlers: dict[str, Any] = {}
        self._metadata: dict[str, ModelMetadata] = {}
        self._lock = asyncio.Lock()
        logger.info("Model registry initialized")

    async def register_model(
        self,
        model_id: str,
        handler: Any,
        model_type: str,
        context_length: int | None = None,
    ) -> None:
        """
        Register a model handler with metadata.

        Parameters
        ----------
        model_id : str
            Unique identifier for the model.
        handler : Any
            Handler instance (e.g., MLXLMHandler, MLXVLMHandler).
        model_type : str
            Type of model (e.g., 'lm', 'multimodal', 'embeddings').
        context_length : int | None, optional
            Maximum context length (if applicable).

        Raises
        ------
        ValueError
            If model_id is already registered.

        Returns
        -------
        None
        """
        async with self._lock:
            if model_id in self._handlers:
                raise ValueError(f"Model '{model_id}' is already registered")

            # Create metadata
            metadata = ModelMetadata(
                id=model_id,
                type=model_type,
                context_length=context_length,
                created_at=int(time.time()),
            )

            # Store handler and metadata
            self._handlers[model_id] = handler
            self._metadata[model_id] = metadata

            logger.info(
                f"Registered model: {model_id} (type={model_type}, context_length={context_length})"
            )

    def get_handler(self, model_id: str) -> Any:
        """
        Get handler for a specific model.

        Parameters
        ----------
        model_id : str
            Model identifier.

        Returns
        -------
        Any
            Handler instance.

        Raises
        ------
        KeyError
            If model_id not found.
        """
        if model_id not in self._handlers:
            raise KeyError(f"Model '{model_id}' not found in registry")
        return self._handlers[model_id]

    def list_models(self) -> list[dict[str, Any]]:
        """
        List all registered models with metadata.

        Returns
        -------
        list[dict[str, Any]]
            List of model metadata dicts in OpenAI API format.
        """
        return [
            {
                "id": metadata.id,
                "object": metadata.object,
                "created": metadata.created_at,
                "owned_by": metadata.owned_by,
            }
            for metadata in self._metadata.values()
        ]

    def get_metadata(self, model_id: str) -> ModelMetadata:
        """
        Get metadata for a specific model.

        Parameters
        ----------
        model_id : str
            Model identifier.

        Returns
        -------
        ModelMetadata
            Model metadata instance.

        Raises
        ------
        KeyError
            If model_id not found.
        """
        if model_id not in self._metadata:
            raise KeyError(f"Model '{model_id}' not found in registry")
        return self._metadata[model_id]

    async def unregister_model(self, model_id: str) -> None:
        """
        Unregister a model (stub for future implementation).

        In Phase 1, this just removes from registry. Future phases will
        implement proper cleanup (handler.cleanup(), memory release, etc.).

        Parameters
        ----------
        model_id : str
            Model identifier.

        Raises
        ------
        KeyError
            If model_id not found.

        Returns
        -------
        None
        """
        async with self._lock:
            if model_id not in self._handlers:
                raise KeyError(f"Model '{model_id}' not found in registry")

            # TODO Phase 2: Call handler.cleanup() before removing
            del self._handlers[model_id]
            del self._metadata[model_id]

            logger.info(f"Unregistered model: {model_id}")

    def has_model(self, model_id: str) -> bool:
        """
        Check if a model is registered.

        Parameters
        ----------
        model_id : str
            Model identifier.

        Returns
        -------
        bool
            True if model is registered, False otherwise.
        """
        return model_id in self._handlers

    def get_model_count(self) -> int:
        """
        Get count of registered models.

        Returns
        -------
        int
            Number of registered models.
        """
        return len(self._handlers)
