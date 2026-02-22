"""Model registry for managing multiple model handlers.

The ``ModelRegistry`` is the central lookup table that maps model IDs
(strings used in the OpenAI-style ``model`` request field) to their
corresponding handler instances. It is thread-safe via an
:class:`asyncio.Lock` and is intended to be stored on
``app.state.registry`` in multi-handler mode.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from loguru import logger

from ..schemas.model import ModelMetadata


class ModelRegistry:
    """Registry for managing model handlers.

    Maintains a thread-safe registry of loaded models and their handlers.
    Handlers are stored in a dictionary keyed by ``model_id`` so that
    incoming requests can be dispatched with a simple lookup.

    Attributes
    ----------
    _handlers : dict[str, Any]
        Mapping of model_id to handler instance.
    _metadata : dict[str, ModelMetadata]
        Mapping of model_id to ``ModelMetadata``.
    _lock : asyncio.Lock
        Async lock for thread-safe mutations.
    """

    def __init__(self) -> None:
        """Initialize empty model registry."""
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
        """Register a model handler with metadata.

        Parameters
        ----------
        model_id : str
            Unique identifier for the model (used in API ``model`` field).
        handler : Any
            Handler instance (``MLXLMHandler``, ``MLXVLMHandler``, etc.).
        model_type : str
            Type of model (``lm``, ``multimodal``, ``embeddings``, etc.).
        context_length : int | None, optional
            Maximum context length (if applicable).

        Raises
        ------
        ValueError
            If ``model_id`` is already registered.
        """
        async with self._lock:
            if model_id in self._handlers:
                raise ValueError(f"Model '{model_id}' is already registered")

            metadata = ModelMetadata(
                id=model_id,
                type=model_type,
                context_length=context_length,
                created_at=int(time.time()),
            )

            self._handlers[model_id] = handler
            self._metadata[model_id] = metadata

            logger.info(
                f"Registered model: {model_id} (type={model_type}, "
                f"context_length={context_length})"
            )

    def get_handler(self, model_id: str) -> Any:
        """Get handler for a specific model.

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
            If ``model_id`` is not found in the registry.
        """
        if model_id not in self._handlers:
            available = ", ".join(sorted(self._handlers.keys())) or "(none)"
            raise KeyError(
                f"Model '{model_id}' not found in registry. "
                f"Available models: {available}"
            )
        return self._handlers[model_id]

    def list_models(self) -> list[dict[str, Any]]:
        """List all registered models with metadata.

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
        """Get metadata for a specific model.

        Parameters
        ----------
        model_id : str
            Model identifier.

        Returns
        -------
        ModelMetadata
            Metadata instance.

        Raises
        ------
        KeyError
            If ``model_id`` is not found.
        """
        if model_id not in self._metadata:
            raise KeyError(f"Model '{model_id}' not found in registry")
        return self._metadata[model_id]

    async def unregister_model(self, model_id: str) -> None:
        """Unregister a model and clean up its handler.

        Parameters
        ----------
        model_id : str
            Model identifier.

        Raises
        ------
        KeyError
            If ``model_id`` is not found.
        """
        async with self._lock:
            if model_id not in self._handlers:
                raise KeyError(f"Model '{model_id}' not found in registry")

            handler = self._handlers[model_id]
            if hasattr(handler, "cleanup"):
                try:
                    await handler.cleanup()
                    logger.info(f"Cleaned up handler for model: {model_id}")
                except Exception as e:
                    logger.error(f"Error cleaning up handler for '{model_id}': {e}")

            del self._handlers[model_id]
            del self._metadata[model_id]
            logger.info(f"Unregistered model: {model_id}")

    async def cleanup_all(self) -> None:
        """Clean up all registered handlers concurrently.

        Spawns cleanup tasks for every handler in parallel using
        ``asyncio.gather`` so that multiple subprocess shutdowns do
        not serialise their timeout windows.  Called during server
        shutdown.
        """
        async with self._lock:
            cleanup_tasks = [
                self._cleanup_single_handler(model_id, handler)
                for model_id, handler in self._handlers.items()
                if hasattr(handler, "cleanup")
            ]
            if cleanup_tasks:
                await asyncio.gather(*cleanup_tasks)

            self._handlers.clear()
            self._metadata.clear()
            logger.info("All models unregistered and cleaned up")

    @staticmethod
    async def _cleanup_single_handler(
        model_id: str, handler: Any
    ) -> None:
        """Clean up a single handler, logging success or failure.

        Parameters
        ----------
        model_id : str
            Model identifier (for logging).
        handler : Any
            Handler instance whose ``cleanup`` method will be awaited.
        """
        try:
            await handler.cleanup()
            logger.info(f"Cleaned up handler for model: {model_id}")
        except Exception as e:
            logger.error(f"Error cleaning up handler for '{model_id}': {e}")

    def has_model(self, model_id: str) -> bool:
        """Check if a model is registered.

        Parameters
        ----------
        model_id : str
            Model identifier.

        Returns
        -------
        bool
            ``True`` if model is registered, ``False`` otherwise.
        """
        return model_id in self._handlers

    def get_model_count(self) -> int:
        """Get count of registered models.

        Returns
        -------
        int
            Number of registered models.
        """
        return len(self._handlers)
