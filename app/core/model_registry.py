"""Model registry for managing multiple model handlers."""

from __future__ import annotations

import asyncio
import time
from typing import Any, cast

from loguru import logger

from ..schemas.model import ModelMetadata

_UNSET = object()


class ModelRegistry:
    """Asyncio event-loop-safe registry for model handlers and metadata."""

    def __init__(self) -> None:
        self._handlers: dict[str, Any | None] = {}
        self._metadata: dict[str, ModelMetadata] = {}
        self._extra: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        logger.info("Model registry initialized")

    async def register_model(
        self,
        model_id: str,
        handler: Any | None,
        model_type: str,
        context_length: int | None = None,
        metadata_extras: dict[str, Any] | None = None,
    ) -> None:
        """Register a model handler with metadata."""

        async with self._lock:
            if model_id in self._handlers:
                raise ValueError(f"Model '{model_id}' is already registered")

            metadata = ModelMetadata(
                id=model_id,
                type=model_type,
                context_length=context_length,
                created_at=int(time.time()),
            )
            base_metadata = {
                "model_type": model_type,
                "context_length": context_length,
                "status": "initialized" if handler else "unloaded",
            }
            if metadata_extras:
                base_metadata.update(metadata_extras)

            self._handlers[model_id] = handler
            self._metadata[model_id] = metadata
            self._extra[model_id] = base_metadata

            logger.info(
                f"Registered model: {model_id} (type={model_type}, context_length={context_length})"
            )

    async def update_model_state(
        self,
        model_id: str,
        *,
        handler: Any | None | object = _UNSET,
        status: str | None = None,
        metadata_updates: dict[str, Any] | None = None,
    ) -> None:
        """Update handler attachment and metadata for a registered model."""

        async with self._lock:
            if model_id not in self._handlers:
                raise KeyError(f"Model '{model_id}' not found in registry")

            if handler is not _UNSET:
                self._handlers[model_id] = cast("Any | None", handler)
                if handler is not None:
                    self._metadata[model_id].created_at = int(time.time())

            entry = self._extra.setdefault(model_id, {})
            if metadata_updates:
                entry.update(metadata_updates)
            if status is not None:
                entry["status"] = status

    async def unregister_model(self, model_id: str) -> None:
        """Remove a model from the registry."""

        async with self._lock:
            if model_id not in self._handlers:
                raise KeyError(f"Model '{model_id}' not found in registry")

            del self._handlers[model_id]
            del self._metadata[model_id]
            self._extra.pop(model_id, None)
            logger.info(f"Unregistered model: {model_id}")

    def get_handler(self, model_id: str) -> Any | None:
        """Return the handler bound to ``model_id`` (may be ``None``)."""

        if model_id not in self._handlers:
            raise KeyError(f"Model '{model_id}' not found in registry")
        return self._handlers[model_id]

    def list_models(self) -> list[dict[str, Any]]:
        """Return OpenAI-compatible metadata for registered models."""

        output: list[dict[str, Any]] = []
        for model_id, metadata in self._metadata.items():
            entry = {
                "id": metadata.id,
                "object": metadata.object,
                "created": metadata.created_at,
                "owned_by": metadata.owned_by,
            }
            extra = self._extra.get(model_id)
            if extra:
                entry["metadata"] = extra
            output.append(entry)
        return output

    def get_metadata(self, model_id: str) -> ModelMetadata:
        """Return the stored metadata for ``model_id``."""

        if model_id not in self._metadata:
            raise KeyError(f"Model '{model_id}' not found in registry")
        return self._metadata[model_id]

    def has_model(self, model_id: str) -> bool:
        """Return ``True`` when the model is registered."""

        return model_id in self._handlers

    def get_model_count(self) -> int:
        """Return how many models are registered."""

        return len(self._handlers)
