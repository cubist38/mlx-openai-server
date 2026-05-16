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

        # On-demand (dynamic swapping) state
        self._on_demand_configs: dict[str, dict[str, Any]] = {}
        self._on_demand_loaded: set[str] = set()
        self._on_demand_load_lock = asyncio.Lock()
        self._on_demand_ref_count: dict[str, int] = {}
        self._on_demand_idle_tasks: dict[str, asyncio.Task] = {}
        self._on_demand_idle_timeouts: dict[str, int] = {}

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
                f"Registered model: {model_id} (type={model_type}, context_length={context_length})"
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
                f"Model '{model_id}' not found in registry. Available models: {available}"
            )
        return self._handlers[model_id]

    def list_model_ids(self) -> list[str]:
        """Return sorted list of all known model IDs (loaded + on-demand)."""
        return sorted(set(self._handlers.keys()) | set(self._on_demand_configs.keys()))

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
        # Cancel any pending on-demand idle unload tasks
        for task in self._on_demand_idle_tasks.values():
            task.cancel()
        self._on_demand_idle_tasks.clear()

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
            self._on_demand_configs.clear()
            self._on_demand_loaded.clear()
            self._on_demand_ref_count.clear()
            self._on_demand_idle_timeouts.clear()
            logger.info("All models unregistered and cleaned up")

    @staticmethod
    async def _cleanup_single_handler(model_id: str, handler: Any) -> None:
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
        """Check if a model is registered (loaded or on-demand).

        Parameters
        ----------
        model_id : str
            Model identifier.

        Returns
        -------
        bool
            ``True`` if model is registered, ``False`` otherwise.
        """
        return model_id in self._handlers or model_id in self._on_demand_configs

    def get_model_count(self) -> int:
        """Get count of registered models (loaded + on-demand).

        Returns
        -------
        int
            Number of registered models.
        """
        return len(self._handlers) + len(self._on_demand_configs.keys() - self._handlers.keys())

    # ------------------------------------------------------------------
    # On-demand (dynamic swapping) support
    # ------------------------------------------------------------------

    async def register_on_demand_model(
        self,
        model_id: str,
        model_cfg_dict: dict[str, Any],
        model_type: str,
        model_path: str,
        context_length: int | None,
        queue_config: dict[str, Any],
        idle_timeout: int = 60,
    ) -> None:
        """Register a model for on-demand loading without spawning it.

        The model will appear in ``list_models()`` but will only be
        loaded into memory when a request arrives for it.

        Parameters
        ----------
        model_id : str
            Unique model identifier.
        model_cfg_dict : dict[str, Any]
            Serialized ``ModelEntryConfig`` fields for subprocess spawning.
        model_type : str
            Model type string.
        model_path : str
            Path / HuggingFace repo for the model.
        context_length : int | None
            Max context length (for metadata).
        queue_config : dict[str, Any]
            Queue/concurrency config forwarded to the handler on spawn.
        idle_timeout : int
            Seconds to wait before unloading an idle on-demand model.
        """
        async with self._lock:
            if model_id in self._handlers or model_id in self._on_demand_configs:
                raise ValueError(f"Model '{model_id}' is already registered")

            self._on_demand_configs[model_id] = {
                "model_cfg_dict": model_cfg_dict,
                "model_type": model_type,
                "model_path": model_path,
                "context_length": context_length,
                "queue_config": queue_config,
            }
            self._on_demand_idle_timeouts[model_id] = idle_timeout

            # Add metadata so the model appears in /v1/models
            self._metadata[model_id] = ModelMetadata(
                id=model_id,
                type=model_type,
                context_length=context_length,
                created_at=int(time.time()),
            )

            logger.info(
                f"Registered on-demand model: {model_id} "
                f"(type={model_type}, idle_timeout={idle_timeout}s)"
            )

    def is_on_demand(self, model_id: str) -> bool:
        """Check if a model is registered as on-demand."""
        return model_id in self._on_demand_configs

    async def ensure_on_demand_loaded(self, model_id: str) -> Any:
        """Load an on-demand model if not already loaded.

        If a different on-demand model is currently loaded and idle,
        it will be unloaded first.  Only one on-demand model is kept
        in memory at a time.

        Parameters
        ----------
        model_id : str
            On-demand model identifier.

        Returns
        -------
        Any
            The handler (``HandlerProcessProxy``) for the model.

        Raises
        ------
        KeyError
            If ``model_id`` is not a registered on-demand model.
        RuntimeError
            If the handler subprocess fails to start.
        """
        if model_id not in self._on_demand_configs:
            raise KeyError(f"Model '{model_id}' is not registered as on-demand")

        async with self._on_demand_load_lock:
            # Already loaded — just bump ref count
            if model_id in self._handlers and model_id in self._on_demand_loaded:
                idle_task = self._on_demand_idle_tasks.pop(model_id, None)
                if idle_task is not None:
                    idle_task.cancel()
                self._on_demand_ref_count[model_id] = self._on_demand_ref_count.get(model_id, 0) + 1
                logger.debug(
                    f"On-demand model '{model_id}' already loaded, "
                    f"ref_count={self._on_demand_ref_count[model_id]}"
                )
                return self._handlers[model_id]

            # Unload idle on-demand models to free memory.
            # Only unload models with no active requests (ref_count == 0).
            for loaded_id in list(self._on_demand_loaded):
                if loaded_id == model_id or loaded_id not in self._handlers:
                    continue
                if self._on_demand_ref_count.get(loaded_id, 0) > 0:
                    logger.info(
                        f"On-demand model '{loaded_id}' still has active requests "
                        f"(ref_count={self._on_demand_ref_count[loaded_id]}), "
                        f"keeping loaded alongside '{model_id}'"
                    )
                    continue
                idle_task = self._on_demand_idle_tasks.pop(loaded_id, None)
                if idle_task is not None:
                    idle_task.cancel()
                old_handler = self._handlers.pop(loaded_id)
                logger.info(
                    f"Unloading on-demand model '{loaded_id}' to make room for '{model_id}'"
                )
                if hasattr(old_handler, "cleanup"):
                    await old_handler.cleanup()
                self._on_demand_ref_count.pop(loaded_id, None)
                self._on_demand_loaded.discard(loaded_id)

            # Spawn the new on-demand handler
            cfg = self._on_demand_configs[model_id]
            logger.info(f"Loading on-demand model '{model_id}' (path={cfg['model_path']})")

            from .handler_process import HandlerProcessProxy

            proxy = HandlerProcessProxy(
                model_cfg_dict=cfg["model_cfg_dict"],
                model_type=cfg["model_type"],
                model_path=cfg["model_path"],
                served_model_name=model_id,
            )
            await proxy.start(cfg["queue_config"])

            self._handlers[model_id] = proxy
            self._on_demand_loaded.add(model_id)
            self._on_demand_ref_count[model_id] = 1

            logger.info(f"On-demand model '{model_id}' loaded successfully")
            return proxy

    async def release_on_demand(self, model_id: str) -> None:
        """Release a reference to an on-demand model after a request completes.

        When the reference count reaches zero, an idle timeout task is
        scheduled to unload the model.

        Parameters
        ----------
        model_id : str
            On-demand model identifier.
        """
        if model_id not in self._on_demand_configs:
            return

        self._on_demand_ref_count[model_id] = max(0, self._on_demand_ref_count.get(model_id, 1) - 1)
        logger.debug(
            f"Released on-demand model '{model_id}', "
            f"ref_count={self._on_demand_ref_count[model_id]}"
        )

        if self._on_demand_ref_count[model_id] == 0:
            timeout = self._on_demand_idle_timeouts.get(model_id, 60)
            old_task = self._on_demand_idle_tasks.pop(model_id, None)
            if old_task is not None:
                old_task.cancel()
            self._on_demand_idle_tasks[model_id] = asyncio.create_task(
                self._idle_unload(model_id, timeout)
            )

    async def unload_on_demand_model(self, model_id: str) -> tuple[bool, str | None]:
        """Unload a registered on-demand model immediately.

        Parameters
        ----------
        model_id : str
            Registered on-demand model identifier.

        Returns
        -------
        tuple[bool, str | None]
            ``(True, None)`` when unload succeeds.

            ``(False, reason)`` when the model is already unloaded or
            cannot be unloaded because it is in use.

        Raises
        ------
        KeyError
            If ``model_id`` is not registered as on-demand.
        """
        if model_id not in self._on_demand_configs:
            raise KeyError(f"Model '{model_id}' is not registered as on-demand")

        # Quick path: if not loaded, avoid locking and return the caller-defined
        # reason used by the API endpoint response.
        if model_id not in self._handlers:
            self._on_demand_idle_tasks.pop(model_id, None)
            return False, "not currently loaded"

        async with self._on_demand_load_lock:
            if model_id not in self._handlers:
                self._on_demand_idle_tasks.pop(model_id, None)
                return False, "not currently loaded"

            if self._on_demand_ref_count.get(model_id, 0) > 0:
                return False, "model currently in use"

            handler = self._handlers.pop(model_id)
            self._on_demand_ref_count.pop(model_id, None)
            self._on_demand_loaded.discard(model_id)
            old_task = self._on_demand_idle_tasks.pop(model_id, None)
            if old_task is not None:
                old_task.cancel()

            if hasattr(handler, "cleanup"):
                await handler.cleanup()

            logger.info(f"Explicitly unloaded on-demand model '{model_id}'")
            return True, None

    async def _idle_unload(self, model_id: str, timeout: int) -> None:
        """Unload an on-demand model after it has been idle.

        Parameters
        ----------
        model_id : str
            On-demand model identifier.
        timeout : int
            Seconds to wait before unloading.
        """
        logger.info(f"On-demand model '{model_id}' idle timer started ({timeout}s)")
        await asyncio.sleep(timeout)

        async with self._on_demand_load_lock:
            # Check that the model is still idle
            if self._on_demand_ref_count.get(model_id, 0) > 0:
                return
            if model_id not in self._handlers:
                return

            handler = self._handlers.pop(model_id)
            if hasattr(handler, "cleanup"):
                await handler.cleanup()
            self._on_demand_loaded.discard(model_id)
            self._on_demand_idle_tasks.pop(model_id, None)
            logger.info(f"Unloaded idle on-demand model '{model_id}'")
