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
from .keep_alive import KeepAlive, parse_keep_alive


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
    _aliases : dict[str, str]
        Mapping of alias (including ``name:version`` forms) to canonical
        model_id. Every lookup resolves through this map, so an alias works
        anywhere a model_id does.
    _lock : asyncio.Lock
        Async lock for thread-safe mutations.
    """

    def __init__(
        self,
        max_loaded_models: int = 1,
        model_load_timeout: float = 300.0,
    ) -> None:
        """Initialize an empty model registry.

        Parameters
        ----------
        max_loaded_models : int
            Maximum number of on-demand model processes kept loaded at once.
            Zero disables the limit. Startup-loaded models do not count.
        model_load_timeout : float
            Maximum time a request waits for an occupied model slot.
        """
        if max_loaded_models < 0:
            raise ValueError("max_loaded_models must be greater than or equal to zero")
        if model_load_timeout <= 0:
            raise ValueError("model_load_timeout must be greater than zero")

        self._handlers: dict[str, Any] = {}
        self._metadata: dict[str, ModelMetadata] = {}
        self._aliases: dict[str, str] = {}
        self._lock = asyncio.Lock()

        # On-demand (dynamic swapping) state
        self._max_loaded_models = max_loaded_models
        self._model_load_timeout = model_load_timeout
        self._on_demand_configs: dict[str, dict[str, Any]] = {}
        self._on_demand_loaded: set[str] = set()
        self._on_demand_load_lock = asyncio.Lock()
        self._on_demand_capacity_changed = asyncio.Event()
        self._on_demand_ref_count: dict[str, int] = {}
        self._on_demand_idle_tasks: dict[str, asyncio.Task] = {}
        self._on_demand_idle_timeouts: dict[str, float | None] = {}
        self._on_demand_loaded_at: dict[str, float] = {}
        self._on_demand_last_used: dict[str, float] = {}
        self._on_demand_expires_at: dict[str, float | None] = {}
        self._on_demand_loading: set[str] = set()
        self._on_demand_last_error: dict[str, str] = {}

        logger.info(
            f"Model registry initialized (max_loaded_models={max_loaded_models}, "
            f"model_load_timeout={model_load_timeout}s)"
        )

    async def register_model(
        self,
        model_id: str,
        handler: Any,
        model_type: str,
        context_length: int | None = None,
        aliases: list[str] | None = None,
        version: str | None = None,
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
        aliases : list[str] | None, optional
            Extra names that route to this model.
        version : str | None, optional
            Version tag recorded in metadata.

        Raises
        ------
        ValueError
            If ``model_id`` or one of its aliases is already registered.
        """
        async with self._lock:
            if model_id in self._handlers:
                raise ValueError(f"Model '{model_id}' is already registered")

            resolved_aliases = self._reserve_aliases_locked(model_id, aliases, version)

            metadata = ModelMetadata(
                id=model_id,
                type=model_type,
                context_length=context_length,
                created_at=int(time.time()),
                version=version,
                aliases=resolved_aliases,
            )

            self._handlers[model_id] = handler
            self._metadata[model_id] = metadata

            logger.info(
                f"Registered model: {model_id} (type={model_type}, context_length={context_length})"
            )
            if resolved_aliases:
                logger.info(f"Model '{model_id}' also answers to: {', '.join(resolved_aliases)}")

    def _reserve_aliases_locked(
        self,
        model_id: str,
        aliases: list[str] | None,
        version: str | None = None,
    ) -> list[str]:
        """Claim alias names for ``model_id``. Caller must hold ``self._lock``.

        A declared ``version`` always yields a ``<model_id>:<version>`` route.
        Deriving it here rather than only in the configuration layer means a
        version tag is addressable however the model was registered.

        Parameters
        ----------
        model_id : str
            Canonical model identifier the aliases point at.
        aliases : list[str] | None
            Candidate alias names. The canonical name and duplicates are
            dropped rather than rejected, since they are harmless no-ops.
        version : str | None
            Version tag to expose as ``<model_id>:<version>``.

        Returns
        -------
        list[str]
            The alias names actually registered, in the order supplied.

        Raises
        ------
        ValueError
            If an alias is already taken by a different model, either as that
            model's canonical name or as one of its aliases.
        """
        candidates: list[str] = []
        if version:
            candidates.append(f"{model_id}:{version}")
        candidates.extend(aliases or [])

        registered: list[str] = []
        for alias in candidates:
            if alias == model_id or alias in registered:
                continue
            owner = self._aliases.get(alias)
            if owner is not None and owner != model_id:
                raise ValueError(f"Alias '{alias}' is already registered for model '{owner}'")
            if alias in self._handlers or alias in self._on_demand_configs:
                raise ValueError(f"Alias '{alias}' collides with registered model '{alias}'")
            self._aliases[alias] = model_id
            registered.append(alias)
        return registered

    def resolve_model_id(self, model_id: str) -> str:
        """Map a requested name to its canonical model identifier.

        Unknown names are returned unchanged so that callers keep raising
        their own not-found errors with the name the client actually sent.

        Parameters
        ----------
        model_id : str
            Name from the request ``model`` field. May be a canonical id, an
            alias, or a ``name:version`` form.

        Returns
        -------
        str
            The canonical model id, or ``model_id`` when it is not an alias.
        """
        return self._aliases.get(model_id, model_id)

    def list_aliases(self, model_id: str) -> list[str]:
        """Return the aliases registered for a canonical model id."""
        canonical = self.resolve_model_id(model_id)
        return sorted(alias for alias, target in self._aliases.items() if target == canonical)

    def get_handler(self, model_id: str) -> Any:
        """Get handler for a specific model.

        Parameters
        ----------
        model_id : str
            Model identifier or alias.

        Returns
        -------
        Any
            Handler instance.

        Raises
        ------
        KeyError
            If ``model_id`` is not found in the registry.
        """
        canonical = self.resolve_model_id(model_id)
        if canonical not in self._handlers:
            available = ", ".join(sorted(self._handlers.keys())) or "(none)"
            raise KeyError(
                f"Model '{model_id}' not found in registry. Available models: {available}"
            )
        return self._handlers[canonical]

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
        status_by_id = {item["id"]: item for item in self.get_model_status()}
        return [
            {
                "id": metadata.id,
                "object": metadata.object,
                "created": metadata.created_at,
                "owned_by": metadata.owned_by,
                "metadata": status_by_id.get(metadata.id),
            }
            for metadata in self._metadata.values()
        ]

    def get_metadata(self, model_id: str) -> ModelMetadata:
        """Get metadata for a specific model.

        Parameters
        ----------
        model_id : str
            Model identifier or alias.

        Returns
        -------
        ModelMetadata
            Metadata instance.

        Raises
        ------
        KeyError
            If ``model_id`` is not found.
        """
        canonical = self.resolve_model_id(model_id)
        if canonical not in self._metadata:
            raise KeyError(f"Model '{model_id}' not found in registry")
        return self._metadata[canonical]

    async def unregister_model(self, model_id: str) -> None:
        """Unregister a model and clean up its handler.

        Parameters
        ----------
        model_id : str
            Model identifier or alias.

        Raises
        ------
        KeyError
            If ``model_id`` is not found.
        """
        async with self._lock:
            canonical = self.resolve_model_id(model_id)
            if canonical not in self._handlers:
                raise KeyError(f"Model '{model_id}' not found in registry")

            handler = self._handlers[canonical]
            if hasattr(handler, "cleanup"):
                try:
                    await handler.cleanup()
                    logger.info(f"Cleaned up handler for model: {canonical}")
                except Exception as e:
                    logger.error(f"Error cleaning up handler for '{canonical}': {e}")

            del self._handlers[canonical]
            del self._metadata[canonical]
            # Aliases must go too, otherwise they would resolve to a name that
            # is no longer registered and every lookup through them would fail
            # with a confusing "not found" for the canonical id.
            for alias in [a for a, target in self._aliases.items() if target == canonical]:
                del self._aliases[alias]
            logger.info(f"Unregistered model: {canonical}")

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
            self._on_demand_loaded_at.clear()
            self._on_demand_last_used.clear()
            self._on_demand_expires_at.clear()
            self._on_demand_loading.clear()
            self._on_demand_last_error.clear()
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
            Model identifier or alias.

        Returns
        -------
        bool
            ``True`` if model is registered, ``False`` otherwise.
        """
        canonical = self.resolve_model_id(model_id)
        return canonical in self._handlers or canonical in self._on_demand_configs

    def get_model_count(self) -> int:
        """Get count of registered models (loaded + on-demand).

        Returns
        -------
        int
            Number of registered models.
        """
        return len(self._handlers) + len(self._on_demand_configs.keys() - self._handlers.keys())

    def get_loaded_model_count(self) -> int:
        """Return the number of currently loaded handlers."""
        return len(self._handlers)

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
        idle_timeout: KeepAlive = 300,
        aliases: list[str] | None = None,
        version: str | None = None,
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
        idle_timeout : KeepAlive
            Default time to retain an idle model. Negative values retain it
            indefinitely.
        aliases : list[str] | None, optional
            Extra names that route to this model.
        version : str | None, optional
            Version tag recorded in metadata.
        """
        parsed_idle_timeout = parse_keep_alive(idle_timeout, 300.0)
        async with self._lock:
            if model_id in self._handlers or model_id in self._on_demand_configs:
                raise ValueError(f"Model '{model_id}' is already registered")

            resolved_aliases = self._reserve_aliases_locked(model_id, aliases, version)

            self._on_demand_configs[model_id] = {
                "model_cfg_dict": model_cfg_dict,
                "model_type": model_type,
                "model_path": model_path,
                "context_length": context_length,
                "queue_config": queue_config,
            }
            self._on_demand_idle_timeouts[model_id] = parsed_idle_timeout

            # Add metadata so the model appears in /v1/models
            self._metadata[model_id] = ModelMetadata(
                id=model_id,
                type=model_type,
                context_length=context_length,
                created_at=int(time.time()),
                version=version,
                aliases=resolved_aliases,
            )

            logger.info(
                f"Registered on-demand model: {model_id} "
                f"(type={model_type}, idle_timeout={parsed_idle_timeout})"
            )
            if resolved_aliases:
                logger.info(f"Model '{model_id}' also answers to: {', '.join(resolved_aliases)}")

    def is_on_demand(self, model_id: str) -> bool:
        """Check if a model (or alias) is registered as on-demand."""
        return self.resolve_model_id(model_id) in self._on_demand_configs

    async def ensure_on_demand_loaded(
        self,
        model_id: str,
        keep_alive: KeepAlive = None,
    ) -> Any:
        """Acquire an on-demand model, loading it when necessary.

        Capacity is enforced using least-recently-used eviction. Active models
        are never evicted; callers wait for a slot until
        ``model_load_timeout`` expires.

        Parameters
        ----------
        model_id : str
            On-demand model identifier or alias.
        keep_alive : KeepAlive
            Per-request retention override, validated when the lease starts
            and applied when it ends.

        Returns
        -------
        Any
            The handler (``HandlerProcessProxy``) for the model.

        Raises
        ------
        KeyError
            If ``model_id`` is not a registered on-demand model.
        RuntimeError
            If the handler subprocess fails to start or no model slot becomes
            available before the configured timeout.
        """
        # Resolve once, then work in canonical ids so lease accounting cannot
        # split between an alias and the name it points at. The requested name
        # is kept for the error message the client will read.
        requested_id = model_id
        model_id = self.resolve_model_id(model_id)
        if model_id not in self._on_demand_configs:
            raise KeyError(f"Model '{requested_id}' is not registered as on-demand")

        default_keep_alive = self._on_demand_idle_timeouts.get(model_id, 300.0)
        parse_keep_alive(keep_alive, default_keep_alive)
        deadline = time.monotonic() + self._model_load_timeout

        while True:
            async with self._on_demand_load_lock:
                if model_id in self._handlers and model_id in self._on_demand_loaded:
                    return self._acquire_loaded_model(model_id)

                victim = self._select_lru_idle_model()
                at_capacity = (
                    self._max_loaded_models > 0
                    and len(self._on_demand_loaded) >= self._max_loaded_models
                )
                if at_capacity and victim is None:
                    self._on_demand_capacity_changed.clear()
                else:
                    if at_capacity and victim is not None:
                        logger.info(f"Evicting idle model '{victim}' to load '{model_id}'")
                        await self._unload_on_demand_locked(victim)
                    return await self._load_on_demand_locked(model_id)

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(
                    f"Timed out waiting for a model slot after {self._model_load_timeout:g} seconds"
                )
            try:
                await asyncio.wait_for(
                    self._on_demand_capacity_changed.wait(),
                    timeout=remaining,
                )
            except TimeoutError as exc:
                raise RuntimeError(
                    f"Timed out waiting for a model slot after {self._model_load_timeout:g} seconds"
                ) from exc

    def _acquire_loaded_model(self, model_id: str) -> Any:
        """Acquire a lease for an already-loaded model."""
        idle_task = self._on_demand_idle_tasks.pop(model_id, None)
        if idle_task is not None:
            idle_task.cancel()
        self._on_demand_expires_at[model_id] = None
        self._on_demand_last_used[model_id] = time.time()
        self._on_demand_ref_count[model_id] = self._on_demand_ref_count.get(model_id, 0) + 1
        logger.debug(
            f"Acquired on-demand model '{model_id}', "
            f"ref_count={self._on_demand_ref_count[model_id]}"
        )
        return self._handlers[model_id]

    async def _load_on_demand_locked(self, model_id: str) -> Any:
        """Spawn and acquire a model while the lifecycle lock is held."""
        cfg = self._on_demand_configs[model_id]
        logger.info(f"Loading on-demand model '{model_id}' (path={cfg['model_path']})")

        from .handler_process import HandlerProcessProxy

        proxy = HandlerProcessProxy(
            model_cfg_dict=cfg["model_cfg_dict"],
            model_type=cfg["model_type"],
            model_path=cfg["model_path"],
            served_model_name=model_id,
        )
        self._on_demand_loading.add(model_id)
        self._on_demand_last_error.pop(model_id, None)
        try:
            await proxy.start(cfg["queue_config"])
        except (Exception, asyncio.CancelledError) as exc:
            self._on_demand_last_error[model_id] = str(exc)
            if hasattr(proxy, "cleanup"):
                try:
                    await proxy.cleanup()
                except Exception as cleanup_exc:
                    logger.error(
                        f"Failed to clean up partially loaded model '{model_id}': {cleanup_exc}"
                    )
            raise
        finally:
            self._on_demand_loading.discard(model_id)

        now = time.time()
        self._handlers[model_id] = proxy
        self._on_demand_loaded.add(model_id)
        self._on_demand_ref_count[model_id] = 1
        self._on_demand_loaded_at[model_id] = now
        self._on_demand_last_used[model_id] = now
        self._on_demand_expires_at[model_id] = None

        logger.info(f"On-demand model '{model_id}' loaded successfully")
        return proxy

    async def release_on_demand(
        self,
        model_id: str,
        keep_alive: KeepAlive = None,
        handler: Any = None,
    ) -> None:
        """Release a reference to an on-demand model after a request completes.

        When the reference count reaches zero, an idle timeout task is
        scheduled to unload the model.

        Parameters
        ----------
        model_id : str
            On-demand model identifier or alias.
        keep_alive : KeepAlive
            Per-request retention override. Zero unloads immediately and a
            negative value disables expiry.
        handler : Any
            Handler the caller leased. When supplied, the release is ignored if
            the worker has since been replaced, so a lease orphaned by a forced
            unload cannot decrement the count of its replacement.
        """
        model_id = self.resolve_model_id(model_id)
        if model_id not in self._on_demand_configs:
            return

        async with self._on_demand_load_lock:
            if model_id not in self._on_demand_loaded:
                return
            if handler is not None and self._handlers.get(model_id) is not handler:
                logger.debug(
                    f"Ignoring stale lease for model '{model_id}': the worker was replaced"
                )
                return
            ref_count = max(0, self._on_demand_ref_count.get(model_id, 1) - 1)
            self._on_demand_ref_count[model_id] = ref_count
            self._on_demand_last_used[model_id] = time.time()
            logger.debug(f"Released on-demand model '{model_id}', ref_count={ref_count}")

            if ref_count > 0:
                return

            self._on_demand_capacity_changed.set()
            timeout = parse_keep_alive(
                keep_alive,
                self._on_demand_idle_timeouts.get(model_id, 300.0),
            )
            old_task = self._on_demand_idle_tasks.pop(model_id, None)
            if old_task is not None:
                old_task.cancel()

            if timeout is None:
                self._on_demand_expires_at[model_id] = None
                logger.info(f"Model '{model_id}' will remain loaded")
                return
            if timeout == 0:
                await self._unload_on_demand_locked(model_id)
                return

            self._on_demand_expires_at[model_id] = time.time() + timeout
            self._on_demand_idle_tasks[model_id] = asyncio.create_task(
                self._idle_unload(model_id, timeout)
            )

    async def _idle_unload(self, model_id: str, timeout: float) -> None:
        """Unload an on-demand model after it has been idle.

        Parameters
        ----------
        model_id : str
            On-demand model identifier.
        timeout : float
            Seconds to wait before unloading.
        """
        logger.info(f"On-demand model '{model_id}' idle timer started ({timeout}s)")
        try:
            await asyncio.sleep(timeout)
            async with self._on_demand_load_lock:
                if self._on_demand_ref_count.get(model_id, 0) > 0:
                    return
                await self._unload_on_demand_locked(model_id)
        except asyncio.CancelledError:
            logger.debug(f"Cancelled idle timer for model '{model_id}'")
            raise
        except Exception as exc:
            # Nothing awaits this task, so the failure must be reported here.
            # The worker stays tracked and is reported through ``last_error``
            # so an explicit unload can retry it.
            logger.error(f"Idle unload of model '{model_id}' failed: {exc}")

    def _select_lru_idle_model(self) -> str | None:
        """Return the least-recently-used idle on-demand model."""
        idle_models = [
            model_id
            for model_id in self._on_demand_loaded
            if self._on_demand_ref_count.get(model_id, 0) == 0
        ]
        if not idle_models:
            return None
        return min(idle_models, key=lambda item: self._on_demand_last_used.get(item, 0.0))

    async def _unload_on_demand_locked(self, model_id: str) -> bool:
        """Unload an idle on-demand model while the lifecycle lock is held."""
        handler = self._handlers.get(model_id)
        if handler is None or model_id not in self._on_demand_loaded:
            return False

        idle_task = self._on_demand_idle_tasks.pop(model_id, None)
        current_task = asyncio.current_task()
        if idle_task is not None and idle_task is not current_task:
            idle_task.cancel()

        if hasattr(handler, "cleanup"):
            try:
                await handler.cleanup()
            except Exception as exc:
                self._on_demand_last_error[model_id] = f"Unload failed: {exc}"
                raise

        self._handlers.pop(model_id, None)
        self._on_demand_loaded.discard(model_id)
        self._on_demand_ref_count.pop(model_id, None)
        self._on_demand_loaded_at.pop(model_id, None)
        self._on_demand_expires_at.pop(model_id, None)
        self._on_demand_capacity_changed.set()

        logger.info(f"Unloaded on-demand model '{model_id}'")
        return True

    async def unload_on_demand(self, model_id: str, force: bool = False) -> bool:
        """Explicitly unload an on-demand model.

        Parameters
        ----------
        model_id : str
            Configured model identifier or alias.
        force : bool
            Allow unloading with active requests. This should normally remain
            false because it interrupts in-flight generation.

        Returns
        -------
        bool
            Whether a loaded model process was stopped.

        Raises
        ------
        KeyError
            If the model is unknown.
        ValueError
            If the model is not on-demand or has active requests.
        """
        if not self.has_model(model_id):
            raise KeyError(f"Model '{model_id}' is not registered")
        if not self.is_on_demand(model_id):
            raise ValueError(f"Model '{model_id}' is not configured for on-demand loading")

        model_id = self.resolve_model_id(model_id)
        async with self._on_demand_load_lock:
            active_requests = self._on_demand_ref_count.get(model_id, 0)
            if active_requests > 0 and not force:
                raise ValueError(f"Model '{model_id}' has {active_requests} active request(s)")
            return await self._unload_on_demand_locked(model_id)

    def get_model_status(self) -> list[dict[str, Any]]:
        """Return lifecycle status for every configured model."""
        result: list[dict[str, Any]] = []
        for model_id in self.list_model_ids():
            metadata = self._metadata.get(model_id)
            is_on_demand = model_id in self._on_demand_configs
            is_loaded = model_id in self._handlers
            handler = self._handlers.get(model_id)
            cfg = self._on_demand_configs.get(model_id, {})
            if model_id in self._on_demand_loading:
                state = "loading"
            elif is_loaded:
                state = "busy" if self._on_demand_ref_count.get(model_id, 0) > 0 else "loaded"
            else:
                state = "unloaded"

            result.append(
                {
                    "id": model_id,
                    "type": metadata.type if metadata is not None else None,
                    "backend": "mlx",
                    "context_length": (metadata.context_length if metadata is not None else None),
                    "version": metadata.version if metadata is not None else None,
                    "aliases": self.list_aliases(model_id),
                    "state": state,
                    "loaded": is_loaded,
                    "on_demand": is_on_demand,
                    "active_requests": self._on_demand_ref_count.get(model_id, 0),
                    "model_path": (
                        cfg.get("model_path")
                        if is_on_demand
                        else getattr(handler, "model_path", None)
                    ),
                    "pid": getattr(handler, "pid", None),
                    "loaded_at": self._on_demand_loaded_at.get(model_id),
                    "last_used": self._on_demand_last_used.get(model_id),
                    "expires_at": self._on_demand_expires_at.get(model_id),
                    "default_keep_alive": self._on_demand_idle_timeouts.get(model_id),
                    "last_error": self._on_demand_last_error.get(model_id),
                }
            )
        return result
