"""Hub controller that manages handler lifecycles for multiple models."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from http import HTTPStatus
from typing import Any

from loguru import logger

from ..config import MLXServerConfig
from ..core.model_registry import ModelRegistry
from ..server import LazyHandlerManager, MLXHandler
from .errors import HubControllerError
from .runtime import HubRuntime

HandlerFactory = Callable[
    [MLXServerConfig, Callable[[MLXHandler | None], None]], LazyHandlerManager
]


class HubController:
    """Coordinate LazyHandlerManagers according to the HubRuntime plan."""

    def __init__(
        self,
        runtime: HubRuntime,
        registry: ModelRegistry,
        *,
        handler_factory: HandlerFactory | None = None,
    ) -> None:
        """Initialize the hub controller.

        Parameters
        ----------
        runtime : HubRuntime
            Runtime that tracks group slots and per-model status.
        registry : ModelRegistry
            Registry that exposes models to downstream consumers.
        handler_factory : HandlerFactory, optional
            Custom factory for creating ``LazyHandlerManager`` instances.
        """
        self.runtime = runtime
        self.registry = registry
        self._handler_factory = handler_factory or self._default_handler_factory
        self._handlers: dict[str, LazyHandlerManager] = {}
        self._base_metadata: dict[str, dict[str, Any]] = {}
        self._registry_tasks: set[asyncio.Task[None]] = set()
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        """Register all models and create handler managers.

        Returns
        -------
        None
            The coroutine completes once bootstrap is finished.

        Raises
        ------
        HubControllerError
            Propagates errors raised while preparing managers.
        """

        async with self._lock:
            for name in self.runtime.model_names():
                config = self.runtime.get_config(name)
                base_metadata = self._build_base_metadata(config)
                self._base_metadata[name] = base_metadata
                manager = self._handler_factory(
                    config,
                    self._make_handler_callback(name),
                )
                self._handlers[name] = manager
                await self.registry.register_model(
                    model_id=name,
                    handler=None,
                    model_type=config.model_type,
                    context_length=config.context_length,
                    metadata_extras=base_metadata,
                )
        await self._auto_bootstrap()

    async def shutdown(self) -> None:
        """Shutdown all handler managers and wait for registry sync.

        Returns
        -------
        None
            The coroutine resolves when all managers and tasks halt.
        """

        async with self._lock:
            managers = list(self._handlers.values())
        await asyncio.gather(*(manager.shutdown() for manager in managers), return_exceptions=True)
        await self.wait_for_registry_idle()

    async def load_model(self, name: str, *, reason: str = "manual") -> None:
        """Load ``name`` using the associated handler manager.

        Parameters
        ----------
        name : str
            Identifier of the model to load.
        reason : str, default="manual"
            Context string recorded in handler lifecycle logs.

        Returns
        -------
        None
            Resolves when the model reaches the loaded state.

        Raises
        ------
        HubControllerError
            If the model cannot load due to constraints or handler errors.
        """

        manager = self._get_manager(name)
        async with self._lock:
            if not self.runtime.can_load(name):
                raise HubControllerError(
                    f"Model '{name}' cannot load due to group constraints or state",
                    status_code=HTTPStatus.TOO_MANY_REQUESTS,
                )
            self.runtime.mark_loading(name)
        try:
            handler = await manager.ensure_loaded(reason)
        except Exception as exc:  # pragma: no cover - handler failures
            await self._handle_load_failure(name, exc)
            raise HubControllerError(
                f"Failed to load model '{name}': {exc}", status_code=HTTPStatus.SERVICE_UNAVAILABLE
            ) from exc
        if handler is None:
            await self._handle_load_failure(name, RuntimeError("handler manager returned None"))
            raise HubControllerError(
                f"Handler manager returned None for model '{name}'",
                status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            )
        async with self._lock:
            self.runtime.mark_loaded(name)

    async def acquire_handler(self, name: str, *, reason: str = "request") -> MLXHandler:
        """Ensure ``name`` is loaded and return its handler.

        Parameters
        ----------
        name : str
            Identifier of the model to acquire.
        reason : str, default="request"
            Context string used in logging and registry updates.

        Returns
        -------
        MLXHandler
            Loaded handler instance.

        Raises
        ------
        HubControllerError
            If the model is unknown or cannot be loaded due to failures.
        """

        manager = self._get_manager(name)
        async with self._lock:
            status = self.runtime.get_status(name)
            handler = manager.current_handler
            if status == "loaded" and handler is None:
                self.runtime.mark_unloaded(name)
                status = "unloaded"

        if status == "loaded" and handler is not None:
            manager.record_activity()
            return handler

        await self.load_model(name, reason=reason)
        handler = manager.current_handler
        if handler is None:
            raise HubControllerError(
                f"Handler missing after loading model '{name}'",
                status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            )
        manager.record_activity()
        return handler

    async def unload_model(self, name: str, *, reason: str = "manual") -> None:
        """Unload ``name`` and free its group slot.

        Parameters
        ----------
        name : str
            Identifier of the model to unload.
        reason : str, default="manual"
            Context string recorded in handler lifecycle logs.

        Returns
        -------
        None
            Completes after the model is marked unloaded.
        """

        manager = self._get_manager(name)
        await manager.unload(reason)
        async with self._lock:
            self.runtime.mark_unloaded(name)

    def get_handler_manager(self, name: str) -> LazyHandlerManager:
        """Return the handler manager for ``name``.

        Parameters
        ----------
        name : str
            Identifier of the requested model.

        Returns
        -------
        LazyHandlerManager
            Manager instance bound to ``name``.

        Raises
        ------
        HubControllerError
            If the model is not registered.
        """

        return self._get_manager(name)

    async def wait_for_registry_idle(self) -> None:
        """Wait until all scheduled registry updates have finished.

        Returns
        -------
        None
            Completes once there are no pending registry update tasks.
        """

        while True:
            async with self._lock:
                pending = tuple(self._registry_tasks)
            if not pending:
                break
            await asyncio.gather(*pending, return_exceptions=True)
            async with self._lock:
                for task in pending:
                    self._registry_tasks.discard(task)

    async def _auto_bootstrap(self) -> None:
        """Load any models flagged for automatic memory bootstrap.

        Returns
        -------
        None
            Completes after attempting the bootstrap set.
        """

        targets = self.runtime.bootstrap_targets()
        for name in targets:
            try:
                await self.load_model(name, reason="bootstrap")
            except HubControllerError as exc:
                logger.warning(f"Bootstrap load skipped for {name}: {exc}")

    async def _handle_load_failure(self, name: str, exc: Exception) -> None:
        """Record failure metadata and update registry state.

        Parameters
        ----------
        name : str
            Identifier of the model that failed to load.
        exc : Exception
            The exception instance raised by the handler manager.

        Returns
        -------
        None
            Completes after recording failure metadata.
        """

        async with self._lock:
            self.runtime.mark_failed(name, str(exc))
        self._schedule_registry_update(name, None, "failed", {"last_error": str(exc)})

    def _get_manager(self, name: str) -> LazyHandlerManager:
        """Return a manager, raising if none exists.

        Parameters
        ----------
        name : str
            Identifier of the model.

        Returns
        -------
        LazyHandlerManager
            Manager instance supervising ``name``.

        Raises
        ------
        HubControllerError
            If ``name`` is unknown.
        """

        manager = self._handlers.get(name)
        if manager is None:
            raise HubControllerError(
                f"Model '{name}' is not registered with the controller",
                status_code=HTTPStatus.NOT_FOUND,
            )
        return manager

    def _default_handler_factory(
        self, config: MLXServerConfig, on_change: Callable[[MLXHandler | None], None]
    ) -> LazyHandlerManager:
        """Construct a default ``LazyHandlerManager`` instance.

        Parameters
        ----------
        config : MLXServerConfig
            Model configuration to control.
        on_change : Callable[[MLXHandler | None], None]
            Callback that receives handler swap notifications.

        Returns
        -------
        LazyHandlerManager
            Newly created manager bound to ``config``.
        """

        return LazyHandlerManager(config, on_change=on_change)

    def _make_handler_callback(self, name: str) -> Callable[[MLXHandler | None], None]:
        """Return a callback used by managers to report handler swaps.

        Parameters
        ----------
        name : str
            Identifier of the model owning the callback.

        Returns
        -------
        Callable[[MLXHandler | None], None]
            Function that schedules registry updates whenever the handler changes.
        """

        def _callback(handler: MLXHandler | None) -> None:
            status = "loaded" if handler else "unloaded"
            self._schedule_registry_update(name, handler, status)

        return _callback

    def _build_base_metadata(self, config: MLXServerConfig) -> dict[str, Any]:
        """Return metadata that is attached to each registry update.

        Parameters
        ----------
        config : MLXServerConfig
            Configuration describing the target model.

        Returns
        -------
        dict[str, Any]
            Metadata blob copied into registry records.
        """

        return {
            "model_path": config.model_identifier,
            "model_type": config.model_type,
            "context_length": config.context_length,
            "jit_enabled": config.jit_enabled,
            "auto_unload_minutes": config.auto_unload_minutes,
            "group": config.group,
            "status": "unloaded",
        }

    def _schedule_registry_update(
        self,
        name: str,
        handler: MLXHandler | None,
        status: str,
        extra_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Schedule an asynchronous registry update for ``name``.

        Parameters
        ----------
        name : str
            Identifier of the model being updated.
        handler : MLXHandler | None
            Active handler when status == "loaded", else ``None``.
        status : str
            Status string persisted to the registry.
        extra_metadata : dict[str, Any] | None, optional
            Additional metadata merged into the base payload.

        Returns
        -------
        None
            The update is enqueued asynchronously.
        """

        metadata = dict(self._base_metadata.get(name, {}))
        if extra_metadata:
            metadata.update(extra_metadata)
        if handler is not None:
            metadata["model_path"] = getattr(handler, "model_path", metadata.get("model_path"))
        task = asyncio.create_task(
            self.registry.update_model_state(
                name,
                handler=handler,
                status=status,
                metadata_updates=metadata,
            )
        )

        def _cleanup(done: asyncio.Task[None]) -> None:
            self._registry_tasks.discard(done)
            exc = done.exception()
            if exc:
                logger.warning(
                    f"Registry update task failed for {name}. {type(exc).__name__}: {exc}"
                )

        self._registry_tasks.add(task)
        task.add_done_callback(_cleanup)
