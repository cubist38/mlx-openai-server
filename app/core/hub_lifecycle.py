"""Shared hub lifecycle scaffolding extracted per plan-streamlineHubLifecycle.

This module is intentionally light on concrete logic for now; it provides the
protocols and helper classes that both the daemon (`HubSupervisor`) and the
single-model server will depend on as we consolidate lifecycle responsibilities.
Future patches will migrate the existing registry/worker code paths to use the
helpers defined here.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
import inspect
from typing import Any, Protocol, cast, runtime_checkable

from loguru import logger

from .model_registry import ModelRegistry


@dataclass(slots=True)
class ModelIdentity:
    """Describe a model known to the hub lifecycle stack.

    Parameters
    ----------
    name : str
        Slug used by hub routes, CLI, and daemon APIs.
    model_path : str or None, optional
        Registry identifier or filesystem path associated with the model.
    group : str or None, optional
        Logical group used for policy enforcement.
    """

    name: str
    model_path: str | None = None
    group: str | None = None


@dataclass(slots=True)
class WorkerTelemetry:
    """Bundle worker-related metadata so registry updates stay consistent.

    Parameters
    ----------
    port : int or None, optional
        Bound TCP port for sidecar exposure.
    pid : int or None, optional
        Process identifier for observability.
    ready : bool or None, optional
        Readiness strobe so callers can avoid redundant probes.
    extras : dict[str, Any], optional
        Additional metadata (e.g., queue size, concurrency limits).
    """

    port: int | None = None
    pid: int | None = None
    ready: bool | None = None
    extras: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class WorkerProtocol(Protocol):
    """Protocol describing the subset of `SidecarWorker` behavior we depend on."""

    ready: bool
    port: int | None
    pid: int | None

    async def start(self) -> None:
        """Start the worker process if it is not already running."""

    async def stop(self) -> None:
        """Stop the worker process if it is running."""


@runtime_checkable
class HubLifecycleService(Protocol):
    """Canonical service surface for hub lifecycle orchestration."""

    async def start_model(self, name: str) -> dict[str, Any]:
        """Ensure a model manager is initialized and its worker is running."""

    async def stop_model(self, name: str) -> dict[str, Any]:
        """Fully stop a model, unloading VRAM and terminating workers."""

    async def load_model(self, name: str) -> dict[str, Any]:
        """Guarantee the associated manager has a loaded handler."""

    async def unload_model(self, name: str) -> dict[str, Any]:
        """Evict the handler from VRAM when possible."""

    async def get_status(self) -> dict[str, Any]:
        """Return a consolidated runtime snapshot for UI/CLI consumers."""

    async def reload_config(self) -> dict[str, Any] | None:
        """Reload configuration and return a summary dict or ``None``."""

    async def shutdown_all(self) -> dict[str, Any] | None:
        """Shutdown all managed handlers and workers, returning metadata."""

    async def acquire_handler(self, name: str, reason: str = "request") -> Any | None:
        """Acquire a model handler for the given name and reason."""

    async def schedule_vram_load(
        self, name: str, settings: Any | None = None
    ) -> dict[str, Any] | None:
        """Schedule a VRAM load for the named model with optional settings."""

    async def schedule_vram_unload(self, name: str) -> dict[str, Any] | None:
        """Schedule a VRAM unload for the named model."""


_HUB_METHODS: tuple[str, ...] = (
    "start_model",
    "stop_model",
    "load_model",
    "unload_model",
    "get_status",
)


CallResult = Awaitable[dict[str, Any]] | dict[str, Any]


@dataclass(slots=True)
class HubServiceAdapter(HubLifecycleService):
    """Thin adapter that wraps arbitrary callables behind the service protocol."""

    start_model_fn: Callable[[str], CallResult]
    stop_model_fn: Callable[[str], CallResult]
    load_model_fn: Callable[[str], CallResult]
    unload_model_fn: Callable[[str], CallResult]
    status_fn: Callable[[], CallResult]
    reload_config_fn: Callable[[], CallResult] | None = None
    shutdown_all_fn: Callable[[], CallResult] | None = None
    acquire_handler_fn: Callable[[str, str], CallResult] | None = None
    schedule_vram_load_fn: Callable[..., CallResult] | None = None
    schedule_vram_unload_fn: Callable[[str], CallResult] | None = None

    async def start_model(self, name: str) -> dict[str, Any]:
        """Invoke the wrapped start callable.

        Parameters
        ----------
        name : str
            Model identifier forwarded to the backend.

        Returns
        -------
        dict[str, Any]
            Adapter-normalized action payload.
        """

        return await self._dispatch(self.start_model_fn, name)

    async def stop_model(self, name: str) -> dict[str, Any]:
        """Invoke the wrapped stop callable.

        Parameters
        ----------
        name : str
            Model identifier forwarded to the backend.

        Returns
        -------
        dict[str, Any]
            Adapter-normalized action payload.
        """

        return await self._dispatch(self.stop_model_fn, name)

    async def load_model(self, name: str) -> dict[str, Any]:
        """Invoke the wrapped load callable.

        Parameters
        ----------
        name : str
            Model identifier forwarded to the backend.

        Returns
        -------
        dict[str, Any]
            Adapter-normalized action payload.
        """

        return await self._dispatch(self.load_model_fn, name)

    async def unload_model(self, name: str) -> dict[str, Any]:
        """Invoke the wrapped unload callable.

        Parameters
        ----------
        name : str
            Model identifier forwarded to the backend.

        Returns
        -------
        dict[str, Any]
            Adapter-normalized action payload.
        """

        return await self._dispatch(self.unload_model_fn, name)

    async def get_status(self) -> dict[str, Any]:
        """Invoke the wrapped callable that returns a service snapshot.

        Returns
        -------
        dict[str, Any]
            Latest snapshot payload from the backend service.
        """

        return await self._dispatch(self.status_fn)

    async def reload_config(self) -> dict[str, Any] | None:
        """Invoke the wrapped `reload_config` callable if present.

        Returns
        -------
        dict[str, Any] | None
            The reload result or ``None`` when the callable is absent.
        """
        if self.reload_config_fn is None:
            return None
        return await self._dispatch(self.reload_config_fn)

    async def shutdown_all(self) -> dict[str, Any] | None:
        """Invoke the wrapped `shutdown_all` callable if present.

        Returns
        -------
        dict[str, Any] | None
            The shutdown result or ``None`` when the callable is absent.
        """
        if self.shutdown_all_fn is None:
            return None
        return await self._dispatch(self.shutdown_all_fn)

    async def acquire_handler(self, name: str, reason: str = "request") -> Any | None:
        """Acquire a handler from the underlying service if available.

        Parameters
        ----------
        name : str
            Model name to acquire a handler for.
        reason : str, default "request"
            Reason forwarded to the underlying acquire call.

        Returns
        -------
        Any | None
            The acquired handler or ``None`` if unsupported.
        """
        if self.acquire_handler_fn is None:
            return None
        return await self._dispatch(self.acquire_handler_fn, name, reason)

    async def schedule_vram_load(
        self, name: str, settings: Any | None = None
    ) -> dict[str, Any] | None:
        """Schedule an asynchronous VRAM load via the wrapped callable.

        Some implementations accept ``settings`` as a keyword-only argument;
        call through a small wrapper for compatibility.
        """
        if self.schedule_vram_load_fn is None:
            return None

        fn = self.schedule_vram_load_fn
        assert fn is not None

        def _wrapper(n: str, s: Any | None = None) -> CallResult:
            return fn(n, settings=s)

        return await self._dispatch(_wrapper, name, settings)

    async def schedule_vram_unload(self, name: str) -> dict[str, Any] | None:
        """Schedule an asynchronous VRAM unload via the wrapped callable.

        Returns
        -------
        dict[str, Any] | None
            Action metadata or ``None`` when unsupported.
        """
        if self.schedule_vram_unload_fn is None:
            return None
        return await self._dispatch(self.schedule_vram_unload_fn, name)

    async def _dispatch(self, func: Callable[..., CallResult], *args: Any) -> dict[str, Any]:
        """Call ``func`` with ``args`` and await when necessary.

        Parameters
        ----------
        func : Callable[..., CallResult]
            Callable to invoke.
        *args : Any
            Arguments forwarded to ``func``.

        Returns
        -------
        dict[str, Any]
            Resulting payload coerced to a mapping.
        """

        result = func(*args)
        if inspect.isawaitable(result):
            res = await result
        else:
            res = result
        return res


@dataclass(slots=True)
class RegistrySyncService:
    """Centralize all `ModelRegistry` mutations tied to lifecycle events.

    The helper exposes narrowly scoped methods so daemon and server paths can
    reuse the same write logic when a handler loads, unloads, or a worker port
    changes. This prevents drift in metadata schemas (e.g., worker ports,
    unload timestamps, VRAM flags).
    """

    registry: ModelRegistry

    async def record_state(
        self,
        model_id: str,
        *,
        handler: Any | None,
        status: str,
        vram_loaded: bool | None = None,
        metadata_updates: dict[str, Any] | None = None,
    ) -> None:
        """Write handler attachment metadata with consistent defaults."""

        payload = dict(metadata_updates or {})
        if vram_loaded is not None:
            payload["vram_loaded"] = vram_loaded

        try:
            await self.registry.update_model_state(
                model_id,
                handler=handler,
                status=status,
                metadata_updates=payload or None,
            )
        except Exception as exc:  # pragma: no cover - defensive logging path
            logger.warning(f"Registry record_state failed for {model_id}: {exc}")

    async def handler_loaded(
        self,
        model_id: str,
        *,
        handler: Any | None,
        metadata_updates: dict[str, Any] | None = None,
    ) -> None:
        """Record a successful handler load inside the registry."""

        await self.record_state(
            model_id,
            handler=handler,
            status="loaded",
            vram_loaded=True,
            metadata_updates=metadata_updates,
        )

    async def handler_unloaded(
        self,
        model_id: str,
        *,
        metadata_updates: dict[str, Any] | None = None,
    ) -> None:
        """Record that a handler has been unloaded and VRAM freed."""

        await self.record_state(
            model_id,
            handler=None,
            status="unloaded",
            vram_loaded=False,
            metadata_updates=metadata_updates,
        )

    async def update_worker_metadata(
        self,
        model_id: str,
        telemetry: WorkerTelemetry,
    ) -> None:
        """Store worker telemetry under the registry metadata block.

        Parameters
        ----------
        model_id : str
            Registry identifier that should be updated.
        telemetry : WorkerTelemetry
            Worker metadata snapshot (port, pid, extras).
        """

        payload: dict[str, Any] = {}
        if telemetry.port is not None:
            payload["worker_port"] = telemetry.port
        if telemetry.pid is not None:
            payload["worker_pid"] = telemetry.pid
        if telemetry.extras:
            payload.update(telemetry.extras)

        try:
            await self.registry.update_model_state(model_id, metadata_updates=payload)
        except Exception as exc:  # pragma: no cover - defensive logging path
            logger.warning(f"Failed to record worker telemetry for {model_id}: {exc}")

    async def clear_worker(self, model_id: str) -> None:
        """Clear worker-specific metadata when a worker terminates."""

        try:
            await self.registry.update_model_state(
                model_id,
                metadata_updates={"worker_port": None, "worker_pid": None},
            )
        except Exception as exc:  # pragma: no cover - defensive logging path
            logger.debug(f"Failed to clear worker metadata for {model_id}: {exc}")


def get_hub_lifecycle_service(container: Any) -> HubLifecycleService | None:
    """Return the first object on ``container`` that satisfies the service protocol."""

    if container is None:
        return None

    def _maybe_extract(obj: Any | None) -> list[Any]:
        """Return hub lifecycle candidates attached to ``obj``."""

        if obj is None:
            return []
        return [
            getattr(obj, attr, None) for attr in ("hub_service", "hub_controller", "supervisor")
        ]

    candidates: list[Any | None] = _maybe_extract(container)
    state = getattr(container, "state", None)
    if state is not None:
        candidates.extend(_maybe_extract(state))

    for candidate in candidates:
        if candidate is None:
            continue

        # If the candidate is already a `HubLifecycleService` implementation
        # return it directly. Otherwise, prefer returning any object that
        # exposes the canonical hub methods to preserve behavior for simple
        # controller objects used in tests and lightweight in-process
        # controllers.
        callable_attrs = _HUB_METHODS
        if all(hasattr(candidate, attr) for attr in callable_attrs):
            # Candidate exposes the full hub method surface; cast to the
            # protocol to satisfy static typing for callers.
            return cast("HubLifecycleService", candidate)

        if isinstance(candidate, HubLifecycleService):
            return candidate

        # Minimal safe defaults for missing lifecycle methods: adapt the
        # candidate into a HubServiceAdapter so routes can call the full
        # lifecycle surface without raising AttributeError. Use explicit
        # typed callables for mypy.
        start_fn: Callable[[str], CallResult] = cast(
            "Callable[[str], CallResult]", getattr(candidate, "start_model", lambda name: {})
        )
        stop_fn: Callable[[str], CallResult] = cast(
            "Callable[[str], CallResult]", getattr(candidate, "stop_model", lambda name: {})
        )
        load_fn: Callable[[str], CallResult] = cast(
            "Callable[[str], CallResult]", getattr(candidate, "load_model", lambda name: {})
        )
        unload_fn: Callable[[str], CallResult] = cast(
            "Callable[[str], CallResult]", getattr(candidate, "unload_model", lambda name: {})
        )
        status_fn: Callable[[], CallResult] = cast(
            "Callable[[], CallResult]", getattr(candidate, "get_status", dict)
        )

        return HubServiceAdapter(
            start_model_fn=start_fn,
            stop_model_fn=stop_fn,
            load_model_fn=load_fn,
            unload_model_fn=unload_fn,
            status_fn=status_fn,
            reload_config_fn=getattr(candidate, "reload_config", None),
            shutdown_all_fn=getattr(candidate, "shutdown_all", None),
            acquire_handler_fn=getattr(candidate, "acquire_handler", None),
            schedule_vram_load_fn=getattr(candidate, "schedule_vram_load", None),
            schedule_vram_unload_fn=getattr(candidate, "schedule_vram_unload", None),
        )
    return None
