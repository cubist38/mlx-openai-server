"""Model registry for managing multiple model handlers.

This module provides an asyncio-safe registry that tracks registered
models, attached per-model manager/handler objects, and VRAM-related
metadata. It exposes methods to attach managers via a loader callable,
request idempotent VRAM load/unload, and a per-request context manager
that increments/decrements an active request counter.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
import time
from typing import Any, TypedDict, cast

from loguru import logger

from ..schemas.model import ModelMetadata
from .manager_protocol import ManagerProtocol

_UNSET = object()


class VRAMStatus(TypedDict, total=False):
    """Per-model VRAM status metadata returned by ``get_vram_status``.

    Attributes
    ----------
    vram_loaded : bool
        Whether the model is currently loaded in VRAM.
    vram_last_load_ts : int or None
        Unix timestamp when the model was last loaded into VRAM, or ``None``.
    vram_last_unload_ts : int or None
        Unix timestamp when the model was last unloaded from VRAM, or ``None``.
    vram_last_request_ts : int or None
        Unix timestamp when the model last served a request, or ``None``.
    vram_load_error : str or None
        Error message from the last failed VRAM load attempt, or ``None``.
    active_requests : int
        Number of currently active requests being served by this model.
    -----
    The registry stores other dynamic keys as needed (e.g. ``_loading_task``),
    so this TypedDict is intentionally non-total to document the common subset
    surfaced to admin/UI code.
    """

    vram_loaded: bool
    vram_last_load_ts: int | None
    vram_last_unload_ts: int | None
    vram_last_request_ts: int | None
    vram_load_error: str | None
    active_requests: int


class ModelRegistry:
    """Asyncio event-loop-safe registry for model managers and metadata.

    The registry stores three parallel structures:
    - ``_handlers``: model_id -> manager object (or ``None`` if not
      attached yet)
    - ``_metadata``: model_id -> ``ModelMetadata`` instance
    - ``_extra``: model_id -> dict with VRAM and runtime metadata
    """

    def __init__(self) -> None:
        self._handlers: dict[str, ManagerProtocol | None] = {}
        self._metadata: dict[str, ModelMetadata] = {}
        # Per-model VRAM/runtime metadata stored for admin/UI surfaces.
        # This is a dynamic mapping; use `VramMetadata` for documentation of
        # the common keys but allow other runtime keys (e.g. ``_loading_task``).
        self._extra: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        # Optional notifier for activity changes. Callable receives model_id.
        self._activity_notifier: Callable[[str], None] | None = None
        logger.info("Model registry initialized")

    def register_activity_notifier(self, notifier: Callable[[str], None]) -> None:
        """Register a synchronous notifier called when active request counts change.

        The notifier should be a lightweight callable (e.g., controller.notify_activity)
        that accepts a single `model_id` string argument.
        """
        self._activity_notifier = notifier

    def register_model(
        self,
        model_id: str,
        handler: ManagerProtocol | None,
        model_type: str,
        context_length: int | None = None,
        metadata_extras: dict[str, Any] | None = None,
    ) -> None:
        """Register a model handler with metadata.

        Parameters
        ----------
        model_id
            Unique model identifier used by the registry and endpoints.
        handler
            Optional pre-attached manager/handler instance (or ``None``).
        model_type
            Human-friendly model type string.
        context_length
            Optional context length reported in metadata.
        metadata_extras
            Optional dictionary of additional metadata to merge.
        """
        if model_id in self._handlers:
            raise ValueError(f"Model '{model_id}' is already registered")

        metadata = ModelMetadata(
            id=model_id,
            type=model_type,
            context_length=context_length,
            created_at=int(time.time()),
        )

        base_metadata: dict[str, Any] = {
            "model_type": model_type,
            "context_length": context_length,
            "status": "initialized" if handler else "unloaded",
        }
        if metadata_extras:
            base_metadata.update(metadata_extras)

        base_metadata.setdefault(
            "vram_loaded",
            bool(handler and getattr(handler, "is_vram_loaded", lambda: False)()),
        )
        base_metadata.setdefault("vram_last_load_ts", None)
        base_metadata.setdefault("vram_last_unload_ts", None)
        base_metadata.setdefault("vram_last_request_ts", None)
        base_metadata.setdefault("vram_load_error", None)
        base_metadata.setdefault("active_requests", 0)

        self._handlers[model_id] = handler
        self._metadata[model_id] = metadata
        self._extra[model_id] = base_metadata

    async def update_model_state(
        self,
        model_id: str,
        *,
        handler: ManagerProtocol | None | object = _UNSET,
        status: str | None = None,
        metadata_updates: dict[str, Any] | None = None,
    ) -> None:
        """Update handler attachment and metadata for a registered model.

        Parameters
        ----------
        model_id
            Registered model identifier.
        handler
            New handler to attach (or ``None``). Use ``_UNSET`` to leave
            the handler unchanged.
        status
            Optional status string to record in the model extras.
        metadata_updates
            Optional dict of extra metadata to merge into ``_extra[model_id]``.
        """
        async with self._lock:
            if model_id not in self._metadata:
                raise KeyError(f"Model '{model_id}' not found in registry")

            entry = self._extra.setdefault(model_id, {})

            if handler is not _UNSET:
                # Handler may be ``object`` when the sentinel _UNSET is allowed;
                # cast to the manager protocol type for the registry storage.
                self._handlers[model_id] = cast("ManagerProtocol | None", handler)
                if handler is not None:
                    # refresh created_at to indicate new attachment
                    self._metadata[model_id].created_at = int(time.time())
                    # Update vram_loaded status
                    try:
                        loaded = bool(getattr(handler, "is_vram_loaded", lambda: False)())
                    except Exception:
                        loaded = False
                    entry["vram_loaded"] = loaded
                    if loaded:
                        entry["vram_last_load_ts"] = int(time.time())
                    else:
                        entry.pop("vram_last_load_ts", None)
                    entry["status"] = "loaded" if loaded else "unloaded"
                else:
                    # Handler detached
                    entry["vram_loaded"] = False
                    entry.pop("vram_last_load_ts", None)
                    entry["status"] = "unloaded"
                    entry["vram_last_unload_ts"] = int(time.time())

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

    def get_handler(self, model_id: str) -> ManagerProtocol | None:
        """Return the handler bound to ``model_id`` (may be ``None``)."""
        if model_id not in self._handlers:
            raise KeyError(f"Model '{model_id}' not found in registry")
        return self._handlers[model_id]

    def list_models(self) -> list[dict[str, Any]]:
        """Return OpenAI-compatible metadata for registered models."""
        output: list[dict[str, Any]] = []
        for mid, metadata in self._metadata.items():
            entry: dict[str, Any] = {
                "id": metadata.id,
                "object": metadata.object,
                "created": metadata.created_at,
                "owned_by": metadata.owned_by,
            }
            extra = self._extra.get(mid)
            if extra:
                entry["metadata"] = extra
            output.append(entry)
        return output

    def get_metadata(self, model_id: str) -> ModelMetadata:
        """Return the stored metadata for ``model_id``."""
        if model_id not in self._metadata:
            raise KeyError(f"Model '{model_id}' not found in registry")
        return self._metadata[model_id]

    async def get_or_attach_manager(
        self,
        model_id: str,
        loader: Callable[[str], Awaitable[ManagerProtocol]],
        *,
        timeout: float | None = None,
    ) -> ManagerProtocol:
        """Return (and attach if necessary) a manager object for ``model_id``.

        Parameters
        ----------
        model_id : str
            Registered model identifier.
        loader : Callable[[str], Awaitable[ManagerProtocol]]
            Async callable that accepts ``model_id`` and returns a manager instance.
        timeout : float | None, optional
            Optional timeout (seconds) to wait for the loader task.

        Returns
        -------
        ManagerProtocol
            The attached or newly created manager instance.

        Raises
        ------
        KeyError
            If the model is not registered.
        RuntimeError
            If the loader fails to produce a manager.
        """
        async with self._lock:
            if model_id not in self._handlers:
                raise KeyError(f"Model '{model_id}' not found in registry")

            manager = self._handlers[model_id]
            if manager is not None:
                return manager

            loading = self._extra.setdefault(model_id, {}).get("_loading_task")
            if loading is None:
                # Use ensure_future to accept any Awaitable (not only coroutines)
                task = asyncio.ensure_future(loader(model_id))
                self._extra[model_id]["_loading_task"] = task
            else:
                task = loading

        try:
            if timeout is not None:
                manager = await asyncio.wait_for(task, timeout=timeout)
            else:
                manager = await task
        except Exception as exc:
            async with self._lock:
                entry = self._extra.setdefault(model_id, {})
                entry["vram_load_error"] = str(exc)
                entry.pop("_loading_task", None)
            raise

        async with self._lock:
            self._handlers[model_id] = manager
            entry = self._extra.setdefault(model_id, {})
            entry.setdefault("active_requests", 0)
            try:
                loaded = bool(getattr(manager, "is_vram_loaded", lambda: False)())
            except Exception:
                loaded = False
            entry["vram_loaded"] = loaded
            if loaded:
                entry["vram_last_load_ts"] = int(time.time())
            entry.pop("_loading_task", None)

        return manager

    async def request_vram_load(
        self,
        model_id: str,
        *,
        force: bool = False,
        timeout: float | None = None,
    ) -> None:
        """Request that the attached manager load model weights into VRAM.

        Parameters
        ----------
        model_id : str
            Registered model identifier.
        force : bool, default False
            If True, force a reload even if the model is already marked loaded.
        timeout : float | None, optional
            Optional timeout (seconds) to wait for the manager operation.

        Raises
        ------
        KeyError
            If the model is not registered or no manager is attached.
        RuntimeError
            If the manager's load operation fails.
        """
        async with self._lock:
            if model_id not in self._handlers:
                raise KeyError(f"Model '{model_id}' not found in registry")
            manager = self._handlers[model_id]

        if manager is None:
            raise KeyError(f"No manager attached for model '{model_id}'")

        coro = manager.ensure_vram_loaded(force=force)
        if timeout is not None:
            await asyncio.wait_for(coro, timeout=timeout)
        else:
            await coro

        async with self._lock:
            entry = self._extra.setdefault(model_id, {})
            entry["vram_loaded"] = True
            entry["vram_last_load_ts"] = int(time.time())
            # Keep human-readable status in sync with VRAM residency
            entry["status"] = "loaded"
            entry.pop("vram_load_error", None)

    async def request_vram_unload(self, model_id: str, *, timeout: float | None = None) -> None:
        """Request that the attached manager release VRAM resources (unload).

        Parameters
        ----------
        model_id : str
            Registered model identifier.
        timeout : float | None, optional
            Optional timeout (seconds) to wait for the manager operation.

        Raises
        ------
        KeyError
            If the model is not registered or no manager is attached.
        RuntimeError
            If the manager's unload operation fails.
        """
        async with self._lock:
            if model_id not in self._handlers:
                raise KeyError(f"Model '{model_id}' not found in registry")
            manager = self._handlers[model_id]

        if manager is None:
            raise KeyError(f"No manager attached for model '{model_id}'")

        coro = manager.release_vram()
        if timeout is not None:
            await asyncio.wait_for(coro, timeout=timeout)
        else:
            await coro

        async with self._lock:
            entry = self._extra.setdefault(model_id, {})
            entry["vram_loaded"] = False
            entry["vram_last_unload_ts"] = int(time.time())
            # Keep human-readable status in sync with VRAM residency
            entry["status"] = "unloaded"
            entry.pop("vram_load_error", None)

    def handler_session(
        self,
        model_id: str,
        *,
        ensure_vram: bool = True,
        ensure_timeout: float | None = None,
    ) -> AbstractAsyncContextManager[ManagerProtocol]:
        """Async context manager for a per-request handler session.

        Parameters
        ----------
        model_id : str
            Registered model identifier.
        ensure_vram : bool, default True
            If True, ensures VRAM residency by calling the manager's
            ``ensure_vram_loaded`` before yielding the manager.
        ensure_timeout : float | None, optional
            Timeout for the manager's ensure_vram call.

        Yields
        ------
        ManagerProtocol
            The attached manager instance for the duration of the request.

        Notes
        -----
        The registry increments the ``active_requests`` counter on entry and
        decrements it on exit; when it drops to zero the configured activity
        notifier (if any) is invoked so the auto-unload controller can act.
        """

        @asynccontextmanager
        async def _session() -> AsyncIterator[Any]:
            async with self._lock:
                if model_id not in self._handlers:
                    raise KeyError(f"Model '{model_id}' not found in registry")
                manager = self._handlers[model_id]
                entry = self._extra.setdefault(model_id, {})
                entry["active_requests"] = entry.get("active_requests", 0) + 1
                entry["vram_last_request_ts"] = int(time.time())

            # Notify activity (reset idle timers) for this model.
            try:
                if self._activity_notifier:
                    self._activity_notifier(model_id)
            except Exception:
                # Notifier should never raise; log and continue.
                logger.exception("Activity notifier raised an exception")

            if manager is None:
                async with self._lock:
                    entry["active_requests"] = max(0, entry.get("active_requests", 1) - 1)
                raise KeyError(f"No manager attached for model '{model_id}'")

            if ensure_vram:
                coro = manager.ensure_vram_loaded()
                if ensure_timeout is not None:
                    await asyncio.wait_for(coro, timeout=ensure_timeout)
                else:
                    await coro

            try:
                yield manager
            finally:
                async with self._lock:
                    entry = self._extra.setdefault(model_id, {})
                    entry["active_requests"] = max(0, entry.get("active_requests", 1) - 1)
                    # If active requests dropped to zero, notify controller so it
                    # can begin idle countdown for this model.
                    if entry.get("active_requests", 0) == 0:
                        try:
                            if self._activity_notifier:
                                self._activity_notifier(model_id)
                        except Exception:
                            logger.exception("Activity notifier raised an exception")

        return _session()

    def get_vram_status(self, model_id: str) -> dict[str, Any]:
        """Return VRAM-related status fields for ``model_id``.

        Parameters
        ----------
        model_id : str
            Registered model identifier.

        Returns
        -------
        dict[str, Any]
            Dictionary containing the keys: ``vram_loaded``, ``vram_last_load_ts``,
            ``vram_last_unload_ts``, ``vram_last_request_ts``, ``vram_load_error``, and ``active_requests``.
        """
        if model_id not in self._extra:
            raise KeyError(f"Model '{model_id}' not found in registry")
        entry = self._extra[model_id]
        return {
            "vram_loaded": bool(entry.get("vram_loaded", False)),
            "vram_last_load_ts": entry.get("vram_last_load_ts"),
            "vram_last_unload_ts": entry.get("vram_last_unload_ts"),
            "vram_last_request_ts": entry.get("vram_last_request_ts"),
            "vram_load_error": entry.get("vram_load_error"),
            "active_requests": int(entry.get("active_requests", 0)),
        }

    def has_model(self, model_id: str) -> bool:
        """Return ``True`` when the model is registered."""
        return model_id in self._handlers

    def get_model_count(self) -> int:
        """Return how many models are registered."""
        return len(self._handlers)
