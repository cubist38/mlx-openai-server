"""Model registry for managing multiple model handlers.

This module provides an asyncio-safe registry that tracks registered
models, attached per-model manager/handler objects, and VRAM-related
metadata. It exposes methods to attach managers via a loader callable,
request idempotent VRAM load/unload, and a per-request context manager
that increments/decrements an active request counter.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from contextlib import AbstractAsyncContextManager, asynccontextmanager, suppress
import time
from typing import Any, TypedDict, cast
import uuid

from loguru import logger

from ..schemas.model import ModelMetadata
from .manager_protocol import ManagerProtocol

_UNSET = object()


def build_group_policy_payload(groups: Iterable[Any] | None) -> dict[str, dict[str, Any]]:
    """Return normalized policy metadata for use with ``set_group_policies``.

    Parameters
    ----------
    groups : Iterable[Any] or None
        Sequence of config objects exposing ``name``, ``max_loaded``, and
        ``idle_unload_trigger_min`` attributes.

    Returns
    -------
    dict[str, dict[str, Any]]
        Mapping of group name to constraint metadata recognized by the registry.
    """

    policies: dict[str, dict[str, Any]] = {}
    if not groups:
        return policies

    for group in groups:
        name = getattr(group, "name", None)
        if not name:
            continue
        policies[name] = {
            "max_loaded": getattr(group, "max_loaded", None),
            "idle_unload_trigger_min": getattr(group, "idle_unload_trigger_min", None),
        }

    return policies


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
    vram_action_id : str | None
        Identifier for the latest VRAM load/unload action.
    vram_action_state : str | None
        State of the latest VRAM action (pending/loading/ready/error).
    vram_action_progress : float | None
        Progress percentage (0-100) for the latest VRAM action.
    vram_action_error : str | None
        Error message associated with the latest VRAM action, if any.
    vram_action_started_ts : int | None
        Timestamp when the latest VRAM action started.
    vram_action_updated_ts : int | None
        Timestamp when the latest VRAM action was last updated.
    worker_port : int | None
        Assigned port for the worker/sidecar serving this model, when applicable.
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
        # Group policy + availability tracking
        self._group_policies: dict[str, dict[str, Any]] = {}
        self._group_snapshots: dict[str, dict[str, Any]] = {}
        self._available_model_ids: set[str] | None = None
        self._availability_snapshot_ts: float | None = None
        logger.info("Model registry initialized")

    # ------------------------------------------------------------------
    # Group policy helpers
    # ------------------------------------------------------------------

    def set_group_policies(self, policies: dict[str, dict[str, Any]]) -> None:
        """Apply group policies used for availability decisions.

        Parameters
        ----------
        policies : dict[str, dict[str, Any]]
            Mapping of group name to policy metadata containing optional
            ``max_loaded`` and ``idle_unload_trigger_min`` keys.
        """

        normalized: dict[str, dict[str, Any]] = {}
        for name, policy in policies.items():
            if not name:
                continue
            max_loaded = self._coerce_positive_int(policy.get("max_loaded"))
            idle_trigger = self._coerce_positive_int(policy.get("idle_unload_trigger_min"))
            entry: dict[str, Any] = {
                "max_loaded": max_loaded,
                "idle_unload_trigger_min": idle_trigger,
            }
            normalized[name] = entry

        self._group_policies = normalized
        if normalized:
            self._recompute_group_availability()
        else:
            self._available_model_ids = None
            self._group_snapshots = {}
            self._availability_snapshot_ts = None

    def get_available_model_ids(self) -> set[str]:
        """Return the most recently computed set of API-visible model ids."""

        if self._available_model_ids is None:
            return set(self._handlers.keys())

        # Check if the availability snapshot is stale (older than 10 seconds)
        # This ensures idle time-based availability is updated periodically
        if self._availability_snapshot_ts is not None:
            now = time.time()
            if now - self._availability_snapshot_ts > 10.0:
                self._recompute_group_availability()

        return set(self._available_model_ids)

    def is_model_available(self, model_id: str) -> bool:
        """Return True when ``model_id`` is currently exposed to clients."""

        return model_id in self.get_available_model_ids()

    def get_group_snapshots(self) -> dict[str, dict[str, Any]]:
        """Return the latest computed per-group availability snapshot."""

        return {name: dict(data) for name, data in self._group_snapshots.items()}

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
        base_metadata.setdefault("vram_action_id", None)
        base_metadata.setdefault("vram_action_state", None)
        base_metadata.setdefault("vram_action_progress", None)
        base_metadata.setdefault("vram_action_error", None)
        base_metadata.setdefault("vram_action_started_ts", None)
        base_metadata.setdefault("vram_action_updated_ts", None)
        base_metadata.setdefault("worker_port", None)
        base_metadata.setdefault("active_requests", 0)

        self._handlers[model_id] = handler
        self._metadata[model_id] = metadata
        self._extra[model_id] = base_metadata
        if self._group_policies:
            self._recompute_group_availability()

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
                # Instrument registry updates for troubleshooting worker_port propagation
                try:
                    if "worker_port" in metadata_updates:
                        logger.debug(
                            f"registry.update_model_state called for '{model_id}' with worker_port={metadata_updates.get('worker_port')}",
                        )
                except Exception:
                    pass
                entry.update(metadata_updates)
            if status is not None:
                entry["status"] = status

        if self._group_policies:
            self._recompute_group_availability()

    async def unregister_model(self, model_id: str) -> None:
        """Remove a model from the registry."""
        async with self._lock:
            if model_id not in self._handlers:
                raise KeyError(f"Model '{model_id}' not found in registry")

            del self._handlers[model_id]
            del self._metadata[model_id]
            self._extra.pop(model_id, None)
            logger.info(f"Unregistered model: {model_id}")

        if self._group_policies:
            self._recompute_group_availability()

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
                entry["vram_action_state"] = "ready"
                entry["vram_action_progress"] = 100.0
                entry["vram_action_error"] = None
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
            entry.setdefault("vram_action_id", uuid.uuid4().hex)
            entry["vram_action_state"] = "ready"
            entry["vram_action_progress"] = 100.0
            entry["vram_action_error"] = None
            entry["vram_action_updated_ts"] = int(time.time())

        if self._group_policies:
            self._recompute_group_availability()

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
            entry.setdefault("vram_action_id", uuid.uuid4().hex)
            entry["vram_action_state"] = "ready"
            entry["vram_action_progress"] = 100.0
            entry["vram_action_error"] = None
            entry["vram_action_updated_ts"] = int(time.time())

        if self._group_policies:
            self._recompute_group_availability()

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
            ``vram_last_unload_ts``, ``vram_last_request_ts``, ``vram_load_error``,
            ``active_requests``, ``vram_action_id``, ``vram_action_state``,
            ``vram_action_progress``, ``vram_action_error``,
            ``vram_action_started_ts``, ``vram_action_updated_ts``, and ``worker_port``.
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
            "vram_action_id": entry.get("vram_action_id"),
            "vram_action_state": entry.get("vram_action_state"),
            "vram_action_progress": entry.get("vram_action_progress"),
            "vram_action_error": entry.get("vram_action_error"),
            "vram_action_started_ts": entry.get("vram_action_started_ts"),
            "vram_action_updated_ts": entry.get("vram_action_updated_ts"),
            "worker_port": entry.get("worker_port"),
        }

    async def start_vram_action(
        self,
        model_id: str,
        *,
        action_id: str | None = None,
        state: str = "pending",
    ) -> str:
        """Record the start of a VRAM load/unload action and return its id."""

        action = action_id or uuid.uuid4().hex
        now = int(time.time())
        async with self._lock:
            if model_id not in self._extra:
                raise KeyError(f"Model '{model_id}' not found in registry")
            entry = self._extra.setdefault(model_id, {})
            entry["vram_action_id"] = action
            entry["vram_action_state"] = state
            entry["vram_action_progress"] = 0.0
            entry["vram_action_error"] = None
            entry["vram_action_started_ts"] = now
            entry["vram_action_updated_ts"] = now
        return action

    async def update_vram_action(
        self,
        model_id: str,
        *,
        action_id: str | None = None,
        state: str | None = None,
        progress: float | None = None,
        error: str | None | object = _UNSET,
        worker_port: int | None | object = _UNSET,
    ) -> str:
        """Update VRAM action metadata for a model and return the action id."""

        now = int(time.time())
        async with self._lock:
            if model_id not in self._extra:
                raise KeyError(f"Model '{model_id}' not found in registry")
            entry = self._extra.setdefault(model_id, {})
            current_action = entry.get("vram_action_id")
            if action_id and current_action and action_id != current_action:
                raise ValueError(
                    f"Action id mismatch for model '{model_id}': expected {current_action}, got {action_id}"
                )
            if current_action is None and action_id is None:
                current_action = uuid.uuid4().hex
                entry["vram_action_id"] = current_action
            elif current_action is None and action_id is not None:
                current_action = action_id
                entry["vram_action_id"] = action_id

            if state is not None:
                entry["vram_action_state"] = state
            if progress is not None:
                entry["vram_action_progress"] = max(0.0, min(float(progress), 100.0))
            if error is not _UNSET:
                entry["vram_action_error"] = error
                if error is not None and state is None:
                    entry["vram_action_state"] = "error"
            if worker_port is not _UNSET:
                with suppress(Exception):
                    logger.debug(
                        f"registry.update_vram_action called for '{model_id}' with worker_port={worker_port}"
                    )
                entry["worker_port"] = worker_port

            entry["vram_action_updated_ts"] = now
            return cast("str", current_action)

    def get_vram_action_status(
        self,
        *,
        model_id: str | None = None,
        action_id: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Return VRAM action status by model_id or action_id."""

        if model_id is None and action_id is None:
            raise ValueError("model_id or action_id is required")

        target_model_id: str | None = model_id
        entry: dict[str, Any] | None = None

        if target_model_id is not None:
            entry = self._extra.get(target_model_id)
            if entry is None:
                raise KeyError(f"Model '{target_model_id}' not found in registry")
            current_action = entry.get("vram_action_id")
            if action_id is not None and current_action and action_id != current_action:
                raise KeyError(f"Action '{action_id}' not found for model '{target_model_id}'")
        else:
            for mid, data in self._extra.items():
                if data.get("vram_action_id") == action_id:
                    target_model_id = mid
                    entry = data
                    break
            if target_model_id is None or entry is None:
                raise KeyError(f"Action '{action_id}' not found")

        action_status = {
            "vram_action_id": entry.get("vram_action_id"),
            "vram_action_state": entry.get("vram_action_state"),
            "vram_action_progress": entry.get("vram_action_progress"),
            "vram_action_error": entry.get("vram_action_error"),
            "vram_action_started_ts": entry.get("vram_action_started_ts"),
            "vram_action_updated_ts": entry.get("vram_action_updated_ts"),
            "worker_port": entry.get("worker_port"),
        }
        return target_model_id, action_status

    def has_model(self, model_id: str) -> bool:
        """Return ``True`` when the model is registered."""
        return model_id in self._handlers

    def get_model_count(self) -> int:
        """Return how many models are registered."""
        return len(self._handlers)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _coerce_positive_int(value: Any) -> int | None:
        """Return ``value`` as a positive int or ``None`` when invalid."""

        if value is None:
            return None
        try:
            candidate = int(value)
        except (TypeError, ValueError):
            return None
        return candidate if candidate > 0 else None

    def _build_group_membership(self) -> dict[str, list[str]]:
        """Return mapping of group -> model ids for configured policies."""

        membership: dict[str, list[str]] = {}
        if not self._group_policies:
            return membership
        for model_id, entry in self._extra.items():
            group_name = entry.get("group")
            if not group_name or group_name not in self._group_policies:
                continue
            membership.setdefault(group_name, []).append(model_id)
        return membership

    def _estimate_idle_seconds(self, model_id: str, *, now: float | None = None) -> float | None:
        """Best-effort idle duration using request/load timestamps."""

        entry = self._extra.get(model_id)
        if not entry:
            return None
        reference = entry.get("vram_last_request_ts")
        if reference is None:
            last_unload = entry.get("vram_last_unload_ts")
            last_load = entry.get("vram_last_load_ts")
            if last_unload is None and last_load is None:
                return None
            reference = max(last_load or 0, last_unload or 0)
        try:
            reference = float(reference)
        except (TypeError, ValueError):
            return None
        current = time.time() if now is None else now
        return max(0.0, current - reference)

    def _recompute_group_availability(self) -> None:
        """Recalculate which models are available based on group policies."""

        if not self._group_policies:
            self._available_model_ids = None
            self._group_snapshots = {}
            self._availability_snapshot_ts = None
            return

        now = time.time()
        membership = self._build_group_membership()
        available: set[str] = set(self._handlers.keys())
        snapshots: dict[str, dict[str, Any]] = {}

        for group_name, members in membership.items():
            policy = self._group_policies.get(group_name) or {}
            max_loaded = self._coerce_positive_int(policy.get("max_loaded"))
            idle_trigger = self._coerce_positive_int(policy.get("idle_unload_trigger_min"))
            loaded_members = [
                mid for mid in members if bool(self._extra.get(mid, {}).get("vram_loaded"))
            ]
            idle_seconds_map: dict[str, float] = {}
            idle_eligible: list[str] = []
            if loaded_members:
                for mid in loaded_members:
                    idle_seconds = self._estimate_idle_seconds(mid, now=now)
                    if idle_seconds is not None:
                        idle_seconds_map[mid] = idle_seconds

            if idle_trigger is not None and loaded_members:
                threshold_seconds = idle_trigger * 60
                for mid, idle_seconds in idle_seconds_map.items():
                    if idle_seconds >= threshold_seconds:
                        idle_eligible.append(mid)

            allow_unloaded = True
            if max_loaded is not None and len(loaded_members) >= max_loaded:
                if idle_trigger is None or not idle_eligible:
                    allow_unloaded = False

            if not allow_unloaded:
                for mid in members:
                    if mid not in loaded_members:
                        available.discard(mid)

            snapshots[group_name] = {
                "name": group_name,
                "members": list(members),
                "loaded_members": list(loaded_members),
                "loaded": len(loaded_members),
                "max_loaded": max_loaded,
                "idle_unload_trigger_min": idle_trigger,
                "idle_eligible": idle_eligible,
                "idle_seconds": idle_seconds_map,
                "mode": "all" if allow_unloaded else "loaded-only",
                "timestamp": now,
            }

        self._available_model_ids = available
        self._group_snapshots = snapshots
        self._availability_snapshot_ts = now
