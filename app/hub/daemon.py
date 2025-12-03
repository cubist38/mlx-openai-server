"""Hub daemon scaffold: FastAPI app factory and HubSupervisor skeleton.

This module provides a non-complete but useful scaffold for the hub daemon
supervisor and HTTP control API. Implementations that require deeper
integration with model handlers should expand the supervisor methods; tests
may mock the supervisor where appropriate.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from http import HTTPStatus
import os
from pathlib import Path
import time
from typing import Any, cast

from fastapi import FastAPI, HTTPException
from loguru import logger

from ..config import MLXServerConfig
from ..const import (
    DEFAULT_API_HOST,
    DEFAULT_AUTO_UNLOAD_MINUTES,
    DEFAULT_BIND_HOST,
    DEFAULT_GROUP,
    DEFAULT_HUB_LOG_PATH,
    DEFAULT_IS_DEFAULT_MODEL,
    DEFAULT_JIT_ENABLED,
    DEFAULT_LOG_LEVEL,
    DEFAULT_MODEL_TYPE,
    DEFAULT_PORT,
)
from ..core.model_registry import ModelRegistry, build_group_policy_payload
from ..server import (
    CentralIdleAutoUnloadController,
    LazyHandlerManager,
    MLXHandler,
    configure_fastapi_app,
    configure_logging,
)
from ..utils.network import is_port_available
from .config import MLXHubConfig, MLXHubGroupConfig, load_hub_config


def create_app_with_config(config_path: str) -> FastAPI:
    """Create FastAPI app with specific config path (for uvicorn factory).

    Parameters
    ----------
    config_path : str
        Path to the hub configuration YAML to pass to :func:`create_app`.

    Returns
    -------
    FastAPI
        Configured FastAPI application instance.
    """

    return create_app(config_path)


@dataclass
class ModelRecord:
    """Record of a model's runtime state."""

    name: str
    config: Any
    manager: Any | None = None  # LazyHandlerManager
    auto_unload_minutes: int | None = None
    group: str | None = None
    is_default: bool = False
    model_path: str | None = None
    started_at: float | None = None
    exit_code: int | None = None


class HubSupervisor:
    """Supervise model handlers and runtime state.

    This manages handler lifecycle with optional JIT loading and unloading.
    """

    def __init__(
        self,
        hub_config: MLXHubConfig,
        registry: ModelRegistry | None = None,
        idle_controller: Any = None,
    ) -> None:
        """Initialize the HubSupervisor and populate model records.

        Parameters
        ----------
        hub_config : MLXHubConfig
            Parsed hub configuration containing model entries and group settings.
        registry : ModelRegistry | None, optional
            Optional model registry used to register and update model state.
        idle_controller : Any, optional
            Optional controller used to compute expected unload timestamps.

        Notes
        -----
        The constructor validates duplicate model names and builds an initial
        ``_models`` mapping of `ModelRecord` instances keyed by slug name.
        """
        self.hub_config = hub_config
        self.registry = registry
        self.idle_controller = idle_controller
        self._models: dict[str, ModelRecord] = {}
        self._lock = asyncio.Lock()
        self._bg_tasks: list[asyncio.Task[None]] = []
        self._shutdown = False

        # Populate model records from hub_config
        for model in getattr(hub_config, "models", []):
            name = getattr(model, "name", None) or str(model)
            if name in self._models:
                existing_record = self._models[name]
                raise ValueError(
                    f"Duplicate model name '{name}' detected in hub configuration. "
                    f"First model: {existing_record.config!r}, "
                    f"Duplicate model: {model!r}",
                )
            record = ModelRecord(
                name=name,
                config=model,
                group=getattr(model, "group", DEFAULT_GROUP),
                is_default=getattr(model, "is_default_model", DEFAULT_IS_DEFAULT_MODEL),
                model_path=getattr(model, "model_path", None),
                auto_unload_minutes=getattr(
                    model,
                    "auto_unload_minutes",
                    DEFAULT_AUTO_UNLOAD_MINUTES,
                ),
            )
            self._models[name] = record

    def _remove_log_sink(self, record: ModelRecord, name: str) -> None:
        """Safely remove a per-model log sink if the manager exposes it.

        Parameters
        ----------
        record : ModelRecord
            The model record whose manager may expose a log sink removal API.
        name : str
            Slug name of the model (used for debug logging).

        Notes
        -----
        This is a best-effort operation; failures are logged at debug level
        and not propagated to callers to avoid disturbing shutdown flows.
        """
        try:
            if record.manager and hasattr(record.manager, "remove_log_sink"):
                record.manager.remove_log_sink()
        except Exception:
            logger.debug(f"Failed to remove log sink for {name}")

    def _check_group_constraints(self, model_name: str) -> bool:
        """Check if loading the given model would violate group max_loaded constraints.

        Parameters
        ----------
        model_name : str
            Slug name of the target model to evaluate.

        Returns
        -------
        bool
            True if loading is allowed, False if it would violate group limits.
        """
        record = self._models[model_name]
        group_name = record.group

        group_config = self._get_group_config(group_name)

        if (
            not group_config
            or not hasattr(group_config, "max_loaded")
            or group_config.max_loaded is None
        ):
            # No constraints for this group
            return True

        max_loaded: int = group_config.max_loaded

        # Count currently loaded models in this group
        loaded_count = 0
        for other_record in self._models.values():
            if (
                other_record.group == group_name
                and other_record.manager
                and other_record.manager.is_vram_loaded()
            ):
                loaded_count += 1

        # Allow loading if we're under the limit, or if this model is already loaded
        # (in which case we're not increasing the count)
        if record.manager and record.manager.is_vram_loaded():
            return True

        return loaded_count < max_loaded

    def _get_group_config(self, group_name: str | None) -> MLXHubGroupConfig | None:
        """Return the configured group definition for ``group_name`` if present.

        Parameters
        ----------
        group_name : str | None
            Name of the group to lookup. If ``None`` returns ``None``.

        Returns
        -------
        MLXHubGroupConfig | None
            The corresponding group config or ``None`` if not found.
        """
        if not group_name:
            return None
        for group in getattr(self.hub_config, "groups", []):
            if getattr(group, "name", None) == group_name:
                return cast("MLXHubGroupConfig", group)
        return None

    def _group_violation_error(self, model_name: str) -> HTTPException:
        """Construct a standardized HTTPException for group capacity violations.

        Parameters
        ----------
        model_name : str
            The model name for which capacity was requested.

        Returns
        -------
        HTTPException
            An exception configured with HTTP 409 and an explanatory detail.
        """
        return HTTPException(
            status_code=409,
            detail=f"Loading model '{model_name}' would violate group max_loaded constraint",
        )

    async def _ensure_group_capacity(self, model_name: str) -> None:
        """Ensure there is group capacity for ``model_name`` or raise HTTPException.

        Parameters
        ----------
        model_name : str
            Name of the model being requested for loading.

        Raises
        ------
        HTTPException
            If loading the model would violate group capacity and eviction
            cannot free sufficient capacity.
        """
        if self._check_group_constraints(model_name):
            return

        record = self._models[model_name]
        group_config = self._get_group_config(record.group)
        if not group_config or group_config.max_loaded is None:
            raise self._group_violation_error(model_name)

        idle_trigger = getattr(group_config, "idle_unload_trigger_min", None)
        if idle_trigger is None:
            raise self._group_violation_error(model_name)

        evicted = await self._attempt_group_eviction(
            group_name=group_config.name,
            threshold_minutes=idle_trigger,
            exclude=model_name,
        )
        if not evicted and not self._check_group_constraints(model_name):
            raise self._group_violation_error(model_name)

    async def _attempt_group_eviction(
        self,
        *,
        group_name: str,
        threshold_minutes: int,
        exclude: str,
    ) -> bool:
        """Attempt to unload the longest-idle model meeting the threshold.

        Parameters
        ----------
        group_name : str
            Group slug to evict from.
        threshold_minutes : int
            Minimum idle time in minutes a model must exceed to be considered.
        exclude : str
            Model name to exclude from eviction consideration.

        Returns
        -------
        bool
            True if an eviction occurred, False otherwise.
        """

        threshold_seconds = max(threshold_minutes, 0) * 60
        candidates: list[tuple[float, ModelRecord]] = []
        for other_record in self._models.values():
            if other_record.name == exclude or other_record.group != group_name:
                continue
            manager = other_record.manager
            if not manager or not manager.is_vram_loaded():
                continue
            seconds_since: float | None
            try:
                seconds_since = float(manager.seconds_since_last_activity())
            except Exception:
                seconds_since = None
            if seconds_since is None or seconds_since < threshold_seconds:
                continue
            candidates.append((seconds_since, other_record))

        if not candidates:
            logger.info(
                f"Group '{group_name}' at capacity for '{exclude}'; no models idle >= {threshold_minutes} minute(s)",
            )
            return False

        candidates.sort(key=lambda item: item[0], reverse=True)
        for idle_seconds, other_record in candidates:
            manager = other_record.manager
            if manager is None:
                continue
            try:
                unloaded = await manager.unload("group-capacity")
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    f"Failed to unload '{other_record.name}' while freeing group '{group_name}' capacity: {exc}",
                )
                continue
            if unloaded:
                idle_minutes = idle_seconds / 60
                logger.info(
                    f"Unloaded '{other_record.name}' after {idle_minutes:.1f} idle minute(s) to free group "
                    f"'{group_name}' capacity for '{exclude}'",
                )
                # Update registry to reflect the unloaded state
                if self.registry and other_record.model_path is not None:
                    try:
                        await self.registry.update_model_state(
                            other_record.model_path, handler=None
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to update registry after evicting model {other_record.name}: {e}"
                        )
                return True

        logger.warning(
            f"Unable to free capacity in group '{group_name}' for '{exclude}' despite eligible candidates",
        )
        return False

    async def start_model(self, name: str) -> dict[str, Any]:
        """Load the model's handler.

        The supervisor will attempt to load the handler for the named model.

        Parameters
        ----------
        name : str
            Slug name of the model to start.

        Returns
        -------
        dict[str, Any]
            A status dictionary indicating the resulting action (e.g.,
            ``{"status": "loaded"}`` or ``{"status": "started"}``).

        Raises
        ------
        HTTPException
            If the model is not defined in the supervisor or registry updates fail.
        RuntimeError
            If registry updates cannot be performed due to missing model id.
        """
        async with self._lock:
            if name not in self._models:
                raise HTTPException(status_code=404, detail="model not found")
            record = self._models[name]

            if record.manager and record.manager.is_vram_loaded():
                return {"status": "already_loaded", "name": name}

            # Store the original registered model identifier before potentially overwriting
            original_model_path = record.model_path

            if not record.manager:
                # Create the manager. Prefer using the parsed MLXServerConfig
                # from the hub config (record.config) so fields like `log_file`
                # and `log_level` are preserved. If record.config is not already
                # an MLXServerConfig, build one and inherit hub defaults.
                if isinstance(record.config, MLXServerConfig):
                    cfg = record.config
                else:
                    cfg = MLXServerConfig(
                        model_path=str(
                            getattr(record.config, "model_path", None)
                            or getattr(record.config, "model", None)
                            or "",
                        ),
                        model_type=getattr(record.config, "model_type", None) or DEFAULT_MODEL_TYPE,
                        host=getattr(record.config, "host", DEFAULT_API_HOST),
                        port=getattr(record.config, "port", None) or 0,
                        jit_enabled=bool(
                            getattr(record.config, "jit_enabled", DEFAULT_JIT_ENABLED)
                        ),
                        auto_unload_minutes=getattr(
                            record.config,
                            "auto_unload_minutes",
                            DEFAULT_AUTO_UNLOAD_MINUTES,
                        ),
                        name=getattr(record.config, "name", None)
                        or getattr(record.config, "model_name", None),
                    )

                # Ensure model log level falls back to hub log level when unspecified
                if not getattr(cfg, "log_level", None):
                    cfg.log_level = getattr(self.hub_config, "log_level", DEFAULT_LOG_LEVEL)

                # Resolve default per-model log file when not explicitly set and
                # file logging is enabled. Use hub log path to derive the filename.
                if not getattr(cfg, "no_log_file", False) and not getattr(cfg, "log_file", None):
                    model_name = getattr(cfg, "name", None)
                    if model_name and getattr(self.hub_config, "log_path", None):
                        try:
                            cfg.log_file = str(Path(self.hub_config.log_path) / f"{model_name}.log")
                        except Exception:
                            cfg.log_file = None

                record.manager = LazyHandlerManager(cfg)
                # Preserve the original registered model id. Do not overwrite
                # `record.model_path` here — registry updates intentionally
                # prefer the original registered id (see `model_id_for_registry`
                # selection below).
                # Also emit the start event to the model-specific log sink
                try:
                    model_logger = getattr(record.manager, "_logger", None)
                    if model_logger:
                        model_logger.info(f"Started model {name} (JIT enabled, not loaded)")
                except Exception:
                    logger.debug(f"Failed to write model-specific start log for {name}")

            # Only load immediately if JIT is disabled
            if not record.manager.jit_enabled:
                await self._ensure_group_capacity(name)
                await record.manager.ensure_loaded("start")
                if self.registry:
                    # Use the registered model_id (original_model_path) for update_model_state
                    model_id_for_registry = (
                        original_model_path if original_model_path else record.model_path
                    )
                    if model_id_for_registry is None:
                        logger.error(
                            f"Registry update skipped: missing model id for registry. "
                            f"model={name!r} original_model_path={original_model_path!r} "
                            f"record.model_path={record.model_path!r} manager={getattr(record, 'manager', None)!r}"
                        )
                        raise RuntimeError(
                            f"Cannot update registry for model '{name}': model id is None"
                        )
                    await self.registry.update_model_state(
                        model_id_for_registry, handler=record.manager
                    )
                # Also emit the load event to the model-specific log sink
                try:
                    model_logger = getattr(record.manager, "_logger", None)
                    if model_logger:
                        model_logger.info(f"Loaded model {name}")
                except Exception:
                    logger.debug(f"Failed to write model-specific load log for {name}")
                return {"status": "loaded", "name": name}

            # JIT enabled: just create the manager, don't load yet
            if self.registry:
                # Use the registered model_id (original_model_path) for update_model_state
                model_id_for_registry = (
                    original_model_path if original_model_path else record.model_path
                )
                if model_id_for_registry is None:
                    logger.error(
                        f"Registry update skipped: missing model id for registry (JIT start). "
                        f"model={name!r} original_model_path={original_model_path!r} "
                        f"record.model_path={record.model_path!r} manager={getattr(record, 'manager', None)!r}"
                    )
                    raise RuntimeError(
                        f"Cannot update registry for model '{name}' (JIT start): model id is None"
                    )
                await self.registry.update_model_state(
                    model_id_for_registry, handler=record.manager
                )
            return {"status": "started", "name": name}

    async def stop_model(self, name: str) -> dict[str, Any]:
        """Stop a supervised model process.

        Parameters
        ----------
        name : str
            Slug name of the model to stop.

        Returns
        -------
        dict[str, Any]
            A status dict describing the result (not_running/stopped).

        Raises
        ------
        HTTPException
            If the model is not found.
        """
        async with self._lock:
            if name not in self._models:
                raise HTTPException(status_code=404, detail="model not found")
            record = self._models[name]
            if record.manager is None:
                return {"status": "not_running", "name": name}

            unloaded = await record.manager.unload("stop")
            if unloaded:
                # Also emit the unload event to the model-specific log sink
                try:
                    model_logger = getattr(record.manager, "_logger", None)
                    if model_logger:
                        model_logger.info(f"Unloaded model {name}")
                except Exception:
                    logger.debug(f"Failed to write model-specific unload log for {name}")
                # Remove per-model log sink if present
                self._remove_log_sink(record, name)
                record.manager = None  # Clear the manager so the model is fully stopped
                # Update registry to reflect the stopped state
                if self.registry and record.model_path is not None:
                    try:
                        await self.registry.update_model_state(record.model_path, handler=None)
                    except Exception as e:
                        logger.warning(f"Failed to update registry for stopped model {name}: {e}")
                return {"status": "stopped", "name": name}

            # If unload returned False it usually means the manager existed but
            # there was no loaded handler (JIT manager with no active process).
            # Treat a stop request as removing the manager in this case so the
            # supervisor and UI reflect the not-running state.
            # Also emit the removal to the model-specific log sink if possible
            try:
                model_logger = getattr(record.manager, "_logger", None)
                if model_logger:
                    model_logger.info(f"Removing manager for {name} (no handler loaded)")
            except Exception:
                logger.debug(f"Failed to write model-specific removal log for {name}")
            try:
                # Remove per-model log sink if present
                self._remove_log_sink(record, name)
                record.manager = None
                if self.registry and record.model_path is not None:
                    await self.registry.update_model_state(record.model_path, handler=None)
            except Exception as e:
                logger.warning(f"Failed to update registry while removing manager for {name}: {e}")
            return {"status": "stopped", "name": name}

    async def get_handler(self, name: str, reason: str = "request") -> MLXHandler | None:
        """Get the current handler for a model, loading it if necessary.

        Parameters
        ----------
        name : str
            The model name.
        reason : str, default "request"
            Reason for loading the handler (for logging/debugging).

        Returns
        -------
        MLXHandler | None
            The handler instance, or None if not found or failed to load.
        """
        async with self._lock:
            record = self._models.get(name)
            if record is None or record.manager is None:
                return None
            if not record.manager.is_vram_loaded():
                await self._ensure_group_capacity(name)
            return await record.manager.ensure_loaded(reason)

    async def acquire_handler(self, name: str, reason: str = "request") -> MLXHandler | None:
        """Acquire a handler for the given model name.

        This is an alias for :meth:`get_handler` for compatibility with the hub
        controller interface.

        Parameters
        ----------
        name : str
            The model name to acquire a handler for.
        reason : str, default "request"
            Context string passed to :meth:`get_handler`.

        Returns
        -------
        MLXHandler | None
            The acquired handler or ``None`` if unavailable.
        """
        return await self.get_handler(name, reason)

    async def load_model(self, name: str) -> dict[str, Any]:
        """Ensure a model's handler is loaded into memory.

        This loads the handler if not already loaded.

        Parameters
        ----------
        name : str
            Slug name of the model to load.

        Returns
        -------
        dict[str, Any]
            Status dict indicating the load outcome.

        Raises
        ------
        HTTPException
            If the model is not found or manager is uninitialized.
        """
        async with self._lock:
            if name not in self._models:
                raise HTTPException(status_code=404, detail="model not found")
            record = self._models[name]
            if record.manager is None:
                raise HTTPException(
                    status_code=400,
                    detail="model manager not initialized; call start_model first",
                )
            if record.manager.is_vram_loaded():
                return {"status": "already_loaded", "name": name}

            await self._ensure_group_capacity(name)
            handler = await record.manager.ensure_loaded("load")
            if handler:
                # Also emit the handler-loaded event to the model-specific log sink
                try:
                    model_logger = getattr(record.manager, "_logger", None)
                    if model_logger:
                        model_logger.info(f"Loaded model {name} handler")
                except Exception:
                    logger.debug(f"Failed to write model-specific handler load log for {name}")
                # Update registry to reflect the loaded state
                if self.registry and record.model_path is not None:
                    try:
                        await self.registry.update_model_state(
                            record.model_path, handler=record.manager
                        )
                    except Exception as e:
                        logger.warning(f"Failed to update registry for loaded model {name}: {e}")
                return {"status": "loaded", "name": name}
            # Handler loading failed - log details and raise appropriate exception
            manager_state = "unknown"
            with suppress(Exception):
                manager_state = "loaded" if record.manager.is_vram_loaded() else "not_loaded"
            logger.warning(
                f"Failed to load model handler for {name}: ensure_loaded returned None, "
                f"manager exists with state '{manager_state}'"
            )
            raise HTTPException(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                detail="failed to load model handler; check logs for details",
            )

    async def unload_model(self, name: str) -> dict[str, Any]:
        """Unload a model's handler from memory.

        This unloads the handler if loaded.

        Parameters
        ----------
        name : str
            Slug name of the model to unload.

        Returns
        -------
        dict[str, Any]
            Status dict indicating the unload outcome.
        """
        async with self._lock:
            if name not in self._models:
                raise HTTPException(status_code=404, detail="model not found")
            record = self._models[name]
            if record.manager is None:
                return {"status": "not_loaded", "name": name}
            unloaded = await record.manager.unload("unload")
            if unloaded:
                # Also emit the unload event to the model-specific log sink
                try:
                    model_logger = getattr(record.manager, "_logger", None)
                    if model_logger:
                        model_logger.info(f"Unloaded model {name} handler")
                except Exception:
                    logger.debug(f"Failed to write model-specific handler unload log for {name}")
                # Remove per-model log sink if present
                self._remove_log_sink(record, name)
                # Update registry to reflect unloaded state (but keep manager alive)
                if self.registry and record.model_path is not None:
                    try:
                        await self.registry.update_model_state(record.model_path, handler=None)
                    except Exception as e:
                        logger.warning(f"Failed to update registry for unloaded model {name}: {e}")
                return {"status": "unloaded", "name": name}

            return {"status": "not_loaded", "name": name}

    async def reload_config(self) -> dict[str, Any]:
        """Reload the hub configuration and reconcile models.

        This implementation performs a best-effort reload: it replaces the
        in-memory ``hub_config`` and returns a simple diff describing started,
        stopped, and unchanged models.

        Returns
        -------
        dict[str, Any]
            A dictionary with keys ``started``, ``stopped``, and ``unchanged``.
        """
        new_hub = load_hub_config(self.hub_config.source_path)
        old_names = set(self._models.keys())
        new_names = {getattr(m, "name", str(m)) for m in getattr(new_hub, "models", [])}

        started = list(new_names - old_names)
        stopped = list(old_names - new_names)
        unchanged = list(old_names & new_names)

        # Rebuild records for new config (do actual mutation under lock to
        # prevent races with start/stop/watch/shutdown operations).
        models_list = list(getattr(new_hub, "models", []))

        async with self._lock:
            # Preserve existing records
            old_models = dict(self._models)
            self.hub_config = new_hub
            self._models = {}
            for model in models_list:
                name = getattr(model, "name", None) or str(model)
                # Detect duplicate names in the reloaded configuration (same
                # behavior as the constructor which validates duplicates).
                if name in self._models:
                    first_existing_record = self._models[name]
                    raise ValueError(
                        f"Duplicate model name '{name}' detected in hub configuration. "
                        f"First model: {first_existing_record.config!r}, "
                        f"Duplicate model: {model!r}",
                    )
                # Preserve existing record if it exists
                existing_record = old_models.get(name)
                if existing_record:
                    # Check if model_path is changing for a started model
                    new_model_path = getattr(model, "model_path", None)
                    if (
                        existing_record.manager is not None
                        and existing_record.model_path != new_model_path
                    ):
                        raise HTTPException(
                            status_code=400,
                            detail=f"Cannot change model_path for started model '{name}' from '{existing_record.model_path}' to '{new_model_path}'. Stop the model first.",
                        )
                    # Update config but keep manager and loaded state
                    existing_record.config = model
                    existing_record.group = getattr(model, "group", DEFAULT_GROUP)
                    existing_record.is_default = getattr(
                        model,
                        "is_default_model",
                        DEFAULT_IS_DEFAULT_MODEL,
                    )
                    # Check if model_path is changing and update registry for stopped models
                    old_model_path = existing_record.model_path
                    existing_record.model_path = new_model_path
                    if (
                        old_model_path != new_model_path
                        and existing_record.manager is None
                        and self.registry is not None
                    ):
                        if old_model_path is not None:
                            try:
                                await self.registry.unregister_model(old_model_path)
                            except KeyError:
                                logger.warning(
                                    f"Failed to unregister old model path '{old_model_path}' from registry"
                                )
                        if new_model_path is not None:
                            try:
                                self.registry.register_model(
                                    model_id=new_model_path,
                                    handler=None,
                                    model_type=getattr(model, "model_type", "unknown"),
                                    context_length=getattr(model, "context_length", None),
                                    metadata_extras={
                                        "group": getattr(model, "group", None),
                                        "model_path": new_model_path,
                                    },
                                )
                            except ValueError:
                                logger.warning(
                                    f"Failed to register new model path '{new_model_path}' in registry"
                                )
                    if self.registry is not None and existing_record.model_path is not None:
                        try:
                            await self.registry.update_model_state(
                                existing_record.model_path,
                                metadata_updates={"group": existing_record.group},
                            )
                        except Exception as exc:
                            logger.warning(
                                f"Failed to update registry group metadata for '{existing_record.model_path}': {exc}"
                            )
                    existing_record.auto_unload_minutes = getattr(
                        model,
                        "auto_unload_minutes",
                        DEFAULT_AUTO_UNLOAD_MINUTES,
                    )
                    # If auto_unload_minutes changed and model is loaded, recalculate unload_timestamp
                    record = existing_record
                else:
                    record = ModelRecord(
                        name=name,
                        config=model,
                        group=getattr(model, "group", DEFAULT_GROUP),
                        is_default=getattr(model, "is_default_model", DEFAULT_IS_DEFAULT_MODEL),
                        model_path=getattr(model, "model_path", None),
                        auto_unload_minutes=getattr(
                            model,
                            "auto_unload_minutes",
                            DEFAULT_AUTO_UNLOAD_MINUTES,
                        ),
                    )
                    # Register new model with registry if available
                    if self.registry is not None:
                        model_id = getattr(model, "model_path", None)
                        if model_id is None:
                            raise ValueError(f"Model {model} has no model_path")
                        try:
                            self.registry.register_model(
                                model_id=model_id,
                                handler=None,
                                model_type=getattr(model, "model_type", "unknown"),
                                context_length=getattr(model, "context_length", None),
                                metadata_extras={
                                    "group": getattr(model, "group", None),
                                    "model_path": model_id,
                                },
                            )
                        except ValueError:
                            logger.warning(
                                f"Failed to register new model path '{model_id}' in registry"
                            )
                self._models[name] = record

            logger.debug(f"Reloaded hub config: started={started} stopped={stopped}")

        if self.registry is not None:
            self.registry.set_group_policies(
                build_group_policy_payload(getattr(self.hub_config, "groups", None))
            )

        return {"started": started, "stopped": stopped, "unchanged": unchanged}

    async def get_status(self) -> dict[str, Any]:
        """Return a serializable snapshot of supervisor state.

        The returned dict includes a ``timestamp`` and a ``models`` list where
        each model object contains keys used by the CLI and status UI.

        Returns
        -------
        dict[str, Any]
            Snapshot payload containing model runtime summaries and group info.
        """
        snapshot: dict[str, Any] = {
            "timestamp": time.time(),
            "models": [],
        }

        group_configs = {
            cast("str", getattr(group, "name", None)): group
            for group in getattr(self.hub_config, "groups", [])
            if getattr(group, "name", None)
        }
        group_state: dict[str, dict[str, Any]] = {}

        # Take a brief, locked snapshot of the model records to avoid races
        # while allowing the lock to be held only for a short time.
        async with self._lock:
            # Capture manager state while holding the lock to avoid TOCTOU races
            models_snapshot = []
            for name, rec in self._models.items():
                manager_ref = rec.manager
                manager_is_vram_loaded = manager_ref.is_vram_loaded() if manager_ref else False
                models_snapshot.append(
                    {
                        "name": name,
                        "record": rec,
                        "manager": manager_ref,
                        "is_vram_loaded": manager_is_vram_loaded,
                    }
                )

        for item in models_snapshot:
            name = cast("str", item["name"])
            rec = cast("ModelRecord", item["record"])
            manager = cast("LazyHandlerManager | None", item["manager"])
            is_vram_loaded = cast("bool", item["is_vram_loaded"])

            unload_timestamp = None
            if is_vram_loaded and rec.auto_unload_minutes:
                # Get unload timestamp from the idle controller if available
                if self.idle_controller and hasattr(
                    self.idle_controller, "get_expected_unload_timestamp"
                ):
                    # Use model_path as the registry model_id, fallback to name if not available
                    model_id = rec.model_path or name
                    unload_timestamp = self.idle_controller.get_expected_unload_timestamp(model_id)

            state = "running" if manager else "stopped"
            snapshot["models"].append(
                {
                    "name": name,
                    "state": state,
                    "pid": None,
                    "port": None,
                    "started_at": rec.started_at,
                    "exit_code": rec.exit_code,
                    "memory_loaded": is_vram_loaded,
                    "group": rec.group,
                    "is_default_model": rec.is_default,
                    "model_path": rec.model_path,
                    "auto_unload_minutes": rec.auto_unload_minutes,
                    "unload_timestamp": unload_timestamp,
                },
            )

            group_name = rec.group
            if not group_name:
                continue
            cfg = group_configs.get(group_name)
            entry = group_state.setdefault(
                group_name,
                {
                    "name": group_name,
                    "max_loaded": getattr(cfg, "max_loaded", None) if cfg else None,
                    "idle_unload_trigger_min": getattr(
                        cfg,
                        "idle_unload_trigger_min",
                        None,
                    )
                    if cfg
                    else None,
                    "loaded": 0,
                    "models": [],
                },
            )
            entry["models"].append(name)
            if is_vram_loaded:
                entry["loaded"] += 1

        for group_name, cfg in group_configs.items():
            group_state.setdefault(
                group_name,
                {
                    "name": group_name,
                    "max_loaded": getattr(cfg, "max_loaded", None),
                    "idle_unload_trigger_min": getattr(cfg, "idle_unload_trigger_min", None),
                    "loaded": 0,
                    "models": [],
                },
            )

        if group_state:
            snapshot["groups"] = sorted(group_state.values(), key=lambda entry: entry["name"])
        return snapshot

    def add_background_task(self, task: asyncio.Task[None]) -> None:
        """Add a background task to be tracked by the supervisor.

        Parameters
        ----------
        task : asyncio.Task[None]
            Task to track; it will be removed when completed via callbacks.
        """
        self._bg_tasks.append(task)

    def remove_background_task(self, task: asyncio.Task[None]) -> None:
        """Remove a background task from tracking.

        Parameters
        ----------
        task : asyncio.Task[None]
            Task previously added with :meth:`add_background_task`.
        """
        self._bg_tasks.remove(task)

    async def shutdown_all(self) -> None:
        """Gracefully stop all supervised model processes.

        This performs a best-effort shutdown of each supervised process and
        logs failures without raising to the caller.
        """
        logger.info("Shutting down all supervised model handlers")
        async with self._lock:
            names = list(self._models.keys())
        for name in names:
            try:
                await self.stop_model(name)
            except Exception as e:  # pragma: no cover - best-effort shutdown
                logger.exception(f"Error stopping model {name}: {e}")


async def _schedule_default_model_starts(supervisor: HubSupervisor) -> None:
    """Schedule auto-start tasks for default models.

    Parameters
    ----------
    supervisor : HubSupervisor
        Supervisor instance used to start model handlers and track background
        tasks.

    Returns
    -------
    None
        This function schedules asynchronous background tasks and does not
        return a value.

    Notes
    -----
    The function iterates configured models and schedules an auto-start task
    for each model marked as a default. Errors during scheduling or during
    individual auto-start attempts are logged and do not interrupt other
    scheduling operations.
    """
    try:
        # Iterate the configured models so we can read their JIT flag
        for model in getattr(supervisor.hub_config, "models", []):
            try:
                if not getattr(model, "is_default_model", False):
                    continue
                name = getattr(model, "name", None)
                if not name:
                    continue

                async def _autostart(cfg_model: Any) -> None:
                    mname = getattr(cfg_model, "name", "<unknown>")
                    try:
                        # Always attempt to start a supervised handler
                        # for default models. The handler will decide
                        # whether to eagerly load based on JIT configuration.
                        await supervisor.start_model(mname)
                        logger.info(f"Auto-started model handler: {mname}")
                    except Exception as e:  # pragma: no cover - best-effort
                        # Starting the supervised handler failed; log the failure and continue.
                        logger.warning(f"Failed to auto-start model handler '{mname}': {e}")

                t = asyncio.create_task(_autostart(model))
                supervisor.add_background_task(t)

                def _cleanup_task(
                    fut: asyncio.Task[None],
                    model_name: str = getattr(model, "name", "<unknown>"),
                ) -> None:
                    supervisor.remove_background_task(fut)
                    if fut.exception():
                        logger.exception(
                            f"Autostart task failed for {model_name}",
                            exc_info=fut.exception(),
                        )

                t.add_done_callback(_cleanup_task)
                logger.info(f"Scheduled auto-start for default model: {name}")
            except Exception as e:  # pragma: no cover - best-effort
                logger.warning(f"Failed to schedule auto-start for model entry {model!r}: {e}")
    except Exception as e:  # pragma: no cover - defensive
        logger.exception(f"Error while scheduling default model starts: {e}")


def create_app(hub_config_path: str | None = None) -> FastAPI:
    """Create and configure the FastAPI application for the hub daemon.

    Parameters
    ----------
    hub_config_path : str | None
        Optional path to the hub YAML configuration. When None, the default
        path from `app/hub/config.py` is used, or the path from the
        MLX_HUB_CONFIG_PATH environment variable if set.

    Returns
    -------
    FastAPI
        Configured FastAPI app instance with supervisor attached at
        `app.state.supervisor`.
    """

    # Create FastAPI app (lifespan will be set after hub_config and supervisor are created)
    app = FastAPI(title="mlx hub daemon")

    # Configure templates directory (fall back to inline rendering if missing)
    # Templates folder lives at the repository root `templates/`

    # Use environment variable if no path provided
    if hub_config_path is None:
        hub_config_path = os.environ.get("MLX_HUB_CONFIG_PATH")

    hub_config = load_hub_config(hub_config_path)

    # Cache the loaded hub configuration on the FastAPI app state so
    # in-process endpoints resolve the same hub config path that the
    # daemon started with. This prevents mismatches when the daemon
    # and the in-process API use different resolution heuristics.
    try:
        app.state.server_config = hub_config
        # Also store the resolved path when available for compatibility with
        # other resolution helpers that prefer a Path on ``app.state``.
        try:
            resolved_path = getattr(hub_config, "source_path", None) or hub_config_path
            if resolved_path is not None:
                app.state.hub_config_path = Path(resolved_path)
        except Exception:
            # Best-effort: do not fail startup due to state caching issues
            pass
    except Exception:
        # Avoid letting state caching errors stop the daemon; log at debug
        # level and continue with the loaded config.
        with suppress(Exception):
            logger.debug("Failed to cache hub config on app.state")

    # Ensure hub log path exists and configure logging for hub-level logs
    # Prefer the configured hub log path; fall back to DEFAULT_HUB_LOG_PATH
    try:
        hub_log_path = Path(getattr(hub_config, "log_path", DEFAULT_HUB_LOG_PATH))
    except Exception:
        hub_log_path = Path(DEFAULT_HUB_LOG_PATH)
    # Expand user (~) if present
    try:
        hub_log_path = hub_log_path.expanduser()
    except Exception:
        hub_log_path = Path.cwd() / "logs"
    with suppress(Exception):
        hub_log_path.mkdir(parents=True, exist_ok=True)
    try:
        configure_logging(
            log_file=str(hub_log_path / "app.log"),
            no_log_file=False,
            log_level=getattr(hub_config, "log_level", "INFO"),
        )
    except Exception:
        logger.warning("Failed to configure hub logging; continuing with defaults")

    # Create and populate model registry
    registry = ModelRegistry()
    app.state.model_registry = registry
    for model in getattr(hub_config, "models", []):
        model_id = getattr(model, "model_path", None)
        if model_id is None:
            raise ValueError(f"Model {model} has no model_path")
        registry.register_model(
            model_id=model_id,
            handler=None,  # Will be set when started
            model_type=getattr(model, "model_type", "unknown"),
            context_length=getattr(model, "context_length", None),
            metadata_extras={
                "group": getattr(model, "group", None),
                "model_path": model_id,
            },
        )

    registry.set_group_policies(build_group_policy_payload(getattr(hub_config, "groups", None)))

    supervisor = HubSupervisor(hub_config, registry)
    app.state.supervisor = supervisor
    app.state.hub_controller = supervisor

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        # Startup
        logger.info("Hub daemon starting up")

        # Check if the configured port is available
        configured_port = getattr(hub_config, "port", DEFAULT_PORT)
        configured_host = getattr(hub_config, "host", DEFAULT_BIND_HOST)
        if not is_port_available(port=configured_port, host=configured_host):
            raise RuntimeError(
                f"Port {configured_port} is already in use on host {configured_host}. Please stop the service using that port.",
            )

        # Auto-start models marked as default in configuration
        await _schedule_default_model_starts(supervisor)

        # Start central auto-unload controller for hub models
        central_controller = CentralIdleAutoUnloadController(registry)
        registry.register_activity_notifier(central_controller.notify_activity)
        central_controller.start()

        # Give the supervisor access to the controller for unload timestamp calculations
        supervisor.idle_controller = central_controller

        # Yield to start serving requests while background tasks proceed
        yield
        # Shutdown
        logger.info("Hub daemon shutting down")
        await central_controller.stop()
        await supervisor.shutdown_all()
        # Remove transient CLI runtime state file if present so future CLI
        # invocations don't pick up a stale PID/port. The runtime file lives
        # under the hub log path and is named `hub_runtime.json`.
        try:
            runtime_file = (
                Path(getattr(hub_config, "log_path", Path.cwd() / "logs")) / "hub_runtime.json"
            )
            if runtime_file.exists():
                try:
                    runtime_file.unlink()
                    logger.debug(f"Removed hub runtime state file: {runtime_file}")
                except Exception as e:  # pragma: no cover - best-effort cleanup
                    logger.warning(f"Failed to remove hub runtime state file {runtime_file}: {e}")
        except Exception:
            # Defensive: do not allow cleanup failures to raise during shutdown
            logger.debug("Error while attempting to clean up runtime state file")

    # Set the lifespan on the FastAPI app now that hub_config and supervisor are available
    app.router.lifespan_context = lifespan

    # Configure OpenAI API routes, middleware, and hub admin routes
    # The daemon mounts the canonical `hub_router` so there is a single
    # implementation for all `/hub/...` endpoints. This avoids duplicate
    # handlers and ensures consistent behavior between in-process and daemon modes.
    configure_fastapi_app(app, include_hub_routes=True)

    return app
