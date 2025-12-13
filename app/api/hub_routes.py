"""Dedicated FastAPI routes and helpers for hub-specific functionality."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
import contextlib
from http import HTTPStatus
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Literal, cast

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from loguru import logger

from ..const import DEFAULT_ENABLE_STATUS_PAGE, DEFAULT_HUB_CONFIG_PATH
from ..hub.config import HubConfigError, MLXHubConfig, load_hub_config
from ..schemas.openai import (
    HubControlActionResponse,
    HubControlLoadRequest,
    HubControlStatusResponse,
    HubControlUnloadRequest,
    HubGroupStatus,
    HubModelActionRequest,
    HubModelActionResponse,
    HubServiceActionResponse,
    HubStatusCounts,
    HubStatusResponse,
    Model,
)
from ..utils.errors import create_error_response
from .availability import guard_model_availability

# `is_port_available` is intentionally not used here; the daemon will perform
# its own availability checks when starting. Keep import removed to avoid
# accidental local scanning.


hub_router = APIRouter()

# Keep references to any fire-and-forget background tasks created by
# this module so they are not garbage-collected before completion.
# Tasks are removed from the set when they finish via a done callback.
_background_tasks: set[asyncio.Task[Any]] = set()


def _retain_task(task: asyncio.Task[Any]) -> None:
    """Retain a background task until completion.

    Adds ``task`` to a module-level set and registers a done-callback
    which removes the task from the set when it finishes. This ensures
    background tasks created here are not garbage-collected and thus
    are not silently cancelled.

    Parameters
    ----------
    task : asyncio.Task[Any]
        Task to retain until completion.

    Returns
    -------
    None
        This function does not return a value.
    """
    _background_tasks.add(task)

    def _on_done(t: asyncio.Task[Any]) -> None:
        # Always remove the finished task from the tracking set. Use a
        # suppress block to avoid cascading errors during cleanup.
        with contextlib.suppress(Exception):
            _background_tasks.discard(t)

        # Surface cancellations and exceptions so failures are visible in logs.
        # Use narrow handling and suppress any errors from the logger itself
        # to avoid causing further failures in the done-callback.
        if t.cancelled():
            with contextlib.suppress(Exception):
                logger.warning(f"Background task cancelled: {t!r}")
            return

        try:
            exc = t.exception()
        except asyncio.CancelledError:
            # Already handled by t.cancelled() above; nothing further to do.
            return

        if exc is not None:
            # Capture and log the full traceback for easier diagnosis of
            # background task failures (e.g., BrokenPipeError originating
            # in third-party native code). Keep logging in a suppress block
            # to avoid raising from the done-callback itself.
            tb_str = ""
            try:
                tb_str = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            except Exception:
                tb_str = f"(failed to format traceback for {type(exc).__name__})"

            with contextlib.suppress(Exception):
                logger.warning(
                    f"Background task raised exception: {t!r} -> {type(exc).__name__}: {exc}"
                )
                # Log the detailed traceback at debug level to avoid noise
                logger.debug(f"Background task exception traceback:\n{tb_str}")

    task.add_done_callback(_on_done)


# Jinja2 templates directory (project root / `templates`)
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parents[2] / "templates"))


def _resolve_hub_config_path(raw_request: Request) -> Path:
    """Resolve the hub configuration file path from request state.

    Parameters
    ----------
    raw_request : Request
        The incoming FastAPI request.

    Returns
    -------
    Path
        Path to the hub configuration file.
    """
    override = getattr(raw_request.app.state, "hub_config_path", None)
    if override:
        return Path(str(override)).expanduser()

    server_config = getattr(raw_request.app.state, "server_config", None)
    if isinstance(server_config, MLXHubConfig) and server_config.source_path is not None:
        return server_config.source_path

    source_path = getattr(server_config, "source_path", None)
    if source_path:
        return Path(str(source_path)).expanduser()

    return DEFAULT_HUB_CONFIG_PATH


def _stop_controller_process(
    raw_request: Request, background_tasks: BackgroundTasks | None = None
) -> bool:
    """Stop the in-process hub controller (if present).

    Adjusts local state by scheduling a shutdown of an in-process
    controller/supervisor when present.

    Parameters
    ----------
    raw_request : Request
        FastAPI request used to access `app.state` and locate the
        in-process controller/supervisor.

    Returns
    -------
    bool
        True if a local controller was present and a shutdown was scheduled,
        False if no in-process controller was found.
    """
    controller = getattr(raw_request.app.state, "hub_controller", None)
    if controller is None:
        controller = getattr(raw_request.app.state, "supervisor", None)

    if controller is None:
        return False

    # If a FastAPI BackgroundTasks object is provided, schedule the
    # shutdown via background tasks so it mirrors the behavior of the
    # `/hub/shutdown` endpoint and allows the response to be returned
    # to the client before the shutdown proceeds. Also schedule a
    # delayed process exit so the daemon terminates after shutting down
    # managed models.
    try:
        if background_tasks is not None:
            background_tasks.add_task(controller.shutdown_all)

            async def _shutdown_server() -> None:
                await asyncio.sleep(1)
                sys.exit(0)

            background_tasks.add_task(_shutdown_server)
        else:
            # Fallback: schedule as a retained asyncio task so it isn't
            # garbage-collected prematurely.
            task = asyncio.create_task(controller.shutdown_all())
            _retain_task(task)
    except Exception:
        logger.exception("Failed to schedule shutdown for in-process controller")
    return True


def _get_controller(raw_request: Request) -> Any | None:
    """Return the in-process hub controller or supervisor when available."""

    controller = getattr(raw_request.app.state, "hub_controller", None)
    if controller is None:
        controller = getattr(raw_request.app.state, "supervisor", None)
    return controller


def _resolve_registry_model_id(config: MLXHubConfig | None, model_name: str) -> str | None:
    """Return the registry model identifier associated with ``model_name``.

    Parameters
    ----------
    config : MLXHubConfig or None
        Hub configuration containing model definitions.
    model_name : str
        Slug or identifier provided by the hub endpoint.

    Returns
    -------
    str or None
        The corresponding registry ``model_path`` or ``None`` when unknown.
    """

    if config is None:
        return None

    target = model_name.strip()
    if not target:
        return None

    for server_cfg in getattr(config, "models", []) or []:
        candidates = [
            getattr(server_cfg, "name", None),
            getattr(server_cfg, "model_identifier", None),
            getattr(server_cfg, "model_path", None),
        ]
        if target in {value for value in candidates if isinstance(value, str)}:
            return getattr(server_cfg, "model_path", None)
    return None


def _guard_hub_action_availability(
    raw_request: Request,
    config: MLXHubConfig | None,
    model_name: str,
    *,
    deny_detail: str,
    log_context: str,
) -> JSONResponse | None:
    """Return an error when a hub action violates cached availability policies.

    Parameters
    ----------
    raw_request : Request
        Incoming FastAPI request referencing the shared registry.
    config : MLXHubConfig or None
        Hub configuration used to map hub names to registry identifiers.
    model_name : str
        Name supplied by the admin endpoint.
    deny_detail : str
        Detail string included in the rate limit response.
    log_context : str
        Human-friendly label emitted in structured logs.

    Returns
    -------
    JSONResponse or None
        ``JSONResponse`` when the model is currently unavailable, otherwise ``None``.
    """

    registry_id = _resolve_registry_model_id(config, model_name) or model_name
    return guard_model_availability(
        raw_request,
        registry_id,
        deny_detail=deny_detail,
        log_context=log_context,
    )


def _get_cached_hub_config(raw_request: Request) -> MLXHubConfig | None:
    """Return the cached hub configuration, loading it if absent.

    Parameters
    ----------
    raw_request : Request
        The incoming FastAPI request used to access application state.

    Returns
    -------
    MLXHubConfig or None
        The cached ``MLXHubConfig`` if present or successfully loaded,
        otherwise ``None`` when the configuration cannot be loaded.
    """

    config = getattr(raw_request.app.state, "hub_config", None)
    if isinstance(config, MLXHubConfig):
        return config
    with contextlib.suppress(HubConfigError):
        return _load_hub_config_from_request(raw_request)
    return None


def _load_hub_config_from_request(raw_request: Request) -> MLXHubConfig:
    """Load hub configuration from disk and cache it on ``app.state``.

    Parameters
    ----------
    raw_request : Request
        The incoming FastAPI request.

    Returns
    -------
    MLXHubConfig
        Loaded hub configuration.
    """

    config = load_hub_config(_resolve_hub_config_path(raw_request))
    setattr(raw_request.app.state, "hub_config", config)
    return config


def _hub_config_error_response(reason: str) -> JSONResponse:
    """Create a JSON error response for hub configuration errors.

    Parameters
    ----------
    reason : str
        The reason for the configuration error.

    Returns
    -------
    JSONResponse
        Formatted error response.
    """
    return JSONResponse(
        content=create_error_response(
            f"Hub configuration unavailable: {reason}",
            "invalid_request_error",
            HTTPStatus.BAD_REQUEST,
        ),
        status_code=HTTPStatus.BAD_REQUEST,
    )


def _controller_unavailable_response() -> JSONResponse:
    """Return a standardized response when the hub controller is missing.

    Returns
    -------
    JSONResponse
        Error response indicating controller unavailability.
    """
    return JSONResponse(
        content=create_error_response(
            "Hub controller is not available. Ensure the hub server is running before issuing memory actions.",
            "service_unavailable",
            HTTPStatus.SERVICE_UNAVAILABLE,
        ),
        status_code=HTTPStatus.SERVICE_UNAVAILABLE,
    )


def _controller_error_response(exc: Exception) -> JSONResponse:
    """Convert a HubControllerError into a JSON API response.

    Parameters
    ----------
    exc : HubControllerError
        The controller error to convert.

    Returns
    -------
    JSONResponse
        JSON error response.
    """
    status = getattr(exc, "status_code", HTTPStatus.INTERNAL_SERVER_ERROR)
    error_type = "invalid_request_error"
    if status == HTTPStatus.TOO_MANY_REQUESTS:
        error_type = "rate_limit_error"
    elif status >= HTTPStatus.INTERNAL_SERVER_ERROR:
        error_type = "service_unavailable"
    # If the controller raised an HTTPException (or similar) include the
    # detail in the response so clients/tests can observe controller-level
    # status and messages. For other exceptions we keep a friendly message
    # and log the full details to server logs to avoid leaking low-level IO
    # errors to the UI.
    logger.debug(f"Controller error converted to response: {exc}")
    if isinstance(exc, HTTPException):
        # fastapi.HTTPException uses .detail for human-friendly info
        detail = exc.detail
        try:
            message = f"{exc.status_code}: {detail}"
        except Exception:
            message = str(detail)
        return JSONResponse(
            content=create_error_response(message, error_type, status),
            status_code=status,
        )

    friendly_message = (
        "Controller failed to execute the requested action. See server logs for details."
    )
    return JSONResponse(
        content=create_error_response(friendly_message, error_type, status),
        status_code=status,
    )


def _registry_unavailable_response() -> JSONResponse:
    """Create a JSON error response for missing ModelRegistry on app.state."""
    return JSONResponse(
        content=create_error_response(
            "Model registry is not available. Ensure the server is configured with a registry.",
            "service_unavailable",
            HTTPStatus.SERVICE_UNAVAILABLE,
        ),
        status_code=HTTPStatus.SERVICE_UNAVAILABLE,
    )


def _registry_error_response(exc: Exception) -> JSONResponse:
    """Convert registry exceptions into JSON responses."""
    status = getattr(exc, "status_code", HTTPStatus.INTERNAL_SERVER_ERROR)
    error_type = "invalid_request_error"
    if status == HTTPStatus.TOO_MANY_REQUESTS:
        error_type = "rate_limit_error"
    elif status >= HTTPStatus.INTERNAL_SERVER_ERROR:
        error_type = "service_unavailable"
    return JSONResponse(
        content=create_error_response(str(exc), error_type, status),
        status_code=status,
    )


# Manager availability is determined directly from the in-process controller.


def _normalize_model_name(model_name: str) -> str:
    """Sanitize a model target provided via the API.

    Parameters
    ----------
    model_name : str
        The model name to normalize.

    Returns
    -------
    str
        The normalized model name.
    """
    return model_name.strip()


def _model_created_timestamp(config: MLXHubConfig) -> int:
    """Get the creation timestamp for models in the config.

    Parameters
    ----------
    config : MLXHubConfig
        The hub configuration.

    Returns
    -------
    int
        Unix timestamp of model creation.
    """
    source_path = config.source_path
    if source_path is not None and source_path.exists():
        try:
            return int(source_path.stat().st_mtime)
        except OSError:  # pragma: no cover - filesystem race
            return int(time.time())
    return int(time.time())


def _build_models_from_config(
    config: MLXHubConfig,
    live_snapshot: dict[str, Any] | None,
) -> tuple[list[Model], HubStatusCounts]:
    """Build model list and status counts from hub config and live snapshot.

    Parameters
    ----------
    config : MLXHubConfig
        The hub configuration.
    live_snapshot : dict[str, Any] | None
        Live status snapshot from the service; may be ``None`` when the
        manager is unavailable.

    Returns
    -------
    tuple[list[Model], HubStatusCounts]
        A tuple containing the rendered list of ``Model`` objects and a
        ``HubStatusCounts`` instance summarizing registered, started, and
        loaded counts.
    """
    live_entries = {}
    if live_snapshot is not None:
        models = live_snapshot.get("models")
        if isinstance(models, list):
            for entry in models:
                name = entry.get("name")
                if isinstance(name, str):
                    live_entries[name] = entry

    created_ts = _model_created_timestamp(config)
    rendered: list[Model] = []
    process_running = 0
    memory_loaded_count = 0
    for server_cfg in config.models:
        name = server_cfg.name or server_cfg.model_identifier
        live = live_entries.get(name, {})
        state = str(live.get("state") or "inactive").lower()
        # Read memory state from live entry
        memory_state_raw = live.get("memory_state") or live.get("memory")
        if memory_state_raw is not None:
            memory_state = str(memory_state_raw).lower()
        elif "memory_loaded" in live:
            memory_loaded_bool = live["memory_loaded"]
            memory_state = "loaded" if memory_loaded_bool else "unloaded"
        else:
            memory_state = None
        memory_loaded = (
            memory_state == "loaded" if memory_state is not None else (state == "running")
        )
        if state == "running":
            process_running += 1
        if memory_loaded:
            memory_loaded_count += 1
        metadata = {
            "status": state,
            "process_state": state,
            "memory_state": memory_state,
            "group": server_cfg.group,
            "default": server_cfg.is_default_model,
            "model_type": server_cfg.model_type,
            "model_path": server_cfg.model_path,
            "log_path": live.get("log_path") or server_cfg.log_file,
            "pid": live.get("pid"),
            "port": live.get("port") or server_cfg.port,
            "started_at": live.get("started_at"),
            "stopped_at": live.get("stopped_at"),
            "exit_code": live.get("exit_code"),
            "auto_unload_minutes": server_cfg.auto_unload_minutes,
            "unload_timestamp": live.get("unload_timestamp"),
        }
        rendered.append(
            Model(
                id=name,
                object="model",
                created=created_ts,
                owned_by="hub",
                metadata=metadata,
            ),
        )

    loaded_count = memory_loaded_count
    counts = HubStatusCounts(
        registered=len(rendered),
        started=process_running,
        loaded=loaded_count,
    )
    return rendered, counts


def _build_groups_from_config(
    config: MLXHubConfig,
    live_snapshot: dict[str, Any] | None,
) -> list[HubGroupStatus]:
    """Build group summaries combining config metadata and live status."""

    configured_groups = list(getattr(config, "groups", []) or [])
    if not configured_groups:
        return []

    live_lookup: dict[str, dict[str, Any]] = {}
    if live_snapshot is not None:
        groups = live_snapshot.get("groups")
        if isinstance(groups, list):
            for entry in groups:
                name = entry.get("name")
                if isinstance(name, str):
                    live_lookup[name] = entry

    members_by_group: dict[str, list[str]] = {}
    for server_cfg in getattr(config, "models", []):
        group_name = getattr(server_cfg, "group", None)
        if not group_name:
            continue
        members_by_group.setdefault(group_name, []).append(server_cfg.name or "<unnamed>")

    summaries: list[HubGroupStatus] = []
    for cfg_group in configured_groups:
        name = cfg_group.name
        live = live_lookup.get(name, {})
        live_members = live.get("models")
        configured_members = members_by_group.get(name, [])
        member_list = configured_members
        if isinstance(live_members, list) and live_members:
            member_list = [str(entry) for entry in live_members]
        summaries.append(
            HubGroupStatus(
                name=name,
                max_loaded=cfg_group.max_loaded,
                idle_unload_trigger_min=getattr(cfg_group, "idle_unload_trigger_min", None),
                loaded=int(live.get("loaded", 0) or 0),
                models=member_list,
            ),
        )
    return summaries


async def get_running_hub_models(raw_request: Request) -> set[str] | None:
    """Return the set of model names whose handlers are currently running."""

    controller = getattr(raw_request.app.state, "hub_controller", None)
    if controller is None:
        controller = getattr(raw_request.app.state, "supervisor", None)
    if controller is None:
        return None

    try:
        snapshot = await controller.get_status()
    except Exception as exc:  # pragma: no cover - controller failures surface elsewhere
        logger.debug(f"Failed to read controller status: {exc}")
        return None

    running: set[str] = set()
    models = snapshot.get("models") if isinstance(snapshot, dict) else None
    if not isinstance(models, list):
        return running

    for entry in models:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        state = str(entry.get("state") or "").lower()
        if isinstance(name, str) and state == "running":
            running.add(name)

    return running


def get_cached_model_metadata(raw_request: Request) -> dict[str, Any] | None:
    """Fetch cached model metadata from application state, if available.

    Parameters
    ----------
    raw_request : Request
        The incoming FastAPI request.

    Returns
    -------
    dict[str, Any] or None
        Cached model metadata or None.
    """
    metadata_cache = getattr(raw_request.app.state, "model_metadata", None)
    if isinstance(metadata_cache, list) and metadata_cache:
        entry = metadata_cache[0]
        if isinstance(entry, dict):
            return entry
    return None


def get_configured_model_id(raw_request: Request) -> str | None:
    """Return the configured model identifier from config or cache.

    Parameters
    ----------
    raw_request : Request
        The incoming FastAPI request.

    Returns
    -------
    str or None
        Model identifier or None.
    """
    config = getattr(raw_request.app.state, "server_config", None)
    if config is not None:
        identifier = cast("str | None", getattr(config, "model_identifier", None))
        if identifier:
            return identifier
        return getattr(config, "model_path", None)

    cached = get_cached_model_metadata(raw_request)
    if cached is not None:
        return cached.get("id")
    return None


@hub_router.get("/hub/status", response_model=HubStatusResponse)
async def hub_status(raw_request: Request) -> HubStatusResponse:
    """Return hub controller status."""

    warnings: list[str] = []
    snapshot: dict[str, Any] | None = None

    controller = getattr(raw_request.app.state, "hub_controller", None)
    if controller is None:
        controller = getattr(raw_request.app.state, "supervisor", None)

    if controller is None:
        warnings.append("Hub controller is not available.")
        controller_available = False
        config: MLXHubConfig | None = None
    else:
        controller_available = True
        with contextlib.suppress(Exception):
            await controller.reload_config()
        try:
            snapshot = await controller.get_status()
        except Exception as exc:  # pragma: no cover - controller diagnostics
            warnings.append(f"Failed to read controller status: {exc}")
        config = getattr(controller, "hub_config", None)

    if config is None:
        try:
            config = _load_hub_config_from_request(raw_request)
        except HubConfigError as exc:
            warnings.append(f"Hub configuration unavailable: {exc}")

    if config is None:
        return HubStatusResponse(
            status="degraded",
            timestamp=int(time.time()),
            host=None,
            port=None,
            models=[],
            counts=HubStatusCounts(registered=0, started=0, loaded=0),
            warnings=warnings,
            controller_available=controller_available,
        )

    models, counts = _build_models_from_config(config, snapshot)
    groups = _build_groups_from_config(config, snapshot)
    response_timestamp = int(time.time())
    if snapshot is not None:
        timestamp_value = snapshot.get("timestamp")
        if isinstance(timestamp_value, (int, float)):
            response_timestamp = int(timestamp_value)

    return HubStatusResponse(
        status="ok" if snapshot is not None else "degraded",
        timestamp=response_timestamp,
        host=config.host,
        port=config.port,
        models=models,
        counts=counts,
        warnings=warnings,
        controller_available=controller_available,
        groups=groups,
    )


@hub_router.get("/health")
async def health(raw_request: Request) -> dict[str, str]:
    """Lightweight health check for the hub daemon.

    Returns a simple 200 OK when the daemon process is running so callers
    (including CLI helpers) can detect availability quickly.

    Parameters
    ----------
    raw_request : Request
        The incoming FastAPI request (unused but kept for handler
        signature compatibility).

    Returns
    -------
    dict[str, str]
        A simple mapping containing ``{"status": "ok"}`` when healthy.
    """
    return {"status": "ok"}


@hub_router.post("/hub/reload", response_model=None)
async def hub_reload(raw_request: Request) -> dict[str, Any] | JSONResponse:
    """Reload the hub configuration inside the running daemon.

    This endpoint is the canonical implementation for ``/hub/reload`` and
    will call the in-process controller when the daemon runs in unified
    mode. When no controller is present the endpoint reports that the
    manager is unavailable.

    Parameters
    ----------
    raw_request : Request
        The incoming FastAPI request used to access application state
        and configuration.

    Returns
    -------
    dict[str, Any] or JSONResponse
        A dictionary describing the reload diff when successful, or a
        ``JSONResponse`` containing a structured error payload when the
        operation fails.
    """
    try:
        _load_hub_config_from_request(raw_request)
    except HubConfigError as exc:
        return _hub_config_error_response(str(exc))

    controller = getattr(raw_request.app.state, "hub_controller", None)
    if controller is None:
        controller = getattr(raw_request.app.state, "supervisor", None)
    if controller is None:
        return _controller_unavailable_response()

    try:
        result: dict[str, Any] = await controller.reload_config()
    except Exception as exc:
        return _controller_error_response(exc)
    return result


@hub_router.get("/hub", response_class=HTMLResponse)
async def hub_status_page(raw_request: Request) -> HTMLResponse:
    """Serve a lightweight HTML dashboard for hub operators.

    Parameters
    ----------
    raw_request : Request
        The incoming FastAPI request.

    Returns
    -------
    HTMLResponse
        HTML response with the dashboard.

    Raises
    ------
    HTTPException
        If the status page is disabled.
    """
    try:
        config = _load_hub_config_from_request(raw_request)
    except HubConfigError as e:
        # If config can't be loaded, default to disabled
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND,
            detail="Hub status page is disabled in configuration.",
        ) from e

    enabled = bool(getattr(config, "enable_status_page", DEFAULT_ENABLE_STATUS_PAGE))
    if not enabled:
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND,
            detail="Hub status page is disabled in configuration.",
        )

    # Serve the inline hub dashboard HTML directly on the main API port.
    # Do not proxy or fetch a separate page from the hub daemon; the
    # dashboard communicates with the daemon (if present) via the
    # `/hub/status` and service endpoints exposed by this API.
    context = {"request": raw_request}
    return templates.TemplateResponse("hub_status.html.jinja", context)


@hub_router.post("/hub/service/start", response_model=HubServiceActionResponse)
async def hub_service_start(raw_request: Request) -> HubServiceActionResponse | JSONResponse:
    """Start the background hub manager service if it is not already running.

    Parameters
    ----------
    raw_request : Request
        The incoming FastAPI request.

    Returns
    -------
    HubServiceActionResponse or JSONResponse
        Response indicating the result of the start action.

    Raises
    ------
    HubConfigError
        If the hub configuration cannot be loaded.
    """
    try:
        _load_hub_config_from_request(raw_request)
    except HubConfigError as exc:
        return _hub_config_error_response(str(exc))

    controller = _get_controller(raw_request)
    if controller is None:
        return _controller_unavailable_response()

    snapshot: dict[str, Any] | None = None
    try:
        with contextlib.suppress(Exception):
            await controller.reload_config()
        snapshot = await controller.get_status()
    except Exception as exc:
        return _controller_error_response(exc)

    details: dict[str, Any] = {}
    if isinstance(snapshot, dict):
        details["models"] = snapshot.get("models", [])

    return HubServiceActionResponse(
        status="ok",
        action="start",
        message="Hub controller is already running.",
        details=details,
    )


@hub_router.post("/hub/service/stop", response_model=HubServiceActionResponse)
async def hub_service_stop(
    raw_request: Request, background_tasks: BackgroundTasks
) -> HubServiceActionResponse | JSONResponse:
    """Stop the hub controller and manager service when present.

    Parameters
    ----------
    raw_request : Request
        The incoming FastAPI request.

    Returns
    -------
    HubServiceActionResponse | JSONResponse
        Response indicating the result of the stop operation.

    """
    controller = _get_controller(raw_request)
    if controller is None:
        return _controller_unavailable_response()

    controller_stopped = _stop_controller_process(raw_request, background_tasks)

    return HubServiceActionResponse(
        status="ok",
        action="stop",
        message="Hub controller shutdown requested"
        if controller_stopped
        else "Hub controller was not running",
        details={
            "controller_stopped": controller_stopped,
        },
    )


@hub_router.post("/hub/service/reload", response_model=HubServiceActionResponse)
async def hub_service_reload(raw_request: Request) -> HubServiceActionResponse | JSONResponse:
    """Reload hub.yaml inside the running manager service and return the diff.

    Parameters
    ----------
    raw_request : Request
        The incoming FastAPI request.

    Returns
    -------
    HubServiceActionResponse or JSONResponse
        Response indicating the result of the reload action.

    """
    reload_result = await hub_reload(raw_request)
    if isinstance(reload_result, JSONResponse):
        return reload_result

    return HubServiceActionResponse(
        status="ok",
        action="reload",
        message="Hub configuration reloaded",
        details=reload_result,
    )


@hub_router.post("/hub/models/{model_name}/start", response_model=HubModelActionResponse)
async def hub_start_model(
    model_name: str,
    raw_request: Request,
    payload: HubModelActionRequest | None = None,
) -> HubModelActionResponse | JSONResponse:
    """Request that the hub manager start ``model_name``.

    Parameters
    ----------
    model_name : str
        The name of the model to start.
    raw_request : Request
        The incoming request.
    payload : HubModelActionRequest, optional
        Additional payload for the request.

    Returns
    -------
    HubModelActionResponse or JSONResponse
        Response indicating the result of the start action.
    """
    _ = payload  # reserved for future compatibility
    target = _normalize_model_name(model_name)
    if not target:
        return JSONResponse(
            content=create_error_response(
                "Model name cannot be empty",
                "invalid_request_error",
                HTTPStatus.BAD_REQUEST,
            ),
            status_code=HTTPStatus.BAD_REQUEST,
        )

    # Check availability against the registry before attempting to start
    config = _get_cached_hub_config(raw_request)
    availability_error = _guard_hub_action_availability(
        raw_request,
        config,
        model_name,
        deny_detail="Model is currently unavailable",
        log_context="start",
    )
    if availability_error is not None:
        return availability_error

    controller = getattr(raw_request.app.state, "hub_controller", None)
    if controller is None:
        controller = getattr(raw_request.app.state, "supervisor", None)

    if controller is None:
        return _controller_unavailable_response()

    try:
        result = await controller.start_model(target)
        # For service-backed controllers the concrete start operation may
        # be a no-op or simply mark the model run state; ensure we also
        # trigger a VRAM load when the controller exposes that API so the
        # model becomes available to clients.
        # Only trigger schedule_vram_load for service-backed controllers
        # (i.e., not the in-process HubSupervisor) to avoid duplicating
        # behavior when the local supervisor is used in-process.
        # Do not call schedule_vram_load on an in-process HubSupervisor
        # (avoids circular import at module load time); use a name-based
        # check to detect the native supervisor implementation.
        if (
            hasattr(controller, "schedule_vram_load")
            and getattr(getattr(controller, "__class__", None), "__name__", "") != "HubSupervisor"
        ):
            # Ensure the attribute is actually an async callable (not a
            # plain MagicMock attribute created lazily) before attempting
            # to await it.
            schedule_fn = getattr(controller, "schedule_vram_load")
            is_async_callable = asyncio.iscoroutinefunction(
                schedule_fn
            ) or asyncio.iscoroutinefunction(getattr(schedule_fn, "__call__", None))
            if not is_async_callable:
                schedule_fn = None
        else:
            schedule_fn = None

        if schedule_fn is not None:
            try:
                sv_result = await schedule_fn(target, settings=None)
                # Merge fields when available (e.g., action_id/worker_port)
                if isinstance(sv_result, dict):
                    # Prefer schedule_vram_load payload when present
                    result = sv_result
            except HTTPException as exc:
                # Propagate controller surface errors as HTTP responses
                return _controller_error_response(exc)
        # Support controller implementations that return either a
        # dict-like action summary or ``None`` when the operation has
        # been accepted without additional payload (e.g., service client
        # stubs). Safely extract known keys when present.
        safe = result or {}
        status = safe.get("status", "started")
        message = f"Model {target} {status}"
        return HubModelActionResponse(
            status="ok",
            action="start",
            model=target,
            message=message,
            state=safe.get("state", "ready"),
            action_id=safe.get("action_id"),
            progress=safe.get("progress"),
            worker_port=safe.get("worker_port"),
            error=None,
        )
    except HTTPException as exc:
        return _controller_error_response(exc)
    except Exception as exc:
        logger.exception(f"Unexpected error starting model {target}: {exc}")
        return JSONResponse(
            content=create_error_response(
                f"Failed to start model: {exc}",
                "internal_error",
                HTTPStatus.INTERNAL_SERVER_ERROR,
            ),
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
        )


@hub_router.post("/hub/models/{model_name}/stop", response_model=HubModelActionResponse)
async def hub_stop_model(
    model_name: str,
    raw_request: Request,
    payload: HubModelActionRequest | None = None,
) -> HubModelActionResponse | JSONResponse:
    """Request that the hub manager stop ``model_name``.

    Parameters
    ----------
    model_name : str
        The name of the model to unload.
    raw_request : Request
        The incoming request.
    payload : HubModelActionRequest, optional
        Additional payload for the request.

    Returns
    -------
    HubModelActionResponse or JSONResponse
        Response indicating the result of the unload action.
    """
    _ = payload  # reserved for future compatibility
    return await _hub_memory_controller_action(
        raw_request,
        model_name,
        "unload",
        log_context="Hub stop",
        message_builder=_stop_action_message,
        response_action="stop",
    )


@hub_router.post("/hub/models/{model_name}/load", response_model=HubModelActionResponse)
async def hub_load_model(
    model_name: str,
    raw_request: Request,
) -> HubModelActionResponse | JSONResponse:
    """Request that the in-process controller load ``model_name`` into memory.

    Parameters
    ----------
    model_name : str
        The name of the model to load.
    raw_request : Request
        The incoming FastAPI request.

    Returns
    -------
    HubModelActionResponse or JSONResponse
        Response indicating the result of the load action.
    """
    return await _hub_memory_controller_action(
        raw_request,
        model_name,
        "load",
        log_context="Hub load",
    )


@hub_router.post("/hub/models/{model_name}/unload", response_model=HubModelActionResponse)
async def hub_unload_model(
    model_name: str,
    raw_request: Request,
) -> HubModelActionResponse | JSONResponse:
    """Request that the in-process controller unload ``model_name`` from memory.

    Parameters
    ----------
    model_name : str
        The name of the model to unload.
    raw_request : Request
        The incoming FastAPI request.

    Returns
    -------
    HubModelActionResponse or JSONResponse
        Response indicating the result of the unload action.
    """
    return await _hub_memory_controller_action(
        raw_request,
        model_name,
        "unload",
        log_context="Hub unload",
    )


@hub_router.post("/hub/models/{model_name}/vram/load", response_model=HubModelActionResponse)
async def hub_vram_load(
    model_name: str,
    raw_request: Request,
    payload: HubModelActionRequest | None = None,
) -> HubModelActionResponse | JSONResponse:
    _ = payload  # reserved for future compatibility
    return await _hub_memory_controller_action(
        raw_request,
        model_name,
        "load",
        log_context="Hub VRAM load",
        message_builder=_vram_action_message,
    )


@hub_router.post("/hub/models/{model_name}/vram/unload", response_model=HubModelActionResponse)
async def hub_vram_unload(
    model_name: str,
    raw_request: Request,
    payload: HubModelActionRequest | None = None,
) -> HubModelActionResponse | JSONResponse:
    _ = payload  # reserved for future compatibility
    return await _hub_memory_controller_action(
        raw_request,
        model_name,
        "unload",
        log_context="Hub VRAM unload",
        message_builder=_vram_action_message,
    )


@hub_router.post("/control/load", response_model=HubControlActionResponse)
async def hub_control_load(
    payload: HubControlLoadRequest,
    raw_request: Request,
) -> HubControlActionResponse | JSONResponse:
    """Schedule VRAM load via the control plane without blocking the hub."""

    model_id = _normalize_model_name(payload.model_id)
    if not model_id:
        return JSONResponse(
            content=create_error_response(
                "Model id cannot be empty",
                "invalid_request_error",
                HTTPStatus.BAD_REQUEST,
            ),
            status_code=HTTPStatus.BAD_REQUEST,
        )

    controller = getattr(raw_request.app.state, "hub_controller", None)
    if controller is None:
        controller = getattr(raw_request.app.state, "supervisor", None)
    if controller is None or not hasattr(controller, "schedule_vram_load"):
        return _controller_unavailable_response()

    registry = getattr(raw_request.app.state, "model_registry", None)

    try:
        result = await controller.schedule_vram_load(model_id, settings=payload.settings)
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        return _controller_error_response(exc)

    action_id = cast("str | None", result.get("action_id"))
    state = cast("str | None", result.get("state"))
    progress = cast("float | None", result.get("progress"))
    worker_port: int | None = None
    error: str | None = None

    if registry is not None:
        with contextlib.suppress(Exception):
            _, action = registry.get_vram_action_status(action_id=action_id, model_id=model_id)
            state = cast("str | None", action.get("vram_action_state", state))
            progress = cast("float | None", action.get("vram_action_progress", progress))
            worker_port = cast("int | None", action.get("worker_port"))
            error = cast("str | None", action.get("vram_action_error"))

    return HubControlActionResponse(
        status="accepted",
        action="vram_load",
        model=model_id,
        action_id=action_id,
        state=state,
        progress=progress,
        worker_port=worker_port,
        message=cast("str | None", result.get("message")),
        error=error,
    )


@hub_router.post("/control/unload", response_model=HubControlActionResponse)
async def hub_control_unload(
    payload: HubControlUnloadRequest,
    raw_request: Request,
) -> HubControlActionResponse | JSONResponse:
    """Schedule VRAM unload via the control plane without blocking the hub."""

    model_id = _normalize_model_name(payload.model_id)
    if not model_id:
        return JSONResponse(
            content=create_error_response(
                "Model id cannot be empty",
                "invalid_request_error",
                HTTPStatus.BAD_REQUEST,
            ),
            status_code=HTTPStatus.BAD_REQUEST,
        )

    controller = getattr(raw_request.app.state, "hub_controller", None)
    if controller is None:
        controller = getattr(raw_request.app.state, "supervisor", None)
    if controller is None or not hasattr(controller, "schedule_vram_unload"):
        return _controller_unavailable_response()

    registry = getattr(raw_request.app.state, "model_registry", None)

    try:
        result = await controller.schedule_vram_unload(model_id)
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        return _controller_error_response(exc)

    action_id = cast("str | None", result.get("action_id"))
    state = cast("str | None", result.get("state"))
    progress = cast("float | None", result.get("progress"))
    worker_port: int | None = None
    error: str | None = None

    if registry is not None:
        with contextlib.suppress(Exception):
            _, action = registry.get_vram_action_status(action_id=action_id, model_id=model_id)
            state = cast("str | None", action.get("vram_action_state", state))
            progress = cast("float | None", action.get("vram_action_progress", progress))
            worker_port = cast("int | None", action.get("worker_port"))
            error = cast("str | None", action.get("vram_action_error"))

    return HubControlActionResponse(
        status="accepted",
        action="vram_unload",
        model=model_id,
        action_id=action_id,
        state=state,
        progress=progress,
        worker_port=worker_port,
        message=cast("str | None", result.get("message")),
        error=error,
    )


@hub_router.get("/control/status", response_model=HubControlStatusResponse)
async def hub_control_status(
    raw_request: Request,
    model_id: str | None = None,
    action_id: str | None = None,
) -> HubControlStatusResponse | JSONResponse:
    """Return control-plane status for a model or action id."""

    if not model_id and not action_id:
        return JSONResponse(
            content=create_error_response(
                "model_id or action_id is required",
                "invalid_request_error",
                HTTPStatus.BAD_REQUEST,
            ),
            status_code=HTTPStatus.BAD_REQUEST,
        )

    registry = getattr(raw_request.app.state, "model_registry", None)
    if registry is None:
        return _registry_unavailable_response()

    try:
        target_model_id, action = registry.get_vram_action_status(
            model_id=model_id if model_id else None,
            action_id=action_id if action_id else None,
        )
    except Exception as exc:
        return _registry_error_response(exc)

    return HubControlStatusResponse(
        model=target_model_id,
        action_id=cast("str | None", action.get("vram_action_id")),
        state=cast("str | None", action.get("vram_action_state")),
        progress=cast("float | None", action.get("vram_action_progress")),
        error=cast("str | None", action.get("vram_action_error")),
        worker_port=cast("int | None", action.get("worker_port")),
        started_ts=cast("int | None", action.get("vram_action_started_ts")),
        updated_ts=cast("int | None", action.get("vram_action_updated_ts")),
    )


@hub_router.post("/hub/shutdown", response_model=HubServiceActionResponse)
async def hub_shutdown(
    raw_request: Request,
    background_tasks: BackgroundTasks,
) -> HubServiceActionResponse | JSONResponse:
    """Request the hub daemon to shutdown all managed models and exit.

    This endpoint forwards the shutdown request to the running hub manager
    process. If the hub manager is not running an appropriate error is returned.
    """
    # Log caller information to help diagnose accidental or spurious shutdown
    try:
        peer = getattr(raw_request, "client", None)
        if peer is not None:
            client_host = getattr(peer, "host", None)
            client_port = getattr(peer, "port", None)
            logger.info(f"Hub shutdown requested by {client_host}:{client_port}")
        else:
            logger.info("Hub shutdown requested (client info unavailable)")
    except Exception:
        logger.exception("Failed to log shutdown caller info")

    controller = _get_controller(raw_request)
    if controller is None:
        return _controller_unavailable_response()

    # Allow callers to request full process exit by providing ?exit=1 or
    # header `X-Hub-Exit: 1`. By default we will shutdown managed models
    # but keep the daemon process alive to avoid accidental termination.
    exit_param = raw_request.query_params.get("exit")
    exit_header = raw_request.headers.get("X-Hub-Exit")
    should_exit = str(exit_param or "").lower() in {"1", "true", "yes"} or str(
        exit_header or ""
    ).lower() in {"1", "true", "yes"}

    controller_stopped = _stop_controller_process(raw_request, background_tasks)
    if controller_stopped and should_exit:
        # If the caller explicitly requested process exit, schedule it.
        try:

            async def _shutdown_server() -> None:
                await asyncio.sleep(1)
                sys.exit(0)

            background_tasks.add_task(_shutdown_server)
            logger.info("Daemon process exit scheduled by caller request")
        except Exception:
            logger.exception("Failed to schedule daemon exit")
    if not controller_stopped:
        return _controller_unavailable_response()

    message = "Shutdown requested"
    if controller_stopped and not should_exit:
        message = "Shutdown requested (managed models stopped; daemon remains running)"

    return HubServiceActionResponse(
        status="ok",
        action="stop",
        message=message,
        details={},
    )


MessageBuilder = Callable[[str, str], str]


def _default_model_action_message(name: str, action: str) -> str:
    """Return the default human-readable summary for memory actions."""

    return f"Model '{name}' {action} requested"


def _vram_action_message(name: str, action: str) -> str:
    """Return the VRAM-specific summary for memory actions."""

    return f"VRAM {action} requested for '{name}'"


def _start_action_message(name: str, _action: str) -> str:
    """Return the hub start summary for scheduled VRAM loads."""

    return f"Model '{name}' start requested"


def _stop_action_message(name: str, _action: str) -> str:
    """Return the hub stop summary for scheduled VRAM unloads."""

    return f"Model '{name}' stop requested"


async def _hub_memory_controller_action(
    raw_request: Request,
    model_name: str,
    action: Literal["load", "unload"],
    *,
    message_builder: MessageBuilder | None = None,
    log_context: str = "Hub load",
    guard_deny_detail: str = "Group capacity exceeded. Unload another model or wait for auto-unload.",
    settings: dict[str, Any] | None = None,
    response_action: Literal["start", "stop", "load", "unload"] | None = None,
) -> HubModelActionResponse | JSONResponse:
    """Schedule a VRAM load/unload via the controller and return action metadata."""

    target = _normalize_model_name(model_name)
    if not target:
        return JSONResponse(
            content=create_error_response(
                "Model name cannot be empty",
                "invalid_request_error",
                HTTPStatus.BAD_REQUEST,
            ),
            status_code=HTTPStatus.BAD_REQUEST,
        )

    config = _get_cached_hub_config(raw_request)
    if action == "load":
        availability_error = _guard_hub_action_availability(
            raw_request,
            config,
            target,
            deny_detail=guard_deny_detail,
            log_context=log_context,
        )
        if availability_error is not None:
            return availability_error

    controller = getattr(raw_request.app.state, "hub_controller", None)
    if controller is None:
        controller = getattr(raw_request.app.state, "supervisor", None)

    schedule_attr = "schedule_vram_load" if action == "load" else "schedule_vram_unload"
    schedule_fn = getattr(controller, schedule_attr, None) if controller is not None else None
    if schedule_fn is None:
        return _controller_unavailable_response()

    registry = getattr(raw_request.app.state, "model_registry", None)

    try:
        if action == "load":
            schedule_result = await schedule_fn(target, settings=settings)
        else:
            schedule_result = await schedule_fn(target)
    except HTTPException as exc:  # pragma: no cover - propagate as structured error
        logger.exception(
            f"Controller rejected {action} for {target}. {type(exc).__name__}: {exc}",
        )
        return _controller_error_response(exc)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception(
            f"Unexpected failure while scheduling {action} for {target}. {type(exc).__name__}: {exc}",
        )
        return _controller_error_response(exc)

    action_id = cast("str | None", schedule_result.get("action_id"))
    state = cast("str | None", schedule_result.get("state"))
    progress = cast("float | None", schedule_result.get("progress"))
    worker_port = cast("int | None", schedule_result.get("worker_port"))
    error = cast("str | None", schedule_result.get("error"))

    if registry is not None and action_id is not None:
        with contextlib.suppress(Exception):
            _, action_meta = registry.get_vram_action_status(action_id=action_id)
            state = cast("str | None", action_meta.get("vram_action_state", state))
            progress = cast("float | None", action_meta.get("vram_action_progress", progress))
            worker_port = cast("int | None", action_meta.get("worker_port", worker_port))
            error = cast("str | None", action_meta.get("vram_action_error", error))

    effective_action = response_action or action
    builder = message_builder or _default_model_action_message
    message = cast("str | None", schedule_result.get("message")) or builder(
        target, effective_action
    )

    return HubModelActionResponse(
        status="ok",
        action=effective_action,
        model=target,
        message=message,
        action_id=action_id,
        state=state,
        progress=progress,
        worker_port=worker_port,
        error=error,
    )


__all__ = [
    "get_cached_model_metadata",
    "get_configured_model_id",
    "hub_router",
]
