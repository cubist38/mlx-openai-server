"""Dedicated FastAPI routes and helpers for hub-specific functionality."""

from __future__ import annotations

import asyncio
import contextlib
from http import HTTPStatus
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import IO, Any, Literal, cast

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import httpx
from loguru import logger

from ..const import (
    DEFAULT_API_HOST,
    DEFAULT_BIND_HOST,
    DEFAULT_HUB_CONFIG_PATH,
    DEFAULT_MODEL_STARTING_PORT,
)
from ..hub.config import HubConfigError, MLXHubConfig, load_hub_config
from ..schemas.openai import (
    HubModelActionRequest,
    HubModelActionResponse,
    HubServiceActionResponse,
    HubStatusCounts,
    HubStatusResponse,
    Model,
)
from ..utils.errors import create_error_response

# `is_port_available` is intentionally not used here; the daemon will perform
# its own availability checks when starting. Keep import removed to avoid
# accidental local scanning.


class HubServiceError(RuntimeError):
    """Raised when the daemon reports a service-level failure.

    Parameters
    ----------
    message : str
        Human-friendly error message.
    status_code : int | None
        Optional HTTP status code associated with the error.
    """

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def start_hub_service_process(
    config_path: str,
    *,
    host: str | None = None,
    port: int | None = None,
) -> int:
    """Start the hub daemon process (development helper).

    This helper launches a uvicorn process in the background using the same
    Python interpreter. It returns the spawned PID. It is intended as a
    development convenience for the API's `/hub/service/start` endpoint.
    """
    host_val = host or DEFAULT_BIND_HOST
    # Use the configured port directly. If the port is unavailable the
    # daemon's own startup checks (`create_app` / is_port_available) will
    # raise a clear error; avoiding local scanning prevents the CLI from
    # starting the daemon on a different port than the configuration.
    starting_port = port or DEFAULT_MODEL_STARTING_PORT
    port_val = starting_port

    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "app.hub.daemon:create_app",
        "--factory",
        "--host",
        host_val,
        "--port",
        str(port_val),
    ]

    # Set environment variable for the hub daemon to use the specified config
    env = os.environ.copy()
    env["MLX_HUB_CONFIG_PATH"] = config_path

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)

    # Start background threads to log subprocess output
    def _log_output(stream: IO[bytes], level: str, prefix: str) -> None:
        """Log output from subprocess stream."""
        try:
            for line in iter(stream.readline, b""):
                line_str = line.decode("utf-8", errors="replace").rstrip()
                if line_str:
                    if level == "info":
                        logger.info(f"{prefix}: {line_str}")
                    elif level == "error":
                        logger.error(f"{prefix}: {line_str}")
                    else:
                        logger.debug(f"{prefix}: {line_str}")
        except Exception as e:
            logger.warning(f"Error reading subprocess {prefix} output: {e}")

    # Start threads to read stdout and stderr
    if proc.stdout:
        stdout_thread = threading.Thread(
            target=_log_output,
            args=(proc.stdout, "info", f"hub-daemon[{proc.pid}].stdout"),
            daemon=True,
        )
        stdout_thread.start()

    if proc.stderr:
        stderr_thread = threading.Thread(
            target=_log_output,
            args=(proc.stderr, "error", f"hub-daemon[{proc.pid}].stderr"),
            daemon=True,
        )
        stderr_thread.start()

    return proc.pid


hub_router = APIRouter()

# Keep references to any fire-and-forget background tasks created by
# this module so they are not garbage-collected before completion.
# Tasks are removed from the set when they finish via a done callback.
_background_tasks: set[asyncio.Task[Any]] = set()


def _retain_task(task: asyncio.Task[Any]) -> None:
    """Add `task` to the module-level set and register a callback to remove it when complete.

    This ensures background tasks created here are retained until they
    finish and avoids silent cancellation via GC.
    """
    _background_tasks.add(task)

    def _on_done(t: asyncio.Task[Any]) -> None:
        with contextlib.suppress(Exception):
            _background_tasks.discard(t)

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


def _load_hub_config_from_request(raw_request: Request) -> MLXHubConfig:
    """Load hub configuration from the resolved config path.

    Parameters
    ----------
    raw_request : Request
        The incoming FastAPI request.

    Returns
    -------
    MLXHubConfig
        Loaded hub configuration.
    """
    return load_hub_config(_resolve_hub_config_path(raw_request))


def _daemon_base_url(config: MLXHubConfig) -> str:
    """Return the base HTTP URL for the hub daemon for the given config."""
    host = (config.host or DEFAULT_BIND_HOST).strip()
    if host in {"0.0.0.0", "::", "[::]"}:
        host = DEFAULT_API_HOST
    # Normalize IPv6: trim any surrounding brackets if present, then add brackets once if IPv6
    while host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    if ":" in host:
        host = f"[{host}]"
    port = config.port
    return f"http://{host}:{port}"


async def _call_daemon_api_async(
    config: MLXHubConfig,
    method: str,
    path: str,
    *,
    json: object | None = None,
    timeout: float = 5.0,
) -> dict[str, object] | None:
    """Async call to the hub daemon HTTP API and return parsed JSON.

    Raises HubServiceError on non-2xx responses.
    """
    base = _daemon_base_url(config)
    url = f"{base.rstrip('/')}{path}"
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.request(method, url, json=json)
    except Exception as e:  # pragma: no cover - network error handling
        raise HubServiceError(f"Failed to contact hub daemon at {base}: {e}") from e

    if resp.status_code >= 400:
        # Try to parse JSON error
        payload: object | None
        try:
            payload = resp.json()
        except Exception:
            payload = resp.text
        raise HubServiceError(
            f"Daemon responded {resp.status_code}: {payload}",
            status_code=resp.status_code,
        )

    if resp.content:
        try:
            payload = resp.json()
        except Exception:
            return {"raw": resp.text}
        if isinstance(payload, dict):
            return payload
        # Wrap non-dict payloads in a dict to maintain the declared return type
        return {"raw": payload}
    return None


def _call_daemon_api_sync(
    config: MLXHubConfig,
    method: str,
    path: str,
    *,
    json: object | None = None,
    timeout: float = 1.0,
) -> dict[str, object] | None:
    """Synchronous HTTP call to the hub daemon used by sync code paths.

    Raises
    ------
    HubServiceError
        On connectivity failures or non-2xx responses.
    """
    base = _daemon_base_url(config)
    url = f"{base.rstrip('/')}{path}"
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.request(method, url, json=json)
    except Exception as e:  # pragma: no cover - network error handling
        raise HubServiceError(f"Failed to contact hub daemon at {base}: {e}") from e

    if resp.status_code >= 400:
        try:
            payload = resp.json()
        except Exception:
            payload = resp.text
        raise HubServiceError(
            f"Daemon responded {resp.status_code}: {payload}",
            status_code=resp.status_code,
        )

    if resp.content:
        try:
            payload = resp.json()
        except Exception:
            return {"raw": resp.text}
        if isinstance(payload, dict):
            return payload
        # Wrap non-dict payloads in a dict to maintain the declared return type
        return {"raw": payload}
    return None


def _service_error_response(action: str, exc: HubServiceError) -> JSONResponse:
    """Create a JSON error response for hub service errors.

    Parameters
    ----------
    action : str
        The action that failed.
    exc : HubServiceError
        The service error exception.

    Returns
    -------
    JSONResponse
        Formatted error response.
    """
    status = exc.status_code or HTTPStatus.SERVICE_UNAVAILABLE
    error_type = (
        "rate_limit_error" if status == HTTPStatus.TOO_MANY_REQUESTS else "service_unavailable"
    )
    # Log the full exception for diagnostics but avoid exposing low-level
    # transport errors (e.g. BrokenPipeError) directly to the UI.
    logger.debug(f"Hub service error while attempting {action}: {exc}")
    friendly_message = f"Failed to {action} via hub manager. See server logs for details."
    return JSONResponse(
        content=create_error_response(
            friendly_message,
            error_type,
            status,
        ),
        status_code=status,
    )


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


def _manager_unavailable_response() -> JSONResponse:
    """Create a JSON error response for unavailable hub manager.

    Returns
    -------
    JSONResponse
        Formatted error response.
    """
    return JSONResponse(
        content=create_error_response(
            "Hub manager is not running. Start it via /hub/service/start or the CLI before issuing actions.",
            "service_unavailable",
            HTTPStatus.SERVICE_UNAVAILABLE,
        ),
        status_code=HTTPStatus.SERVICE_UNAVAILABLE,
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


# Manager availability is determined by proxying to the hub daemon's
# `/health` endpoint using `_call_daemon_api_async` where appropriate.


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
    live_snapshot : dict[str, Any] or None
        Live status snapshot from the service.
    runtime : HubRuntime, optional
        Runtime reference used to enrich metadata with memory lifecycle states.

    Returns
    -------
    tuple[list[Model], HubStatusCounts]
        Models list and status counts.
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


def get_running_hub_models(raw_request: Request) -> set[str] | None:
    """Return the set of model names whose processes are currently running.

    Parameters
    ----------
    raw_request : Request
        FastAPI request containing hub server state.

    Returns
    -------
    set[str] | None
        Names of running models, or ``None`` when the service is unavailable.
    """
    server_config = getattr(raw_request.app.state, "server_config", None)
    if not isinstance(server_config, MLXHubConfig):
        return None

    try:
        config = _load_hub_config_from_request(raw_request)
    except HubConfigError:
        return None

    try:
        snapshot = _call_daemon_api_sync(config, "GET", "/hub/status", timeout=1.0)
    except HubServiceError as e:
        logger.debug(
            f"Hub manager status unavailable; skipping running model filter. {type(e).__name__}: {e}",
        )
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
    """Return hub status derived from the hub manager service when available.

    Returns
    -------
    HubStatusResponse
        The hub status response.
    """
    warnings: list[str] = []
    snapshot: dict[str, Any] | None = None
    config: MLXHubConfig | None = None

    # Check if controller is available directly (unified daemon mode)
    controller = getattr(raw_request.app.state, "hub_controller", None)
    if controller is None:
        controller = getattr(raw_request.app.state, "supervisor", None)

    if controller is not None:
        # Use controller directly in unified daemon mode
        try:
            # Ensure config is up to date
            with contextlib.suppress(Exception):
                await controller.reload_config()
            config = controller.hub_config
            snapshot = await controller.get_status()
        except Exception as e:
            warnings.append(f"Failed to get status from controller: {e}")
            # Fallback: try to load config manually
            try:
                config = _load_hub_config_from_request(raw_request)
            except HubConfigError as e2:
                warnings.append(f"Hub configuration unavailable: {e2}")
    else:
        # Fall back to daemon API calls (legacy mode)
        try:
            config = _load_hub_config_from_request(raw_request)
            # Try to reconcile via daemon then fetch status snapshot
            with contextlib.suppress(HubServiceError):
                await _call_daemon_api_async(config, "POST", "/hub/reload")
            snapshot = await _call_daemon_api_async(config, "GET", "/hub/status")
        except (HubConfigError, HubServiceError) as e:
            warnings.append(f"Hub manager unavailable: {e}")

    if config is None:
        return HubStatusResponse(
            status="degraded",
            timestamp=int(time.time()),
            host=None,
            port=None,
            models=[],
            counts=HubStatusCounts(registered=0, started=0, loaded=0),
            warnings=warnings,
            controller_available=controller is not None,
        )

    models, counts = _build_models_from_config(config, snapshot)
    response_timestamp = int(time.time())
    if snapshot is not None:
        timestamp_value = snapshot.get("timestamp")
        if isinstance(timestamp_value, (int, float)):
            response_timestamp = int(timestamp_value)

    controller_available = controller is not None

    return HubStatusResponse(
        status="ok" if snapshot is not None else "degraded",
        timestamp=response_timestamp,
        host=config.host,
        port=config.port,
        models=models,
        counts=counts,
        warnings=warnings,
        controller_available=controller_available,
    )


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

    enabled = bool(getattr(config, "enable_status_page", False))
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
        config = _load_hub_config_from_request(raw_request)
    except HubConfigError as exc:
        return _hub_config_error_response(str(exc))

    # If daemon is already responding, return early
    try:
        await _call_daemon_api_async(config, "GET", "/health", timeout=1.0)
        return HubServiceActionResponse(
            status="ok",
            action="start",
            message="Hub manager already running.",
        )
    except HubServiceError:
        pass

    if config.source_path is None:
        return _hub_config_error_response(
            "Hub configuration must be saved to disk before starting the manager.",
        )

    pid = start_hub_service_process(str(config.source_path), host=config.host, port=config.port)

    # Wait for daemon to become available
    deadline = time.time() + 10.0
    available = False
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        try:
            await _call_daemon_api_async(config, "GET", "/health", timeout=1.0)
            available = True
            break
        except HubServiceError:
            remaining_seconds = deadline - time.time()
            logger.info(
                f"Waiting for hub daemon to start... attempt {attempts}, {remaining_seconds:.1f}s remaining",
            )
            await asyncio.sleep(0.25)

    if not available:
        return JSONResponse(
            content=create_error_response(
                "Hub manager failed to start within 10 seconds.",
                "service_unavailable",
                HTTPStatus.SERVICE_UNAVAILABLE,
            ),
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
        )

    try:
        await _call_daemon_api_async(config, "POST", "/hub/reload")
        snapshot = await _call_daemon_api_async(config, "GET", "/hub/status")
    except HubServiceError as e:
        snapshot = None
        logger.warning(f"Hub manager started (pid={pid}) but status fetch failed: {e}")

    details: dict[str, Any] = {"pid": pid}
    if snapshot is not None:
        details["models"] = snapshot.get("models", [])
    return HubServiceActionResponse(
        status="ok",
        action="start",
        message="Hub manager started",
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

    Raises
    ------
    HubConfigError
        If the hub configuration cannot be loaded.
    HubServiceError
        If there is an error communicating with the hub service.
    """
    try:
        config = _load_hub_config_from_request(raw_request)
    except HubConfigError as exc:
        return _hub_config_error_response(str(exc))

    # Stop any in-process controller (schedules shutdown via BackgroundTasks
    # so response can be returned before shutdown proceeds). This mirrors the
    # behavior of the `/hub/shutdown` endpoint.
    controller_stopped = _stop_controller_process(raw_request, background_tasks)
    manager_shutdown = controller_stopped  # In unified mode, controller is the manager

    if not controller_stopped:
        # Only attempt daemon calls in legacy mode (no in-process controller)
        try:
            # Ask daemon to reload before shutdown; if unavailable we still proceed
            with contextlib.suppress(HubServiceError):
                await _call_daemon_api_async(config, "POST", "/hub/reload")

            await _call_daemon_api_async(config, "POST", "/hub/shutdown")
            manager_shutdown = True
        except HubServiceError:
            # If daemon unreachable, treat as not running
            manager_shutdown = False

    message_parts = [
        "Hub controller stop requested" if controller_stopped else "Hub controller was not running",
        "Hub manager shutdown requested" if manager_shutdown else "Hub manager was not running",
    ]

    return HubServiceActionResponse(
        status="ok",
        action="stop",
        message=". ".join(message_parts),
        details={
            "controller_stopped": controller_stopped,
            "manager_shutdown": manager_shutdown,
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

    Raises
    ------
    HubConfigError
        If the hub configuration cannot be loaded.
    HubServiceError
        If there is an error communicating with the hub service.
    """
    try:
        config = _load_hub_config_from_request(raw_request)
    except HubConfigError as exc:
        return _hub_config_error_response(str(exc))

    # Check if controller is available directly (unified daemon mode)
    controller = getattr(raw_request.app.state, "hub_controller", None)
    if controller is None:
        controller = getattr(raw_request.app.state, "supervisor", None)

    if controller is not None:
        # Use controller directly in unified daemon mode
        try:
            diff = await controller.reload_config()
        except Exception as exc:
            return _service_error_response("reload hub configuration", HubServiceError(str(exc)))
    else:
        # Fall back to daemon API calls (legacy mode)
        try:
            # Ensure daemon responds to health check
            await _call_daemon_api_async(config, "GET", "/health")
        except HubServiceError:
            return _manager_unavailable_response()

        try:
            diff = await _call_daemon_api_async(config, "POST", "/hub/reload") or {}
        except HubServiceError as exc:
            return _service_error_response("reload hub configuration", exc)

    return HubServiceActionResponse(
        status="ok",
        action="reload",
        message="Hub configuration reloaded",
        details=diff,
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
        The name of the model to load.
    raw_request : Request
        The incoming request.
    payload : HubModelActionRequest, optional
        Additional payload for the request.

    Returns
    -------
    HubModelActionResponse or JSONResponse
        Response indicating the result of the load action.
    """
    _ = payload  # reserved for future compatibility
    return await _hub_model_service_action(raw_request, model_name, "start")


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
    return await _hub_model_service_action(raw_request, model_name, "stop")


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
    return await _hub_memory_controller_action(raw_request, model_name, "load")


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
    return await _hub_memory_controller_action(raw_request, model_name, "unload")


@hub_router.post("/hub/models/{model_name}/vram/load", response_model=HubModelActionResponse)
async def hub_vram_load(
    model_name: str,
    raw_request: Request,
    payload: HubModelActionRequest | None = None,
) -> HubModelActionResponse | JSONResponse:
    """Admin endpoint to request VRAM residency for a registered model via the registry."""
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

    registry = getattr(raw_request.app.state, "model_registry", None)
    if registry is None:
        return _registry_unavailable_response()

    try:
        await registry.request_vram_load(
            target,
            force=bool(getattr(payload, "force", False)),
            timeout=getattr(payload, "timeout", None),
        )
        message = f"VRAM load requested for '{target}'"
    except Exception as exc:
        return _registry_error_response(exc)

    return HubModelActionResponse(status="ok", action="load", model=target, message=message)


@hub_router.post("/hub/models/{model_name}/vram/unload", response_model=HubModelActionResponse)
async def hub_vram_unload(
    model_name: str,
    raw_request: Request,
    payload: HubModelActionRequest | None = None,
) -> HubModelActionResponse | JSONResponse:
    """Admin endpoint to request VRAM release for a registered model via the registry."""
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

    registry = getattr(raw_request.app.state, "model_registry", None)
    if registry is None:
        return _registry_unavailable_response()

    try:
        await registry.request_vram_unload(target, timeout=getattr(payload, "timeout", None))
        message = f"VRAM unload requested for '{target}'"
    except Exception as exc:
        return _registry_error_response(exc)

    return HubModelActionResponse(status="ok", action="unload", model=target, message=message)


@hub_router.post("/hub/shutdown", response_model=HubServiceActionResponse)
async def hub_shutdown(
    raw_request: Request,
    background_tasks: BackgroundTasks,
) -> HubServiceActionResponse | JSONResponse:
    """Request the hub daemon to shutdown all managed models and exit.

    This endpoint forwards the shutdown request to the running hub manager
    process. If the hub manager is not running an appropriate error is returned.
    """
    try:
        config = _load_hub_config_from_request(raw_request)
    except HubConfigError as exc:
        return _hub_config_error_response(str(exc))

    controller = getattr(raw_request.app.state, "hub_controller", None)
    if controller is None:
        controller = getattr(raw_request.app.state, "supervisor", None)
    if controller is not None:
        background_tasks.add_task(controller.shutdown_all)

        # Schedule server shutdown after model shutdown
        async def shutdown_server() -> None:
            await asyncio.sleep(1)  # Give time for response to be sent
            sys.exit(0)

        background_tasks.add_task(shutdown_server)
        return HubServiceActionResponse(
            status="ok",
            action="stop",
            message="Shutdown requested",
            details={},
        )

    try:
        await _call_daemon_api_async(config, "POST", "/hub/shutdown")
    except HubServiceError as exc:
        return _service_error_response("shutdown hub manager", exc)

    return HubServiceActionResponse(
        status="ok",
        action="stop",
        message="Shutdown requested",
        details={},
    )


async def _hub_model_service_action(
    raw_request: Request,
    model_name: str,
    action: Literal["start", "stop"],
) -> HubModelActionResponse | JSONResponse:
    """Execute a load or unload action on a model via the hub service.

    Parameters
    ----------
    raw_request : Request
        The incoming FastAPI request.
    model_name : str
        Name of the model to act on.
    action : Literal["start", "stop"]
        The action to perform.

    Returns
    -------
    HubModelActionResponse or JSONResponse
        Action response or error response.
    """
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

    try:
        config = _load_hub_config_from_request(raw_request)
    except HubConfigError as exc:
        return _hub_config_error_response(str(exc))

    controller = getattr(raw_request.app.state, "hub_controller", None)
    if controller is None:
        controller = getattr(raw_request.app.state, "supervisor", None)
    if controller is not None:
        try:
            if action == "start":
                await controller.start_model(target)
                message = f"Model '{target}' start requested"
            else:
                await controller.stop_model(target)
                message = f"Model '{target}' stop requested"
        except Exception as exc:
            return _controller_error_response(exc)
        return HubModelActionResponse(status="ok", action=action, model=target, message=message)

    try:
        await _call_daemon_api_async(config, "GET", "/health")
    except HubServiceError:
        return _manager_unavailable_response()

    try:
        await _call_daemon_api_async(config, "POST", "/hub/reload")
    except HubServiceError as exc:
        return _service_error_response("reload before executing the model action", exc)

    try:
        if action == "start":
            await _call_daemon_api_async(config, "POST", f"/hub/models/{target}/start")
            message = f"Model '{target}' start requested"
        else:
            await _call_daemon_api_async(config, "POST", f"/hub/models/{target}/stop")
            message = f"Model '{target}' stop requested"
    except HubServiceError as exc:
        friendly = action.replace("-", " ")
        return _service_error_response(f"{friendly} for model '{target}'", exc)

    return HubModelActionResponse(status="ok", action=action, model=target, message=message)


async def _hub_memory_controller_action(
    raw_request: Request,
    model_name: str,
    action: Literal["load", "unload"],
) -> HubModelActionResponse | JSONResponse:
    """Execute a memory load/unload request using the in-process controller.

    Parameters
    ----------
    raw_request : Request
        The incoming FastAPI request.
    model_name : str
        The name of the model to act on.
    action : Literal["load", "unload"]
        The action to perform.

    Returns
    -------
    HubModelActionResponse or JSONResponse
        Response indicating the result of the action.
    """
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

    controller = getattr(raw_request.app.state, "hub_controller", None)
    if controller is None:
        return _controller_unavailable_response()

    try:
        if action == "load":
            # Shield the long-running load so cancellation of the request
            # (for example due to a client disconnect) doesn't cancel the
            # underlying model initialization. If the await is cancelled
            # we schedule the operation in the background and return a
            # "requested" response to the client.
            coro = controller.load_model(target)
            try:
                await asyncio.shield(coro)
                message = f"Model '{target}' load requested"
            except asyncio.CancelledError:
                # Request task was cancelled (client disconnected). Ensure
                # the load continues in background and return a requested
                # response. Create a fresh coroutine for the background
                # task since coroutine objects cannot be awaited/used
                # twice.
                task = asyncio.create_task(controller.load_model(target))
                _retain_task(task)
                message = f"Model '{target}' load requested (running in background)"
        else:
            coro = controller.unload_model(target)
            try:
                await asyncio.shield(coro)
                message = f"Model '{target}' unload requested"
            except asyncio.CancelledError:
                # Create a fresh coroutine for the background unload task.
                task = asyncio.create_task(controller.unload_model(target))
                _retain_task(task)
                message = f"Model '{target}' unload requested (running in background)"
    except BrokenPipeError as e:  # pragma: no cover - defensive handling
        # Broken pipe can surface from low-level I/O during heavy init; log
        # and schedule the operation in background to avoid exposing raw
        # socket errors to the UI while still performing the work.
        logger.exception(
            f"Broken pipe while executing {action} for {target}: {e}",
        )
        try:
            task = asyncio.create_task(
                controller.load_model(target)
                if action == "load"
                else controller.unload_model(target)
            )
            _retain_task(task)
        except Exception:
            logger.debug("Failed to schedule background model action after BrokenPipeError")
        message = f"Model '{target}' {action} requested (running in background)"
    except Exception as e:  # pragma: no cover - defensive logging
        logger.exception(
            f"Unexpected failure while executing {action} for {target}. {type(e).__name__}: {e}",
        )
        return _controller_error_response(e)

    return HubModelActionResponse(status="ok", action=action, model=target, message=message)


__all__ = [
    "get_cached_model_metadata",
    "get_configured_model_id",
    "hub_router",
]
