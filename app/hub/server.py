"""FastAPI/uvicorn harness for hub-managed deployments."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager, suppress
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import TYPE_CHECKING

from fastapi import FastAPI
from loguru import logger
import uvicorn

from ..api.hub_routes import hub_router
from ..core.model_registry import ModelRegistry
from ..server import configure_fastapi_app, configure_logging
from ..version import __version__
from .config import MLXHubConfig, load_hub_config
from .runtime import HubRuntime

if TYPE_CHECKING:
    from .controller import HubController

_CONTROLLER_PID_NAME = "hub-server.pid"


def _controller_pid_path(config: MLXHubConfig) -> Path:
    """Return the path to the hub controller PID file.

    Parameters
    ----------
    config : MLXHubConfig
        The hub configuration.

    Returns
    -------
    Path
        Path to the PID file.
    """
    return config.log_path / _CONTROLLER_PID_NAME


def _pid_alive(pid: int) -> bool:
    """Check if a process with the given PID is alive.

    Parameters
    ----------
    pid : int
        Process ID to check.

    Returns
    -------
    bool
        True if the process is alive, False otherwise.
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # pragma: no cover - extremely rare
        return True
    else:
        return True


def is_hub_controller_running(config: MLXHubConfig) -> bool:
    """Check if the hub controller process is currently running.

    Parameters
    ----------
    config : MLXHubConfig
        Hub configuration used to locate the PID file.

    Returns
    -------
    bool
        True if the controller is running, False otherwise.
    """
    path = _controller_pid_path(config)
    if not path.exists():
        return False
    try:
        pid = int(path.read_text(encoding="utf-8").strip())
    except ValueError:
        with suppress(Exception):
            path.unlink()
        return False
    alive = _pid_alive(pid)
    if not alive:
        with suppress(Exception):
            path.unlink()
    return alive


def start_hub_controller_process(config_path: Path | str) -> int:
    """Start the hub controller process in the background.

    Parameters
    ----------
    config_path : Path | str
        Path to the hub configuration file.

    Returns
    -------
    int
        PID of the started controller process.
    """
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "from app.hub.server import _controller_process_entrypoint; "
                f"_controller_process_entrypoint('{config_path}')"
            ),
        ],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=Path.cwd(),
    )
    return proc.pid


def stop_hub_controller_process(config: MLXHubConfig, *, timeout: float = 10.0) -> bool:
    """Stop the hub controller process.

    Parameters
    ----------
    config : MLXHubConfig
        Hub configuration used to locate the PID file.
    timeout : float, default=10.0
        Maximum time to wait for the process to stop.

    Returns
    -------
    bool
        True if the process was stopped, False otherwise.
    """
    path = _controller_pid_path(config)
    if not path.exists():
        return False
    try:
        pid = int(path.read_text(encoding="utf-8").strip())
    except ValueError:
        with suppress(Exception):
            path.unlink()
        return False

    with suppress(ProcessLookupError):
        os.kill(pid, signal.SIGTERM)

    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _pid_alive(pid):
            break
        time.sleep(0.2)
    else:  # pragma: no cover - fallback
        with suppress(ProcessLookupError):
            os.kill(pid, signal.SIGKILL)

    with suppress(FileNotFoundError):
        path.unlink()
    return True


def setup_hub_server(runtime: HubRuntime) -> uvicorn.Config:
    """Create the FastAPI application used by hub deployments.

    Parameters
    ----------
    runtime : HubRuntime
        Runtime object encapsulating hub configuration and state trackers.

    Returns
    -------
    uvicorn.Config
        Configured uvicorn configuration ready to be passed to ``uvicorn.Server``.
    """

    from .controller import HubController  # noqa: PLC0415

    log_file = runtime.config.log_path / "hub.log"
    configure_logging(log_file=str(log_file), no_log_file=False, log_level=runtime.config.log_level)

    registry = ModelRegistry()
    controller = None
    try:
        controller = HubController(runtime, registry)
    except Exception as e:
        logger.error(f"Failed to create HubController. {type(e).__name__}: {e}")
        controller = None
    app = FastAPI(
        title="MLX Hub API",
        description="Multi-model hub for the MLX OpenAI-compatible server",
        version=__version__,
        lifespan=_create_hub_lifespan(controller) if controller else None,
    )

    app.state.registry = registry
    app.state.server_config = runtime.config
    app.state.hub_controller = controller
    app.state.hub_runtime = runtime
    app.state.model_metadata = []

    configure_fastapi_app(app)

    app.include_router(hub_router)

    logger.info(
        f"Starting hub server on {runtime.config.host}:{runtime.config.port} with {len(runtime.model_names())} configured model(s)",
    )

    return uvicorn.Config(
        app=app,
        host=runtime.config.host,
        port=runtime.config.port,
        log_level=runtime.config.log_level.lower(),
        access_log=True,
    )


def _create_hub_lifespan(
    controller: HubController,
) -> Callable[[FastAPI], AbstractAsyncContextManager[None]]:
    """Return a FastAPI lifespan context that owns the hub controller.

    Parameters
    ----------
    controller : HubController
        Controller responsible for coordinating handler managers.

    Returns
    -------
    Callable[[FastAPI], AbstractAsyncContextManager[None]]
        ``FastAPI`` lifespan hook that starts/stops the controller.
    """

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
        try:
            await controller.start()
            yield
        finally:
            await controller.shutdown()

    return lifespan


def print_hub_startup_banner(config: MLXHubConfig) -> None:
    """Emit a concise startup banner for hub deployments.

    Parameters
    ----------
    config : MLXHubConfig
        Hub configuration describing host/port/logging defaults.
    """

    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info(f"✨ MLX Hub Server v{__version__} Starting ✨")
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info(f"🌐 Host: {config.host}")
    logger.info(f"🔌 Port: {config.port}")
    logger.info(f"📝 Log Level: {config.log_level}")
    logger.info(f"📁 Log Path: {config.log_path}")
    logger.info(f"📦 Models: {len(config.models)} configured")
    defaults: list[str] = []
    for model in config.models:
        if not model.is_default_model:
            continue
        name = model.name
        if name:
            defaults.append(name)
    if defaults:
        logger.info(f"⭐ Default Models: {', '.join(defaults)}")
    else:
        logger.info("⭐ Default Models: none (all on-demand)")
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


def _controller_process_entrypoint(config_path: str) -> None:
    """Entrypoint for the hub controller process.

    Parameters
    ----------
    config_path : str
        Path to the hub configuration file.
    """
    config = load_hub_config(config_path)
    pid_path = _controller_pid_path(config)
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(str(os.getpid()), encoding="utf-8")
    runtime = HubRuntime(config)
    try:
        asyncio.run(start_hub(runtime))
    finally:
        with suppress(FileNotFoundError):
            pid_path.unlink()


async def start_hub(runtime: HubRuntime) -> None:
    """Configure and launch the hub Uvicorn server.

    Parameters
    ----------
    runtime : HubRuntime
        Prepared runtime used to seed the controller and FastAPI app.
    """

    try:
        print_hub_startup_banner(runtime.config)
        uvconfig = setup_hub_server(runtime)
        server = uvicorn.Server(uvconfig)
        await server.serve()
    except KeyboardInterrupt:
        logger.info("Hub server shutdown requested by user. Exiting...")
    except Exception:
        logger.exception("Hub server startup failed")
        sys.exit(1)
