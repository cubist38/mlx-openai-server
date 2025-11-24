"""Hub daemon scaffold: FastAPI app factory and HubSupervisor skeleton.

This module provides a non-complete but useful scaffold for the hub daemon
supervisor and HTTP control API. Implementations that require deeper
integration with model handlers should expand the supervisor methods; tests
may mock the supervisor where appropriate.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
import contextlib
from contextlib import asynccontextmanager
from dataclasses import dataclass
import os
from pathlib import Path
import sys
import time
from typing import Any, cast

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from loguru import logger

from .config import MLXHubConfig, load_hub_config


def create_app_with_config(config_path: str) -> FastAPI:
    """Create FastAPI app with specific config path (for uvicorn factory)."""
    return create_app(config_path)


@dataclass
class ModelRecord:
    """Runtime record for a supervised model.

    Attributes
    ----------
    name : str
        Slug name of the model.
    config : Any
        The model configuration object (from `MLXHubConfig`).
    process : asyncio.subprocess.Process | None
        The subprocess instance if started.
    pid : int | None
        OS process id when running.
    port : int | None
        Assigned port for the model process.
    started_at : float | None
        Epoch timestamp when process was started.
    exit_code : int | None
        Last exit code, if the process exited.
    memory_loaded : bool
        Whether the model's runtime/memory is currently loaded.
    auto_unload_minutes : int | None
        Optional idle minutes after which memory should be auto-unloaded.
    group : str | None
        Group slug for capacity accounting.
    is_default : bool
        Whether this model is marked as a default auto-start.
    model_path : str | None
        Configured model path.
    """

    name: str
    config: Any
    process: asyncio.subprocess.Process | None = None
    pid: int | None = None
    port: int | None = None
    started_at: float | None = None
    exit_code: int | None = None
    memory_loaded: bool = False
    auto_unload_minutes: int | None = None
    group: str | None = None
    is_default: bool = False
    model_path: str | None = None


class HubSupervisor:
    """Supervise model worker processes and runtime state.

    This is a conservative scaffold that implements non-blocking process
    management patterns. Long-running operations should be scheduled as
    background tasks when invoked from FastAPI endpoints.
    """

    def __init__(self, hub_config: MLXHubConfig) -> None:
        self.hub_config = hub_config
        self._models: dict[str, ModelRecord] = {}
        self._lock = asyncio.Lock()
        self._bg_tasks: list[asyncio.Task[None]] = []
        self._shutdown = False

        # Populate model records from hub_config (best-effort)
        for model in getattr(hub_config, "models", []):
            name = getattr(model, "name", None) or str(model)
            record = ModelRecord(
                name=name,
                config=model,
                port=getattr(model, "port", None),
                group=getattr(model, "group", None),
                is_default=getattr(model, "is_default_model", False),
                model_path=getattr(model, "model_path", None),
            )
            self._models[name] = record

    async def start_model(self, name: str) -> dict[str, Any]:
        """Start the model worker as an OS subprocess.

        The supervisor will attempt to spawn a worker process for the named
        model. If the model configuration does not supply an explicit command
        the supervisor constructs a sensible default invocation that runs the
        package's single-model server with the model's configured settings.
        """

        async with self._lock:
            if name not in self._models:
                raise HTTPException(status_code=404, detail="model not found")
            record = self._models[name]

            if record.process is not None and record.process.returncode is None:
                return {"status": "already_running", "name": name, "pid": record.pid}

            cmd = None
            cfg = record.config
            # Construct a default invocation to spawn the single-model
            # server for this model. The command uses the package CLI
            # (`python -m app.main launch`) with model settings from the
            # hub configuration.
            # Build default invocation: `python -m app.main launch ...`
            py = sys.executable
            model_path = getattr(cfg, "model_path", None) or getattr(cfg, "model", None)
            model_type = getattr(cfg, "model_type", None) or "lm"
            host = getattr(cfg, "host", "127.0.0.1")
            port = record.port or getattr(cfg, "port", None) or 0
            cmd = [
                py,
                "-m",
                "app.main",
                "launch",
                "--model-path",
                str(model_path),
                "--model-type",
                str(model_type),
                "--host",
                str(host),
                "--port",
                str(int(port)),
            ]
            if bool(getattr(cfg, "jit_enabled", False)):
                cmd.append("--jit")
            if getattr(cfg, "auto_unload_minutes", None) is not None:
                cmd.extend(["--auto-unload-minutes", str(getattr(cfg, "auto_unload_minutes"))])

            # Spawn subprocess without blocking the event loop
            proc = await asyncio.create_subprocess_exec(*cmd)
            record.process = proc
            record.pid = proc.pid
            record.started_at = time.time()
            record.exit_code = None

            # schedule a watcher
            task = asyncio.create_task(self._watch_process(name, proc))
            self._bg_tasks.append(task)

            # If a port is configured (non-zero), wait until the spawned
            # process is listening on host:port before returning. This makes
            # `hub start` and controller start operations synchronous from
            # the operator perspective: they only return once the worker
            # is reachable, or fail if the process exits or a timeout is
            # reached.
            async def _wait_for_listen(
                host: str,
                port: int,
                proc: asyncio.subprocess.Process,
                timeout: float = 30.0,
                interval: float = 0.2,
            ) -> None:
                start = time.time()
                while True:
                    # If process exited while we were waiting, abort
                    if proc.returncode is not None:
                        raise HTTPException(
                            status_code=500,
                            detail=f"process exited before listening (exit_code={proc.returncode})",
                        )
                    try:
                        reader, writer = await asyncio.open_connection(host, port)
                        # Connected successfully — close and return
                        try:
                            writer.close()
                            with contextlib.suppress(Exception):
                                await writer.wait_closed()
                        except Exception:
                            pass
                    except Exception as e:
                        # Not listening yet — check timeout then sleep
                        if (time.time() - start) >= timeout:
                            raise HTTPException(
                                status_code=504,
                                detail=f"timed out waiting for process to listen on {host}:{port}",
                            ) from e
                        await asyncio.sleep(interval)
                    else:
                        return

            # If we have a port to check, wait until the worker accepts
            # connections. If port is zero (ephemeral) we can't reliably
            # determine the bound port here, so skip the wait in that case.
            try:
                bound_port = int(port) if port is not None else 0
            except Exception:
                bound_port = 0

            if bound_port and bound_port > 0:
                await _wait_for_listen(host, bound_port, proc)
                logger.info(f"Started model {name} pid={proc.pid} listening={host}:{bound_port}")
            else:
                logger.info(f"Started model {name} pid={proc.pid} (no port check)")

            return {"status": "started", "name": name, "pid": proc.pid, "port": record.port}

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
            proc = record.process
            if proc is None or proc.returncode is not None:
                return {"status": "not_running", "name": name}

            with contextlib.suppress(ProcessLookupError):
                proc.terminate()

        # Wait outside the lock
        try:
            await asyncio.wait_for(proc.wait(), timeout=10.0)
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            await proc.wait()

        record.exit_code = proc.returncode
        record.process = None
        record.pid = None
        logger.info(f"Stopped model {name} exit_code={record.exit_code}")
        return {"status": "stopped", "name": name, "exit_code": record.exit_code}

    async def load_model(self, name: str) -> dict[str, Any]:
        """Mark a model's memory/runtime as loaded.

        This is a lightweight marker; heavy loading work should be scheduled
        separately as background tasks integrated with model handlers.
        """

        async with self._lock:
            if name not in self._models:
                raise HTTPException(status_code=404, detail="model not found")
            record = self._models[name]
            # Mark loaded and schedule any heavy lifting as a background task
            record.memory_loaded = True
            logger.info(f"Marked model {name} memory_loaded")
            return {"status": "memory_loaded", "name": name}

    async def unload_model(self, name: str) -> dict[str, Any]:
        """Mark a model's memory/runtime as unloaded.

        The supervisor will not attempt to free in-process handler state here;
        this method provides a consistent API surface for the CLI and tests.
        """

        async with self._lock:
            if name not in self._models:
                raise HTTPException(status_code=404, detail="model not found")
            record = self._models[name]
            record.memory_loaded = False
            logger.info(f"Marked model {name} memory_unloaded")
            return {"status": "memory_unloaded", "name": name}

    async def reload_config(self) -> dict[str, Any]:
        """Reload the hub configuration and reconcile models.

        This implementation performs a best-effort reload: it replaces the
        in-memory hub_config and returns a simple diff.
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
            self.hub_config = new_hub
            self._models = {}
            for model in models_list:
                name = getattr(model, "name", None) or str(model)
                record = ModelRecord(
                    name=name,
                    config=model,
                    port=getattr(model, "port", None),
                    group=getattr(model, "group", None),
                    is_default=getattr(model, "is_default_model", False),
                    model_path=getattr(model, "model_path", None),
                    auto_unload_minutes=getattr(model, "auto_unload_minutes", None),
                )
                self._models[name] = record

            logger.info(f"Reloaded hub config: started={started} stopped={stopped}")

        return {"started": started, "stopped": stopped, "unchanged": unchanged}

    def get_status(self) -> dict[str, Any]:
        """Return a serializable snapshot of supervisor state.

        The returned dict includes a `timestamp` and a `models` list where
        each model object contains keys used by the CLI and status UI.
        """

        snapshot: dict[str, Any] = {
            "timestamp": time.time(),
            "models": [],
        }
        for name, rec in self._models.items():
            state = "running" if rec.process and rec.process.returncode is None else "stopped"
            snapshot["models"].append(
                {
                    "name": name,
                    "state": state,
                    "pid": rec.pid,
                    "port": rec.port,
                    "started_at": rec.started_at,
                    "exit_code": rec.exit_code,
                    "memory_loaded": rec.memory_loaded,
                    "group": rec.group,
                    "is_default_model": rec.is_default,
                    "model_path": rec.model_path,
                    "auto_unload_minutes": rec.auto_unload_minutes,
                }
            )
        return snapshot

    async def _watch_process(self, name: str, proc: asyncio.subprocess.Process) -> None:
        try:
            await proc.wait()
            async with self._lock:
                rec = self._models.get(name)
                if rec is not None:
                    rec.exit_code = proc.returncode
                    rec.process = None
                    rec.pid = None
                    logger.info(f"Model process exited: {name} code={proc.returncode}")
        except asyncio.CancelledError:
            logger.debug(f"Watcher cancelled for {name}")

    async def shutdown_all(self) -> None:
        """Gracefully stop all supervised model processes.

        This performs a best-effort shutdown of each supervised process and
        logs failures without raising to the caller.
        """

        logger.info("Shutting down all supervised model processes")
        async with self._lock:
            names = list(self._models.keys())
        for name in names:
            try:
                await self.stop_model(name)
            except Exception as exc:  # pragma: no cover - best-effort shutdown
                logger.exception(f"Error stopping model {name}: {exc}")


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

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        # Startup
        logger.info("Hub daemon starting up")

        # Auto-start models marked as default in configuration. Schedule
        # start operations as background tasks so the daemon becomes
        # responsive quickly while model processes spin up.
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
                            # Always attempt to start a supervised worker process
                            # for default models. The worker process itself will
                            # decide whether to eagerly load handlers based on
                            # its JIT configuration and the LazyHandlerManager.
                            await supervisor.start_model(mname)
                            logger.info(f"Auto-started model process: {mname}")
                        except Exception as exc:  # pragma: no cover - best-effort
                            # Starting the supervised worker failed; do not
                            # attempt to mark the model as memory-loaded here.
                            # Handler loading is the responsibility of the
                            # worker process via LazyHandlerManager (or of the
                            # operator). Log the failure and continue.
                            logger.warning(f"Failed to auto-start model process '{mname}': {exc}")

                    asyncio.create_task(_autostart(model))
                    logger.info(f"Scheduled auto-start for default model: {name}")
                except Exception as exc:  # pragma: no cover - best-effort
                    logger.warning(
                        f"Failed to schedule auto-start for model entry {model!r}: {exc}"
                    )
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception(f"Error while scheduling default model starts: {exc}")

        # Yield to start serving requests while background tasks proceed
        yield
        # Shutdown
        logger.info("Hub daemon shutting down")
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
                except Exception as exc:  # pragma: no cover - best-effort cleanup
                    logger.warning(f"Failed to remove hub runtime state file {runtime_file}: {exc}")
        except Exception:
            # Defensive: do not allow cleanup failures to raise during shutdown
            logger.debug("Error while attempting to clean up runtime state file")

    app = FastAPI(title="mlx hub daemon", lifespan=lifespan)

    # Configure templates directory (fall back to inline rendering if missing)
    # Templates folder lives at the repository root `templates/`
    templates = Jinja2Templates(directory=Path(__file__).parent.parent.parent / "templates")

    # Use environment variable if no path provided
    if hub_config_path is None:
        hub_config_path = os.environ.get("MLX_HUB_CONFIG_PATH")

    hub_config = load_hub_config(hub_config_path)
    supervisor = HubSupervisor(hub_config)
    app.state.supervisor = supervisor
    supervisor = cast("HubSupervisor", app.state.supervisor)  # Allow tests to override

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/hub/status")
    async def hub_status() -> dict[str, Any]:
        return supervisor.get_status()

    @app.post("/hub/reload")
    async def hub_reload() -> dict[str, Any]:
        return await supervisor.reload_config()

    @app.post("/hub/shutdown")
    async def hub_shutdown(background_tasks: BackgroundTasks, request: Request) -> dict[str, str]:
        supervisor = cast("HubSupervisor", request.app.state.supervisor)
        background_tasks.add_task(supervisor.shutdown_all)
        return {"status": "shutdown_scheduled"}

    @app.post("/hub/models/{name}/start")
    async def model_start(name: str, request: Request) -> dict[str, Any]:
        supervisor = cast("HubSupervisor", request.app.state.supervisor)
        return await supervisor.start_model(name)

    @app.post("/hub/models/{name}/stop")
    async def model_stop(name: str, request: Request) -> dict[str, Any]:
        supervisor = cast("HubSupervisor", request.app.state.supervisor)
        return await supervisor.stop_model(name)

    @app.post("/hub/models/{name}/load")
    async def model_load(name: str, request: Request) -> dict[str, Any]:
        supervisor = cast("HubSupervisor", request.app.state.supervisor)
        return await supervisor.load_model(name)

    @app.post("/hub/models/{name}/unload")
    async def model_unload(name: str, request: Request) -> dict[str, Any]:
        supervisor = cast("HubSupervisor", request.app.state.supervisor)
        return await supervisor.unload_model(name)

    @app.get("/hub", response_class=HTMLResponse)
    async def hub_status_page(request: Request) -> HTMLResponse:
        """Render the hub status page using Jinja templates.

        We pre-format timestamps and model fields to avoid relying on custom
        Jinja filters (keeps template simple and safe).
        """
        supervisor = cast("HubSupervisor", request.app.state.supervisor)
        status = supervisor.get_status()
        hub_cfg = supervisor.hub_config

        timestamp = status.get("timestamp")
        ts_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp)) if timestamp else "-"

        models = [
            {
                "name": m.get("name", "-"),
                "state": m.get("state", "-"),
                "pid": m.get("pid"),
                "port": m.get("port"),
                "memory_loaded": bool(m.get("memory_loaded")),
                "model_path": m.get("model_path"),
                "auto_unload_minutes": m.get("auto_unload_minutes"),
            }
            for m in status.get("models", [])
        ]

        context = {
            "request": request,
            "hub_config": hub_cfg,
            "models": models,
            "timestamp": ts_str,
        }
        return templates.TemplateResponse("hub_status.html", context)

    return app
