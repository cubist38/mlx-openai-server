"""Background service utilities for the process-based hub manager.

This module exposes the HubService and HubServiceClient helpers used for IPC
(Inter-Process Communication). IPC refers to mechanisms that allow multiple
processes to exchange data and coordinate actions; this project uses a UNIX
domain socket via Python's multiprocessing connection APIs (Client/Listener).
"""

from __future__ import annotations

from contextlib import suppress
from dataclasses import asdict, dataclass
from multiprocessing.connection import Client, Connection, Listener
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from types import FrameType
from typing import Any, cast

from loguru import logger

from .config import MLXHubConfig, load_hub_config
from .errors import HubControllerError
from .manager import HubManager, ProcessFactory

HUB_SERVICE_AUTH_KEY = b"mlx_hub_service_v1"
_SERVICE_SOCKET_NAME = "hub-manager.sock"
_SERVICE_PID_NAME = "hub-manager.pid"
_SERVICE_LOG_NAME = "hub-manager.log"


@dataclass(slots=True)
class HubServicePaths:
    """Filesystem paths used by the hub manager service."""

    socket_path: Path
    pid_path: Path
    log_path: Path


class HubServiceError(RuntimeError):
    """Raised when the CLI cannot communicate with the hub service."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class HubService:
    """Long-lived server that wraps :class:`HubManager` behind IPC commands.

    The HubService implements an IPC server (UNIX-domain socket) which accepts
    dictionary-encoded commands from the `HubServiceClient` used by the CLI,
    FastAPI routes, and any other control surface. Supported actions include
    `ping`, `status`, `reload`, `start_model`, `stop_model`, and `shutdown`.
    """

    def __init__(
        self, config_path: Path | str, *, process_factory: ProcessFactory | None = None
    ) -> None:
        """Initialize the hub service.

        Parameters
        ----------
        config_path : Path or str
            Path to the hub configuration file.
        process_factory : ProcessFactory, optional
            Factory for creating managed processes.
        """

        self._config_path = Path(config_path).expanduser()
        self._manager = HubManager(self._config_path, process_factory=process_factory)
        self._stop_event = threading.Event()
        self._listener: Listener | None = None
        self._paths: HubServicePaths | None = None
        self._log_level = "INFO"

    def serve_forever(self, *, ready_event: threading.Event | None = None) -> None:
        """Run the service loop until a shutdown command or signal arrives.

        Parameters
        ----------
        ready_event : threading.Event, optional
            Event to set when the service is ready to accept connections.
        """

        config = load_hub_config(self._config_path)
        self._log_level = config.log_level
        self._paths = get_service_paths(config)
        _configure_service_logging(self._paths.log_path, self._log_level)
        self._install_signal_handlers()
        self._prepare_filesystem()

        logger.info(f"Starting hub manager service with {len(config.models)} configured model(s)")

        self._manager.reload()

        try:
            address = str(self._paths.socket_path)
            with Listener(address, family="AF_UNIX", authkey=HUB_SERVICE_AUTH_KEY) as listener:
                self._listener = listener
                if ready_event is not None:
                    ready_event.set()
                while not self._stop_event.is_set():
                    try:
                        conn = listener.accept()
                    except (OSError, EOFError):
                        if self._stop_event.is_set():
                            break
                        continue
                    threading.Thread(target=self._handle_client, args=(conn,), daemon=True).start()
        finally:
            self._cleanup()
            logger.info("Hub manager service stopped")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _install_signal_handlers(self) -> None:
        """Install signal handlers for graceful shutdown and reload."""

        def _stop_handler(signum: int, _frame: FrameType | None) -> None:
            logger.info(f"Received signal {signum}; shutting down hub manager")
            self._request_stop()

        def _reload_handler(signum: int, _frame: FrameType | None) -> None:
            logger.info(f"Received signal {signum}; reloading hub configuration")
            try:
                self._manager.reload()
            except Exception as exc:  # pragma: no cover - defensive path
                logger.error(f"Hub reload failed: {exc}")

        signal.signal(signal.SIGTERM, _stop_handler)
        signal.signal(signal.SIGINT, _stop_handler)
        if hasattr(signal, "SIGHUP"):
            signal.signal(signal.SIGHUP, _reload_handler)

    def _prepare_filesystem(self) -> None:
        """Prepare the filesystem by creating directories and cleaning up old sockets."""

        assert self._paths is not None
        socket_path = self._paths.socket_path
        if socket_path.exists():
            socket_path.unlink()
        pid_path = self._paths.pid_path
        pid_path.write_text(str(os.getpid()), encoding="utf-8")

    def _handle_client(self, conn: Connection) -> None:
        """Handle a client connection by processing the command and sending response.

        Parameters
        ----------
        conn : Connection
            The multiprocessing connection to the client.
        """

        try:
            payload = conn.recv()
            response = self._dispatch_command(payload)
        except HubControllerError as exc:
            response = {"status": "error", "error": str(exc), "code": exc.status_code}
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.exception("Hub service command failed")
            response = {"status": "error", "error": str(exc)}
        finally:
            with suppress(Exception):  # pragma: no cover - defensive guard
                conn.send(response)
            conn.close()

    def _dispatch_command(self, payload: Any) -> dict[str, Any]:
        """Dispatch a command payload and return the response.

        Parameters
        ----------
        payload : Any
            The command payload dictionary.

        Returns
        -------
        dict[str, Any]
            The response dictionary.

        Raises
        ------
        TypeError
            If payload is invalid.
        ValueError
            If action is unknown.
        """

        if not isinstance(payload, dict):
            raise TypeError("Hub service payloads must be dictionaries")
        action = payload.get("action")
        if not isinstance(action, str):
            raise TypeError("Hub service payloads require an 'action' key")

        if action == "ping":
            return {"status": "ok", "data": {"pid": os.getpid()}}

        if action == "status":
            statuses = [asdict(status) for status in self._manager.get_status()]
            return {"status": "ok", "data": {"models": statuses, "timestamp": time.time()}}

        if action == "reload":
            result = asdict(self._manager.reload())
            return {"status": "ok", "data": result}

        if action == "start_model":
            name = payload.get("name")
            if not isinstance(name, str) or not name.strip():
                raise TypeError("start_model payload requires non-empty 'name'")
            self._manager.start_model(name.strip())
            return {"status": "ok", "data": {"started": name.strip()}}

        if action == "stop_model":
            name = payload.get("name")
            if not isinstance(name, str) or not name.strip():
                raise TypeError("stop_model payload requires non-empty 'name'")
            self._manager.stop_model(name.strip())
            return {"status": "ok", "data": {"stopped": name.strip()}}

        if action == "shutdown":
            self._request_stop()
            return {"status": "ok", "data": {"message": "shutdown requested"}}

        raise ValueError(f"Unknown hub service action '{action}'")

    def _request_stop(self) -> None:
        """Request the service to stop gracefully."""

        if self._stop_event.is_set():
            return
        self._stop_event.set()
        try:
            self._manager.shutdown()
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.error(f"Failed to shutdown hub manager cleanly: {exc}")
        if self._listener is not None:
            with suppress(Exception):
                self._listener.close()

    def _cleanup(self) -> None:
        """Clean up filesystem artifacts."""

        if self._paths is None:
            return
        if self._paths.socket_path.exists():
            with suppress(Exception):
                self._paths.socket_path.unlink()
        if self._paths.pid_path.exists():
            with suppress(Exception):
                self._paths.pid_path.unlink()


def _configure_service_logging(log_path: Path, log_level: str) -> None:
    logger.remove()
    logger.add(
        log_path,
        rotation="25 MB",
        retention="5 days",
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    )


def get_service_paths(config: MLXHubConfig) -> HubServicePaths:
    """Return socket/pid/log paths derived from the provided hub config.

    Parameters
    ----------
    config : MLXHubConfig
        The hub configuration.

    Returns
    -------
    HubServicePaths
        Paths for service files.
    """

    base = config.log_path
    return HubServicePaths(
        socket_path=base / _SERVICE_SOCKET_NAME,
        pid_path=base / _SERVICE_PID_NAME,
        log_path=base / _SERVICE_LOG_NAME,
    )


class HubServiceClient:
    """Thin IPC client for issuing hub manager commands."""

    def __init__(self, socket_path: Path | str, *, timeout: float = 5.0) -> None:
        """Initialize the hub service client.

        Parameters
        ----------
        socket_path : Path or str
            Path to the service socket.
        timeout : float, default=5.0
            Timeout for requests in seconds.
        """

        self._socket_path = str(Path(socket_path).expanduser())
        self._timeout = timeout

    def is_available(self) -> bool:
        """Return ``True`` when the background service responds to pings."""

        try:
            self._request({"action": "ping"}, timeout=1.0)
        except HubServiceError:
            return False
        else:
            return True

    def reload(self) -> dict[str, Any]:
        """Reload hub.yaml inside the running service and return the diff.

        Returns
        -------
        dict[str, Any]
            Response data from the service including reload details.
        """

        return self._command("reload")

    def status(self) -> dict[str, Any]:
        """Return the current process table maintained by the service.

        Returns
        -------
        dict[str, Any]
            Response data from the service including model statuses.
        """

        return self._command("status")

    def start_model(self, name: str) -> dict[str, Any]:
        """Start the specified model process if it is not already running.

        Parameters
        ----------
        name : str
            The name of the model to start.

        Returns
        -------
        dict[str, Any]
            Response data from the service.
        """

        return self._command("start_model", name=name)

    def stop_model(self, name: str) -> dict[str, Any]:
        """Stop the specified model process if it is running.

        Parameters
        ----------
        name : str
            The name of the model to stop.

        Returns
        -------
        dict[str, Any]
            Response data from the service.
        """

        return self._command("stop_model", name=name)

    def shutdown(self) -> dict[str, Any]:
        """Request a graceful shutdown of the background hub service.

        Returns
        -------
        dict[str, Any]
            Response data from the service.
        """

        return self._command("shutdown")

    def wait_until_available(self, timeout: float = 10.0) -> bool:
        """Poll for availability until ``timeout`` seconds have elapsed.

        Parameters
        ----------
        timeout : float, default=10.0
            Maximum time in seconds to wait for availability.

        Returns
        -------
        bool
            True if the service became available, False if timeout elapsed.
        """

        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.is_available():
                return True
            time.sleep(0.2)
        return False

    def _command(self, action: str, **payload: Any) -> dict[str, Any]:
        """Send a command to the service and return the data response.

        Parameters
        ----------
        action : str
            The action to perform.
        **payload : Any
            Additional payload data.

        Returns
        -------
        dict[str, Any]
            The data from the response.

        Raises
        ------
        HubServiceError
            If the service returns an error.
        """

        message = {"action": action}
        message.update(payload)
        response = self._request(message, timeout=self._timeout)
        status = response.get("status")
        if status != "ok":
            error = response.get("error", "unknown error")
            code = response.get("code")
            raise HubServiceError(str(error), status_code=code)
        data = response.get("data")
        if data is None:
            return {}
        if not isinstance(data, dict):  # pragma: no cover - defensive guard
            raise HubServiceError("Hub service returned malformed data payload")
        return cast("dict[str, Any]", data)

    def _request(self, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
        """Send a request to the service and return the response.

        Parameters
        ----------
        payload : dict[str, Any]
            The request payload.
        timeout : float
            Timeout for the request.

        Returns
        -------
        dict[str, Any]
            The response dictionary.

        Raises
        ------
        HubServiceError
            If unable to reach the service.
        """

        start = time.time()
        last_error: Exception | None = None
        while time.time() - start < timeout:
            try:
                with Client(self._socket_path, authkey=HUB_SERVICE_AUTH_KEY) as conn:
                    conn.send(payload)
                    raw_response = conn.recv()
                    if not isinstance(raw_response, dict):  # pragma: no cover - defensive
                        raise HubServiceError("Hub service returned invalid payload")
                    return cast("dict[str, Any]", raw_response)
            except FileNotFoundError as exc:
                last_error = exc
                time.sleep(0.2)
            except ConnectionRefusedError as exc:
                last_error = exc
                time.sleep(0.2)
            except OSError as exc:
                last_error = exc
                time.sleep(0.2)
        raise HubServiceError(
            f"Unable to reach hub manager at {self._socket_path}: {last_error or 'timeout'}"
        )


def start_hub_service_process(config_path: Path | str) -> int:
    """Spawn the hub service in a dedicated process and return its PID.

    Parameters
    ----------
    config_path : Path or str
        Path to the hub configuration file.

    Returns
    -------
    int
        The process ID of the spawned service.
    """

    # Use subprocess.Popen with detached process group for proper background execution
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            f"from app.hub.service import _service_process_entrypoint; _service_process_entrypoint('{config_path}')",
        ],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=Path.cwd(),
    )
    return proc.pid


def _service_process_entrypoint(config_path: str) -> None:
    """Entrypoint for the hub service process.

    Parameters
    ----------
    config_path : str
        Path to the hub configuration file.
    """

    service = HubService(config_path)
    service.serve_forever()
