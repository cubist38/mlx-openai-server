"""Process-based hub manager that reloads hub.yaml and orchestrates models."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
import importlib
import multiprocessing
from pathlib import Path
import sys
import threading
import time
from typing import cast

from loguru import logger

from ..config import MLXServerConfig
from .config import MLXHubConfig, load_hub_config
from .errors import HubManagerError, HubProcessError
from .observability import HubModelContext, HubObservabilitySink, LoggingHubObservabilitySink


class ManagedProcess:
    """Minimal interface implemented by hub-managed subprocesses."""

    def start(self) -> None:  # pragma: no cover - interface definition
        """Start the underlying process if it is not already running."""

        raise NotImplementedError

    def stop(self, timeout: float = 10.0) -> None:  # pragma: no cover - interface definition
        """Stop the underlying process, waiting up to ``timeout`` seconds."""

        raise NotImplementedError

    def poll(self) -> int | None:  # pragma: no cover - interface definition
        """Return the exit code if the process ended, else ``None``."""

        raise NotImplementedError

    @property
    def pid(self) -> int | None:  # pragma: no cover - interface definition
        """Return the operating system PID for the managed process."""

        raise NotImplementedError


def _import_launch_single_model() -> Callable[[MLXServerConfig], Awaitable[None]]:
    """Import ``start`` lazily to avoid circular imports."""

    # Add the repository root to sys.path to ensure 'app' module is importable
    repo_root = Path(__file__).resolve().parent.parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    module = importlib.import_module("app.main")
    return cast("Callable[[MLXServerConfig], Awaitable[None]]", module.start)


class MultiprocessingModelProcess(ManagedProcess):
    """Spawn a background process that runs the standard single-model server."""

    def __init__(self, config: MLXServerConfig, log_path: Path | None) -> None:
        self._config = config
        self._log_path = log_path
        ctx = multiprocessing.get_context("spawn")
        self._process = ctx.Process(
            target=_model_process_entry,
            args=(config, str(log_path) if log_path else None),
            daemon=True,
        )
        self._started = False

    def start(self) -> None:
        """Start the configured multiprocessing target."""

        if self._started:
            return
        self._process.start()
        self._started = True

    def stop(self, timeout: float = 10.0) -> None:
        """Terminate the process and wait for it to exit."""

        if not self._started:
            return
        if not self._process.is_alive():
            self._process.join(timeout=0)
            return
        self._process.terminate()
        self._process.join(timeout)
        if self._process.is_alive():
            self._process.kill()
            self._process.join()

    def poll(self) -> int | None:
        """Return the exit code once available."""

        return self._process.exitcode

    @property
    def pid(self) -> int | None:
        """Return the PID once the process has started."""

        return self._process.pid if self._started else None


def _model_process_entry(config: MLXServerConfig, log_path: str | None) -> None:
    """Entrypoint executed inside each managed process."""

    launch_single_model = _import_launch_single_model()

    async def _run() -> None:
        await launch_single_model(config)

    stack = ExitStack()
    stream = None
    try:
        if log_path:
            path = Path(log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            stream = path.open("a", encoding="utf-8")
            stack.enter_context(stream)
            stack.enter_context(redirect_stdout(stream))
            stack.enter_context(redirect_stderr(stream))
            # Configure loguru to output to the stream
            logger.remove()
            logger.add(
                stream, level="DEBUG", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
            )
        asyncio.run(_run())
    finally:
        stack.close()


@dataclass(slots=True)
class HubProcessRecord:
    """Represents a single managed model process and its telemetry."""

    name: str
    config: MLXServerConfig
    process: ManagedProcess
    log_path: Path | None
    started_at: float = field(default_factory=time.time)
    stopped_at: float | None = None
    exit_code: int | None = None
    exit_handled: bool = False

    def refresh(self) -> None:
        """Update cached exit code information from the underlying process."""

        if self.exit_code is not None:
            return
        code = self.process.poll()
        if code is not None:
            self.exit_code = code
            self.stopped_at = time.time()

    @property
    def state(self) -> str:
        """Return a coarse state string for status reporting."""

        self.refresh()
        if self.exit_code is None:
            return "running"
        if self.exit_code == 0:
            return "stopped"
        return "failed"

    @property
    def pid(self) -> int | None:
        """Return the PID of the managed process."""

        return self.process.pid

    def context(self) -> HubModelContext:
        """Return immutable context describing this model for logging hooks."""

        log_path = str(self.log_path) if self.log_path else None
        return HubModelContext(name=self.name, group=self.config.group, log_path=log_path)


@dataclass(slots=True)
class HubReloadResult:
    """Summary describing what changed after a hub config reload."""

    started: list[str] = field(default_factory=list)
    stopped: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)


@dataclass(slots=True)
class HubModelStatus:
    """Serializable snapshot describing a managed model."""

    name: str
    state: str
    group: str | None
    log_path: str | None
    pid: int | None
    port: int | None
    exit_code: int | None
    started_at: float | None
    stopped_at: float | None


ProcessFactory = Callable[[MLXServerConfig], ManagedProcess]


class HubManager:
    """Reloads hub.yaml and supervises per-model background processes."""

    def __init__(
        self,
        config_path: Path | str,
        *,
        process_factory: ProcessFactory | None = None,
        observability_sink: HubObservabilitySink | None = None,
    ) -> None:
        """Initialize the hub manager.

        Parameters
        ----------
        config_path : Path or str
            Path to the hub configuration file.
        process_factory : ProcessFactory, optional
            Factory for creating managed processes.
        observability_sink : HubObservabilitySink, optional
            Sink for observability events.
        """

        self._config_path = Path(config_path).expanduser()
        self._process_factory = process_factory or self._default_process_factory
        self._lock = threading.Lock()
        self._records: dict[str, HubProcessRecord] = {}
        self._model_configs: dict[str, MLXServerConfig] = {}
        self._hub_config: MLXHubConfig | None = None
        self._observability: HubObservabilitySink = (
            observability_sink or LoggingHubObservabilitySink()
        )

    def reload(self) -> HubReloadResult:
        """Reload hub.yaml, starting/stopping processes as needed.

        Returns
        -------
        HubReloadResult
            Summary of changes made during reload.
        """
        with self._lock:
            persisted_ports = {
                name: config.port
                for name, config in self._model_configs.items()
                if config.port is not None
            }

        config = load_hub_config(self._config_path, persisted_ports=persisted_ports)

        with self._lock:
            previous_configs = dict(self._model_configs)
            self._refresh_all_records_locked()
            return self._apply_config(config, previous_configs)

    def start_model(self, name: str) -> None:
        """Ensure the specified model process is running.

        Parameters
        ----------
        name : str
            The name of the model to start.
        """

        with self._lock:
            self._ensure_config_loaded()
            if name in self._records:
                record = self._records[name]
                self._refresh_record_locked(record)
                if record.exit_code is None:
                    logger.info(f"Model '{name}' already running")
                    return
                self._stop_model_locked(name)
            config = self._model_configs.get(name)
            if config is None:
                raise HubManagerError(f"Model '{name}' not defined in hub config")
            self._start_model_locked(config)

    def stop_model(self, name: str) -> None:
        """Stop a running model process.

        Parameters
        ----------
        name : str
            The name of the model to stop.
        """

        with self._lock:
            self._refresh_all_records_locked()
            if name not in self._records:
                logger.info(f"Model '{name}' is not running")
                return
            self._stop_model_locked(name)

    def get_status(self) -> list[HubModelStatus]:
        """Return per-model status snapshots.

        Returns
        -------
        list[HubModelStatus]
            List of status snapshots for all models.
        """

        with self._lock:
            self._refresh_all_records_locked()
            snapshots = []
            for name, record in self._records.items():
                snapshots.append(
                    HubModelStatus(
                        name=name,
                        state=record.state,
                        group=record.config.group,
                        log_path=str(record.log_path) if record.log_path else None,
                        pid=record.pid,
                        port=record.config.port,
                        exit_code=record.exit_code,
                        started_at=record.started_at,
                        stopped_at=record.stopped_at,
                    )
                )
            return snapshots

    def shutdown(self) -> None:
        """Stop all running model processes and clean up resources."""

        with self._lock:
            names = list(self._records)
            for name in names:
                self._stop_model_locked(name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_config(
        self, config: MLXHubConfig, previous_configs: dict[str, MLXServerConfig]
    ) -> HubReloadResult:
        """Apply a new hub configuration, starting/stopping processes as needed.

        Parameters
        ----------
        config : MLXHubConfig
            The new hub configuration.

        Returns
        -------
        HubReloadResult
            Summary of changes made.
        """

        self._hub_config = config

        desired = {model.name: model for model in config.models if model.name}
        current_names = set(self._records)
        desired_names = set(desired)

        removed = current_names - desired_names
        added = desired_names - current_names
        maybe_updated = desired_names & current_names

        result = HubReloadResult()
        previous_configs = previous_configs or {}

        for name in removed:
            self._stop_model_locked(name)
            result.stopped.append(name)

        for name in maybe_updated:
            record = self._records[name]
            new_config = desired[name]
            needs_restart = record.exit_code is not None or not self._configs_match(
                record.config, new_config
            )
            if needs_restart:
                self._stop_model_locked(name)
                result.stopped.append(name)
                self._start_model_locked(new_config)
                result.started.append(name)
            else:
                result.unchanged.append(name)

        for name in added:
            config_obj = desired[name]
            previously_known = name in previous_configs
            if config_obj.is_default_model and not previously_known:
                self._start_model_locked(config_obj)
                result.started.append(name)
            else:
                logger.info(
                    f"Model '{name}' configured as on-demand; skipping automatic start",
                )
                result.unchanged.append(name)

        self._model_configs = desired
        return result

    def _start_model_locked(self, config: MLXServerConfig) -> None:
        """Start a model process with the given configuration.

        Parameters
        ----------
        config : MLXServerConfig
            The configuration for the model to start.

        Raises
        ------
        HubManagerError
            If the model name is missing.
        HubProcessError
            If starting the process fails.
        """

        name = config.name
        if not name:
            raise HubManagerError("Cannot start unnamed model entry")
        process = self._process_factory(config)
        log_path = Path(config.log_file).expanduser() if config.log_file else None
        try:
            process.start()
        except Exception as exc:  # pragma: no cover - defensive guard
            raise HubProcessError(f"Failed to start process for model '{name}': {exc}") from exc
        record = HubProcessRecord(name=name, config=config, process=process, log_path=log_path)
        self._records[name] = record
        self._observability.model_started(record.context(), pid=process.pid)

    def _stop_model_locked(self, name: str, *, update_structures: bool = True) -> None:
        """Stop a model process.

        Parameters
        ----------
        name : str
            The name of the model to stop.
        update_structures : bool, default=True
            Whether to update internal data structures.
        """

        record = self._records.get(name)
        if record is None:
            return
        try:
            record.process.stop()
        except Exception as exc:  # pragma: no cover - defensive guard
            raise HubProcessError(f"Failed to stop process for model '{name}': {exc}") from exc
        record.refresh()
        if not record.exit_handled:
            self._observability.model_stopped(record.context(), exit_code=record.exit_code)
        if update_structures:
            self._records.pop(name, None)

    def _ensure_config_loaded(self) -> None:
        """Ensure the hub configuration is loaded.

        Raises
        ------
        HubManagerError
            If configuration is not loaded.
        """

        if self._hub_config is None:
            raise HubManagerError("Hub configuration not loaded; call reload() first")

    def _default_process_factory(self, config: MLXServerConfig) -> ManagedProcess:
        """Create a default multiprocessing model process.

        Parameters
        ----------
        config : MLXServerConfig
            The server configuration.

        Returns
        -------
        ManagedProcess
            The managed process instance.
        """

        log_path = Path(config.log_file).expanduser() if config.log_file else None
        return MultiprocessingModelProcess(config, log_path)

    @staticmethod
    def _configs_match(a: MLXServerConfig, b: MLXServerConfig) -> bool:
        """Check if two configurations match.

        Parameters
        ----------
        a : MLXServerConfig
            First configuration.
        b : MLXServerConfig
            Second configuration.

        Returns
        -------
        bool
            True if configurations match.
        """

        return a.__dict__ == b.__dict__

    def _refresh_all_records_locked(self) -> None:
        """Refresh all process records."""

        for record in list(self._records.values()):
            self._refresh_record_locked(record)

    def _refresh_record_locked(self, record: HubProcessRecord) -> None:
        """Refresh a single process record.

        Parameters
        ----------
        record : HubProcessRecord
            The record to refresh.
        """

        previous_exit = record.exit_code
        record.refresh()
        if record.exit_code is not None and previous_exit is None:
            self._handle_process_exit_locked(record)

    def _handle_process_exit_locked(self, record: HubProcessRecord) -> None:
        """Handle process exit for a record.

        Parameters
        ----------
        record : HubProcessRecord
            The record whose process exited.
        """

        if record.exit_handled:
            return
        record.exit_handled = True
        if record.exit_code == 0:
            self._observability.model_stopped(record.context(), exit_code=record.exit_code)
        else:
            self._observability.model_failed(record.context(), exit_code=record.exit_code)
