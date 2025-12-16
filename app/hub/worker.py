"""Sidecar worker process management utilities."""

from __future__ import annotations

import asyncio
from asyncio import subprocess as aio_subprocess
from collections.abc import Sequence
from contextlib import suppress
import os
from pathlib import Path
import socket
import sys
import time
from typing import Any

import httpx
from loguru import logger

from ..config import MLXServerConfig
from ..const import (
    DEFAULT_API_HOST,
    DEFAULT_SIDECAR_HEALTH_INTERVAL,
    DEFAULT_SIDECAR_HEALTH_TIMEOUT,
    DEFAULT_SIDECAR_SHUTDOWN_TIMEOUT,
)


class SidecarWorkerError(RuntimeError):
    """Raised when a sidecar worker process fails to start or remain healthy."""


def _comma_join(values: Sequence[object] | None) -> str | None:
    if not values:
        return None
    rendered = [str(value).strip() for value in values if value is not None]
    joined = ",".join(entry for entry in rendered if entry)
    return joined or None


def build_worker_command(config: MLXServerConfig) -> list[str]:
    """Return CLI arguments used to launch a worker for ``config``."""

    cmd: list[str] = [
        sys.executable,
        "-m",
        "app.main",
        "launch",
        "--model-path",
        config.model_path,
        "--model-type",
        config.model_type,
        "--context-length",
        str(config.context_length),
        "--port",
        str(config.port),
        "--host",
        config.host or DEFAULT_API_HOST,
        "--max-concurrency",
        str(config.max_concurrency),
        "--queue-timeout",
        str(config.queue_timeout),
        "--queue-size",
        str(config.queue_size),
        "--quantize",
        str(config.quantize),
        "--log-level",
        config.log_level,
    ]

    if config.config_name:
        cmd.extend(["--config-name", config.config_name])

    lora_paths = _comma_join(getattr(config, "lora_paths", None))
    if lora_paths:
        cmd.extend(["--lora-paths", lora_paths])

    lora_scales = _comma_join(getattr(config, "lora_scales", None))
    if lora_scales:
        cmd.extend(["--lora-scales", lora_scales])

    if getattr(config, "disable_auto_resize", False):
        cmd.append("--disable-auto-resize")

    if config.log_file:
        cmd.extend(["--log-file", config.log_file])

    if getattr(config, "no_log_file", False):
        cmd.append("--no-log-file")

    if getattr(config, "enable_auto_tool_choice", False):
        cmd.append("--enable-auto-tool-choice")

    if config.tool_call_parser:
        cmd.extend(["--tool-call-parser", config.tool_call_parser])

    if config.reasoning_parser:
        cmd.extend(["--reasoning-parser", config.reasoning_parser])

    if config.draft_model_path:
        cmd.extend(["--draft-model-path", config.draft_model_path])

    if config.draft_tokens is not None:
        cmd.extend(["--draft-tokens", str(config.draft_tokens)])

    if getattr(config, "trust_remote_code", False):
        cmd.append("--trust-remote-code")

    if getattr(config, "jit_enabled", False):
        cmd.append("--jit")

    if config.auto_unload_minutes is not None:
        cmd.extend(["--auto-unload-minutes", str(config.auto_unload_minutes)])

    if config.worker_extra_args:
        cmd.extend([arg for arg in config.worker_extra_args if arg])

    return cmd


class SidecarWorker:
    """Manage the lifecycle of a single worker subprocess."""

    def __init__(
        self,
        *,
        name: str,
        config: MLXServerConfig,
        health_interval: float = DEFAULT_SIDECAR_HEALTH_INTERVAL,
        health_timeout: float = DEFAULT_SIDECAR_HEALTH_TIMEOUT,
        shutdown_timeout: float = DEFAULT_SIDECAR_SHUTDOWN_TIMEOUT,
    ) -> None:
        self.name = name
        self.config = config
        self._health_interval = max(health_interval, 0.1)
        self._health_timeout = max(health_timeout, self._health_interval)
        self._shutdown_timeout = max(shutdown_timeout, 1.0)
        self._process: aio_subprocess.Process | None = None
        self._ready = False
        self._ready_at: float | None = None
        self._cwd = Path(__file__).resolve().parents[2]
        self._stdout_task: asyncio.Task[Any] | None = None
        self._stderr_task: asyncio.Task[Any] | None = None

    @property
    def port(self) -> int:
        """Return the TCP port assigned to the worker."""

        return int(self.config.port)

    @property
    def pid(self) -> int | None:
        """Return the process identifier for the worker if running."""

        return self._process.pid if self._process else None

    @property
    def ready(self) -> bool:
        """Indicate whether the worker passed its health check."""

        return self._ready

    @property
    def ready_at(self) -> float | None:
        """Return the epoch timestamp when the worker became healthy."""

        return self._ready_at

    @property
    def returncode(self) -> int | None:
        """Return the exit code of the worker process, if available."""

        if self._process is None:
            return None
        return self._process.returncode

    def is_running(self) -> bool:
        """Return True when the worker subprocess is alive."""

        return self._process is not None and self._process.returncode is None

    async def start(self) -> None:
        """Start the worker process and wait for its health endpoint."""

        if self.is_running():
            if not self._ready:
                await self._wait_for_health()
            return

        # Preflight: check whether the configured port is already in use so we
        # fail fast with a helpful message instead of spawning a process that
        # will immediately exit with a bind error.
        try:
            host = self._health_host()
            port = int(self.config.port)
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.5)
                try:
                    s.connect((host, port))
                except OSError:
                    # Connection refused / no listener — port appears free
                    port_in_use = False
                else:
                    port_in_use = True

            if port_in_use:
                # Always attempt a fallback to the next free port(s). This
                # improves robustness when the configured port is transiently
                # occupied by another process.
                max_fallbacks = 10
                found: int | None = None
                for offset in range(1, max_fallbacks + 1):
                    candidate = port + offset
                    try:
                        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s2:
                            s2.settimeout(0.5)
                            try:
                                s2.connect((host, candidate))
                            except OSError:
                                # candidate free
                                found = candidate
                                break
                            else:
                                continue
                    except Exception:
                        continue
                if found is None:
                    raise SidecarWorkerError(
                        f"Port {port} on host {host} is in use and no fallback port was found within {max_fallbacks} attempts",
                    )
                # Update the worker config to use the fallback port
                old = self.config.port
                self.config.port = found
                logger.info(
                    f"Port {old} in use; falling back to port {found} for worker '{self.name}'",
                )
        except SidecarWorkerError:
            raise
        except Exception:
            # Best-effort preflight; don't prevent startup for unexpected
            # socket errors — let the subprocess report bind failures.
            pass

        command = build_worker_command(self.config)
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        try:
            self._process = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(self._cwd),
                env=env,
                stdout=aio_subprocess.PIPE,
                stderr=aio_subprocess.PIPE,
            )
        except OSError as exc:  # pragma: no cover - passthrough logging
            raise SidecarWorkerError(f"Failed to spawn worker process: {exc}") from exc

        logger.info(
            f"Started sidecar worker '{self.name}' (pid={self.pid}, port={self.port})",
        )
        # Start background readers for stdout/stderr so we capture any startup
        # diagnostics emitted by the worker process.
        if self._process.stdout is not None:
            self._stdout_task = asyncio.create_task(
                self._drain_stream(self._process.stdout, "stdout")
            )
        if self._process.stderr is not None:
            self._stderr_task = asyncio.create_task(
                self._drain_stream(self._process.stderr, "stderr")
            )

        await self._wait_for_health()

    async def stop(self) -> None:
        """Terminate the worker process, waiting for graceful shutdown."""

        if self._process is None:
            self._ready = False
            return

        if self._process.returncode is not None:
            self._ready = False
            return

        self._process.terminate()
        try:
            await asyncio.wait_for(self._process.wait(), timeout=self._shutdown_timeout)
        except TimeoutError:
            logger.warning(
                f"Worker '{self.name}' did not exit within {self._shutdown_timeout}s; killing",
            )
            self._process.kill()
            await self._process.wait()
        finally:
            logger.info(
                f"Stopped sidecar worker '{self.name}' (pid={self.pid}, code={self._process.returncode})",
            )
            self._ready = False
            # Cancel background stream readers
            for task in (self._stdout_task, self._stderr_task):
                if task is not None and not task.done():
                    task.cancel()
                    with suppress(Exception):
                        await task

    async def _wait_for_health(self) -> None:
        """Poll the worker's health endpoint until it reports ready."""

        deadline = time.time() + self._health_timeout
        url = f"http://{self._health_host()}:{self.port}/health"
        async with httpx.AsyncClient(timeout=self._health_interval + 1.0) as client:
            while time.time() < deadline:
                if not self.is_running():
                    raise SidecarWorkerError(
                        f"Worker exited before becoming healthy (code={self.returncode})",
                    )
                try:
                    response = await client.get(url)
                except httpx.HTTPError:
                    response = None
                if response is not None and response.status_code == 200:
                    self._ready = True
                    self._ready_at = time.time()
                    logger.debug(f"Worker '{self.name}' is healthy at {url}")
                    return
                await asyncio.sleep(self._health_interval)

        raise SidecarWorkerError(
            f"Timed out waiting for worker '{self.name}' health at {url}",
        )

    async def _drain_stream(self, stream: asyncio.StreamReader, stream_name: str) -> None:
        """Read lines from ``stream`` and log them with a worker prefix.

        This helps capture startup errors or early exit diagnostics from the
        child process so the hub operator can debug failures more easily.
        """
        prefix = f"worker[{self.name}].{stream_name}"
        bound_logger = logger.bind(model=self.name)
        try:
            while True:
                line = await stream.readline()
                if not line:
                    break
                try:
                    text = line.decode("utf-8", errors="replace").rstrip()
                except Exception:
                    text = str(line)
                if text:
                    # Send worker logs to the model-specific sink by binding
                    # the `model` extra. Use INFO for stderr (diagnostic
                    # messages) instead of ERROR so they do not surface as
                    # global errors, and DEBUG for stdout.
                    if stream_name == "stderr":
                        bound_logger.info(f"{prefix}: {text}")
                    else:
                        bound_logger.debug(f"{prefix}: {text}")
        except asyncio.CancelledError:
            return
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning(f"Error reading {prefix}: {exc}")

    def _health_host(self) -> str:
        host = (self.config.host or DEFAULT_API_HOST).strip()
        if host in {"0.0.0.0", "::", "[::]"}:
            return DEFAULT_API_HOST
        if host.startswith("[") and host.endswith("]"):
            return host[1:-1]
        return host
