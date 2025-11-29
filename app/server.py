"""Application server helpers with JIT-aware handler management.

This module provides the core server infrastructure for the MLX OpenAI-compatible
API server, including FastAPI application setup, handler lifecycle management,
and JIT (Just-In-Time) model loading with automatic unloading capabilities.

The server supports multiple MLX model types including language models, multimodal
models, image generation, embeddings, and audio transcription. It features
configurable concurrency limits, memory management, and optional automatic
model unloading during idle periods to optimize resource usage.

Key Components
--------------
LazyHandlerManager : Manages model handler lifecycle with optional JIT loading
CentralIdleAutoUnloadController : Central background task for automatic model unloading
FastAPI Integration : REST API endpoints with OpenAI-compatible interface
Memory Management : Periodic cleanup and MLX cache management

Notes
-----
The server uses loguru for structured logging and supports both console and
rotating file output. All handlers implement async initialization and cleanup
protocols for proper resource management.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager, suppress
import gc
from http import HTTPStatus
from pathlib import Path
import sys
import time
from typing import Any, TypeAlias, cast

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
import mlx.core as mx
from starlette.responses import Response
import uvicorn

from .api.endpoints import router
from .config import MLXServerConfig
from .core.manager_protocol import ManagerProtocol
from .core.model_registry import ModelRegistry
from .handler import MFLUX_AVAILABLE, MLXFluxHandler
from .handler.mlx_embeddings import MLXEmbeddingsHandler
from .handler.mlx_lm import MLXLMHandler
from .handler.mlx_vlm import MLXVLMHandler
from .handler.mlx_whisper import MLXWhisperHandler
from .middleware import RequestTrackingMiddleware
from .version import __version__

# Type alias for MLX handlers
MLXHandler: TypeAlias = (
    MLXVLMHandler | MLXFluxHandler | MLXEmbeddingsHandler | MLXWhisperHandler | MLXLMHandler
)

# Supported mflux configuration names per feature for centralized validation
ALLOWED_IMAGE_GENERATION_CONFIGS: frozenset[str] = frozenset(
    {"flux-schnell", "flux-dev", "flux-krea-dev"},
)
ALLOWED_IMAGE_EDIT_CONFIGS: frozenset[str] = frozenset({"flux-kontext-dev"})


def configure_logging(
    log_file: str | None = None,
    no_log_file: bool = False,
    log_level: str = "INFO",
) -> None:
    """Set up loguru handlers used by the server.

    This helper replaces the default loguru handler with a console
    handler using a compact, colored format. When ``no_log_file`` is
    False a rotating file handler is also added using ``log_file`` or
    a default path.

    Parameters
    ----------
    log_file : str, optional
        Optional filesystem path where logs should be written. When
        ``None`` and file logging is enabled a sensible default
        (``logs/app.log``) is used.
    no_log_file : bool, default False
        When True, file logging is disabled and only console logs are
        emitted.
    log_level : str, default "INFO"
        Minimum log level to emit (e.g. "DEBUG", "INFO").
    """
    logger.remove()  # Remove default handler

    # Ensure log directory exists when a file path is specified (or default)
    if log_file:
        with suppress(Exception):
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    else:
        with suppress(Exception):
            Path("logs").mkdir(parents=True, exist_ok=True)

    # Add console handler. Exclude records that are specific to a model
    # (they will be written to per-model log files instead).
    def _global_filter(record: dict[str, Any]) -> bool:  # pragma: no cover - tiny helper
        try:
            return record.get("extra", {}).get("model") is None
        except Exception:
            return True

    logger.add(
        sys.stdout,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "✦ <level>{message}</level>",
        colorize=True,
        enqueue=True,
        filter=cast("Callable[[Any], bool]", _global_filter),
    )
    if not no_log_file:
        file_path = log_file if log_file else "logs/app.log"
        with suppress(Exception):
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            file_path,
            rotation="500 MB",
            retention="10 days",
            level=log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
            enqueue=True,
            filter=cast("Callable[[Any], bool]", _global_filter),
        )


def get_model_identifier(config_args: MLXServerConfig) -> str:
    """Compute the identifier passed to MLX handlers.

    Presently the identifier is the raw model path supplied on the
    command line. This helper centralizes that logic so it can be
    changed in a single place later (for example, to map shortcuts to
    real paths).

    Parameters
    ----------
    config_args : MLXServerConfig
        Configuration object produced by the CLI. The attribute
        ``model_path`` is read to produce the identifier.

    Returns
    -------
    str
        Value that identifies the model for handler initialization.
    """
    return config_args.model_path


def get_registry_model_id(config_args: MLXServerConfig) -> str:
    """Return the identifier used when registering models with the registry."""
    return config_args.name or config_args.model_identifier


def validate_mflux_config(
    config_name: str,
    allowed_configs: frozenset[str],
    feature_name: str,
) -> None:
    """Ensure mflux is available and the provided config name is supported.

    Parameters
    ----------
    config_name : str
        The configuration name supplied via CLI/args.
    allowed_configs : frozenset[str]
        The allowed configuration names for the feature.
    feature_name : str
        Human-friendly name of the feature (e.g., "Image generation").

    Raises
    ------
    ValueError
        Raised when mflux is missing or the configuration name is unsupported.
    """
    if not MFLUX_AVAILABLE:
        raise ValueError(
            f"{feature_name} requires mflux. Install with: pip install git+https://github.com/cubist38/mflux.git",
        )

    if config_name not in allowed_configs:
        allowed_values = ", ".join(sorted(allowed_configs))
        raise ValueError(
            f"Invalid config name: {config_name}. Supported configs for {feature_name.lower()} are: {allowed_values}.",
        )


async def instantiate_handler(config_args: MLXServerConfig) -> MLXHandler:
    """Instantiate and initialize the MLX handler for the given config.

    Based on the model type in the configuration, this function creates
    the appropriate handler instance (e.g., MLXLMHandler for language
    models, MLXVLMHandler for multimodal, etc.) and initializes it with
    the provided settings.

    Parameters
    ----------
    config_args : MLXServerConfig
        Configuration object containing model type, path, and other
        handler initialization parameters.

    Returns
    -------
    handler
        An initialized MLX handler instance ready for use.

    Raises
    ------
    ValueError
        If the model type is unsupported or required dependencies are
        missing (e.g., mflux for image generation).
    Exception
        If handler initialization fails for any reason.
    """
    model_identifier = get_model_identifier(config_args)
    if config_args.model_type == "image-generation":
        logger.info(f"Initializing MLX handler with model name: {model_identifier}")
    else:
        logger.info(f"Initializing MLX handler with model path: {model_identifier}")

    try:
        handler: MLXHandler
        if config_args.model_type == "multimodal":
            handler = MLXVLMHandler(
                model_path=model_identifier,
                context_length=config_args.context_length,
                max_concurrency=config_args.max_concurrency,
                disable_auto_resize=config_args.disable_auto_resize,
                enable_auto_tool_choice=config_args.enable_auto_tool_choice,
                tool_call_parser=config_args.tool_call_parser,
                reasoning_parser=config_args.reasoning_parser,
                trust_remote_code=config_args.trust_remote_code,
            )
        elif config_args.model_type == "image-generation":
            if config_args.config_name is None:
                raise ValueError("config_name is required for image-generation models")
            validate_mflux_config(
                config_args.config_name,
                ALLOWED_IMAGE_GENERATION_CONFIGS,
                "Image generation",
            )
            handler = MLXFluxHandler(
                model_path=model_identifier,
                max_concurrency=config_args.max_concurrency,
                quantize=config_args.quantize,
                config_name=config_args.config_name,
                lora_paths=config_args.lora_paths,
                lora_scales=config_args.lora_scales,
            )
        elif config_args.model_type == "embeddings":
            handler = MLXEmbeddingsHandler(
                model_path=model_identifier,
                max_concurrency=config_args.max_concurrency,
            )
        elif config_args.model_type == "image-edit":
            if config_args.config_name is None:
                raise ValueError("config_name is required for image-edit models")
            validate_mflux_config(
                config_args.config_name,
                ALLOWED_IMAGE_EDIT_CONFIGS,
                "Image editing",
            )
            handler = MLXFluxHandler(
                model_path=model_identifier,
                max_concurrency=config_args.max_concurrency,
                quantize=config_args.quantize,
                config_name=config_args.config_name,
                lora_paths=config_args.lora_paths,
                lora_scales=config_args.lora_scales,
            )
        elif config_args.model_type == "whisper":
            handler = MLXWhisperHandler(
                model_path=model_identifier,
                max_concurrency=config_args.max_concurrency,
            )
        elif config_args.model_type == "lm":
            handler = MLXLMHandler(
                model_path=model_identifier,
                context_length=config_args.context_length,
                max_concurrency=config_args.max_concurrency,
                enable_auto_tool_choice=config_args.enable_auto_tool_choice,
                tool_call_parser=config_args.tool_call_parser,
                reasoning_parser=config_args.reasoning_parser,
                trust_remote_code=config_args.trust_remote_code,
            )
        else:
            raise ValueError(
                f"Invalid model_type: {config_args.model_type!r}. "
                f"Supported types are: lm, multimodal, image-generation, image-edit, embeddings, whisper",
            )

        await handler.initialize(
            {
                "max_concurrency": config_args.max_concurrency,
                "timeout": config_args.queue_timeout,
                "queue_size": config_args.queue_size,
            },
        )
        logger.info("MLX handler initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize MLX handler. {type(e).__name__}: {e}")
        raise
    else:
        return handler


class LazyHandlerManager(ManagerProtocol):
    """Manage handler lifecycle with optional JIT loading and unloading."""

    _handler: MLXHandler | None

    def __init__(
        self,
        config_args: MLXServerConfig,
        *,
        on_change: Callable[[MLXHandler | None], None] | None = None,
        on_activity: Callable[[], None] | None = None,
    ) -> None:
        """Initialize the LazyHandlerManager.

        Parameters
        ----------
        config_args : MLXServerConfig
            Configuration arguments for the server.
        on_change : Callable[[MLXHandler | None], None], optional
            Callback called when the handler changes.
        on_activity : Callable[[], None], optional
            Callback called on activity.
        """
        self.config_args = config_args
        self._handler = None
        self._lock = asyncio.Lock()
        self._shutdown = False
        self._on_change = on_change
        self._on_activity = on_activity
        self._last_activity = time.monotonic()
        self._background_tasks: set[asyncio.Task[None]] = set()
        # Optional per-model log sink id (registered with loguru)
        # Bound logger for manager-scoped messages (adds `model` extra)
        bound_model_name = getattr(self.config_args, "name", None) or getattr(
            self.config_args, "model_identifier", None
        )
        self._logger = logger.bind(model=bound_model_name) if bound_model_name else logger
        self._log_sink_id: int | None = None

        # If model-level file logging is enabled, add a dedicated sink
        log_file: str | None = getattr(self.config_args, "log_file", None)
        if not getattr(self.config_args, "no_log_file", False) and log_file:
            try:
                # Ensure parent dir exists
                with suppress(Exception):
                    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
                model_name: str | None = getattr(self.config_args, "name", None) or getattr(
                    self.config_args, "model_identifier", None
                )

                # Add a model-specific file sink that only receives records
                # bound to this model via the `model` extra.
                def _model_filter(record: dict[str, Any]) -> bool:  # pragma: no cover - tiny helper
                    try:
                        extra = record.get("extra", {})
                        if not isinstance(extra, dict):
                            return False
                        # Cast final result to bool to avoid returning Any
                        return bool(extra.get("model") == model_name)
                    except Exception:
                        return False

                self._log_sink_id = logger.add(
                    log_file,
                    level=getattr(self.config_args, "log_level", "INFO"),
                    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
                    enqueue=True,
                    filter=cast("Callable[[Any], bool]", _model_filter),
                )
            except Exception:  # pragma: no cover - best-effort logging
                # Log the exception with the bound logger so model context is preserved
                self._logger.exception("Failed to add model log sink")

        if self._on_change:
            self._on_change(None)

    @property
    def auto_unload_minutes(self) -> int | None:
        """Get the auto-unload timeout in minutes from configuration."""
        return self.config_args.auto_unload_minutes

    @property
    def jit_enabled(self) -> bool:
        """Check if JIT loading is enabled."""
        return self.config_args.jit_enabled

    @property
    def current_handler(self) -> MLXHandler | None:
        """Get the currently loaded handler instance, or None if unloaded."""
        return self._handler

    def record_activity(self) -> None:
        """Record that activity has occurred for auto-unload tracking.

        This method updates the last activity timestamp and calls the
        on_activity callback if provided.
        """
        self._last_activity = time.monotonic()
        if self._on_activity:
            self._on_activity()

    def set_activity_callback(self, callback: Callable[[], None] | None) -> None:
        """Set the activity callback function.

        Parameters
        ----------
        callback : Callable[[], None] or None
            Function to call when activity occurs, or None to disable.
        """
        self._on_activity = callback

    def seconds_since_last_activity(self) -> float:
        """Return the number of seconds since the last recorded activity.

        Returns
        -------
        float
            Seconds since last activity.
        """
        return time.monotonic() - self._last_activity

    def idle_timeout_seconds(self) -> int | None:
        """Return the idle timeout in seconds, or None if auto-unload is disabled.

        Returns
        -------
        int or None
            Idle timeout in seconds, or None if disabled.
        """
        if self.auto_unload_minutes is None:
            return None
        return self.auto_unload_minutes * 60

    async def ensure_loaded(self, reason: str = "request") -> MLXHandler | None:
        """Ensure the handler is loaded, loading it if necessary.

        If the handler is already loaded and not shutting down, records
        activity and returns it. Otherwise, instantiates a new handler
        and loads it.

        Parameters
        ----------
        reason : str, default "request"
            Reason for loading, used in log messages.

        Returns
        -------
        MLXHandler or None
            The loaded handler instance, or None if shutting down.
        """
        async with self._lock:
            if self._handler and not self._shutdown:
                self.record_activity()
                return self._handler

            if self._shutdown:
                return None

            # Use the bound per-model logger so per-model sinks receive records
            self._logger.info(self._format_log_message("Loading model", reason))
            handler = await instantiate_handler(self.config_args)
            self._handler = handler
            self.record_activity()
            if self._on_change:
                self._on_change(handler)
            self._schedule_memory_cleanup()
            self._logger.info(self._format_log_message("Model loaded", reason))
            return handler

    async def unload(self, reason: str = "manual") -> bool:
        """Unload the current handler if loaded.

        Cleans up the handler resources and clears memory caches.

        Parameters
        ----------
        reason : str, default "manual"
            Reason for unloading, used in log messages.

        Returns
        -------
        bool
            True if a handler was unloaded, False if none was loaded.
        """
        if not self._handler:
            return False

        async with self._lock:
            if not self._handler:
                return False
            handler = self._handler
            self._handler = None
            if self._on_change:
                self._on_change(None)

        try:
            self._logger.info(self._format_log_message("Unloading model", reason))
            await handler.cleanup()
            self._logger.info(self._format_log_message("Model unloaded", reason))
        finally:
            mx.clear_cache()
            gc.collect()
        return True

    async def shutdown(self) -> None:
        """Shutdown the handler manager and unload any loaded handler.

        Sets the shutdown flag and unloads the handler.
        """
        self._shutdown = True
        await self.unload("shutdown")
        # Remove any per-model log sink when the manager is shut down
        with suppress(Exception):
            self.remove_log_sink()

    # ManagerProtocol adapter methods -------------------------------------------------
    def is_vram_loaded(self) -> bool:
        """Return True when the handler is currently loaded into memory/VRAM."""
        return self._handler is not None

    async def ensure_vram_loaded(
        self,
        *,
        force: bool = False,
        timeout: float | None = None,
    ) -> None:
        """Ensure the handler is loaded (idempotent).

        Maps the ManagerProtocol call onto the existing ``ensure_loaded`` behavior.
        """
        if self._shutdown:
            raise RuntimeError("Manager is shutting down")

        if not self._handler or force:
            coro = self.ensure_loaded("ensure_vram")
            if timeout is not None:
                await asyncio.wait_for(coro, timeout=timeout)
            else:
                await coro

    async def release_vram(self, *, timeout: float | None = None) -> None:
        """Release VRAM resources by unloading the handler (idempotent)."""
        coro = self.unload("release_vram")
        if timeout is not None:
            await asyncio.wait_for(coro, timeout=timeout)
        else:
            await coro

    def request_session(
        self,
        *,
        ensure_vram: bool = True,
        ensure_timeout: float | None = None,
    ) -> AbstractAsyncContextManager[Any]:
        """Return an async context manager for per-request sessions.

        This adapter notifies activity and optionally ensures VRAM before yielding
        the manager (self). It is intentionally lightweight because the
        authoritative active-request counter lives in `ModelRegistry`.
        """

        @asynccontextmanager
        async def _session() -> AsyncGenerator[Any, None]:
            # Record activity so central controller timers are reset
            self.record_activity()

            if ensure_vram:
                coro = self.ensure_vram_loaded()
                if ensure_timeout is not None:
                    await asyncio.wait_for(coro, timeout=ensure_timeout)
                else:
                    await coro

            try:
                yield self
            finally:
                # Mark activity on exit (helps with last-activity heuristics)
                self.record_activity()

        return _session()

    def remove_log_sink(self) -> None:
        """Remove per-model log sink if one was registered.

        This is intentionally best-effort: failures to remove the sink are
        logged but do not raise.
        """
        if self._log_sink_id is None:
            return
        try:
            logger.remove(self._log_sink_id)
        except Exception as e:  # pragma: no cover - best-effort cleanup
            logger.warning(f"Failed to remove model log sink: {e}")
        finally:
            self._log_sink_id = None

    def _format_log_message(self, action: str, reason: str) -> str:
        """Format a log message with model identifier.

        Parameters
        ----------
        action : str
            The action being performed.
        reason : str
            The reason for the action.

        Returns
        -------
        str
            Formatted log message.
        """
        identifier = f"{self.config_args.model_type}:{self.config_args.model_path}"
        return f"[{identifier}] {action} ({reason})"

    def _schedule_memory_cleanup(self) -> None:
        """Run cache clearing in the background to avoid blocking requests."""

        async def _run() -> None:
            try:
                await asyncio.to_thread(mx.clear_cache)
                await asyncio.to_thread(gc.collect)
            except Exception as e:  # pragma: no cover - best-effort logging
                logger.warning(f"Background memory cleanup failed. {type(e).__name__}: {e}")

        task = asyncio.create_task(_run())

        def _on_complete(done: asyncio.Task[None]) -> None:
            self._background_tasks.discard(done)
            e = done.exception()
            if e:
                logger.warning(f"Background memory cleanup raised. {type(e).__name__}: {e}")

        self._background_tasks.add(task)
        task.add_done_callback(_on_complete)


class CentralIdleAutoUnloadController:
    """Central controller that monitors `ModelRegistry` and unloads idle models.

    The controller scans registered models and triggers `registry.request_vram_unload`
    when a model's idle period exceeds its configured auto-unload timeout and the
    model has no active requests.
    """

    WATCH_LOOP_MAX_WAIT_SECONDS = 5

    def __init__(self, registry: ModelRegistry) -> None:
        self.registry = registry
        self._event = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._active = False
        # Per-model backoff map (model_id -> next_allowed_time)
        self._backoff: dict[str, float] = {}

    def start(self) -> None:
        """Start the central idle auto-unload background task."""
        if self._active:
            return
        self._active = True
        self._task = asyncio.create_task(self._watch_loop())

    async def stop(self) -> None:
        """Stop the controller and wait for background task to finish."""
        if not self._active:
            return
        self._active = False
        self._event.set()
        if self._task:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task

    def notify_activity(self, model_id: str) -> None:
        """Called by the registry when activity occurs or when a model becomes idle.

        The controller uses this to reset timers and wake the watch loop.
        """
        # model_id is intentionally unused; controller queries registry for current state
        # Wake the loop; controller will query registry for current state
        if self._active:
            self._event.set()

    async def _watch_loop(self) -> None:
        try:
            while self._active:
                # Iterate registered models and consider unloading
                try:
                    models = [m["id"] for m in self.registry.list_models()]
                    # Keep the list_models payload so we can read per-model metadata
                    entries = self.registry.list_models()
                except Exception:
                    await asyncio.sleep(self.WATCH_LOOP_MAX_WAIT_SECONDS)
                    continue

                now = time.time()
                for mid in models:
                    # Skip if backoff in effect
                    next_allowed = self._backoff.get(mid, 0)
                    if now < next_allowed:
                        continue

                    try:
                        status = self.registry.get_vram_status(mid)
                    except KeyError:
                        continue

                    # active_requests prevents unload
                    if status.get("active_requests", 0) > 0:
                        continue

                    handler = None
                    try:
                        handler = self.registry.get_handler(mid)
                    except KeyError:
                        handler = None

                    # Determine auto-unload timeout (minutes) from handler or metadata
                    minutes = None
                    if handler is not None:
                        minutes = getattr(handler, "auto_unload_minutes", None)
                    if minutes is None:
                        # Fall back to metadata extras if present
                        try:
                            meta: dict[str, Any] = next(
                                (e.get("metadata", {}) for e in entries if e.get("id") == mid),
                                {},
                            )
                            minutes = meta.get("auto_unload_minutes")
                        except Exception:
                            minutes = None

                    if minutes is None:
                        continue

                    timeout_secs = int(minutes) * 60

                    # If handler provides last-activity metric, use it
                    idle_elapsed = None
                    if handler is not None and hasattr(handler, "seconds_since_last_activity"):
                        try:
                            idle_elapsed = handler.seconds_since_last_activity()
                        except Exception:
                            idle_elapsed = None

                    # If we don't have per-handler idle metric, derive from vram timestamps
                    if idle_elapsed is None:
                        last_request = status.get("vram_last_request_ts")
                        if last_request is not None:
                            # Prefer actual request activity timestamp
                            last_activity_ts = float(last_request)
                        else:
                            # Fall back to load/unload times if no request activity recorded
                            last_unload = status.get("vram_last_unload_ts") or 0
                            last_load = status.get("vram_last_load_ts") or 0
                            last_activity_ts = max(last_load, last_unload)
                        idle_elapsed = max(0, now - last_activity_ts)

                    if idle_elapsed >= timeout_secs and status.get("vram_loaded", False):
                        try:
                            await self.registry.request_vram_unload(mid)
                        except Exception:
                            # On failure, backoff to avoid tight error loops
                            self._backoff[mid] = time.time() + 30
                            logger.exception(f"Failed to auto-unload model {mid}")

                # Wait for activity or timeout
                try:
                    await asyncio.wait_for(
                        self._event.wait(),
                        timeout=self.WATCH_LOOP_MAX_WAIT_SECONDS,
                    )
                except TimeoutError:
                    pass
                finally:
                    self._event.clear()
        except asyncio.CancelledError:
            pass


def create_lifespan(
    config_args: MLXServerConfig,
) -> Callable[[FastAPI], AbstractAsyncContextManager[None]]:
    """Create an async FastAPI lifespan context manager bound to configuration.

    The returned context manager sets up JIT-aware handler management and
    auto-unload functionality during application startup:

    - Creates a LazyHandlerManager to manage model loading/unloading
    - Creates a CentralIdleAutoUnloadController for automatic unloading when idle
    - If JIT is disabled, loads the handler immediately at startup
    - Starts the auto-unload background task if configured
    - Performs initial memory cleanup

    During shutdown:
    - Stops the auto-unload controller
    - Shuts down the handler manager and cleans up resources
    - Performs final memory cleanup

    Parameters
    ----------
    config_args : MLXServerConfig
        Configuration object containing CLI settings for
        model type, path, JIT settings, auto-unload configuration, etc.

    Returns
    -------
    Callable
        An async context manager usable as FastAPI ``lifespan``.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        """FastAPI lifespan that wires the LazyHandlerManager into app state.

        Sets up the handler manager and auto-unload controller during
        application startup, and performs cleanup during shutdown.
        Manages the application lifecycle for JIT-aware model loading.

        Parameters
        ----------
        app : FastAPI
            The FastAPI application instance.
        """
        registry = ModelRegistry()
        app.state.model_registry = registry
        registry_model_id = get_registry_model_id(config_args)
        base_registry_metadata = {
            "model_path": config_args.model_identifier,
            "model_type": config_args.model_type,
            "context_length": config_args.context_length,
            "jit_enabled": config_args.jit_enabled,
            "auto_unload_minutes": config_args.auto_unload_minutes,
            "group": config_args.group,
        }
        registry.register_model(
            model_id=registry_model_id,
            handler=None,
            model_type=config_args.model_type,
            context_length=config_args.context_length,
            metadata_extras=base_registry_metadata,
        )
        registry_tasks: set[asyncio.Task[None]] = set()

        async def _sync_registry_update(handler: MLXHandler | None) -> None:
            metadata_payload = dict(base_registry_metadata)
            metadata_payload["model_path"] = getattr(
                handler,
                "model_path",
                config_args.model_identifier,
            )
            status = "initialized" if handler else "unloaded"
            try:
                await registry.update_model_state(
                    registry_model_id,
                    handler=handler,
                    status=status,
                    metadata_updates=metadata_payload,
                )
            except Exception as e:  # pragma: no cover - defensive logging
                logger.warning(
                    f"Failed to synchronize model registry for {registry_model_id}. "
                    f"{type(e).__name__}: {e}",
                )

        def _update_model_metadata(handler: MLXHandler | None) -> None:
            model_metadata = getattr(app.state, "model_metadata", None)
            if not model_metadata:
                return

            entry = model_metadata[0]
            metadata_block = entry.setdefault("metadata", {})
            metadata_block["status"] = "initialized" if handler else "unloaded"
            if handler:
                entry["created"] = getattr(handler, "model_created", int(time.time()))
                metadata_block["model_path"] = getattr(
                    handler,
                    "model_path",
                    metadata_block.get("model_path"),
                )

        def _update_handler(handler: MLXHandler | None) -> None:
            app.state.handler = handler
            _update_model_metadata(handler)
            task = asyncio.create_task(_sync_registry_update(handler))

            def _cleanup(done: asyncio.Task[None]) -> None:
                registry_tasks.discard(done)
                e = done.exception()
                if e:
                    logger.warning(f"Registry update task raised. {type(e).__name__}: {e}")

            registry_tasks.add(task)
            task.add_done_callback(_cleanup)

        handler_manager = LazyHandlerManager(config_args, on_change=_update_handler)
        app.state.handler_manager = handler_manager

        # Central controller watches the ModelRegistry and can auto-unload any
        # registered model when idle. Register the registry notifier so
        # handler sessions and other registry-driven activity can reset timers.
        central_controller = CentralIdleAutoUnloadController(registry)
        registry.register_activity_notifier(central_controller.notify_activity)
        app.state.idle_controller = central_controller

        # Wire manager-level activity into the central controller so a single
        # controller handles auto-unload decisions for all models.
        handler_manager.set_activity_callback(
            lambda: central_controller.notify_activity(registry_model_id),
        )

        try:
            if not config_args.jit_enabled:
                await handler_manager.ensure_loaded("startup")
        except Exception:
            await handler_manager.shutdown()
            raise

        central_controller.start()

        # Initial memory cleanup
        mx.clear_cache()
        gc.collect()

        yield

        # Shutdown
        logger.info("Shutting down application")
        try:
            await central_controller.stop()
        except Exception as e:
            logger.error(f"Error stopping central auto-unload controller. {type(e).__name__}: {e}")

        try:
            await handler_manager.shutdown()
            logger.info("Resources cleaned up successfully")
        except Exception as e:
            logger.error(f"Error during shutdown. {type(e).__name__}: {e}")

        if registry_tasks:
            await asyncio.gather(*registry_tasks, return_exceptions=True)

        # Final memory cleanup
        mx.clear_cache()
        gc.collect()

    return lifespan


def setup_server(config_args: MLXServerConfig) -> uvicorn.Config:
    """Create and configure the FastAPI app and return a Uvicorn config.

    This function sets up logging, constructs the FastAPI application with
    a configured lifespan, registers routes and middleware, and returns a
    :class:`uvicorn.Config` ready to be used to run the server.

    Parameters
    ----------
    config_args : MLXServerConfig
        Configuration object usually produced by the CLI. Expected
        to have attributes like ``host``, ``port``, ``log_level``,
        and logging-related fields.

    Returns
    -------
    uvicorn.Config
        A configuration object that can be passed to
        ``uvicorn.Server(config).run()`` to start the application.
    """
    # Configure logging based on CLI parameters
    configure_logging(
        log_file=config_args.log_file,
        no_log_file=config_args.no_log_file,
        log_level=config_args.log_level,
    )

    # Create FastAPI app with the configured lifespan
    app = FastAPI(
        title="OpenAI-compatible API",
        description="API for OpenAI-compatible chat completion and text embedding",
        version=__version__,
        lifespan=create_lifespan(config_args),
    )
    app.state.server_config = config_args
    app.state.model_metadata = [
        {
            "id": config_args.model_identifier,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "local",
            "metadata": {
                "model_type": config_args.model_type,
                "context_length": config_args.context_length,
                "jit_enabled": config_args.jit_enabled,
                "auto_unload_minutes": config_args.auto_unload_minutes,
                "max_concurrency": config_args.max_concurrency,
                "status": "unloaded",
                "model_path": config_args.model_identifier,
            },
        },
    ]

    configure_fastapi_app(app)

    logger.info(f"Starting server on {config_args.host}:{config_args.port}")
    return uvicorn.Config(
        app=app,
        host=config_args.host,
        port=config_args.port,
        log_level=config_args.log_level.lower(),
        access_log=True,
    )


def configure_fastapi_app(app: FastAPI) -> None:
    """Register routers, middleware, and global handlers on ``app``.

    This helper centralizes FastAPI configuration that is shared between the
    standard single-model server and the hub-aware server so both surfaces
    expose identical middleware, routing, and error handling behavior.
    """
    app.include_router(router)
    # Ensure a ModelRegistry is available on the application state so
    # hub-aware endpoints and admin routes can access model metadata
    # and request VRAM operations without requiring explicit wiring.
    if not getattr(app.state, "model_registry", None):
        app.state.model_registry = ModelRegistry()
    app.add_middleware(RequestTrackingMiddleware)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def add_process_time_header(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Attach timing metadata and trigger periodic memory cleanup."""
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)

        request_count = getattr(request.app.state, "request_count", 0) + 1
        request.app.state.request_count = request_count

        if request_count % 50 == 0:
            mx.clear_cache()
            gc.collect()
            logger.debug(f"Performed memory cleanup after {request_count} requests")

        return response

    @app.exception_handler(Exception)
    async def global_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
        """Log unexpected exceptions and emit a generic payload."""
        logger.exception(f"Global exception handler caught. {type(exc).__name__}: {exc}")
        return JSONResponse(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            content={"error": {"message": "Internal server error", "type": "internal_error"}},
        )
