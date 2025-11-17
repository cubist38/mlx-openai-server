"""Application server helpers with JIT-aware handler management."""

import asyncio
import gc
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any

import mlx.core as mx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from .api.endpoints import router
from .config import MLXServerConfig
from .handler import MFLUX_AVAILABLE, MLXFluxHandler
from .handler.mlx_embeddings import MLXEmbeddingsHandler
from .handler.mlx_lm import MLXLMHandler
from .handler.mlx_vlm import MLXVLMHandler
from .handler.mlx_whisper import MLXWhisperHandler
from .version import __version__


def configure_logging(
    log_file: str | None = None, no_log_file: bool = False, log_level: str = "INFO"
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

    # Add console handler
    logger.add(
        lambda msg: print(msg),
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "✦ <level>{message}</level>",
        colorize=True,
    )
    if not no_log_file:
        file_path = log_file if log_file else "logs/app.log"
        logger.add(
            file_path,
            rotation="500 MB",
            retention="10 days",
            level=log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
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


async def instantiate_handler(config_args: MLXServerConfig):
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
            if not MFLUX_AVAILABLE:
                raise ValueError(
                    "Image generation requires mflux. Install with: pip install git+https://github.com/cubist38/mflux.git"
                )
            if config_args.config_name not in ["flux-schnell", "flux-dev", "flux-krea-dev"]:
                raise ValueError(
                    f"Invalid config name: {config_args.config_name}. Only flux-schnell, flux-dev, and flux-krea-dev are supported for image generation."
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
                model_path=model_identifier, max_concurrency=config_args.max_concurrency
            )
        elif config_args.model_type == "image-edit":
            if not MFLUX_AVAILABLE:
                raise ValueError(
                    "Image editing requires mflux. Install with: pip install git+https://github.com/cubist38/mflux.git"
                )
            if config_args.config_name != "flux-kontext-dev":
                raise ValueError(
                    f"Invalid config name: {config_args.config_name}. Only flux-kontext-dev is supported for image edit."
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
                model_path=model_identifier, max_concurrency=config_args.max_concurrency
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
                f"Supported types are: lm, multimodal, image-generation, image-edit, embeddings, whisper"
            )

        await handler.initialize(
            {
                "max_concurrency": config_args.max_concurrency,
                "timeout": config_args.queue_timeout,
                "queue_size": config_args.queue_size,
            }
        )
        logger.info("MLX handler initialized successfully")
        return handler
    except Exception as exc:
        logger.error(f"Failed to initialize MLX handler: {exc}")
        raise


class LazyHandlerManager:
    """Manage handler lifecycle with optional JIT loading and unloading."""

    def __init__(
        self,
        config_args: MLXServerConfig,
        *,
        on_change: Callable[[Any | None], None] | None = None,
        on_activity: Callable[[], None] | None = None,
    ) -> None:
        """Initialize the LazyHandlerManager.

        Parameters
        ----------
        config_args : MLXServerConfig
            Configuration arguments for the server.
        on_change : Callable[[Any | None], None], optional
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
    def current_handler(self):
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

    async def ensure_loaded(self, reason: str = "request"):
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
        Any
            The loaded handler instance.
        """
        if self._handler and not self._shutdown:
            self.record_activity()
            return self._handler

        async with self._lock:
            if self._handler or self._shutdown:
                if self._handler and not self._shutdown:
                    self.record_activity()
                return self._handler

            logger.info(self._format_log_message("Loading model", reason))
            handler = await instantiate_handler(self.config_args)
            self._handler = handler
            self.record_activity()
            if self._on_change:
                self._on_change(handler)
            mx.clear_cache()
            gc.collect()
            logger.info(self._format_log_message("Model loaded", reason))
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
            logger.info(self._format_log_message("Unloading model", reason))
            await handler.cleanup()
            logger.info(self._format_log_message("Model unloaded", reason))
        finally:
            mx.clear_cache()
            gc.collect()
        return True

    async def shutdown(self):
        """Shutdown the handler manager and unload any loaded handler.

        Sets the shutdown flag and unloads the handler.
        """
        self._shutdown = True
        await self.unload("shutdown")

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


class IdleAutoUnloadController:
    """Background helper that unloads handlers after idle periods."""

    def __init__(self, handler_manager: LazyHandlerManager) -> None:
        """Initialize the IdleAutoUnloadController.

        Parameters
        ----------
        handler_manager : LazyHandlerManager
            The handler manager to monitor for idle periods.
        """
        self.handler_manager = handler_manager
        self._event = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._active = False

    @property
    def auto_unload_minutes(self) -> int | None:
        """Expose auto-unload configuration for logging."""
        return self.handler_manager.auto_unload_minutes

    def start(self) -> None:
        """Start the auto-unload background task if configured."""
        if not self.handler_manager.jit_enabled:
            return
        if self.handler_manager.auto_unload_minutes is None:
            return
        if self._active:
            return

        self._active = True
        self._task = asyncio.create_task(self._watch_loop())

    async def stop(self) -> None:
        """Stop the auto-unload background task."""
        if not self._active:
            return

        self._active = False
        self._event.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def notify_activity(self) -> None:
        """Notify the background task of recent activity."""
        if self._active:
            self._event.set()

    async def _wait_for_event(self) -> None:
        """Wait for an activity event and clear it.

        This method waits for the event to be set and then clears it.
        """
        await self._event.wait()
        self._event.clear()

    async def _watch_loop(self) -> None:
        """Background loop that monitors for idle periods and unloads handlers."""
        try:
            while self._active:
                timeout = self.handler_manager.idle_timeout_seconds()
                if timeout is None:
                    break

                if not self.handler_manager.current_handler:
                    await asyncio.sleep(10)  # Wait a bit before checking again
                    continue

                if self.handler_manager.seconds_since_last_activity() >= timeout:
                    await self.handler_manager.unload(
                        f"Idle for {self.auto_unload_minutes} minutes"
                    )
                    continue

                await asyncio.sleep(10)  # Poll every 10 seconds
        except asyncio.CancelledError:
            pass


def create_lifespan(config_args: MLXServerConfig):
    """Create an async FastAPI lifespan context manager bound to configuration.

    The returned context manager sets up JIT-aware handler management and
    auto-unload functionality during application startup:

    - Creates a LazyHandlerManager to manage model loading/unloading
    - Creates an IdleAutoUnloadController for automatic unloading when idle
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
    async def lifespan(app: FastAPI) -> None:
        """FastAPI lifespan that wires the LazyHandlerManager into app state."""

        def _update_handler(handler) -> None:
            app.state.handler = handler

        handler_manager = LazyHandlerManager(config_args, on_change=_update_handler)
        app.state.handler_manager = handler_manager
        auto_unload_controller = IdleAutoUnloadController(handler_manager)
        app.state.auto_unload_controller = auto_unload_controller
        handler_manager._on_activity = auto_unload_controller.notify_activity

        try:
            if not config_args.jit_enabled:
                await handler_manager.ensure_loaded("startup")
        except Exception:
            await handler_manager.shutdown()
            raise

        auto_unload_controller.start()

        # Initial memory cleanup
        mx.clear_cache()
        gc.collect()

        yield

        # Shutdown
        logger.info("Shutting down application")
        try:
            await auto_unload_controller.stop()
        except Exception as exc:
            logger.error(f"Error stopping auto-unload controller: {exc}")

        try:
            await handler_manager.shutdown()
            logger.info("Resources cleaned up successfully")
        except Exception as exc:
            logger.error(f"Error during shutdown: {exc}")

        # Final memory cleanup
        mx.clear_cache()
        gc.collect()

    return lifespan


# App instance will be created during setup with the correct lifespan
app = None


def setup_server(config_args: MLXServerConfig) -> uvicorn.Config:
    """Create and configure the FastAPI app and return a Uvicorn config.

    This function sets up logging, constructs the FastAPI application with
    a configured lifespan, registers routes and middleware, and returns a
    :class:`uvicorn.Config` ready to be used to run the server.

    Note: This function mutates the module-level ``app`` global variable.

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
    global app

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
    app.state.model_metadata = {"created": int(time.time())}

    app.include_router(router)

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        """Middleware to add processing time header and run cleanup.

        Measures request processing time, appends an ``X-Process-Time``
        header, and increments a simple request counter used to trigger
        periodic memory cleanup for long-running processes.
        """
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)

        # Periodic memory cleanup for long-running processes
        if hasattr(request.app.state, "request_count"):
            request.app.state.request_count += 1
        else:
            request.app.state.request_count = 1

        # Clean up memory every 50 requests
        if request.app.state.request_count % 50 == 0:
            mx.clear_cache()
            gc.collect()
            logger.debug(
                f"Performed memory cleanup after {request.app.state.request_count} requests"
            )

        return response

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Global exception handler that logs and returns a 500 payload.

        Logs the exception (with traceback) and returns a generic JSON
        response with a 500 status code so internal errors do not leak
        implementation details to clients.
        """
        logger.error(f"Global exception handler caught: {str(exc)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": {"message": "Internal server error", "type": "internal_error"}},
        )

    logger.info(f"Starting server on {config_args.host}:{config_args.port}")
    return uvicorn.Config(
        app=app,
        host=config_args.host,
        port=config_args.port,
        log_level=config_args.log_level.lower(),
        access_log=True,
    )
