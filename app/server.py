"""Application server helpers.

This module provides utilities to configure logging and to create the
FastAPI application with the correct lifespan for MLX model handlers.

Key exports:
- ``configure_logging``: sets up loguru handlers based on CLI flags.
- ``create_lifespan``: returns an asynccontextmanager that initializes
    the appropriate MLX handler on startup and performs cleanup on
    shutdown.
- ``setup_server``: builds the FastAPI app and returns a Uvicorn config
    ready to run.
"""

import gc
import time
from contextlib import asynccontextmanager

import mlx.core as mx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from .api.endpoints import router
from .config import MLXServerConfig
from .handler import MLXFluxHandler
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
    log_file:
        Optional filesystem path where logs should be written. When
        ``None`` and file logging is enabled a sensible default
        (``logs/app.log``) is used.
    no_log_file:
        When True, file logging is disabled and only console logs are
        emitted.
    log_level:
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
    config_args:
        Configuration object produced by the CLI. The attribute
        ``model_path`` is read to produce the identifier.

    Returns
    -------
    str
        Value that identifies the model for handler initialization.
    """

    return config_args.model_path


def create_lifespan(config_args: MLXServerConfig):
    """Create an async FastAPI lifespan context manager bound to configuration.

    The returned context manager performs the following actions during
    application startup:

    - Determine the model identifier from the provided ``config_args``
    - Instantiate the appropriate MLX handler based on ``model_type``
      (multimodal, image-generation, image-edit, embeddings, whisper, or
      text LM)
    - Initialize the handler (including queuing and concurrency setup)
    - Perform an initial memory cleanup

    During shutdown the lifespan will attempt to call the handler's
    ``cleanup`` method and perform final memory cleanup.

    Args:
        config_args: Object containing CLI configuration attributes used
            to initialize handlers (e.g., model_type, model_path,
            max_concurrency, queue_timeout, etc.).

    Returns:
        Callable: An asynccontextmanager usable as FastAPI ``lifespan``.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> None:
        """FastAPI lifespan callable that initializes MLX handlers.

        On startup this function selects and initializes the correct
        model handler based on ``config_args`` and attaches it to
        ``app.state.handler``. It also performs an initial memory
        cleanup. On shutdown it invokes the handler's ``cleanup``
        method and runs a final memory cleanup.

        Parameters
        ----------
        app:
            FastAPI application instance being started.
        """
        try:
            model_identifier = get_model_identifier(config_args)
            if config_args.model_type == "image-generation":
                logger.info(f"Initializing MLX handler with model name: {model_identifier}")
            else:
                logger.info(f"Initializing MLX handler with model path: {model_identifier}")

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
                    chat_template_file=config_args.chat_template_file,
                )
            elif config_args.model_type == "image-generation":
                if config_args.config_name not in ["flux-schnell", "flux-dev", "flux-krea-dev", "qwen-image", "z-image-turbo", "fibo"]:
                    raise ValueError(
                        f"Invalid config name: {config_args.config_name}. Only flux-schnell, flux-dev, flux-krea-dev, qwen-image, z-image-turbo, and fibo are supported for image generation."
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
                if config_args.config_name not in ["flux-kontext-dev", "qwen-image-edit"]:
                    raise ValueError(
                        f"Invalid config name: {config_args.config_name}. Only flux-kontext-dev and qwen-image-edit are supported for image edit."
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
            else:
                handler = MLXLMHandler(
                    model_path=model_identifier,
                    context_length=config_args.context_length,
                    max_concurrency=config_args.max_concurrency,
                    enable_auto_tool_choice=config_args.enable_auto_tool_choice,
                    tool_call_parser=config_args.tool_call_parser,
                    reasoning_parser=config_args.reasoning_parser,
                    trust_remote_code=config_args.trust_remote_code,
                    chat_template_file=config_args.chat_template_file,
                )
            # Initialize queue
            await handler.initialize(
                {
                    "max_concurrency": config_args.max_concurrency,
                    "timeout": config_args.queue_timeout,
                    "queue_size": config_args.queue_size,
                }
            )
            logger.info("MLX handler initialized successfully")
            app.state.handler = handler

        except Exception as e:
            logger.error(f"Failed to initialize MLX handler: {str(e)}")
            raise

        # Initial memory cleanup
        mx.clear_cache()
        gc.collect()

        yield

        # Shutdown
        logger.info("Shutting down application")
        if hasattr(app.state, "handler") and app.state.handler:
            try:
                # Use the proper cleanup method which handles both request queue and image processor
                logger.info("Cleaning up resources")
                await app.state.handler.cleanup()
                logger.info("Resources cleaned up successfully")
            except Exception as e:
                logger.error(f"Error during shutdown: {str(e)}")

        # Final memory cleanup
        mx.clear_cache()
        gc.collect()

    return lifespan


# App instance will be created during setup with the correct lifespan
app = None


def setup_server(config_args: MLXServerConfig) -> uvicorn.Config:
    global app

    """Create and configure the FastAPI app and return a Uvicorn config.

    This function sets up logging, constructs the FastAPI application with
    a configured lifespan, registers routes and middleware, and returns a
    :class:`uvicorn.Config` ready to be used to run the server.

    Note: This function mutates the module-level ``app`` global variable.

    Args:
        args: Configuration object usually produced by the CLI. Expected
            to have attributes like ``host``, ``port``, ``log_level``,
            and logging-related fields.

    Returns:
        uvicorn.Config: A configuration object that can be passed to
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
