"""Application server helpers.

This module provides utilities to configure logging and to create the
FastAPI application with the correct lifespan for MLX model handlers.

Key exports:
- ``configure_logging``: sets up loguru handlers based on CLI flags.
- ``create_lifespan``: returns an asynccontextmanager that initializes
    the appropriate MLX handler on startup and performs cleanup on
    shutdown.
- ``create_multi_lifespan``: returns an asynccontextmanager for
    multi-handler mode, registering all models from a
    ``MultiModelServerConfig``.
- ``setup_server``: builds the FastAPI app and returns a Uvicorn config
    ready to run.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import gc
import time
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
import mlx.core as mx
import uvicorn

from .api.endpoints import router
from .config import MLXServerConfig, ModelEntryConfig, MultiModelServerConfig
from .core.handler_process import HandlerProcessProxy
from .core.model_registry import ModelRegistry
from .handler import MLXFluxHandler
from .handler.mlx_embeddings import MLXEmbeddingsHandler
from .handler.mlx_lm import MLXLMHandler
from .handler.mlx_vlm import MLXVLMHandler
from .handler.mlx_whisper import MLXWhisperHandler
from .version import __version__

MFLUX_INSTALL_HINT = (
    "Image generation and editing require the `mflux` package. "
    "Install it with `pip install mflux==0.17.0`."
)


def ensure_image_handler_available(model_type: str) -> None:
    """Validate that optional image generation support is installed."""
    if model_type not in {"image-generation", "image-edit"}:
        return

    if MLXFluxHandler is not None:
        return

    raise RuntimeError(MFLUX_INSTALL_HINT)


_SAMPLING_DEFAULT_FIELDS: tuple[str, ...] = (
    "default_max_tokens",
    "default_temperature",
    "default_top_p",
    "default_top_k",
    "default_min_p",
    "default_repetition_penalty",
    "default_presence_penalty",
    "default_xtc_probability",
    "default_xtc_threshold",
    "default_seed",
    "default_repetition_context_size",
)


def _attach_sampling_defaults(handler: Any, config: Any) -> Any:
    """Attach configured sampling-default attributes to a handler or proxy."""

    for field_name in _SAMPLING_DEFAULT_FIELDS:
        setattr(handler, field_name, getattr(config, field_name, None))
    setattr(handler, "_uses_model_sampling_defaults", True)
    return handler


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
        print,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "✦ <level>{message}</level>",
        colorize=True,
    )
    if not no_log_file:
        file_path = log_file or "logs/app.log"
        logger.add(
            file_path,
            rotation="500 MB",
            retention="10 days",
            level=log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        )


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
            queue_timeout, etc.).

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
            model_cfg = config_args.to_model_entry_config()
            ensure_image_handler_available(model_cfg.model_type)

            logger.info(f"Initializing MLX handler with model path: {model_cfg.model_path}")

            handler = create_handler_from_config(model_cfg)
            await handler.initialize(
                {
                    "timeout": config_args.queue_timeout,
                    "queue_size": config_args.queue_size,
                }
            )

            # Override the model name exposed via /v1/models and accepted
            # in the request ``model`` field when --served-model-name is set.
            if config_args.served_model_name:
                handler.model_path = config_args.served_model_name
                logger.info(f"Serving model as '{config_args.served_model_name}'")
            handler.served_model_name = model_cfg.served_model_name

            logger.info("MLX handler initialized successfully")
            app.state.handler = handler

        except Exception as e:
            logger.error(f"Failed to initialize MLX handler: {e!s}")
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
                logger.error(f"Error during shutdown: {e!s}")

        # Final memory cleanup
        mx.clear_cache()
        gc.collect()

    return lifespan


def create_handler_from_config(model_cfg: ModelEntryConfig) -> Any:
    """Instantiate the correct handler for a single model entry.

    This factory function mirrors the type-based dispatch in
    ``create_lifespan`` but works with the per-model
    ``ModelEntryConfig`` dataclass used in YAML configs.

    Parameters
    ----------
    model_cfg : ModelEntryConfig
        Configuration for a single model.

    Returns
    -------
    Any
        An initialized (but not yet queue-started) handler instance.

    Raises
    ------
    ValueError
        If the ``model_type`` is unrecognized or the configuration
        is invalid for the given type.
    """
    model_path = model_cfg.model_path

    if model_cfg.model_type == "multimodal":
        return _attach_sampling_defaults(
            MLXVLMHandler(
                model_path=model_path,
                context_length=model_cfg.context_length,
                disable_auto_resize=model_cfg.disable_auto_resize,
                enable_auto_tool_choice=model_cfg.enable_auto_tool_choice,
                tool_call_parser=model_cfg.tool_call_parser,
                reasoning_parser=model_cfg.reasoning_parser,
                message_converter=model_cfg.message_converter,
                trust_remote_code=model_cfg.trust_remote_code,
                chat_template_file=model_cfg.chat_template_file,
                debug=model_cfg.debug,
                kv_bits=model_cfg.kv_bits,
                kv_group_size=model_cfg.kv_group_size,
                quantized_kv_start=model_cfg.quantized_kv_start,
            ),
            model_cfg,
        )

    if model_cfg.model_type == "image-generation":
        ensure_image_handler_available(model_cfg.model_type)
        return _attach_sampling_defaults(
            MLXFluxHandler(
                model_path=model_path,
                quantize=model_cfg.quantize,
                config_name=model_cfg.config_name,
                lora_paths=model_cfg.lora_paths,
                lora_scales=model_cfg.lora_scales,
            ),
            model_cfg,
        )

    if model_cfg.model_type == "image-edit":
        ensure_image_handler_available(model_cfg.model_type)
        return _attach_sampling_defaults(
            MLXFluxHandler(
                model_path=model_path,
                quantize=model_cfg.quantize,
                config_name=model_cfg.config_name,
                lora_paths=model_cfg.lora_paths,
                lora_scales=model_cfg.lora_scales,
            ),
            model_cfg,
        )

    if model_cfg.model_type == "embeddings":
        return _attach_sampling_defaults(
            MLXEmbeddingsHandler(
                model_path=model_path,
            ),
            model_cfg,
        )

    if model_cfg.model_type == "whisper":
        return _attach_sampling_defaults(
            MLXWhisperHandler(
                model_path=model_path,
            ),
            model_cfg,
        )

    # Default: language model ("lm")
    return _attach_sampling_defaults(
        MLXLMHandler(
            model_path=model_path,
            context_length=model_cfg.context_length,
            enable_auto_tool_choice=model_cfg.enable_auto_tool_choice,
            tool_call_parser=model_cfg.tool_call_parser,
            reasoning_parser=model_cfg.reasoning_parser,
            message_converter=model_cfg.message_converter,
            trust_remote_code=model_cfg.trust_remote_code,
            chat_template_file=model_cfg.chat_template_file,
            debug=model_cfg.debug,
            prompt_cache_size=model_cfg.prompt_cache_size,
            prompt_cache_max_bytes=model_cfg.prompt_cache_max_bytes,
            prompt_cache_dir=model_cfg.prompt_cache_dir,
            draft_model_path=model_cfg.draft_model_path,
            num_draft_tokens=model_cfg.num_draft_tokens,
            kv_bits=model_cfg.kv_bits,
            kv_group_size=model_cfg.kv_group_size,
            quantized_kv_start=model_cfg.quantized_kv_start,
            batch_completion_size=model_cfg.batch_completion_size,
            batch_prefill_size=model_cfg.batch_prefill_size,
            batch_prefill_step_size=model_cfg.batch_prefill_step_size,
            disable_batching=model_cfg.disable_batching,
        ),
        model_cfg,
    )


def create_multi_lifespan(config: MultiModelServerConfig):
    """Create a FastAPI lifespan for multi-handler mode.

    Each model entry in ``config.models`` is spawned in a dedicated
    subprocess using ``multiprocessing.get_context("spawn")``, preventing
    MLX Metal/GPU semaphore leaks (see
    `<https://github.com/ml-explore/mlx/issues/2457>`_).  A
    ``HandlerProcessProxy`` in the main process forwards requests to
    the child via multiprocessing queues.

    The proxies are registered in a ``ModelRegistry`` and attached to
    ``app.state.registry``.  For backward compatibility the first
    proxy is also stored as ``app.state.handler``.

    Parameters
    ----------
    config : MultiModelServerConfig
        Parsed multi-model configuration (typically from YAML).

    Returns
    -------
    Callable
        An asynccontextmanager usable as FastAPI ``lifespan``.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """FastAPI lifespan that spawns handler subprocesses.

        Each model from the YAML config is loaded in its own spawned
        subprocess, keeping MLX Metal runtime state fully isolated.
        A ``HandlerProcessProxy`` is registered in the
        ``ModelRegistry`` for each model.

        Parameters
        ----------
        app : FastAPI
            FastAPI application instance being started.
        """
        registry = ModelRegistry()

        try:
            for model_cfg in config.models:
                model_id = model_cfg.served_model_name  # guaranteed non-None after __post_init__

                # Serialize the dataclass config to a plain dict for
                # pickling across the spawn boundary.
                from dataclasses import asdict

                model_cfg_dict = asdict(model_cfg)

                queue_config = {
                    "timeout": model_cfg.queue_timeout,
                    "queue_size": model_cfg.queue_size,
                }

                if model_cfg.on_demand:
                    # Register on-demand model without spawning a subprocess.
                    # It will be loaded dynamically when a request arrives.
                    await registry.register_on_demand_model(
                        model_id=model_id,
                        model_cfg_dict=model_cfg_dict,
                        model_type=model_cfg.model_type,
                        model_path=model_cfg.model_path,
                        context_length=model_cfg.context_length,
                        queue_config=queue_config,
                        idle_timeout=model_cfg.on_demand_idle_timeout,
                    )
                    logger.info(
                        f"Model '{model_id}' registered as on-demand "
                        f"(will load when first requested)"
                    )
                    continue

                logger.info(
                    f"Spawning handler process for model '{model_id}' "
                    f"(type={model_cfg.model_type}, path={model_cfg.model_path})"
                )

                proxy = HandlerProcessProxy(
                    model_cfg_dict=model_cfg_dict,
                    model_type=model_cfg.model_type,
                    model_path=model_cfg.model_path,
                    served_model_name=model_id,
                )

                # Spawn the child process and wait for it to load the model.
                await proxy.start(queue_config)

                await registry.register_model(
                    model_id=model_id,
                    handler=proxy,
                    model_type=model_cfg.model_type,
                    context_length=model_cfg.context_length,
                )
                logger.info(f"Model '{model_id}' spawned and registered successfully")

            # Store registry on app state for endpoint access
            app.state.registry = registry

            # Backward compatibility: expose first non-on-demand handler
            # as app.state.handler
            for _cfg in config.models:
                if not _cfg.on_demand:
                    app.state.handler = registry.get_handler(_cfg.served_model_name)
                    break

            logger.info(
                f"Multi-handler initialization complete. "
                f"{registry.get_model_count()} model(s) spawned."
            )

        except Exception as e:
            logger.error(f"Failed to initialize multi-handler setup: {e}")
            # Cleanup any handlers that were already registered
            await registry.cleanup_all()
            raise

        # Initial memory cleanup (main process only)
        mx.clear_cache()
        gc.collect()

        yield

        # Shutdown
        logger.info("Shutting down multi-handler application")
        await registry.cleanup_all()

        # Final memory cleanup
        mx.clear_cache()
        gc.collect()

    return lifespan


# App instance will be created during setup with the correct lifespan
app = None


def setup_server(config_args: MLXServerConfig | MultiModelServerConfig) -> uvicorn.Config:
    """Create and configure the FastAPI app and return a Uvicorn config.

    This function sets up logging, constructs the FastAPI application with
    a configured lifespan, registers routes and middleware, and returns a
    :class:`uvicorn.Config` ready to be used to run the server.

    When ``config_args`` is a ``MultiModelServerConfig`` the multi-handler
    lifespan is used, which registers all models in a ``ModelRegistry``.

    Note: This function mutates the module-level ``app`` global variable.

    Parameters
    ----------
    config_args : MLXServerConfig | MultiModelServerConfig
        Configuration object produced by the CLI or by YAML loading.

    Returns
    -------
    uvicorn.Config
        A configuration object that can be passed to
        ``uvicorn.Server(config).run()`` to start the application.
    """
    global app  # noqa: PLW0603

    # Extract logging parameters (available on both config types)
    log_file = getattr(config_args, "log_file", None)
    no_log_file = getattr(config_args, "no_log_file", False)
    log_level = getattr(config_args, "log_level", "INFO")

    # Configure logging based on CLI parameters
    configure_logging(
        log_file=log_file,
        no_log_file=no_log_file,
        log_level=log_level,
    )

    # Choose the correct lifespan based on config type
    if isinstance(config_args, MultiModelServerConfig):
        lifespan_fn = create_multi_lifespan(config_args)
    else:
        lifespan_fn = create_lifespan(config_args)

    # Create FastAPI app with the configured lifespan
    app = FastAPI(
        title="OpenAI-compatible API",
        description="API for OpenAI-compatible chat completion and text embedding",
        version=__version__,
        lifespan=lifespan_fn,
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
        logger.error(f"Global exception handler caught: {exc!s}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": {"message": "Internal server error", "type": "internal_error"}},
        )

    host = config_args.host
    port = config_args.port
    logger.info(f"Starting server on {host}:{port}")
    return uvicorn.Config(
        app=app,
        host=host,
        port=port,
        log_level=log_level.lower(),
        access_log=True,
    )
