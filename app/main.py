import argparse
import asyncio
import gc
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

import mlx.core as mx
from app.handler.mlx_vlm import MLXVLMHandler
from app.handler.mlx_lm import MLXLMHandler
from app.handler.mlx_embeddings import MLXEmbeddingsHandler
from app.handler.mlx_whisper import MLXWhisperHandler
from app.handler import MLXFluxHandler, MFLUX_AVAILABLE 
from app.api.endpoints import router
from app.version import __version__

def configure_logging(log_file=None, no_log_file=False, log_level="INFO"):
    """Configure loguru logging based on CLI parameters."""
    logger.remove()  # Remove default handler
    
    # Add console handler
    logger.add(
        lambda msg: print(msg), 
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
               "✦ <level>{message}</level>",
        colorize=True
    )
    
    # Add file handler if not disabled
    if not no_log_file:
        file_path = log_file if log_file else "logs/app.log"
        logger.add(
            file_path,
            rotation="500 MB",
            retention="10 days",
            level=log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
        )

# Logging will be configured in setup_server() with CLI arguments

def parse_args():
    parser = argparse.ArgumentParser(description="MLX OpenAI Compatible Server")
    parser.add_argument("--model-path", type=str, help="Path to the model (required for lm, multimodal, image-generation, image-edit, embeddings, whisper model types). With `image-generation` or `image-edit` model types, it should be the local path to the model.")
    parser.add_argument("--model-type", type=str, default="lm", choices=["lm", "multimodal", "image-generation", "image-edit", "embeddings", "whisper"], help="Model type")
    parser.add_argument("--context-length", type=int, default=None, help="Context length for language models. Only works with `lm` or `multimodal` model types.")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on")
    parser.add_argument("--max-concurrency", type=int, default=1, help="Maximum number of concurrent requests")
    parser.add_argument("--queue-timeout", type=int, default=300, help="Request timeout in seconds")
    parser.add_argument("--queue-size", type=int, default=100, help="Maximum queue size for pending requests")
    parser.add_argument("--quantize", type=int, default=8, help="Quantization level for the model. Only used for image-generation and image-edit Flux models.")
    parser.add_argument("--config-name", type=str, default=None, choices=["flux-schnell", "flux-dev", "flux-krea-dev", "flux-kontext-dev"], help="Config name of the model. Only used for image-generation and image-edit Flux models.")
    parser.add_argument("--lora-paths", type=str, default=None, help="Path to the LoRA file(s). Multiple paths should be separated by commas.")
    parser.add_argument("--lora-scales", type=str, default=None, help="Scale factor for the LoRA file(s). Multiple scales should be separated by commas.")
    parser.add_argument("--disable-auto-resize", action="store_true", help="Disable automatic model resizing. Only work for Vision Language Models.")
    parser.add_argument("--log-file", type=str, default=None, help="Path to log file. If not specified, logs will be written to 'logs/app.log' by default.")
    parser.add_argument("--no-log-file", action="store_true", help="Disable file logging entirely. Only console output will be shown.")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Set the logging level. Default is INFO.")
    
    args = parser.parse_args()
    
    return args


def get_model_identifier(args):
    """Get the appropriate model identifier based on model type."""
    return args.model_path

def create_lifespan(config_args):
    """Factory function to create a lifespan context manager with access to config args."""
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            model_identifier = get_model_identifier(config_args)
            if config_args.model_type == "image-generation":
                logger.info(f"Initializing MLX handler with model name: {model_identifier}")
            else:
                logger.info(f"Initializing MLX handler with model path: {model_identifier}")
            
            if config_args.model_type == "multimodal":
                handler = MLXVLMHandler(
                    model_path=model_identifier,
                    context_length=getattr(config_args, 'context_length', None),
                    max_concurrency=config_args.max_concurrency,
                    disable_auto_resize=getattr(config_args, 'disable_auto_resize', False),
                    enable_auto_tool_choice=getattr(config_args, 'enable_auto_tool_choice', False),
                    tool_call_parser=getattr(config_args, 'tool_call_parser', None),
                    reasoning_parser=getattr(config_args, 'reasoning_parser', None)
                )
            elif config_args.model_type == "image-generation":
                if not MFLUX_AVAILABLE:
                    raise ValueError("Image generation requires mflux. Install with: pip install git+https://github.com/cubist38/mflux.git")
                if not config_args.config_name in ["flux-schnell", "flux-dev", "flux-krea-dev"]:
                    raise ValueError(f"Invalid config name: {config_args.config_name}. Only flux-schnell, flux-dev, and flux-krea-dev are supported for image generation.")
                handler = MLXFluxHandler(
                    model_path=model_identifier,
                    max_concurrency=config_args.max_concurrency,
                    quantize=getattr(config_args, 'quantize', 8),
                    config_name=config_args.config_name,
                    lora_paths=getattr(config_args, 'lora_paths', None),
                    lora_scales=getattr(config_args, 'lora_scales', None)
                )
            elif config_args.model_type == "embeddings":
                handler = MLXEmbeddingsHandler(
                    model_path=model_identifier,
                    max_concurrency=config_args.max_concurrency
                )
            elif config_args.model_type == "image-edit":
                if not MFLUX_AVAILABLE:
                    raise ValueError("Image editing requires mflux. Install with: pip install git+https://github.com/cubist38/mflux.git")
                if config_args.config_name != "flux-kontext-dev":
                    raise ValueError(f"Invalid config name: {config_args.config_name}. Only flux-kontext-dev is supported for image edit.")
                handler = MLXFluxHandler(
                    model_path=model_identifier,
                    max_concurrency=config_args.max_concurrency,
                    quantize=getattr(config_args, 'quantize', 8),
                    config_name=config_args.config_name,
                    lora_paths=getattr(config_args, 'lora_paths', None),
                    lora_scales=getattr(config_args, 'lora_scales', None)
                )
            elif config_args.model_type == "whisper":
                handler = MLXWhisperHandler(
                    model_path=model_identifier,
                    max_concurrency=config_args.max_concurrency
                )
            else:
                handler = MLXLMHandler(
                    model_path=model_identifier,
                    context_length=getattr(config_args, 'context_length', None),
                    max_concurrency=config_args.max_concurrency,
                    enable_auto_tool_choice=getattr(config_args, 'enable_auto_tool_choice', False),
                    tool_call_parser=getattr(config_args, 'tool_call_parser', None),
                    reasoning_parser=getattr(config_args, 'reasoning_parser', None)
                )       
            # Initialize queue
            await handler.initialize({
                "max_concurrency": config_args.max_concurrency,
                "timeout": config_args.queue_timeout,
                "queue_size": config_args.queue_size
            })
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

async def setup_server(args) -> uvicorn.Config:
    global app
    
    # Configure logging based on CLI parameters
    configure_logging(
        log_file=getattr(args, 'log_file', None),
        no_log_file=getattr(args, 'no_log_file', False),
        log_level=getattr(args, 'log_level', 'INFO')
    )
    
    # Create FastAPI app with the configured lifespan
    app = FastAPI(
        title="OpenAI-compatible API",
        description="API for OpenAI-compatible chat completion and text embedding",
        version=__version__,
        lifespan=create_lifespan(args)
    )
    
    app.include_router(router)
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, replace with specific origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        
        # Periodic memory cleanup for long-running processes
        if hasattr(request.app.state, 'request_count'):
            request.app.state.request_count += 1
        else:
            request.app.state.request_count = 1
        
        # Clean up memory every 50 requests
        if request.app.state.request_count % 50 == 0:
            mx.clear_cache()
            gc.collect()
            logger.debug(f"Performed memory cleanup after {request.app.state.request_count} requests")
        
        return response
    
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Global exception handler caught: {str(exc)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": {"message": "Internal server error", "type": "internal_error"}}
        )
    
    logger.info(f"Starting server on {args.host}:{args.port}")
    config = uvicorn.Config(
        app=app,
        host=args.host,
        port=args.port,
        log_level="info",
        access_log=True
    )
    return config

if __name__ == "__main__":
    args = parse_args()
    config = asyncio.run(setup_server(args))
    uvicorn.Server(config).run()