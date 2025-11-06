import sys
import asyncio
import click
import uvicorn
from loguru import logger
from functools import lru_cache
from app.version import __version__
from app.main import setup_server
from app.handler.parser.factory import PARSER_REGISTRY

class Config:
    """Configuration container for server parameters."""
    def __init__(self, model_path, model_type, context_length, port, host, max_concurrency, queue_timeout, queue_size, disable_auto_resize=False, quantize=8, config_name=None, lora_paths=None, lora_scales=None, log_file=None, no_log_file=False, log_level="INFO", enable_auto_tool_choice=False, tool_call_parser=None, reasoning_parser=None):
        self.model_path = model_path
        self.model_type = model_type
        self.context_length = context_length
        self.port = port
        self.host = host
        self.max_concurrency = max_concurrency
        self.queue_timeout = queue_timeout
        self.queue_size = queue_size
        self.disable_auto_resize = disable_auto_resize
        self.quantize = quantize
        self.config_name = config_name
        self.log_file = log_file
        self.no_log_file = no_log_file
        self.log_level = log_level
        self.enable_auto_tool_choice = enable_auto_tool_choice
        self.tool_call_parser = tool_call_parser
        self.reasoning_parser = reasoning_parser
        
        # Process comma-separated LoRA paths and scales
        if lora_paths:
            self.lora_paths = [path.strip() for path in lora_paths.split(',') if path.strip()]
        else:
            self.lora_paths = None
            
        if lora_scales:
            self.lora_scales = [float(scale.strip()) for scale in lora_scales.split(',') if scale.strip()]
        else:
            self.lora_scales = None


    @property
    def model_identifier(self):
        """Get the appropriate model identifier based on model type."""
        # For Flux models, we always use model_path (local directory path)
        return self.model_path


# Configure basic logging for CLI (will be overridden by main.py)
logger.remove()  # Remove default handler
logger.add(
    sys.stderr, 
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
           "<level>{level: <8}</level> | "
           "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
           "✦ <level>{message}</level>",
    colorize=True,
    level="INFO"
)


@click.group()
@click.version_option(
    version=__version__, 
    message="""
✨ %(prog)s - OpenAI Compatible API Server for MLX models ✨
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🚀 Version: %(version)s
"""
)
def cli():
    """MLX Server - OpenAI Compatible API for MLX models."""
    pass


@lru_cache(maxsize=1)
def get_server_config(model_path, model_type, context_length, port, host, max_concurrency, queue_timeout, queue_size, quantize, config_name, lora_paths, lora_scales, disable_auto_resize, log_file, no_log_file, log_level, enable_auto_tool_choice, tool_call_parser, reasoning_parser):
    """Cache and return server configuration to avoid redundant processing."""
    return Config(
        model_path=model_path,
        model_type=model_type,
        context_length=context_length,
        port=port,
        host=host,
        max_concurrency=max_concurrency,
        queue_timeout=queue_timeout,
        queue_size=queue_size,
        disable_auto_resize=disable_auto_resize,
        quantize=quantize,
        config_name=config_name,
        lora_paths=lora_paths,
        lora_scales=lora_scales,
        log_file=log_file,
        no_log_file=no_log_file,
        log_level=log_level,
        enable_auto_tool_choice=enable_auto_tool_choice,
        tool_call_parser=tool_call_parser,
        reasoning_parser=reasoning_parser
    )


def print_startup_banner(args):
    """Display beautiful startup banner with configuration details."""
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info(f"✨ MLX Server v{__version__} Starting ✨")
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info(f"🔮 Model Path: {args.model_path}")
    logger.info(f"🔮 Model Type: {args.model_type}")
    if args.context_length:
        logger.info(f"🔮 Context Length: {args.context_length}")
    logger.info(f"🌐 Host: {args.host}")
    logger.info(f"🔌 Port: {args.port}")
    logger.info(f"⚡ Max Concurrency: {args.max_concurrency}")
    logger.info(f"⏱️ Queue Timeout: {args.queue_timeout} seconds")
    logger.info(f"📊 Queue Size: {args.queue_size}")
    if args.model_type in ["image-generation", "image-edit"]:
        logger.info(f"🔮 Quantize: {args.quantize}")
        logger.info(f"🔮 Config Name: {args.config_name}")
        if args.lora_paths:
            logger.info(f"🔮 LoRA Paths: {args.lora_paths}")
        if args.lora_scales:
            logger.info(f"🔮 LoRA Scales: {args.lora_scales}")
    if hasattr(args, 'disable_auto_resize') and args.disable_auto_resize and args.model_type == "multimodal":
        logger.info(f"🖼️ Auto-resize: Disabled")
    if args.model_type in ["lm", "multimodal"]:
        if args.enable_auto_tool_choice:
            logger.info(f"🔧 Auto Tool Choice: Enabled")
        if args.tool_call_parser:
            logger.info(f"🔧 Tool Call Parser: {args.tool_call_parser}")
        if args.reasoning_parser:
            logger.info(f"🔧 Reasoning Parser: {args.reasoning_parser}")
    logger.info(f"📝 Log Level: {args.log_level}")
    if args.no_log_file:
        logger.info(f"📝 File Logging: Disabled")
    elif args.log_file:
        logger.info(f"📝 Log File: {args.log_file}")
    else:
        logger.info(f"📝 Log File: logs/app.log (default)")
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

@cli.command()
@click.option(
    "--model-path", 
    help="Path to the model (required for lm, multimodal, embeddings, image-generation, image-edit, whisper model types). With `image-generation` or `image-edit` model types, it should be the local path to the model."
)
@click.option(
    "--model-type",
    default="lm",
    type=click.Choice(["lm", "multimodal", "image-generation", "image-edit", "embeddings", "whisper"]),
    help="Type of model to run (lm: text-only, multimodal: text+vision+audio, image-generation: flux image generation, image-edit: flux image edit, embeddings: text embeddings, whisper: audio transcription)"
)
@click.option(
    "--context-length",
    default=None,
    type=int,
    help="Context length for language models. Only works with `lm` or `multimodal` model types."
)
@click.option(
    "--port", 
    default=8000, 
    type=int, 
    help="Port to run the server on"
)
@click.option(
    "--host", 
    default="0.0.0.0", 
    help="Host to run the server on"
)
@click.option(
    "--max-concurrency", 
    default=1, 
    type=int, 
    help="Maximum number of concurrent requests"
)
@click.option(
    "--queue-timeout", 
    default=300, 
    type=int, 
    help="Request timeout in seconds"
)
@click.option(
    "--queue-size", 
    default=100, 
    type=int, 
    help="Maximum queue size for pending requests"
)
@click.option(
    "--quantize",
    default=8,
    type=int,
    help="Quantization level for the model. Only used for image-generation and image-edit Flux models."
)
@click.option(
    "--config-name",
    default=None,
    type=click.Choice(["flux-schnell", "flux-dev", "flux-krea-dev", "flux-kontext-dev"]),
    help="Config name of the model. Only used for image-generation and image-edit Flux models."
)
@click.option(
    "--lora-paths",
    default=None,
    type=str,
    help="Path to the LoRA file(s). Multiple paths should be separated by commas."
)
@click.option(
    "--lora-scales",
    default=None,
    type=str,
    help="Scale factor for the LoRA file(s). Multiple scales should be separated by commas."
)
@click.option(
    "--disable-auto-resize",
    is_flag=True,
    help="Disable automatic model resizing. Only work for Vision Language Models."
)
@click.option(
    "--log-file",
    default=None,
    type=str,
    help="Path to log file. If not specified, logs will be written to 'logs/app.log' by default."
)
@click.option(
    "--no-log-file",
    is_flag=True,
    help="Disable file logging entirely. Only console output will be shown."
)
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
    help="Set the logging level. Default is INFO."
)
@click.option(
    "--enable-auto-tool-choice",
    is_flag=True,
    help="Enable automatic tool choice. Only works with language models."
)
@click.option(
    "--tool-call-parser",
    default=None,
    type=click.Choice(list(PARSER_REGISTRY.keys())),
    help="Specify tool call parser to use instead of auto-detection. Only works with language models."
)
@click.option(
    "--reasoning-parser",
    default=None,
    type=click.Choice(list(PARSER_REGISTRY.keys())),
    help="Specify reasoning parser to use instead of auto-detection. Only works with language models."
)
def launch(model_path, model_type, context_length, port, host, max_concurrency, queue_timeout, queue_size, quantize, config_name, lora_paths, lora_scales, disable_auto_resize, log_file, no_log_file, log_level, enable_auto_tool_choice, tool_call_parser, reasoning_parser):
    """Launch the MLX server with the specified model."""
    try:
        # Validate that config name is only used with image-generation and image-edit model types
        if config_name and model_type not in ["image-generation", "image-edit"]:
            logger.warning(f"Config name parameter '{config_name}' provided but model type is '{model_type}'. Config name is only used with image-generation and image-edit models.")
        elif model_type == "image-generation" and not config_name:
            logger.warning("Model type is 'image-generation' but no config name specified. Using default 'flux-schnell'.")
            config_name = "flux-schnell"
        elif model_type == "image-edit" and not config_name:
            logger.warning("Model type is 'image-edit' but no config name specified. Using default 'flux-kontext-dev'.")
            config_name = "flux-kontext-dev"
        
        # Get optimized configuration
        args = get_server_config(model_path, model_type, context_length, port, host, max_concurrency, queue_timeout, queue_size, quantize, config_name, lora_paths, lora_scales, disable_auto_resize, log_file, no_log_file, log_level, enable_auto_tool_choice, tool_call_parser, reasoning_parser)
        
        # Display startup information
        print_startup_banner(args)
        
        # Set up and start the server
        config = asyncio.run(setup_server(args))
        logger.info("Server configuration complete.")
        logger.info("Starting Uvicorn server...")
        uvicorn.Server(config).run()
    except KeyboardInterrupt:
        logger.info("Server shutdown requested by user. Exiting...")
    except Exception as e:
        logger.error(f"Server startup failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    cli()