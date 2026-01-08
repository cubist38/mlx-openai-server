"""Command-line interface and helpers for the MLX server.

This module defines the Click command group used by the package and the
``launch`` command which constructs a server configuration and starts
the ASGI server.
"""

from __future__ import annotations

import asyncio
import sys

import click
from loguru import logger

from .config import MLXServerConfig
from .main import start
from .message_converters import MESSAGE_CONVERTER_MAP
from .parsers import REASONING_PARSER_MAP, TOOL_PARSER_MAP, UNIFIED_PARSER_MAP
from .version import __version__


class UpperChoice(click.Choice[str]):
    """Case-insensitive choice type that returns uppercase values.

    This small convenience subclass normalizes user input in a
    case-insensitive way but returns the canonical uppercase option
    value to callers. It is useful for flags like ``--log-level``
    where the internal representation is uppercased.
    """

    def normalize_choice(self, choice: str | None, ctx: click.Context | None) -> str | None:  # type: ignore[override]
        """Return the canonical uppercase choice or raise BadParameter.

        Parameters
        ----------
        choice:
            Raw value supplied by the user (may be ``None``).
        ctx:
            Click context object (unused here but part of the API).

        Returns
        -------
        str | None
            Uppercased canonical choice, or ``None`` if ``choice`` is
            ``None``.
        """
        if choice is None:
            return None
        upperchoice = choice.upper()
        for opt in self.choices:
            if opt.upper() == upperchoice:
                return upperchoice
        raise click.BadParameter(
            f"invalid choice: {choice!r}. (choose from {', '.join(map(repr, self.choices))})"
        )


# Configure basic logging for CLI (will be overridden by main.py)
logger.remove()  # Remove default handler
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "✦ <level>{message}</level>",
    colorize=True,
    level="INFO",
)


@click.group()
@click.version_option(
    version=__version__,
    message="""
✨ %(prog)s - OpenAI Compatible API Server for MLX models ✨
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🚀 Version: %(version)s
""",
)
def cli() -> None:
    """Top-level Click command group for the MLX server CLI.

    Subcommands (such as ``launch``) are registered on this group and
    invoked by the console entry point.
    """


@cli.command(help="Start the MLX OpenAI Server with the supplied flags")
@click.option(
    "--model-path",
    required=True,
    help="Path to the model (required for lm, multimodal, embeddings, image-generation, image-edit, whisper model types). Can be a local path or Hugging Face repository ID (e.g., 'blackforestlabs/FLUX.1-dev').",
)
@click.option(
    "--model-type",
    default="lm",
    type=click.Choice(
        ["lm", "multimodal", "image-generation", "image-edit", "embeddings", "whisper"]
    ),
    help="Type of model to run (lm: text-only, multimodal: text+vision+audio, image-generation: flux image generation, image-edit: flux image edit, embeddings: text embeddings, whisper: audio transcription)",
)
@click.option(
    "--context-length",
    default=32768,
    type=int,
    help="Context length for language models. Only works with `lm` or `multimodal` model types.",
)
@click.option("--port", default=8000, type=int, help="Port to run the server on")
@click.option("--host", default="0.0.0.0", help="Host to run the server on")
@click.option(
    "--max-concurrency", default=1, type=int, help="Maximum number of concurrent requests"
)
@click.option("--queue-timeout", default=300, type=int, help="Request timeout in seconds")
@click.option("--queue-size", default=100, type=int, help="Maximum queue size for pending requests")
@click.option(
    "--quantize",
    default=8,
    type=int,
    help="Quantization level for the model. Only used for image-generation and image-edit Flux models.",
)
@click.option(
    "--config-name",
    default=None,
    type=click.Choice(
        [
            "flux-schnell",
            "flux-dev",
            "flux-krea-dev",
            "flux-kontext-dev",
            "qwen-image",
            "qwen-image-edit",
            "z-image-turbo",
            "fibo",
        ]
    ),
    help="Config name of the model. Only used for image-generation and image-edit models.",
)
@click.option(
    "--lora-paths",
    default=None,
    type=str,
    help="Path to the LoRA file(s). Multiple paths should be separated by commas.",
)
@click.option(
    "--lora-scales",
    default=None,
    type=str,
    help="Scale factor for the LoRA file(s). Multiple scales should be separated by commas.",
)
@click.option(
    "--disable-auto-resize",
    is_flag=True,
    help="Disable automatic model resizing. Only work for Vision Language Models.",
)
@click.option(
    "--log-file",
    default=None,
    type=str,
    help="Path to log file. If not specified, logs will be written to 'logs/app.log' by default.",
)
@click.option(
    "--no-log-file",
    is_flag=True,
    help="Disable file logging entirely. Only console output will be shown.",
)
@click.option(
    "--log-level",
    default="INFO",
    type=UpperChoice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
    help="Set the logging level. Default is INFO.",
)
@click.option(
    "--enable-auto-tool-choice",
    is_flag=True,
    help="Enable automatic tool choice. Only works with language models.",
)
@click.option(
    "--tool-call-parser",
    default=None,
    type=click.Choice(sorted(set(TOOL_PARSER_MAP.keys()) | set(UNIFIED_PARSER_MAP.keys()))),
    help="Specify tool call parser to use instead of auto-detection. Only works with language models.",
)
@click.option(
    "--reasoning-parser",
    default=None,
    type=click.Choice(sorted(set(REASONING_PARSER_MAP.keys()) | set(UNIFIED_PARSER_MAP.keys()))),
    help="Specify reasoning parser to use instead of auto-detection. Only works with language models.",
)
@click.option(
    "--message-converter",
    default=None,
    type=click.Choice(sorted(MESSAGE_CONVERTER_MAP.keys())),
    help="Specify message converter to use for preprocessing messages. Only works with language models.",
)
@click.option(
    "--trust-remote-code",
    is_flag=True,
    help="Enable trust_remote_code when loading models. This allows loading custom code from model repositories.",
)
@click.option(
    "--chat-template-file",
    default=None,
    type=str,
    help="Path to a custom chat template file. Only works with language models (lm) and multimodal models.",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug mode for language models. Only works with language models (lm) and multimodal models.",
)
def launch(
    model_path: str,
    model_type: str,
    context_length: int,
    port: int,
    host: str,
    max_concurrency: int,
    queue_timeout: int,
    queue_size: int,
    quantize: int,
    config_name: str | None,
    lora_paths: str | None,
    lora_scales: str | None,
    disable_auto_resize: bool,
    log_file: str | None,
    no_log_file: bool,
    log_level: str,
    enable_auto_tool_choice: bool,
    tool_call_parser: str | None,
    reasoning_parser: str | None,
    message_converter: str | None,
    trust_remote_code: bool,
    chat_template_file: str | None,
    debug: bool,
) -> None:
    """Start the FastAPI/Uvicorn server with the supplied flags.

    The command builds a server configuration object using
    ``MLXServerConfig`` and then calls the async ``start`` routine
    which handles the event loop and server lifecycle.

    Parameters
    ----------
    model_path : str
        Path to the model (required for lm, multimodal, embeddings, image-generation, image-edit, whisper model types).
    model_type : str
        Type of model to run (lm, multimodal, image-generation, image-edit, embeddings, whisper).
    context_length : int
        Context length for language models.
    port : int
        Port to run the server on.
    host : str
        Host to run the server on.
    max_concurrency : int
        Maximum number of concurrent requests.
    queue_timeout : int
        Request timeout in seconds.
    queue_size : int
        Maximum queue size for pending requests.
    quantize : int
        Quantization level for the model.
    config_name : str or None
        Config name of the model.
    lora_paths : str or None
        Path to the LoRA file(s).
    lora_scales : str or None
        Scale factor for the LoRA file(s).
    disable_auto_resize : bool
        Disable automatic model resizing.
    log_file : str or None
        Path to log file.
    no_log_file : bool
        Disable file logging entirely.
    log_level : str
        Set the logging level.
    enable_auto_tool_choice : bool
        Enable automatic tool choice.
    tool_call_parser : str or None
        Specify tool call parser to use.
    reasoning_parser : str or None
        Specify reasoning parser to use.
    trust_remote_code : bool
        Enable trust_remote_code when loading models.
    """
    args = MLXServerConfig(
        model_path=model_path,
        model_type=model_type,
        context_length=context_length,
        port=port,
        host=host,
        max_concurrency=max_concurrency,
        queue_timeout=queue_timeout,
        queue_size=queue_size,
        quantize=quantize,
        config_name=config_name,
        lora_paths_str=lora_paths,
        lora_scales_str=lora_scales,
        disable_auto_resize=disable_auto_resize,
        log_file=log_file,
        no_log_file=no_log_file,
        log_level=log_level,
        enable_auto_tool_choice=enable_auto_tool_choice,
        tool_call_parser=tool_call_parser,
        reasoning_parser=reasoning_parser,
        message_converter=message_converter,
        trust_remote_code=trust_remote_code,
        chat_template_file=chat_template_file,
        debug=debug,
    )

    asyncio.run(start(args))
