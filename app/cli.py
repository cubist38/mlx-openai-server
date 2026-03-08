"""Command-line interface and helpers for the MLX server.

This module defines the Click command group used by the package and the
``launch`` command which constructs a server configuration and starts
the ASGI server. When a ``--config`` YAML file is supplied the server
runs in multi-handler mode, loading multiple models at once.
"""

from __future__ import annotations

import asyncio
import sys

import click
from loguru import logger

from .config import MLXServerConfig, load_config_from_yaml
from .main import start, start_multi
from .message_converters import MESSAGE_CONVERTER_MAP
from .parsers import REASONING_PARSER_MAP, TOOL_PARSER_MAP, UNIFIED_PARSER_MAP
from .version import __version__

try:
    from .models.mflux import IMAGE_CONFIG_NAMES
except ImportError as exc:
    IMAGE_CONFIG_NAMES: tuple[str, ...] = ()
    MFLUX_AVAILABLE = False
    MFLUX_IMPORT_ERROR: ImportError | None = exc
else:
    MFLUX_AVAILABLE = True
    MFLUX_IMPORT_ERROR = None

MFLUX_INSTALL_HINT = (
    "Image generation and editing require the optional `mflux` package. "
    "Install a compatible build separately, for example "
    "`pip install git+https://github.com/cubist38/mflux.git`."
)


class UpperChoice(click.Choice):
    """Case-insensitive choice type that returns uppercase values.

    This small convenience subclass normalizes user input in a
    case-insensitive way but returns the canonical uppercase option
    value to callers. It is useful for flags like ``--log-level``
    where the internal representation is uppercased.
    """

    def normalize_choice(self, choice, ctx):
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


def validate_image_config_name(
    _ctx: click.Context, _param: click.Parameter, value: str | None
) -> str | None:
    """Validate image config names when optional mflux support is installed."""
    if value is None or not IMAGE_CONFIG_NAMES:
        return value

    if value not in IMAGE_CONFIG_NAMES:
        choices = ", ".join(sorted(IMAGE_CONFIG_NAMES))
        raise click.BadParameter(f"invalid choice: {value!r}. (choose from {choices})")

    return value


def ensure_image_support_available(model_types: set[str]) -> None:
    """Raise a usage error when image features are requested without mflux."""
    if not any(model_type in {"image-generation", "image-edit"} for model_type in model_types):
        return

    if MFLUX_AVAILABLE:
        return

    detail = f" Optional import failed: {MFLUX_IMPORT_ERROR!s}" if MFLUX_IMPORT_ERROR else ""
    raise click.UsageError(f"{MFLUX_INSTALL_HINT}{detail}")


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
def cli():
    """Top-level Click command group for the MLX server CLI.

    Subcommands (such as ``launch``) are registered on this group and
    invoked by the console entry point.
    """


@cli.command()
@click.option(
    "--config",
    "config_file",
    default=None,
    type=click.Path(exists=True),
    help="Path to a YAML config file for multi-handler mode. "
    "When provided, --model-path and other per-model flags are ignored.",
)
@click.option(
    "--model-path",
    required=False,
    default=None,
    help="Path to the model (required for lm, multimodal, embeddings, image-generation, image-edit, whisper model types). With `image-generation` or `image-edit` model types, it should be the local path to the model.",
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
    default=None,
    type=int,
    help="Context length for language models. If not specified, uses model default. Only works with `lm` or `multimodal` model types.",
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
    default=None,
    type=int,
    help="Quantization level for the model. Only used for image-generation and image-edit Flux models.",
)
@click.option(
    "--config-name",
    default=None,
    type=str,
    callback=validate_image_config_name,
    metavar="CONFIG_NAME",
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
    hidden=True,
    help="Deprecated override for message preprocessing. Message converters are auto-detected from parser selection.",
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
@click.option(
    "--prompt-cache-size",
    default=100,
    type=int,
    help="Maximum number of prompt KV cache entries to store. Only works with language models (lm). Default is 10.",
)
@click.option(
    "--max-bytes",
    "prompt_cache_max_bytes",
    default=1 << 63,
    type=int,
    help="Maximum total bytes retained by prompt KV caches before eviction. Only works with language models (lm).",
)
@click.option(
    "--draft-model-path",
    default=None,
    type=str,
    help="Path to the draft model for speculative decoding. Only supported with model type 'lm'. When set, --num-draft-tokens controls how many tokens the draft model generates per step.",
)
@click.option(
    "--num-draft-tokens",
    default=2,
    type=int,
    help="Number of draft tokens per step when using speculative decoding (--draft-model-path). Only supported with model type 'lm'. Default is 2.",
)
# Sampling parameters (defaults used when API request omits them)
@click.option(
    "--max-tokens",
    default=100000,
    type=int,
    help="Default maximum number of tokens to generate.",
)
@click.option("--temperature", default=1.0, type=float, help="Default sampling temperature.")
@click.option(
    "--top-p", default=1.0, type=float, help="Default nucleus sampling (top-p) probability."
)
@click.option("--top-k", default=20, type=int, help="Default top-k sampling parameter.")
@click.option("--min-p", default=0.0, type=float, help="Default min-p sampling parameter.")
@click.option(
    "--presence-penalty",
    default=0.0,
    type=float,
    help="Default presence penalty for token generation.",
)
@click.option(
    "--xtc-probability",
    default=0.0,
    type=float,
    help="Default XTC probability sampling parameter.",
)
@click.option(
    "--xtc-threshold",
    default=0.0,
    type=float,
    help="Default XTC threshold sampling parameter.",
)
@click.option("--seed", default=0, type=int, help="Default random seed for generation.")
@click.option(
    "--repetition-context-size",
    default=20,
    type=int,
    help="Default repetition context size parameter.",
)
def launch(
    config_file,
    model_path,
    model_type,
    context_length,
    port,
    host,
    max_concurrency,
    queue_timeout,
    queue_size,
    quantize,
    config_name,
    lora_paths,
    lora_scales,
    disable_auto_resize,
    log_file,
    no_log_file,
    log_level,
    enable_auto_tool_choice,
    tool_call_parser,
    reasoning_parser,
    message_converter,
    trust_remote_code,
    chat_template_file,
    debug,
    prompt_cache_size,
    prompt_cache_max_bytes,
    draft_model_path,
    num_draft_tokens,
    max_tokens,
    temperature,
    top_p,
    top_k,
    min_p,
    presence_penalty,
    xtc_probability,
    xtc_threshold,
    seed,
    repetition_context_size,
) -> None:
    """Start the FastAPI/Uvicorn server with the supplied flags.

    When ``--config`` is provided the server launches in multi-handler
    mode, loading all models defined in the YAML file. In this mode
    per-model CLI flags (``--model-path``, ``--model-type``, etc.) are
    ignored.

    Otherwise the command builds a single-model ``MLXServerConfig``
    and calls the async ``start`` routine.
    """
    # ---- Multi-handler mode ----
    if config_file is not None:
        logger.info(f"Loading multi-handler config from: {config_file}")
        try:
            multi_config = load_config_from_yaml(config_file)
        except (FileNotFoundError, ValueError) as e:
            raise click.BadParameter(str(e), param_hint="'--config'") from e
        ensure_image_support_available({model.model_type for model in multi_config.models})
        asyncio.run(start_multi(multi_config))
        return

    # ---- Single-handler mode (original behavior) ----
    if model_path is None:
        raise click.UsageError(
            "Either --config (multi-handler YAML) or --model-path (single model) is required."
        )

    ensure_image_support_available({model_type})

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
        prompt_cache_size=prompt_cache_size,
        prompt_cache_max_bytes=prompt_cache_max_bytes,
        draft_model_path=draft_model_path,
        num_draft_tokens=num_draft_tokens,
        default_max_tokens=max_tokens,
        default_temperature=temperature,
        default_top_p=top_p,
        default_top_k=top_k,
        default_min_p=min_p,
        default_presence_penalty=presence_penalty,
        default_xtc_probability=xtc_probability,
        default_xtc_threshold=xtc_threshold,
        default_seed=seed,
        default_repetition_context_size=repetition_context_size,
    )

    asyncio.run(start(args))
