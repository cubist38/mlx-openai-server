"""CLI entrypoint shim for the MLX OpenAI Server package.

This lightweight module allows running the CLI via ``python -m app.main``
while preserving the same behavior as the installed console script. It
normalizes ``sys.argv`` so a missing subcommand implicitly becomes
``launch`` (backwards compatibility) and delegates to the Click-based
``cli`` command group defined in :mod:`app.cli`.

Examples
--------
Run the default launch flow:

    python -m app.main

Forward explicit arguments to the CLI:

    python -m app.main launch --port 8000

Run multi-handler mode from YAML config:

    python -m app.main launch --config config.yaml
"""

from __future__ import annotations

import os
import sys

from loguru import logger
import uvicorn

from .config import MLXServerConfig, ModelEntryConfig, MultiModelServerConfig
from .server import setup_server
from .version import __version__


def _format_bytes(n: int) -> str:
    """Render a byte count in a human-friendly unit.

    The prompt-cache byte budget defaults to ``1 << 63`` as a sentinel for
    "unbounded", which reads as a 19-digit integer in the startup banner.
    This helper turns that (and more ordinary values) into something a human
    can read at a glance.
    """
    if n >= (1 << 60):
        return "unbounded"
    gib = n / (1024**3)
    if gib >= 1:
        return f"{gib:.2f} GiB"
    mib = n / (1024**2)
    if mib >= 1:
        return f"{mib:.2f} MiB"
    return f"{n} B"


def print_startup_banner(config_args: MLXServerConfig) -> None:
    """Log a compact startup banner describing the selected config.

    The function emits human-friendly log messages that summarize the
    runtime configuration (model path/type, host/port, concurrency,
    LoRA settings, and logging options). Intended for the user-facing
    startup output only.

    Parameters
    ----------
    config_args : MLXServerConfig
        Single-model server configuration.
    """
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info(f"✨ MLX Server v{__version__} Starting ✨")
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info(f"🔮 Model Path: {config_args.model_path}")
    if config_args.served_model_name:
        logger.info(f"🔮 Served Model Name: {config_args.served_model_name}")
    logger.info(f"🔮 Model Type: {config_args.model_type}")
    if config_args.context_length:
        logger.info(f"🔮 Context Length: {config_args.context_length}")
    logger.info(f"🌐 Host: {config_args.host}")
    logger.info(f"🔌 Port: {config_args.port}")
    logger.info(f"⏱️ Queue Timeout: {config_args.queue_timeout} seconds")
    logger.info(f"📊 Queue Size: {config_args.queue_size}")
    if config_args.model_type in ["image-generation", "image-edit"]:
        logger.info(f"🔮 Quantize: {config_args.quantize}")
        logger.info(f"🔮 Config Name: {config_args.config_name}")
        if config_args.lora_paths:
            logger.info(f"🔮 LoRA Paths: {config_args.lora_paths}")
        if config_args.lora_scales:
            logger.info(f"🔮 LoRA Scales: {config_args.lora_scales}")
    if (
        hasattr(config_args, "disable_auto_resize")
        and config_args.disable_auto_resize
        and config_args.model_type == "multimodal"
    ):
        logger.info("🖼️ Auto-resize: Disabled")
    if config_args.model_type in ["lm", "multimodal"]:
        if config_args.enable_auto_tool_choice:
            logger.info("🔧 Auto Tool Choice: Enabled")
        if config_args.tool_call_parser:
            logger.info(f"🔧 Tool Call Parser: {config_args.tool_call_parser}")
        if config_args.reasoning_parser:
            logger.info(f"🔧 Reasoning Parser: {config_args.reasoning_parser}")
        if config_args.message_converter:
            logger.info(f"🔧 Message Converter: {config_args.message_converter}")
    if config_args.model_type == "lm":
        logger.info(f"💾 Prompt Cache Size: {config_args.prompt_cache_size} entries")
        logger.info(
            f"💾 Prompt Cache Max Bytes: {_format_bytes(config_args.prompt_cache_max_bytes)}"
        )
        logger.info(
            f"🧵 Batch Scheduler: decode={config_args.batch_completion_size}, "
            f"prefill={config_args.batch_prefill_size}, "
            f"prefill_step={config_args.batch_prefill_step_size}"
        )
    logger.info(f"📝 Log Level: {config_args.log_level}")
    if config_args.no_log_file:
        logger.info("📝 File Logging: Disabled")
    elif config_args.log_file:
        logger.info(f"📝 Log File: {config_args.log_file}")
    else:
        logger.info("📝 Log File: logs/app.log (default)")
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


def _model_entry_extras(m: ModelEntryConfig) -> list[tuple[str, object]]:
    """Return non-default per-model settings worth surfacing in the banner.

    Only fields that differ from their dataclass default are included —
    this keeps the banner quiet for minimal configs and lights up every
    knob that was actually touched (parsers, tool choice, LoRA, draft
    model, KV quant, on-demand, etc.).
    """
    extras: list[tuple[str, object]] = []

    if m.served_model_name and m.served_model_name != m.model_path:
        extras.append(("served_model_name", m.served_model_name))
    if m.context_length is not None:
        extras.append(("context_length", m.context_length))
    if m.enable_auto_tool_choice:
        extras.append(("auto_tool_choice", True))
    if m.tool_call_parser:
        extras.append(("tool_call_parser", m.tool_call_parser))
    if m.reasoning_parser:
        extras.append(("reasoning_parser", m.reasoning_parser))
    if m.message_converter:
        extras.append(("message_converter", m.message_converter))
    if m.chat_template_file:
        extras.append(("chat_template_file", m.chat_template_file))
    if m.trust_remote_code:
        extras.append(("trust_remote_code", True))
    if m.draft_model_path:
        extras.append(("draft_model_path", m.draft_model_path))
        extras.append(("num_draft_tokens", m.num_draft_tokens))
    if m.kv_bits is not None:
        extras.append(("kv_bits", f"{m.kv_bits} (group={m.kv_group_size}, start={m.quantized_kv_start})"))
    if m.quantize is not None:
        extras.append(("quantize", m.quantize))
    if m.config_name:
        extras.append(("config_name", m.config_name))
    if m.lora_paths:
        extras.append(("lora_paths", m.lora_paths))
    if m.lora_scales:
        extras.append(("lora_scales", m.lora_scales))
    if m.disable_auto_resize:
        extras.append(("auto_resize", False))
    if m.on_demand:
        extras.append(("on_demand", f"idle_timeout={m.on_demand_idle_timeout}s"))
    if m.debug:
        extras.append(("debug", True))
    return extras


def print_multi_startup_banner(config: MultiModelServerConfig) -> None:
    """Log a startup banner for multi-handler mode.

    Parameters
    ----------
    config : MultiModelServerConfig
        Multi-model server configuration.
    """
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info(f"✨ MLX Server v{__version__} Starting (Multi-Handler Mode) ✨")
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info(f"🌐 Host: {config.host}")
    logger.info(f"🔌 Port: {config.port}")
    logger.info(f"📝 Log Level: {config.log_level}")
    logger.info(f"🔢 Models to load: {len(config.models)}")
    for idx, m in enumerate(config.models, start=1):
        logger.info(f"  [{idx}] {m.served_model_name} (type={m.model_type}, path={m.model_path})")
        for key, value in _model_entry_extras(m):
            logger.info(f"       • {key}: {value}")
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


def _apply_sampling_env(config: MLXServerConfig) -> None:
    """Set DEFAULT_* env vars from config so model layers use CLI sampling defaults."""
    if config.default_max_tokens is not None:
        os.environ["DEFAULT_MAX_TOKENS"] = str(config.default_max_tokens)
    if config.default_temperature is not None:
        os.environ["DEFAULT_TEMPERATURE"] = str(config.default_temperature)
    if config.default_top_p is not None:
        os.environ["DEFAULT_TOP_P"] = str(config.default_top_p)
    if config.default_top_k is not None:
        os.environ["DEFAULT_TOP_K"] = str(config.default_top_k)
    if config.default_min_p is not None:
        os.environ["DEFAULT_MIN_P"] = str(config.default_min_p)
    if config.default_repetition_penalty is not None:
        os.environ["DEFAULT_REPETITION_PENALTY"] = str(config.default_repetition_penalty)
    if config.default_presence_penalty is not None:
        os.environ["DEFAULT_PRESENCE_PENALTY"] = str(config.default_presence_penalty)
    if config.default_xtc_probability is not None:
        os.environ["DEFAULT_XTC_PROBABILITY"] = str(config.default_xtc_probability)
    if config.default_xtc_threshold is not None:
        os.environ["DEFAULT_XTC_THRESHOLD"] = str(config.default_xtc_threshold)
    if config.default_seed is not None:
        os.environ["DEFAULT_SEED"] = str(config.default_seed)
    if config.default_repetition_context_size is not None:
        os.environ["DEFAULT_REPETITION_CONTEXT_SIZE"] = str(config.default_repetition_context_size)


async def start(config: MLXServerConfig) -> None:
    """Run the ASGI server using the provided configuration.

    This coroutine wires the configuration into the server setup
    routine, logs progress, and starts the Uvicorn server. It handles
    KeyboardInterrupt and logs any startup failures before exiting the
    process with a non-zero code.

    Parameters
    ----------
    config : MLXServerConfig
        Single-model server configuration.
    """
    try:
        _apply_sampling_env(config)
        # Display startup information
        print_startup_banner(config)

        # Set up and start the server
        uvconfig = setup_server(config)
        logger.info("Server configuration complete.")
        logger.info("Starting Uvicorn server...")
        server = uvicorn.Server(uvconfig)
        await server.serve()
    except KeyboardInterrupt:
        logger.info("Server shutdown requested by user. Exiting...")
    except Exception as e:
        logger.error(f"Server startup failed: {e!s}")
        sys.exit(1)


async def start_multi(config: MultiModelServerConfig) -> None:
    """Run the ASGI server in multi-handler mode.

    Similar to ``start`` but works with a ``MultiModelServerConfig``
    which defines multiple models to be loaded concurrently.

    Parameters
    ----------
    config : MultiModelServerConfig
        Multi-model YAML-based configuration.
    """
    try:
        print_multi_startup_banner(config)

        uvconfig = setup_server(config)
        logger.info("Multi-handler server configuration complete.")
        logger.info("Starting Uvicorn server...")
        server = uvicorn.Server(uvconfig)
        await server.serve()
    except KeyboardInterrupt:
        logger.info("Server shutdown requested by user. Exiting...")
    except Exception as e:
        logger.error(f"Multi-handler server startup failed: {e!s}")
        sys.exit(1)


def main():
    """Normalize process args and dispatch to the Click CLI.

    This helper gathers command-line arguments, inserts the "launch"
    subcommand when a subcommand is omitted for backwards compatibility,
    and delegates execution to :func:`app.cli.cli` through
    ``cli.main``.
    """
    from .cli import cli

    args = [str(x) for x in sys.argv[1:]]
    # Keep backwards compatibility: Add 'launch' subcommand if none is provided
    if not args or args[0].startswith("-"):
        args.insert(0, "launch")
    cli.main(args=args)


if __name__ == "__main__":
    main()
