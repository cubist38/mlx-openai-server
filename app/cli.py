"""Command-line interface and helpers for the MLX server.

This module defines the Click command group used by the package and the
``launch`` command which constructs a server configuration and starts
the ASGI server.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from pathlib import Path
import sys
import time
from typing import Any, Literal
from urllib.parse import quote

import click
import httpx
from loguru import logger

from .config import MLXServerConfig
from .handler.parser.factory import PARSER_REGISTRY
from .hub.config import DEFAULT_HUB_CONFIG_PATH, HubConfigError, MLXHubConfig, load_hub_config
from .hub.server import (
    is_hub_controller_running,
    start_hub_controller_process,
    stop_hub_controller_process,
)
from .hub.service import (
    HubServiceClient,
    HubServiceError,
    get_service_paths,
    start_hub_service_process,
)
from .main import start
from .version import __version__


class UpperChoice(click.Choice[str]):
    """Case-insensitive choice type that returns canonical, uppercase values.

    This convenience subclass normalizes user input in a case-insensitive way
    but returns the canonical, uppercase option value from ``self.choices``. It is useful
    for flags like ``--log-level`` where callers expect the stored value to
    exactly match one of the declared choices.
    """

    def normalize_choice(self, choice: str | None, ctx: click.Context | None) -> str | None:  # type: ignore[override]
        """Return the canonical, uppercase choice or raise BadParameter.

        Parameters
        ----------
        choice:
            Raw value supplied by the user (may be ``None``).
        ctx:
            Click context object (unused here but part of the API).

        Returns
        -------
        str | None
            Canonical matching choice, or ``None`` if ``choice`` is ``None``.
        """
        if choice is None:
            return None
        upperchoice = choice.upper()
        for opt in self.choices:
            if opt.upper() == upperchoice:
                return opt  # return the canonical opt
        self.fail(
            f"Invalid choice: {choice}. (choose from {', '.join(self.choices)})",
            param=None,
            ctx=ctx,
        )
        return None


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


def _load_hub_config_or_fail(config_path: str | None) -> MLXHubConfig:
    """Load hub configuration or exit with a CLI error.

    Parameters
    ----------
    config_path : str or None
        Path to the hub config file.

    Returns
    -------
    MLXHubConfig
        Loaded hub configuration.

    Raises
    ------
    click.ClickException
        If the configuration cannot be loaded.
    """
    try:
        return load_hub_config(config_path)
    except HubConfigError as exc:  # pragma: no cover - CLI friendly errors
        raise click.ClickException(str(exc)) from exc


def _build_service_client(config: MLXHubConfig) -> HubServiceClient:
    """Build a hub service client for the given configuration.

    Parameters
    ----------
    config : MLXHubConfig
        The hub configuration.

    Returns
    -------
    HubServiceClient
        Configured service client.
    """
    paths = get_service_paths(config)
    return HubServiceClient(paths.socket_path)


def _print_hub_status(
    config: MLXHubConfig,
    *,
    model_names: Iterable[str] | None = None,
    live_status: dict[str, Any] | None = None,
) -> None:
    """Print hub status information to the console.

    Parameters
    ----------
    config : MLXHubConfig
        The hub configuration.
    model_names : Iterable[str] | None, optional
        Specific model names to display. If None, all models are shown.
    live_status : dict[str, Any] | None, optional
        Live status data from the hub service. If provided, includes runtime state.
    """
    click.echo(f"Hub log path: {config.log_path}")
    click.echo(f"Status page enabled: {'yes' if config.enable_status_page else 'no'}")

    selection = None
    if model_names:
        selection = {name.strip() for name in model_names if name.strip()}
    configured = []
    for model in config.models:
        if selection and model.name not in selection:
            continue
        configured.append(model)

    if not configured:
        click.echo("No matching models in hub config")
        return

    live_lookup: dict[str, dict[str, Any]] = {}
    if live_status:
        for entry in live_status.get("models", []):
            name = entry.get("name")
            if isinstance(name, str):
                live_lookup[name] = entry

    name_width = max(len(model.name or "<unnamed>") for model in configured)
    click.echo("Models:")
    for model in configured:
        name = model.name or "<unnamed>"
        live = live_lookup.get(name)
        state = (live or {}).get("state", "inactive")
        pid = (live or {}).get("pid") or "-"
        log_target = model.log_file or "<hub managed>"
        group = model.group or "<none>"
        default_flag = "⭐ auto-start" if model.is_default_model else "manual"
        click.echo(
            f"  - {name:<{name_width}} | state={state} | pid={pid} | "
            f"group={group} | log={log_target} | {default_flag}"
        )


_FLASH_STYLES: dict[str, tuple[str, str]] = {
    "info": ("[info]", "cyan"),
    "success": ("[ok]", "green"),
    "warning": ("[warn]", "yellow"),
    "error": ("[err]", "red"),
}


def _flash(message: str, tone: Literal["info", "success", "warning", "error"] = "info") -> None:
    """Emit a short, colorized status line for CLI actions.

    Parameters
    ----------
    message : str
        The message to display.
    tone : Literal["info", "success", "warning", "error"], optional
        The tone of the message, defaults to "info".
    """

    prefix, color = _FLASH_STYLES.get(tone, _FLASH_STYLES["info"])
    click.echo(click.style(f"{prefix} {message}", fg=color))


def _format_name_list(values: Iterable[str] | None) -> str:
    """Format a list of names into a comma-separated string.

    Parameters
    ----------
    values : Iterable[str] | None
        The list of names to format.

    Returns
    -------
    str
        Comma-separated string of names, or "none" if empty.
    """
    if not values:
        return "none"
    filtered = [value for value in values if value]
    return ", ".join(filtered) if filtered else "none"


def _emit_reload_summary(diff: dict[str, Any], *, header: str) -> None:
    """Emit a summary of reload changes to the console.

    Parameters
    ----------
    diff : dict[str, Any]
        The reload diff containing started, stopped, and unchanged models.
    header : str
        The header message for the summary.
    """
    started = _format_name_list(diff.get("started"))
    stopped = _format_name_list(diff.get("stopped"))
    unchanged = _format_name_list(diff.get("unchanged"))
    tone: Literal["info", "success"] = (
        "success" if started != "none" or stopped != "none" else "info"
    )
    _flash(f"{header}: started={started} | stopped={stopped} | unchanged={unchanged}", tone=tone)


def _reload_or_fail(client: HubServiceClient, *, header: str) -> dict[str, Any]:
    """Reload the hub service and emit a summary, or fail with an exception.

    Parameters
    ----------
    client : HubServiceClient
        The hub service client.
    header : str
        The header message for the reload summary.

    Returns
    -------
    dict[str, Any]
        The reload diff.

    Raises
    ------
    click.ClickException
        If the reload operation fails.
    """
    try:
        diff = client.reload()
    except HubServiceError as exc:
        raise click.ClickException(f"Hub reload failed: {exc}") from exc
    _emit_reload_summary(diff, header=header)
    return diff


def _require_service_client(config: MLXHubConfig) -> HubServiceClient:
    """Build and validate a hub service client.

    Parameters
    ----------
    config : MLXHubConfig
        The hub configuration.

    Returns
    -------
    HubServiceClient
        The validated service client.

    Raises
    ------
    click.ClickException
        If the hub manager is not running.
    """
    client = _build_service_client(config)
    if not client.is_available():
        raise click.ClickException(
            "Hub manager is not running. Start it via 'mlx-openai-server hub start'."
        )
    return client


def _resolve_controller_base_url(config: MLXHubConfig) -> str:
    """Resolve the base URL for the hub controller.

    Parameters
    ----------
    config : MLXHubConfig
        The hub configuration.

    Returns
    -------
    str
        The base URL for the controller.
    """
    host = (config.host or "127.0.0.1").strip()
    if host in {"0.0.0.0", "::", "[::]"}:
        host = "127.0.0.1"
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    port = config.port or 8000
    return f"http://{host}:{port}"


def _perform_memory_action_request(
    config: MLXHubConfig,
    model_name: str,
    action: Literal["load-model", "unload-model"],
    *,
    reason: str,
) -> tuple[bool, str]:
    """Perform a memory action request to the hub controller.

    Parameters
    ----------
    config : MLXHubConfig
        The hub configuration.
    model_name : str
        The name of the model.
    action : Literal["load-model", "unload-model"]
        The action to perform.
    reason : str
        The reason for the action.

    Returns
    -------
    tuple[bool, str]
        Success flag and message.
    """
    base_url = _resolve_controller_base_url(config)
    url = f"{base_url}/hub/models/{quote(model_name, safe='')}/{action}"
    try:
        response = httpx.post(url, json={"reason": reason}, timeout=10.0)
    except httpx.HTTPError as exc:  # pragma: no cover - network errors
        return False, f"controller unreachable ({exc})"
    try:
        payload = response.json()
    except ValueError:
        payload = {}
    if response.is_error:
        message = (
            payload.get("error", {}).get("message")
            or payload.get("message")
            or f"Failed to {action} {model_name}"
        )
        return False, message
    message = payload.get("message") or f"Memory {action} requested"
    return True, message


def _run_memory_actions(
    config: MLXHubConfig,
    model_names: Iterable[str],
    action: Literal["load-model", "unload-model"],
    *,
    reason: str,
) -> None:
    """Run memory actions for multiple models.

    Parameters
    ----------
    config : MLXHubConfig
        The hub configuration.
    model_names : Iterable[str]
        The names of the models.
    action : Literal["load-model", "unload-model"]
        The action to perform.
    reason : str
        The reason for the action.

    Raises
    ------
    click.ClickException
        If any memory action fails.
    """
    had_error = False
    for raw_name in model_names:
        target = raw_name.strip()
        if not target:
            had_error = True
            _flash("Skipping blank model name entry", tone="warning")
            continue
        ok, message = _perform_memory_action_request(config, target, action, reason=reason)
        if ok:
            verb = "load" if action == "load" else "unload"
            _flash(f"{target}: memory {verb} requested ({message})", tone="success")
        else:
            had_error = True
            _flash(f"{target}: {message}", tone="error")
    if had_error:
        raise click.ClickException("One or more memory actions failed")


def _wait_for_controller_available(config: MLXHubConfig, timeout: float = 20.0) -> bool:
    """Wait for the hub controller to become available.

    Parameters
    ----------
    config : MLXHubConfig
        The hub configuration.
    timeout : float, optional
        Maximum time to wait in seconds, defaults to 20.0.

    Returns
    -------
    bool
        True if the controller became available, False otherwise.
    """
    base_url = _resolve_controller_base_url(config)
    deadline = time.time() + timeout
    last_error: str | None = None
    while time.time() < deadline:
        try:
            response = httpx.get(f"{base_url}/health", timeout=5.0)
            if response.status_code < 500:
                return True
        except httpx.HTTPError as exc:  # pragma: no cover - network timing
            last_error = str(exc)
        time.sleep(0.5)
    if last_error:
        logger.debug(f"Controller readiness check failed: {last_error}")
    return False


def _start_controller_if_needed(config: MLXHubConfig) -> None:
    """Start the hub controller if it's not already running.

    Parameters
    ----------
    config : MLXHubConfig
        The hub configuration.

    Raises
    ------
    click.ClickException
        If the configuration is not saved or the controller fails to start.
    """
    if is_hub_controller_running(config):
        _flash("Hub controller already running", tone="warning")
        return
    if config.source_path is None:
        raise click.ClickException(
            "Hub configuration must be saved to disk before starting the controller."
        )
    pid = start_hub_controller_process(config.source_path)
    _flash(f"Launching hub controller (pid={pid})", tone="info")
    if not _wait_for_controller_available(config):
        raise click.ClickException(
            "Hub controller failed to start within 20 seconds. Inspect hub logs for details."
        )
    _flash("Hub controller is now running", tone="success")


def _stop_controller_if_running(config: MLXHubConfig) -> None:
    """Stop the hub controller if it's running.

    Parameters
    ----------
    config : MLXHubConfig
        The hub configuration.
    """
    if stop_hub_controller_process(config):
        _flash("Hub controller shutdown requested", tone="success")
    else:
        _flash("Hub controller is not running", tone="warning")


def _format_duration(seconds: float | None) -> str:
    """Return a compact human-readable duration string.

    Parameters
    ----------
    seconds : float | None
        The duration in seconds.

    Returns
    -------
    str
        Formatted duration string.
    """

    if seconds is None or seconds < 0:
        return "-"
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _render_watch_table(models: Iterable[dict[str, Any]], *, now: float | None = None) -> str:
    """Return a formatted table describing hub-managed processes.

    Parameters
    ----------
    models : Iterable[dict[str, Any]]
        The model process data.
    now : float | None, optional
        Reference time for uptime calculation, defaults to current time.

    Returns
    -------
    str
        Formatted table string.
    """

    snapshot = list(models)
    if not snapshot:
        return "  (no managed processes)"

    reference = now if isinstance(now, (int, float)) else time.time()
    headers = ["NAME", "STATE", "PID", "GROUP", "UPTIME", "EXIT", "LOG"]
    rows: list[dict[str, str]] = []
    for entry in sorted(snapshot, key=lambda item: str(item.get("name", "?"))):
        name = str(entry.get("name", "?"))
        state = str(entry.get("state", "unknown")).upper()
        pid = str(entry.get("pid") or "-")
        group = entry.get("group") or "-"
        started_at = entry.get("started_at")
        uptime = "-"
        if isinstance(started_at, (int, float)):
            uptime = _format_duration(reference - float(started_at))
        exit_code = entry.get("exit_code")
        exit_display = "-" if exit_code in (None, 0) else str(exit_code)
        log_path = entry.get("log_path")
        log_display = Path(log_path).name if isinstance(log_path, str) else "-"
        rows.append(
            {
                "NAME": name,
                "STATE": state,
                "PID": pid,
                "GROUP": group,
                "UPTIME": uptime,
                "EXIT": exit_display,
                "LOG": log_display,
            }
        )

    widths: dict[str, int] = {header: len(header) for header in headers}
    for row in rows:
        for header in headers:
            widths[header] = max(widths[header], len(row[header]))

    divider = "  " + "-+-".join("-" * widths[header] for header in headers)
    header_line = "  " + " | ".join(header.ljust(widths[header]) for header in headers)
    lines = [header_line, divider]
    lines.extend(
        "  " + " | ".join(row[header].ljust(widths[header]) for header in headers) for row in rows
    )
    return "\n".join(lines)


def _print_watch_snapshot(snapshot: dict[str, Any]) -> None:
    """Print a formatted snapshot of hub-managed processes.

    Parameters
    ----------
    snapshot : dict[str, Any]
        The snapshot data containing models and timestamp.
    """
    timestamp = snapshot.get("timestamp")
    reference = timestamp if isinstance(timestamp, (int, float)) else time.time()
    formatted = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(reference))

    raw_models = snapshot.get("models")
    models: list[dict[str, Any]] = raw_models if isinstance(raw_models, list) else []
    running = sum(1 for entry in models if str(entry.get("state")).lower() == "running")
    stopped = sum(1 for entry in models if str(entry.get("state")).lower() == "stopped")
    failed = sum(
        1
        for entry in models
        if str(entry.get("state")).lower() == "failed" or entry.get("exit_code") not in (None, 0)
    )

    click.echo(
        f"[{formatted}] models={len(models)} running={running} stopped={stopped} failed={failed}"
    )
    click.echo(_render_watch_table(models, now=reference))


@cli.command(help="Start the MLX OpenAI Server with the supplied flags")
@click.option(
    "--model-path",
    required=True,
    type=str,
    help="Path to the model. Accepts local paths or Hugging Face repository IDs (e.g., 'blackforestlabs/FLUX.1-dev').",
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
    type=click.Choice(["flux-schnell", "flux-dev", "flux-krea-dev", "flux-kontext-dev"]),
    help="Config name of the model. Only used for image-generation and image-edit Flux models.",
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
    type=click.Choice(list(PARSER_REGISTRY.keys())),
    help="Specify tool call parser to use instead of auto-detection. Only works with language models.",
)
@click.option(
    "--reasoning-parser",
    default=None,
    type=click.Choice(list(PARSER_REGISTRY.keys())),
    help="Specify reasoning parser to use instead of auto-detection. Only works with language models.",
)
@click.option(
    "--trust-remote-code",
    is_flag=True,
    help="Enable trust_remote_code when loading models. This allows loading custom code from model repositories.",
)
@click.option(
    "--jit",
    "jit_enabled",
    is_flag=True,
    help="Enable just-in-time model loading. Models load on first request instead of startup.",
)
@click.option(
    "--auto-unload-minutes",
    type=click.IntRange(1),
    default=None,
    help="When JIT is enabled, unload the model after idle for this many minutes.",
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
    trust_remote_code: bool,
    jit_enabled: bool,
    auto_unload_minutes: int | None,
) -> None:
    """Start the FastAPI/Uvicorn server with the supplied flags.

    The command builds a server configuration object using
    ``MLXServerConfig`` and then calls the async ``start`` routine
    which handles the event loop and server lifecycle.

    Parameters
    ----------
    model_path : str or None
        Path to the model loaded by the single-model server.
    model_type : str
        Type of model to run (lm: text-only, multimodal: text+vision+audio, image-generation: flux image generation, image-edit: flux image edit, embeddings: text embeddings, whisper: audio transcription).
    context_length : int
        Context length for language models. Only works with `lm` or `multimodal` model types.
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
        Quantization level for the model. Only used for image-generation and image-edit Flux models.
    config_name : str or None
        Config name of the model. Only used for image-generation and image-edit Flux models.
    lora_paths : str or None
        Path to the LoRA file(s). Multiple paths should be separated by commas.
    lora_scales : str or None
        Scale factor for the LoRA file(s). Multiple scales should be separated by commas.
    disable_auto_resize : bool
        Disable automatic model resizing. Only work for Vision Language Models.
    log_file : str or None
        Path to log file. If not specified, logs will be written to 'logs/app.log' by default.
    no_log_file : bool
        Disable file logging entirely. Only console output will be shown.
    log_level : str
        Set the logging level. Default is INFO.
    enable_auto_tool_choice : bool
        Enable automatic tool choice. Only works with language models.
    tool_call_parser : str or None
        Specify tool call parser to use instead of auto-detection. Only works with language models.
    reasoning_parser : str or None
        Specify reasoning parser to use instead of auto-detection. Only works with language models.
    trust_remote_code : bool
        Enable trust_remote_code when loading models. This allows loading custom code from model repositories.
    jit_enabled : bool
        Enable just-in-time model loading. Models load on first request instead of startup.
    auto_unload_minutes : int or None
        When JIT is enabled, unload the model after idle for this many minutes.

    Raises
    ------
    click.BadOptionUsage
        If auto_unload_minutes is set without jit_enabled.
    """
    if auto_unload_minutes is not None and not jit_enabled:
        raise click.BadOptionUsage(
            "--auto-unload-minutes", "--auto-unload-minutes requires --jit to be set."
        )

    assert model_path is not None

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
        trust_remote_code=trust_remote_code,
        jit_enabled=jit_enabled,
        auto_unload_minutes=auto_unload_minutes,
    )

    asyncio.run(start(args))


@cli.group(help="Manage hub-based multi-model deployments", invoke_without_command=True)
@click.option(
    "--config",
    "hub_config_path",
    default=None,
    help=f"Path to hub YAML (default: {DEFAULT_HUB_CONFIG_PATH})",
)
@click.pass_context
def hub(
    ctx: click.Context,
    hub_config_path: str | None,
) -> None:
    """Entry point for hub sub-commands."""

    ctx.ensure_object(dict)
    ctx.obj["hub_config_path"] = hub_config_path
    if ctx.invoked_subcommand is None:
        ctx.invoke(hub_start)


@hub.command(name="start", help="Start the hub manager ")
@click.argument("model_names", nargs=-1)
@click.pass_context
def hub_start(ctx: click.Context, model_names: tuple[str, ...]) -> None:
    """Launch the hub manager and print status."""

    config = _load_hub_config_or_fail(ctx.obj.get("hub_config_path"))
    client = _build_service_client(config)
    models = model_names or None

    if not client.is_available():
        if config.source_path is None:
            raise click.ClickException(
                "Hub configuration must be loaded from disk. Provide --config to specify a file."
            )
        pid = start_hub_service_process(config.source_path)
        _flash(f"Launching hub manager (pid={pid})", tone="info")
        if not client.wait_until_available(timeout=20.0):
            raise click.ClickException("Hub manager failed to start within 20 seconds")
        _flash("Hub manager is now running", tone="success")
    else:
        _flash("Hub manager already running", tone="warning")

    _start_controller_if_needed(config)

    click.echo(f"Status page enabled: {'yes' if config.enable_status_page else 'no'}")
    if config.enable_status_page:
        click.echo(f"Browse to http://{config.host}:{config.port}/hub for the status dashboard")

    snapshot = None
    try:
        snapshot = client.status()
    except HubServiceError as exc:
        _flash(f"Unable to fetch live status: {exc}", tone="warning")
    _print_hub_status(config, model_names=models, live_status=snapshot)


@hub.command(name="status", help="Show hub configuration and running processes")
@click.argument("model_names", nargs=-1)
@click.pass_context
def hub_status(ctx: click.Context, model_names: tuple[str, ...]) -> None:
    """Display configured models and any active processes.

    Parameters
    ----------
    ctx : click.Context
        Click context.
    model_names : tuple[str, ...]
        Names of models to show status for.
    """

    config = _load_hub_config_or_fail(ctx.obj.get("hub_config_path"))
    client = _build_service_client(config)
    snapshot = None
    if client.is_available():
        _reload_or_fail(client, header="Config synced")
        try:
            snapshot = client.status()
        except HubServiceError as exc:
            _flash(f"Unable to fetch live status: {exc}", tone="warning")
    else:
        _flash("Hub manager is not running", tone="warning")
    _print_hub_status(config, model_names=model_names or None, live_status=snapshot)


@hub.command(name="reload", help="Reload hub.yaml and reconcile model processes")
@click.pass_context
def hub_reload(ctx: click.Context) -> None:
    """Force the running hub manager to reload its configuration.

    Parameters
    ----------
    ctx : click.Context
        Click context.
    """

    config = _load_hub_config_or_fail(ctx.obj.get("hub_config_path"))
    client = _require_service_client(config)
    _reload_or_fail(client, header="Hub reload complete")


@hub.command(name="stop", help="Stop the hub manager and all models")
@click.pass_context
def hub_stop(ctx: click.Context) -> None:
    """Shut down the hub manager and terminate all managed models.

    Parameters
    ----------
    ctx : click.Context
        Click context.
    """

    config = _load_hub_config_or_fail(ctx.obj.get("hub_config_path"))
    _stop_controller_if_running(config)

    client = _build_service_client(config)
    if not client.is_available():
        _flash("Hub manager is not running", tone="warning")
        return

    _reload_or_fail(client, header="Config synced before shutdown")
    try:
        client.shutdown()
    except HubServiceError as exc:
        raise click.ClickException(f"Hub shutdown failed: {exc}") from exc
    _flash("Hub manager shutdown requested", tone="success")


@hub.command(name="start-model", help="Start one or more model processes")
@click.argument("model_names", nargs=-1, required=True)
@click.pass_context
def hub_start_model(ctx: click.Context, model_names: tuple[str, ...]) -> None:
    """Trigger process launches for the provided model names.

    Parameters
    ----------
    ctx : click.Context
        Click context.
    model_names : tuple[str, ...]
        Names of models to load.
    """

    config = _load_hub_config_or_fail(ctx.obj.get("hub_config_path"))
    client = _require_service_client(config)
    _reload_or_fail(client, header="Config synced before load")
    for raw_name in model_names:
        target = raw_name.strip()
        if not target:
            _flash("Skipping blank model name entry", tone="warning")
            continue
        try:
            client.start_model(target)
        except HubServiceError as exc:
            _flash(f"{target}: start failed ({exc})", tone="error")
        else:
            _flash(f"{target}: start requested", tone="success")


@hub.command(name="stop-model", help="Stop one or more model processes")
@click.argument("model_names", nargs=-1, required=True)
@click.pass_context
def hub_stop_model(ctx: click.Context, model_names: tuple[str, ...]) -> None:
    """Stop running processes for the provided model names.

    Parameters
    ----------
    ctx : click.Context
        Click context.
    model_names : tuple[str, ...]
        Names of models to unload.
    """

    config = _load_hub_config_or_fail(ctx.obj.get("hub_config_path"))
    client = _require_service_client(config)
    _reload_or_fail(client, header="Config synced before unload")
    for raw_name in model_names:
        target = raw_name.strip()
        if not target:
            _flash("Skipping blank model name entry", tone="warning")
            continue
        try:
            client.stop_model(target)
        except HubServiceError as exc:
            _flash(f"{target}: stop failed ({exc})", tone="error")
        else:
            _flash(f"{target}: stop requested", tone="success")


@hub.command(name="load-model", help="Load handlers for one or more models into memory")
@click.argument("model_names", nargs=-1, required=True)
@click.option(
    "--reason",
    default="cli",
    show_default=True,
    help="Reason string recorded alongside the controller action.",
)
@click.pass_context
def hub_memory_load(ctx: click.Context, model_names: tuple[str, ...], reason: str) -> None:
    """Trigger controller-backed memory loads for the provided models."""

    config = _load_hub_config_or_fail(ctx.obj.get("hub_config_path"))
    _run_memory_actions(config, model_names, "load-model", reason=reason)


@hub.command(name="unload-model", help="Unload handlers for one or more models from memory")
@click.argument("model_names", nargs=-1, required=True)
@click.option(
    "--reason",
    default="cli",
    show_default=True,
    help="Reason string recorded alongside the controller action.",
)
@click.pass_context
def hub_memory_unload(ctx: click.Context, model_names: tuple[str, ...], reason: str) -> None:
    """Trigger controller-backed memory unloads for the provided models."""

    config = _load_hub_config_or_fail(ctx.obj.get("hub_config_path"))
    _run_memory_actions(config, model_names, "unload-model", reason=reason)


@hub.command(name="watch", help="Continuously print live hub manager status")
@click.option(
    "--interval",
    default=5.0,
    show_default=True,
    type=float,
    help="Seconds between refreshes.",
)
@click.pass_context
def hub_watch(ctx: click.Context, interval: float) -> None:
    """Poll the hub manager service until interrupted.

    Parameters
    ----------
    ctx : click.Context
        Click context.
    interval : float
        Seconds between refreshes.
    """

    config = _load_hub_config_or_fail(ctx.obj.get("hub_config_path"))
    client = _build_service_client(config)
    click.echo("Watching hub manager (press Ctrl+C to stop)...")
    sleep_interval = max(interval, 0.5)
    try:
        while True:
            if not client.is_available():
                click.echo(
                    click.style(
                        "Hub manager is not running. Start it via 'mlx-openai-server hub start'.",
                        fg="yellow",
                    )
                )
            else:
                try:
                    snapshot = client.status()
                except HubServiceError as exc:
                    click.echo(click.style(f"[watch] status request failed: {exc}", fg="red"))
                    click.echo(
                        click.style(
                            "        Inspect hub logs (hub status) or restart the service if needed.",
                            fg="yellow",
                        )
                    )
                else:
                    _print_watch_snapshot(snapshot)
            time.sleep(sleep_interval)
    except KeyboardInterrupt:  # pragma: no cover - interactive command
        click.echo("Stopped watching hub manager")
