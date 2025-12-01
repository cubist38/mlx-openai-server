"""Command-line interface and helpers for the MLX server.

This module defines the Click command group used by the package and the
``launch`` command which constructs a server configuration and starts
the ASGI server.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import IO, Any, Literal
from urllib.parse import quote

import click
import httpx
from loguru import logger

from .config import MLXServerConfig
from .const import (
    DEFAULT_BIND_HOST,
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_HUB_CONFIG_PATH,
    DEFAULT_LOG_LEVEL,
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_MODEL_TYPE,
    DEFAULT_PORT,
    DEFAULT_QUANTIZE,
    DEFAULT_QUEUE_SIZE,
    DEFAULT_QUEUE_TIMEOUT,
)
from .handler.parser.factory import PARSER_REGISTRY
from .hub.config import HubConfigError, MLXHubConfig, load_hub_config

# Hub IPC service removed: CLI uses HTTP API to contact the hub daemon
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


def _controller_base_url(config: MLXHubConfig) -> str:
    """Return the base HTTP URL for the hub daemon from the config.

    The daemon host/port values are read from the hub config. This helper
    centralizes where the CLI constructs the daemon base URL.
    """
    # Prefer runtime state file (written by `hub start`) when available
    runtime = _read_hub_runtime_state(config)
    if runtime:
        host = runtime.get("host") or (config.host or DEFAULT_BIND_HOST)
        # runtime may contain untyped values (loaded from JSON). Validate
        # the port before converting to int to keep mypy and runtime checks happy.
        rt_port = runtime.get("port")
        if isinstance(rt_port, (int, str)):
            try:
                port = int(rt_port)
            except Exception:
                port = int(config.port or DEFAULT_PORT)
        else:
            port = int(config.port or DEFAULT_PORT)
        return f"http://{host}:{port}"

    host = config.host or DEFAULT_BIND_HOST
    port = config.port
    return f"http://{host}:{port}"


def _runtime_state_path(config: MLXHubConfig) -> Path:
    """Return path for the transient runtime state file under the configured log path."""
    try:
        log_dir = (
            Path(config.log_path) if getattr(config, "log_path", None) else Path.cwd() / "logs"
        )
    except Exception:
        log_dir = Path.cwd() / "logs"
    return log_dir / "hub_runtime.json"


def _write_hub_runtime_state(config: MLXHubConfig, pid: int) -> None:
    """Persist transient runtime info so other CLI commands can find the running daemon.

    The file is intentionally lightweight and not used for durable configuration.
    """
    path = _runtime_state_path(config)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pid": int(pid),
            "host": config.host or DEFAULT_BIND_HOST,
            "port": int(config.port or DEFAULT_PORT),
            "started_at": datetime.datetime.now(datetime.UTC).isoformat(),
        }
        path.write_text(json.dumps(payload))
        logger.debug(f"Wrote hub runtime state to {path}")
    except Exception as e:  # pragma: no cover - best-effort logging
        logger.warning(f"Failed to write hub runtime state to {path}. {type(e).__name__}: {e}")


def _read_hub_runtime_state(config: MLXHubConfig) -> dict[str, object] | None:
    """Return runtime state dict if valid and the process appears alive, otherwise None."""
    path = _runtime_state_path(config)
    try:
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        pid = int(data.get("pid"))
        host = data.get("host") or DEFAULT_BIND_HOST
        raw_port = data.get("port")
        if raw_port is None:
            port = int(config.port or DEFAULT_PORT)
        else:
            try:
                port = int(raw_port)
            except Exception:
                port = int(config.port or DEFAULT_PORT)
    except Exception:
        return None

    # Check PID alive (best-effort)
    pid_alive = False
    try:
        # os.kill with signal 0 raises OSError if process does not exist
        os.kill(pid, 0)
        pid_alive = True
    except Exception:
        pid_alive = False

    if not pid_alive:
        return None

    return {"pid": pid, "host": host, "port": port}


def _call_daemon_api(
    config: MLXHubConfig,
    method: str,
    path: str,
    *,
    json: object | None = None,
    timeout: float = 5.0,
) -> dict[str, object] | None:
    """Call the hub daemon HTTP API synchronously and return parsed JSON.

    Parameters
    ----------
    config : MLXHubConfig
        Hub configuration used to determine daemon base URL.
    method : str
        HTTP method (GET/POST/etc).
    path : str
        Path part of the URL (should start with '/').
    json : object | None
        JSON body to send for POST/PUT requests.
    timeout : float
        Request timeout in seconds.

    Returns
    -------
    dict[str, object] | None
        Parsed JSON response (if any).

    Raises
    ------
    click.ClickException
        On connectivity or non-2xx responses.
    """
    base = _controller_base_url(config)
    url = f"{base.rstrip('/')}{path}"
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.request(method, url, json=json)
    except httpx.HTTPError as e:  # pragma: no cover - network error handling
        raise click.ClickException(f"Failed to contact hub daemon at {base}: {e}") from e

    if resp.status_code >= 400:
        # Try to include JSON error message if present
        try:
            payload: object = resp.json()
        except ValueError:
            payload = resp.text

        # Extract user-friendly error message from structured error responses
        error_message = payload
        if isinstance(payload, dict) and "error" in payload:
            error_info = payload["error"]
            if isinstance(error_info, dict) and "message" in error_info:
                # Use the clean error message from the API response
                error_message = error_info["message"]

        raise click.ClickException(f"Daemon responded {resp.status_code}: {error_message}")

    if not resp.content:
        return None

    try:
        payload = resp.json()
    except ValueError:
        return {"raw": resp.text}

    # Ensure we return a dict[str, object] as declared; if the JSON is not a
    # mapping, fall back to returning the raw text.
    if isinstance(payload, dict):
        return payload
    return {"raw": resp.text}


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
    group_live_lookup: dict[str, dict[str, Any]] = {}
    if live_status:
        for entry in live_status.get("models", []):
            name = entry.get("id")  # Model objects have "id"
            if isinstance(name, str):
                live_lookup[name] = entry
        for group_entry in live_status.get("groups", []):
            group_name = group_entry.get("name")
            if isinstance(group_name, str):
                group_live_lookup[group_name] = group_entry

    click.echo("Models:")
    headers = ["NAME", "STATE", "LOADED", "AUTO-UNLOAD", "TYPE", "GROUP", "DEFAULT", "MODEL"]
    rows: list[dict[str, str]] = []
    for model in configured:
        name = model.name or "<unnamed>"
        live = live_lookup.get(name)
        metadata = (live or {}).get("metadata", {})
        state = metadata.get("process_state", "inactive")
        pid = metadata.get("pid")
        port = metadata.get("port") or model.port

        # Format state with pid and port if running
        if state == "running" and pid is not None:
            state_display = f"{state} (pid={pid}"
            if port is not None:
                state_display += f", port={port}"
            state_display += ")"
        else:
            state_display = state

        # Loaded in memory: prefer explicit runtime flag when available,
        # based on the memory_state field from runtime metadata.
        memory_flag = metadata.get("memory_state") == "loaded"
        if memory_flag and model.auto_unload_minutes:
            unload_time = datetime.datetime.now() + datetime.timedelta(
                minutes=model.auto_unload_minutes
            )
            loaded_in_memory = f"yes (unload {unload_time.strftime('%Y-%m-%d %H:%M:%S')})"
        else:
            loaded_in_memory = "yes" if memory_flag else "no"

        # Auto-unload
        auto_unload = f"{model.auto_unload_minutes}min" if model.auto_unload_minutes else "-"

        # Model type
        model_type = model.model_type

        # Group
        group = model.group or "-"

        # Default
        default = "✓" if model.is_default_model else "-"

        # Model path
        model_path = model.model_path

        rows.append(
            {
                "NAME": name,
                "STATE": state_display,
                "LOADED": loaded_in_memory,
                "AUTO-UNLOAD": auto_unload,
                "TYPE": model_type,
                "GROUP": group,
                "DEFAULT": default,
                "MODEL": model_path,
            },
        )

    # Calculate column widths
    widths: dict[str, int] = {header: len(header) for header in headers}
    for row in rows:
        for header in headers:
            widths[header] = max(widths[header], len(row[header]))

    # Print header
    header_line = "  " + " | ".join(header.ljust(widths[header]) for header in headers)
    click.echo(header_line)

    # Print divider
    divider = "  " + "-+-".join("-" * widths[header] for header in headers)
    click.echo(divider)

    # Print rows
    for row in rows:
        row_line = "  " + " | ".join(row[header].ljust(widths[header]) for header in headers)
        click.echo(row_line)

    configured_groups = list(getattr(config, "groups", []) or [])
    if not configured_groups:
        return

    click.echo("")
    click.echo("Groups:")
    group_headers = ["NAME", "MAX", "IDLE-UNLOAD", "LOADED", "MODELS"]
    group_widths: dict[str, int] = {header: len(header) for header in group_headers}

    membership: dict[str, list[str]] = {}
    for model in config.models:
        if not model.group:
            continue
        membership.setdefault(model.group, []).append(model.name or "<unnamed>")

    group_rows: list[dict[str, str]] = []
    for group_cfg in configured_groups:
        name = getattr(group_cfg, "name", "<unnamed>")
        live_entry = {}
        if isinstance(group_live_lookup, dict):
            live_entry = group_live_lookup.get(name, {})
        max_loaded = getattr(group_cfg, "max_loaded", None)
        idle_trigger = getattr(group_cfg, "idle_unload_trigger_min", None)
        loaded_count = int(live_entry.get("loaded", 0) or 0)
        members = membership.get(name, [])
        row = {
            "NAME": name,
            "MAX": str(max_loaded) if max_loaded is not None else "-",
            "IDLE-UNLOAD": f"{idle_trigger}min" if idle_trigger is not None else "-",
            "LOADED": str(loaded_count),
            "MODELS": ", ".join(members) if members else "-",
        }
        for header in group_headers:
            group_widths[header] = max(group_widths[header], len(row[header]))
        group_rows.append(row)

    group_header_line = "  " + " | ".join(
        header.ljust(group_widths[header]) for header in group_headers
    )
    click.echo(group_header_line)
    group_divider = "  " + "-+-".join("-" * group_widths[header] for header in group_headers)
    click.echo(group_divider)
    for row in group_rows:
        row_line = "  " + " | ".join(
            row[header].ljust(group_widths[header]) for header in group_headers
        )
        click.echo(row_line)


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


def _reload_or_fail(config: MLXHubConfig, *, header: str) -> dict[str, Any]:
    """Reload the hub daemon and emit a summary, or fail with an exception.

    Parameters
    ----------
    config : MLXHubConfig
        The hub configuration used to contact the daemon.
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
        diff = _call_daemon_api(config, "POST", "/hub/reload") or {}
    except click.ClickException as e:
        raise click.ClickException(f"Hub reload failed: {e}") from e
    _emit_reload_summary(diff, header=header)
    return diff


def _require_service_client(config: MLXHubConfig) -> bool:
    """Build and validate a hub service client.

    Parameters
    ----------
    config : MLXHubConfig
        The hub configuration.

    Returns
    -------
    bool
        True if the daemon is reachable (used to assert availability), otherwise raises.

    Raises
    ------
    click.ClickException
        If the hub manager is not running.
    """
    try:
        _call_daemon_api(config, "GET", "/health", timeout=1.0)
    except click.ClickException as exc:
        raise click.ClickException(
            "Hub manager is not running. Start it via 'mlx-openai-server hub start'.",
        ) from exc
    return True


def _perform_memory_action_request(
    config: MLXHubConfig,
    model_name: str,
    action: Literal["load", "unload"],
) -> tuple[bool, str]:
    """Perform a memory action request to the hub controller.

    Parameters
    ----------
    config : MLXHubConfig
        The hub configuration.
    model_name : str
        The name of the model.
    action : Literal["load", "unload"]
        The action to perform.

    Returns
    -------
    tuple[bool, str]
        Success flag and message.
    """
    try:
        payload = _call_daemon_api(
            config,
            "POST",
            f"/hub/models/{quote(model_name, safe='')}/{action}",
            timeout=10.0,
        )
    except click.ClickException as exc:
        return False, str(exc)
    raw_message = (payload or {}).get("message")
    if raw_message is None:
        message = f"{action} requested"
    else:
        message = str(raw_message)
    return True, message


def _run_memory_actions(
    config: MLXHubConfig,
    model_names: Iterable[str],
    action: Literal["load", "unload"],
) -> None:
    """Run memory actions for multiple models.

    Parameters
    ----------
    config : MLXHubConfig
        The hub configuration.
    model_names : Iterable[str]
        The names of the models.
    action : Literal["load", "unload"]
        The action to perform.

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
        ok, message = _perform_memory_action_request(config, target, action)
        verb = action
        if ok:
            _flash(f"{target}: {verb} requested ({message})", tone="success")
        else:
            had_error = True
            _flash(f"{target}: {message}", tone="error")
    if had_error:
        raise click.ClickException("One or more memory actions failed")


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


def _render_watch_table(models: Iterable[dict[str, Any]]) -> str:
    """Return a formatted table describing hub-managed models, matching hub status format.

    Parameters
    ----------
    models : Iterable[dict[str, Any]]
        The model data.

    Returns
    -------
    str
        Formatted table string.
    """
    snapshot = list(models)
    if not snapshot:
        return "  (no managed models)"

    headers = ["NAME", "STATE", "LOADED", "AUTO-UNLOAD", "TYPE", "GROUP", "DEFAULT", "MODEL"]
    rows: list[dict[str, str]] = []
    for entry in sorted(snapshot, key=lambda item: str(item.get("name", "?"))):
        name = str(entry.get("name", "?"))
        state = str(entry.get("state", "inactive"))
        loaded = str(entry.get("loaded", "no"))
        auto_unload = str(entry.get("auto_unload", "-"))
        model_type = str(entry.get("type", "-"))
        group = str(entry.get("group", "-"))
        default = str(entry.get("default", "-"))
        model_path = str(entry.get("model", "-"))

        rows.append(
            {
                "NAME": name,
                "STATE": state,
                "LOADED": loaded,
                "AUTO-UNLOAD": auto_unload,
                "TYPE": model_type,
                "GROUP": group,
                "DEFAULT": default,
                "MODEL": model_path,
            },
        )

    # Calculate column widths
    widths: dict[str, int] = {header: len(header) for header in headers}
    for row in rows:
        for header in headers:
            widths[header] = max(widths[header], len(row[header]))

    # Print header
    header_line = "  " + " | ".join(header.ljust(widths[header]) for header in headers)

    # Print divider
    divider = "  " + "-+-".join("-" * widths[header] for header in headers)

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
    models: list[dict[str, Any]] = []
    if isinstance(raw_models, list):
        for model in raw_models:
            if isinstance(model, dict) and "id" in model:
                # Convert Model dict to format expected by _render_watch_table
                metadata = model.get("metadata") or {}
                name = model["id"]
                state = metadata.get("process_state", "inactive")
                pid = metadata.get("pid")
                port = metadata.get("port")

                # Format state with pid and port if running
                if state == "running" and pid is not None:
                    state_display = f"{state} (pid={pid}"
                    if port is not None:
                        state_display += f", port={port}"
                    state_display += ")"
                else:
                    state_display = state

                # Loaded in memory
                memory_flag = metadata.get("memory_state") == "loaded"
                loaded_in_memory = "yes" if memory_flag else "no"

                # Auto-unload
                auto_unload_minutes = metadata.get("auto_unload_minutes")
                auto_unload = f"{auto_unload_minutes}min" if auto_unload_minutes else "-"

                # Model type
                model_type = metadata.get("model_type", "-")

                # Group
                group = metadata.get("group") or "-"

                # Default
                is_default = metadata.get("default", False)
                default = "✓" if is_default else "-"

                # Model path
                model_path = metadata.get("model_path", "-")

                models.append(
                    {
                        "name": name,
                        "state": state_display,
                        "loaded": loaded_in_memory,
                        "auto_unload": auto_unload,
                        "type": model_type,
                        "group": group,
                        "default": default,
                        "model": model_path,
                    }
                )
            elif isinstance(model, dict):
                # Fallback for dict format (legacy)
                models.append(model)

    def _normalize_state(entry: dict[str, Any]) -> str:
        return str(entry.get("state", "")).strip().lower()

    running = sum(1 for entry in models if _normalize_state(entry).startswith("running"))
    stopped = sum(1 for entry in models if _normalize_state(entry).startswith("stopped"))
    failed = sum(
        1
        for entry in models
        if _normalize_state(entry).startswith("failed") or entry.get("exit_code") not in (None, 0)
    )

    click.echo(
        f"[{formatted}] models={len(models)} running={running} stopped={stopped} failed={failed}",
    )
    click.echo(_render_watch_table(models))


@cli.command(help="Start the MLX OpenAI Server with the supplied flags")
@click.option(
    "--model-path",
    required=True,
    type=str,
    help="Path to the model. Accepts local paths or Hugging Face repository IDs (e.g., 'blackforestlabs/FLUX.1-dev').",
)
@click.option(
    "--model-type",
    default=DEFAULT_MODEL_TYPE,
    type=click.Choice(
        ["lm", "multimodal", "image-generation", "image-edit", "embeddings", "whisper"],
    ),
    help="Type of model to run (lm: text-only, multimodal: text+vision+audio, image-generation: flux image generation, image-edit: flux image edit, embeddings: text embeddings, whisper: audio transcription)",
)
@click.option(
    "--context-length",
    default=DEFAULT_CONTEXT_LENGTH,
    type=int,
    help="Context length for language models. Only works with `lm` or `multimodal` model types.",
)
@click.option("--port", default=DEFAULT_PORT, type=int, help="Port to run the server on")
@click.option("--host", default=DEFAULT_BIND_HOST, help="Host to run the server on")
@click.option(
    "--max-concurrency",
    default=DEFAULT_MAX_CONCURRENCY,
    type=int,
    help="Maximum number of concurrent requests",
)
@click.option(
    "--queue-timeout",
    default=DEFAULT_QUEUE_TIMEOUT,
    type=int,
    help="Request timeout in seconds",
)
@click.option(
    "--queue-size",
    default=DEFAULT_QUEUE_SIZE,
    type=int,
    help="Maximum queue size for pending requests",
)
@click.option(
    "--quantize",
    default=DEFAULT_QUANTIZE,
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
    default=DEFAULT_LOG_LEVEL,
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
    model_path : str
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
            "--auto-unload-minutes",
            "--auto-unload-minutes requires --jit to be set.",
        )

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


def _start_hub_daemon(config: MLXHubConfig) -> subprocess.Popen[bytes] | None:
    """Start the hub daemon subprocess if not already running.

    Returns the subprocess.Popen object if started, None if already running.
    Raises click.ClickException on failure.
    """
    # Check daemon availability
    try:
        _call_daemon_api(config, "GET", "/health", timeout=2.0)
    except click.ClickException:
        pass  # Not running, proceed to start
    else:
        return None  # Already running

    if config.source_path is None:
        raise click.ClickException(
            "Hub configuration must be saved to disk before starting the manager.",
        )

    click.echo("Starting hub manager...")
    host_val = config.host or DEFAULT_BIND_HOST
    port_val = str(config.port)

    # Set environment variable for daemon to use the same config
    env = os.environ.copy()
    if config.source_path:
        env["MLX_HUB_CONFIG_PATH"] = str(config.source_path)

    cmd = [
        sys.executable,  # Use the same Python executable
        "-m",
        "uvicorn",
        "app.hub.daemon:create_app",
        "--factory",
        "--host",
        host_val,
        "--port",
        port_val,
    ]

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            cwd=Path.cwd(),
        )

        # Start background threads to log subprocess output
        def _log_output(stream: IO[bytes], level: str, prefix: str) -> None:
            """Log output from subprocess stream."""
            try:
                for line in iter(stream.readline, b""):
                    line_str = line.decode("utf-8", errors="replace").rstrip()
                    if line_str:
                        if level == "info":
                            logger.info(f"{prefix}: {line_str}")
                        elif level == "error":
                            logger.error(f"{prefix}: {line_str}")
                        else:
                            logger.debug(f"{prefix}: {line_str}")
            except Exception as e:
                logger.warning(f"Error reading subprocess {prefix} output: {e}")

        # Start threads to read stdout and stderr
        if proc.stdout:
            stdout_thread = threading.Thread(
                target=_log_output,
                args=(proc.stdout, "debug", f"hub-daemon[{proc.pid}].stdout"),
                daemon=True,
            )
            stdout_thread.start()

        if proc.stderr:
            stderr_thread = threading.Thread(
                target=_log_output,
                args=(
                    proc.stderr,
                    "debug",
                    f"hub-daemon[{proc.pid}].stderr",
                ),  # MLX-LM outputs to stderr so treat as debug
                daemon=True,
            )
            stderr_thread.start()

        click.echo(f"Hub manager process started (PID: {proc.pid})")
    except Exception as e:
        raise click.ClickException(f"Failed to start hub manager: {e}") from e

    # Wait for daemon to become available
    deadline = time.time() + 20.0
    while time.time() < deadline:
        try:
            _call_daemon_api(config, "GET", "/health", timeout=1.0)
            click.echo("Hub manager is now running.")
            break
        except click.ClickException:
            time.sleep(0.5)
    else:
        raise click.ClickException("Hub manager failed to start within 20 seconds.")

    return proc


def _auto_start_default_models(config: MLXHubConfig) -> None:
    """Auto-start any models marked as default in the configuration."""
    try:
        # Refresh the controller state so it sees the latest config
        _call_daemon_api(config, "POST", "/hub/reload")
        for model in config.models:
            try:
                if not getattr(model, "is_default_model", False):
                    continue
                name = model.name
                jit = bool(getattr(model, "jit_enabled", False))
                click.echo(f"Requesting process start for default model: {name}")
                try:
                    _call_daemon_api(
                        config,
                        "POST",
                        f"/hub/models/{quote(str(name), safe='')}/start",
                    )
                    _flash(f"{name}: start requested", tone="success")
                except click.ClickException as exc_start:
                    # If start failed and the model is non-JIT, fall back
                    # to requesting a memory load so the configured default
                    # ends up available in the controller view.
                    if not jit:
                        try:
                            _call_daemon_api(
                                config,
                                "POST",
                                f"/hub/models/{quote(str(name), safe='')}/load",
                            )
                            _flash(f"{name}: load requested (fallback)", tone="success")
                        except click.ClickException as exc_load:
                            _flash(f"{name}: load failed ({exc_load})", tone="error")
                    else:
                        _flash(f"{name}: start failed ({exc_start})", tone="error")
            except Exception as e:  # pragma: no cover - best-effort
                logger.debug(f"Error while auto-starting default model. {type(e).__name__}: {e}")
    except Exception as e:
        # Ignore failures here; user can start models manually
        logger.exception("Failed to auto-start default models, continuing anyway", exc_info=e)


@hub.command(name="start", help="Start the hub manager ")
@click.argument("model_names", nargs=-1)
@click.pass_context
def hub_start(ctx: click.Context, model_names: tuple[str, ...]) -> None:
    """Launch the hub manager and print status."""
    config = _load_hub_config_or_fail(ctx.obj.get("hub_config_path"))
    models = model_names or None

    proc = _start_hub_daemon(config)

    if proc is not None:
        # Persist runtime state for other CLI invocations to find this daemon
        try:
            _write_hub_runtime_state(config, proc.pid)
        except Exception:
            # Best-effort; do not fail start if writing runtime state fails
            logger.debug("Failed to write hub runtime state after start")

        _auto_start_default_models(config)

    click.echo(f"Status page enabled: {'yes' if config.enable_status_page else 'no'}")
    if config.enable_status_page:
        host_display = "localhost" if config.host == "0.0.0.0" else config.host
        click.echo(f"Browse to http://{host_display}:{config.port}/hub for the status dashboard")

    snapshot = None
    try:
        snapshot = _call_daemon_api(config, "GET", "/hub/status") or {}
    except click.ClickException as e:
        _flash(f"Unable to fetch live status: {e}", tone="warning")
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
    snapshot = None
    try:
        snapshot = _call_daemon_api(config, "GET", "/hub/status") or {}
    except click.ClickException:
        _flash("Hub daemon is not running", tone="warning")
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
    try:
        diff = _call_daemon_api(config, "POST", "/hub/reload") or {}
    except click.ClickException as e:
        raise click.ClickException(f"Hub reload failed: {e}") from e
    _emit_reload_summary(diff, header="Hub reload complete")


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
    try:
        _call_daemon_api(config, "POST", "/hub/reload")
    except click.ClickException as e:
        # If we can't contact the daemon it's likely not running; be friendly
        msg = str(e)
        if "Failed to contact hub daemon" in msg:
            _flash("Hub manager is not running; nothing to stop", tone="info")
            return
        raise click.ClickException(f"Config sync failed before shutdown: {e}") from e

    try:
        _call_daemon_api(config, "POST", "/hub/shutdown")
    except click.ClickException as e:
        msg = str(e)
        if "Failed to contact hub daemon" in msg:
            _flash("Hub manager is not running; nothing to stop", tone="info")
            return
        raise click.ClickException(f"Hub shutdown failed: {e}") from e

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
    if not model_names or all(not str(n).strip() for n in model_names):
        raise click.UsageError("Missing argument 'MODEL_NAMES'.")
    try:
        _call_daemon_api(config, "POST", "/hub/reload")
    except click.ClickException as e:
        raise click.ClickException(f"Config sync failed before load: {e}") from e
    for raw_name in model_names:
        target = raw_name.strip()
        if not target:
            _flash("Skipping blank model name entry", tone="warning")
            continue
        try:
            _call_daemon_api(config, "POST", f"/hub/models/{quote(target, safe='')}/start")
        except click.ClickException as e:
            _flash(f"{target}: start failed ({e})", tone="error")
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
    if not model_names or all(not str(n).strip() for n in model_names):
        raise click.UsageError("Missing argument 'MODEL_NAMES'.")
    try:
        _call_daemon_api(config, "POST", "/hub/reload")
    except click.ClickException as e:
        raise click.ClickException(f"Config sync failed before unload: {e}") from e
    for raw_name in model_names:
        target = raw_name.strip()
        if not target:
            _flash("Skipping blank model name entry", tone="warning")
            continue
        try:
            _call_daemon_api(config, "POST", f"/hub/models/{quote(target, safe='')}/stop")
        except click.ClickException as e:
            _flash(f"{target}: stop failed ({e})", tone="error")
        else:
            _flash(f"{target}: stop requested", tone="success")


@hub.command(name="load-model", help="Load handlers for one or more models into memory")
@click.argument("model_names", nargs=-1, required=True)
@click.pass_context
def hub_load_model(ctx: click.Context, model_names: tuple[str, ...]) -> None:
    """Trigger controller-backed memory loads for the provided models."""
    config = _load_hub_config_or_fail(ctx.obj.get("hub_config_path"))
    _run_memory_actions(config, model_names, "load")


@hub.command(name="unload-model", help="Unload handlers for one or more models from memory")
@click.argument("model_names", nargs=-1, required=True)
@click.pass_context
def hub_unload_model(ctx: click.Context, model_names: tuple[str, ...]) -> None:
    """Trigger controller-backed memory unloads for the provided models."""
    config = _load_hub_config_or_fail(ctx.obj.get("hub_config_path"))
    _run_memory_actions(config, model_names, "unload")


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
    click.echo("Watching hub manager (press Ctrl+C to stop)...")
    sleep_interval = max(interval, 0.5)
    try:
        while True:
            try:
                snapshot = _call_daemon_api(config, "GET", "/hub/status") or {}
            except click.ClickException:
                click.echo(
                    click.style(
                        "Hub daemon is not running. Start it via 'mlx-openai-server hub start'.",
                        fg="yellow",
                    ),
                )
            else:
                _print_watch_snapshot(snapshot)
            time.sleep(sleep_interval)
    except KeyboardInterrupt:  # pragma: no cover - interactive command
        click.echo("Stopped watching hub manager")
