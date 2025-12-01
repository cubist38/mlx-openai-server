"""Hub configuration parsing utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any

from loguru import logger
import yaml

from ..config import MLXServerConfig
from ..const import (
    DEFAULT_BIND_HOST,
    DEFAULT_ENABLE_STATUS_PAGE,
    DEFAULT_HUB_CONFIG_PATH,
    DEFAULT_HUB_LOG_PATH,
    DEFAULT_LOG_LEVEL,
    DEFAULT_MODEL_STARTING_PORT,
    DEFAULT_PORT,
)
from ..utils.network import is_port_available

PORT_MIN = 1024
PORT_MAX = 65535


class HubConfigError(RuntimeError):
    """Raised when the hub configuration file is invalid."""


_slug_pattern = re.compile(r"^[a-z0-9]+(?:[-_][a-z0-9]+)*$", re.IGNORECASE)


def _ensure_slug(value: str, *, field_name: str) -> str:
    """Validate that ``value`` is already a compliant slug without altering it."""
    candidate = value.strip()
    if not candidate:
        raise HubConfigError(f"{field_name} cannot be empty")
    if not _slug_pattern.fullmatch(candidate):
        raise HubConfigError(
            f"{field_name} must be alphanumeric with optional hyphen/underscore separators",
        )
    return candidate


@dataclass(slots=True)
class MLXHubGroupConfig:
    """Configuration for a logical model group."""

    name: str
    max_loaded: int | None = None
    idle_unload_trigger_min: int | None = None

    def __post_init__(self) -> None:
        """Validate group names and ensure group constraints are sane."""
        self.name = _ensure_slug(self.name, field_name="group name")
        if self.max_loaded is not None and self.max_loaded < 1:
            raise HubConfigError("max_loaded must be a positive integer when provided")
        if self.idle_unload_trigger_min is not None:
            if self.idle_unload_trigger_min < 1:
                raise HubConfigError("idle_unload_trigger_min must be a positive integer")
            if self.max_loaded is None:
                raise HubConfigError(
                    "idle_unload_trigger_min requires max_loaded to be configured for the group",
                )


@dataclass(slots=True)
class MLXHubConfig:
    """Top-level hub configuration derived from YAML."""

    host: str = DEFAULT_BIND_HOST
    port: int = DEFAULT_PORT
    model_starting_port: int = DEFAULT_MODEL_STARTING_PORT
    log_level: str = DEFAULT_LOG_LEVEL
    log_path: Path = field(default_factory=lambda: DEFAULT_HUB_LOG_PATH)
    enable_status_page: bool = DEFAULT_ENABLE_STATUS_PAGE
    models: list[MLXServerConfig] = field(default_factory=list)
    groups: list[MLXHubGroupConfig] = field(default_factory=list)
    source_path: Path | None = None

    def __post_init__(self) -> None:
        """Normalize hub defaults."""
        self.log_level = self.log_level.upper()
        self.log_path = self.log_path.expanduser()


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load and parse a YAML file.

    Parameters
    ----------
    path : Path
        Path to the YAML file.

    Returns
    -------
    dict[str, Any]
        The parsed YAML data.

    Raises
    ------
    HubConfigError
        If the file is not found or parsing fails.
    """
    try:
        with path.open("r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
    except FileNotFoundError as e:
        raise HubConfigError(f"Hub config file not found: {path}") from e
    except yaml.YAMLError as e:
        raise HubConfigError(f"Failed to parse hub config '{path}': {e}") from e

    if not isinstance(loaded, dict):
        raise HubConfigError("Hub config root must be a mapping")
    return loaded


def _build_groups(raw_groups: list[dict[str, Any]] | None) -> list[MLXHubGroupConfig]:
    """Build group configurations from raw data.

    Parameters
    ----------
    raw_groups : list[dict[str, Any]] or None
        Raw group data from YAML.

    Returns
    -------
    list[MLXHubGroupConfig]
        List of group configurations.

    Raises
    ------
    HubConfigError
        If group data is invalid.
    """
    groups: list[MLXHubGroupConfig] = []
    if not raw_groups:
        return groups

    seen: set[str] = set()
    for idx, group_data in enumerate(raw_groups, start=1):
        if not isinstance(group_data, dict):
            raise HubConfigError(f"Group entry #{idx} must be a mapping")
        if "name" not in group_data:
            raise HubConfigError(f"Group entry #{idx} is missing required 'name'")
        max_loaded_value = group_data.get("max_loaded")
        max_loaded = None
        if max_loaded_value is not None:
            try:
                max_loaded = int(max_loaded_value)
            except (TypeError, ValueError) as exc:
                raise HubConfigError("max_loaded must be an integer") from exc

        idle_trigger_value = group_data.get("idle_unload_trigger_min")
        idle_trigger = None
        if idle_trigger_value is not None:
            try:
                idle_trigger = int(idle_trigger_value)
            except (TypeError, ValueError) as exc:
                raise HubConfigError("idle_unload_trigger_min must be an integer") from exc

        group = MLXHubGroupConfig(
            name=str(group_data["name"]),
            max_loaded=max_loaded,
            idle_unload_trigger_min=idle_trigger,
        )
        if group.name in seen:
            raise HubConfigError(f"Duplicate group name '{group.name}' detected")
        seen.add(group.name)
        groups.append(group)
    return groups


def _resolve_model_log_file(server_config: MLXServerConfig, hub_log_path: Path) -> MLXServerConfig:
    """Resolve the log file path for a model configuration.

    Parameters
    ----------
    server_config : MLXServerConfig
        The server configuration.
    hub_log_path : Path
        The hub log directory path.

    Returns
    -------
    MLXServerConfig
        The updated server configuration.

    Raises
    ------
    HubConfigError
        If the model name is missing.
    """
    if server_config.no_log_file:
        return server_config

    if server_config.log_file:
        return server_config

    if not server_config.name:
        raise HubConfigError("Each hub model requires a 'name' to derive default log paths")

    log_file = hub_log_path / f"{server_config.name}.log"
    server_config.log_file = str(log_file)
    return server_config


def _build_models(
    raw_models: list[dict[str, Any]] | None,
    base_host: str,
    base_port: int,
    starting_port: int,
    base_log_level: str,
    hub_log_path: Path,
    group_lookup: dict[str, MLXHubGroupConfig],
    persisted_ports: dict[str, int] | None = None,
    additional_reserved_ports: set[int] | None = None,
) -> list[MLXServerConfig]:
    """Build model configurations from raw data.

    Parameters
    ----------
    raw_models : list[dict[str, Any]] or None
        Raw model data from YAML.
    base_host : str
        Base host for models.
    base_port : int
        Base port for models.
    starting_port : int
        The first port number to try/assign when auto-allocating model server
        ports. Must be a positive integer (typically >= 1024 and <= 65535)
        and should not collide with reserved ports (for example the hub
        controller `base_port` or any entries in ``additional_reserved_ports``).
        When ``persisted_ports`` is provided, previously assigned ports are
        preferred and ``starting_port`` only influences allocation for models
        without a persisted assignment.
    base_log_level : str
        Base log level for models.
    hub_log_path : Path
        Hub log directory path.
    group_lookup : dict[str, MLXHubGroupConfig]
        Lookup of group configurations.
    persisted_ports : dict[str, int] | None, optional
        Mapping of model names to previously assigned ports. When provided, models
        without an explicit ``port`` reuse their prior assignment even if the
        socket currently appears busy.

    Returns
    -------
    list[MLXServerConfig]
        List of model configurations.

    Raises
    ------
    HubConfigError
        If model data is invalid.
    """
    if not raw_models:
        raise HubConfigError("Hub config must include at least one model entry")

    models: list[MLXServerConfig] = []
    seen_names: set[str] = set()
    reserved_ports: set[int] = {base_port}
    if additional_reserved_ports:
        reserved_ports.update(additional_reserved_ports)
    next_auto_port = max(starting_port, PORT_MIN)

    persisted_ports = persisted_ports or {}

    for idx, raw_model in enumerate(raw_models, start=1):
        if not isinstance(raw_model, dict):
            raise HubConfigError(f"Model entry #{idx} must be a mapping")

        if "model_path" not in raw_model:
            raise HubConfigError(f"Model entry #{idx} is missing required 'model_path'")
        if "name" not in raw_model:
            raise HubConfigError(f"Model entry #{idx} is missing required 'name'")

        name = _ensure_slug(str(raw_model["name"]), field_name="model name")
        group_value = raw_model.get("group")
        group_slug = _ensure_slug(str(group_value), field_name="group") if group_value else None
        if group_slug and group_lookup and group_slug not in group_lookup:
            raise HubConfigError(
                f"Model '{name}' references group '{group_slug}' which is not defined in hub config",
            )

        model_payload = dict(raw_model)
        default_flag = bool(model_payload.pop("default", False))

        model_payload["host"] = model_payload.get("host", base_host)
        model_payload["log_level"] = model_payload.get("log_level", base_log_level)
        model_payload["name"] = name
        model_payload["group"] = group_slug

        port_value = model_payload.get("port")
        host_value = str(model_payload["host"])
        persisted_port = persisted_ports.get(name)
        if port_value is None:
            if persisted_port is not None:
                candidate_port = _coerce_port_value(name, persisted_port)
                if candidate_port in reserved_ports:
                    raise HubConfigError(
                        f"Persisted port {candidate_port} for model '{name}' conflicts with another entry",
                    )
                model_payload["port"] = candidate_port
            else:
                candidate_port, next_auto_port = _allocate_port(
                    name,
                    host_value,
                    next_auto_port,
                    reserved_ports,
                )
                model_payload["port"] = candidate_port
        else:
            candidate_port = _coerce_port_value(name, port_value)
            if candidate_port in reserved_ports:
                raise HubConfigError(
                    f"Model '{name}' port {candidate_port} conflicts with another model or the controller",
                )
            # Skip availability check when reloading with the same port (already bound by current process)
            if persisted_port != candidate_port and not is_port_available(
                host_value,
                candidate_port,
            ):
                raise HubConfigError(
                    f"Model '{name}' port {candidate_port} is already in use on host '{host_value}'",
                )
            model_payload["port"] = candidate_port
        reserved_ports.add(candidate_port)

        server_config = MLXServerConfig(**model_payload)
        server_config.is_default_model = default_flag
        if not server_config.name:
            raise HubConfigError("Each hub model must have a slug-compliant name")
        if server_config.name in seen_names:
            raise HubConfigError(f"Duplicate model name '{server_config.name}' detected")
        seen_names.add(server_config.name)
        server_config = _resolve_model_log_file(server_config, hub_log_path)
        models.append(server_config)

    return models


def load_hub_config(
    config_path: Path | str | None = None,
    *,
    persisted_ports: dict[str, int] | None = None,
) -> MLXHubConfig:
    """Load and validate a hub configuration file.

    Parameters
    ----------
    config_path : Path, str, or None, optional
        Path to the hub configuration file. If None, uses default path.
    persisted_ports : dict[str, int] | None, optional
        Mapping of model names to previously assigned ports. Useful when reloading
        while hub-managed workers are still bound to their sockets.

    Returns
    -------
    MLXHubConfig
        The loaded and validated hub configuration.
    """
    if config_path is None:
        path = DEFAULT_HUB_CONFIG_PATH
    else:
        path = Path(config_path).expanduser()

    data = _load_yaml(path)

    port_value = data.get("port", DEFAULT_PORT)
    try:
        port = int(port_value)
    except (TypeError, ValueError) as exc:
        raise HubConfigError("Hub port must be an integer") from exc

    model_starting_port_value = data.get("model_starting_port", DEFAULT_MODEL_STARTING_PORT)
    try:
        model_starting_port = int(model_starting_port_value)
    except (TypeError, ValueError) as exc:
        raise HubConfigError("model_starting_port must be an integer") from exc
    if not (PORT_MIN <= model_starting_port <= PORT_MAX):
        raise HubConfigError(
            f"model_starting_port must be between {PORT_MIN} and {PORT_MAX}",
        )

    hub = MLXHubConfig(
        host=str(data.get("host", DEFAULT_BIND_HOST)),
        port=port,
        model_starting_port=model_starting_port,
        log_level=str(data.get("log_level", DEFAULT_LOG_LEVEL)),
        log_path=Path(str(data.get("log_path", DEFAULT_HUB_LOG_PATH))),
        enable_status_page=data.get("enable_status_page", DEFAULT_ENABLE_STATUS_PAGE),
        source_path=path,
    )

    # Ensure log directory exists
    hub.log_path.mkdir(parents=True, exist_ok=True)

    hub.groups = _build_groups(data.get("groups"))
    group_lookup = {group.name: group for group in hub.groups}

    hub.models = _build_models(
        raw_models=data.get("models"),
        base_host=hub.host,
        base_port=hub.port,
        starting_port=hub.model_starting_port,  # Start model allocation from starting port
        base_log_level=hub.log_level,
        hub_log_path=hub.log_path,
        group_lookup=group_lookup,
        persisted_ports=persisted_ports,
        additional_reserved_ports=set(),
    )

    # Ensure all models inherit the hub's status page setting
    for model in hub.models:
        model.enable_status_page = hub.enable_status_page

    logger.debug(
        f"Loaded hub config from {path} with {len(hub.models)} model(s) and {len(hub.groups)} group(s)",
    )
    return hub


def _coerce_port_value(name: str, port_value: Any) -> int:
    try:
        candidate_port = int(port_value)
    except (TypeError, ValueError) as exc:
        raise HubConfigError(f"Model '{name}' port must be an integer") from exc
    if not (PORT_MIN <= candidate_port <= PORT_MAX):
        raise HubConfigError(
            f"Model '{name}' port must be between {PORT_MIN} and {PORT_MAX}",
        )
    return candidate_port


def _allocate_port(
    name: str,
    host: str,
    starting_port: int,
    reserved_ports: set[int],
) -> tuple[int, int]:
    port = max(starting_port, PORT_MIN)
    while port <= PORT_MAX:
        if port not in reserved_ports and is_port_available(host, port):
            return port, port + 1
        port += 1
    raise HubConfigError(
        f"Unable to find an available port for model '{name}' starting at {starting_port}",
    )
