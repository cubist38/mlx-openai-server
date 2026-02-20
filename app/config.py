"""Server configuration dataclass and helpers.

This module exposes ``MLXServerConfig``, a dataclass that holds all CLI
configuration values for the server. The dataclass performs minimal
normalization in ``__post_init__`` (parsing comma-separated LoRA
arguments and applying small model-type-specific defaults).

It also provides ``ModelEntryConfig`` and ``MultiModelServerConfig``
for YAML-based multi-handler configurations, along with the
``load_config_from_yaml`` helper that parses a YAML file into
these structures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger


@dataclass
class MLXServerConfig:
    """Container for server CLI configuration values.

    The class mirrors the Click CLI options and normalizes a few fields
    during initialization (for example converting comma-separated
    strings into lists and setting sensible defaults for image model
    configurations).
    """

    model_path: str
    model_type: str = "lm"
    context_length: int | None = None
    port: int = 8000
    host: str = "0.0.0.0"
    max_concurrency: int = 1
    queue_timeout: int = 300
    queue_size: int = 100
    disable_auto_resize: bool = False
    quantize: int | None = None
    config_name: str | None = None
    lora_paths: list[str] | None = field(default=None, init=False)
    lora_scales: list[float] | None = field(default=None, init=False)
    log_file: str | None = None
    no_log_file: bool = False
    log_level: str = "INFO"
    enable_auto_tool_choice: bool = False
    tool_call_parser: str | None = None
    reasoning_parser: str | None = None
    message_converter: str | None = None
    trust_remote_code: bool = False
    chat_template_file: str | None = None
    debug: bool = False
    prompt_cache_size: int = 10
    draft_model_path: str | None = None
    num_draft_tokens: int = 2

    # Default sampling parameters (override DEFAULT_* env when set via CLI)
    default_max_tokens: int | None = None
    default_temperature: float | None = None
    default_top_p: float | None = None
    default_top_k: int | None = None

    # Used to capture raw CLI input before processing
    lora_paths_str: str | None = None
    lora_scales_str: str | None = None

    def __post_init__(self):
        """Normalize certain CLI fields after instantiation.

        - Convert comma-separated ``lora_paths`` and ``lora_scales`` into
          lists when provided.
        - Apply small model-type-specific defaults for ``config_name``
          and emit warnings when values appear inconsistent.
        """

        # Process comma-separated LoRA paths and scales into lists (or None)
        if self.lora_paths_str:
            self.lora_paths = [p.strip() for p in self.lora_paths_str.split(",") if p.strip()]

        if self.lora_scales_str:
            try:
                self.lora_scales = [
                    float(s.strip()) for s in self.lora_scales_str.split(",") if s.strip()
                ]
            except ValueError:
                # If parsing fails, log and set to None
                logger.warning("Failed to parse lora_scales into floats; ignoring lora_scales")
                self.lora_scales = None

        # Validate that config name is only used with image-generation and
        # image-edit model types. If missing for those types, set defaults.
        if self.config_name and self.model_type not in ["image-generation", "image-edit"]:
            logger.warning(
                "Config name parameter '%s' provided but model type is '%s'. "
                "Config name is only used with image-generation "
                "and image-edit models.",
                self.config_name,
                self.model_type,
            )
        elif self.model_type == "image-generation" and not self.config_name:
            logger.warning(
                "Model type is 'image-generation' but no config name "
                "specified. Using default 'flux-schnell'."
            )
            self.config_name = "flux-schnell"
        elif self.model_type == "image-edit" and not self.config_name:
            logger.warning(
                "Model type is 'image-edit' but no config name "
                "specified. Using default 'flux-kontext-dev'."
            )
            self.config_name = "flux-kontext-dev"

        # Speculative decoding (draft model) is only supported for lm model type
        if self.draft_model_path and self.model_type != "lm":
            logger.warning(
                "Draft model / num-draft-tokens are only supported for model type 'lm'. "
                "Ignoring speculative decoding options."
            )
            self.draft_model_path = None
            self.num_draft_tokens = 2

    @property
    def model_identifier(self) -> str:
        """Get the appropriate model identifier based on model type.

        For Flux models, we always use model_path (local directory path).
        """
        return self.model_path


# ---------------------------------------------------------------------------
# Multi-model YAML configuration
# ---------------------------------------------------------------------------

VALID_MODEL_TYPES = frozenset(
    {"lm", "multimodal", "image-generation", "image-edit", "embeddings", "whisper"}
)


@dataclass
class ModelEntryConfig:
    """Configuration for a single model entry in a multi-model YAML config.

    Each entry maps to exactly one handler that will be registered in
    the ``ModelRegistry``.  The ``model_id`` defaults to ``model_path``
    when not set explicitly, giving callers a short alias they can use
    in API requests.
    """

    model_path: str
    model_type: str = "lm"
    model_id: str | None = None

    # Common options
    context_length: int | None = None
    max_concurrency: int = 1
    queue_timeout: int = 300
    queue_size: int = 100

    # Image-generation / image-edit options
    quantize: int | None = None
    config_name: str | None = None

    # LoRA options
    lora_paths: list[str] | None = None
    lora_scales: list[float] | None = None

    # LM / multimodal options
    disable_auto_resize: bool = False
    enable_auto_tool_choice: bool = False
    tool_call_parser: str | None = None
    reasoning_parser: str | None = None
    message_converter: str | None = None
    trust_remote_code: bool = False
    chat_template_file: str | None = None
    debug: bool = False
    prompt_cache_size: int = 10
    draft_model_path: str | None = None
    num_draft_tokens: int = 2

    def __post_init__(self) -> None:
        """Resolve ``model_id`` and validate ``model_type``."""
        if self.model_id is None:
            self.model_id = self.model_path

        if self.model_type not in VALID_MODEL_TYPES:
            msg = (
                f"Invalid model_type '{self.model_type}' for model '{self.model_path}'. "
                f"Must be one of {sorted(VALID_MODEL_TYPES)}."
            )
            raise ValueError(msg)

        # Apply image-generation / image-edit defaults (same as MLXServerConfig)
        if self.model_type == "image-generation" and not self.config_name:
            logger.warning(
                "Model '%s' (image-generation) has no config_name. "
                "Defaulting to 'flux-schnell'.",
                self.model_path,
            )
            self.config_name = "flux-schnell"
        elif self.model_type == "image-edit" and not self.config_name:
            logger.warning(
                "Model '%s' (image-edit) has no config_name. "
                "Defaulting to 'flux-kontext-dev'.",
                self.model_path,
            )
            self.config_name = "flux-kontext-dev"

        # Speculative decoding is LM-only
        if self.draft_model_path and self.model_type != "lm":
            logger.warning(
                "Draft model is only supported for 'lm'. "
                "Ignoring for model '%s'.",
                self.model_path,
            )
            self.draft_model_path = None
            self.num_draft_tokens = 2


@dataclass
class MultiModelServerConfig:
    """Top-level configuration for running multiple models from a YAML file.

    The ``server`` section holds host/port/logging settings, while
    ``models`` is a list of ``ModelEntryConfig`` entries – each of
    which will be loaded as a separate handler at startup.
    """

    models: list[ModelEntryConfig]
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    log_file: str | None = None
    no_log_file: bool = False


def load_config_from_yaml(config_path: str) -> MultiModelServerConfig:
    """Parse a YAML config file into a ``MultiModelServerConfig``.

    Parameters
    ----------
    config_path : str
        Filesystem path to the YAML configuration file.

    Returns
    -------
    MultiModelServerConfig
        Parsed and validated configuration.

    Raises
    ------
    FileNotFoundError
        If ``config_path`` does not exist.
    ValueError
        If the YAML is missing required keys or contains invalid values.
    """
    import yaml

    path = Path(config_path)
    if not path.exists():
        msg = f"Config file not found: {config_path}"
        raise FileNotFoundError(msg)

    with path.open("r") as fh:
        raw: dict = yaml.safe_load(fh)

    if not isinstance(raw, dict):
        msg = f"Config file must be a YAML mapping, got {type(raw).__name__}"
        raise ValueError(msg)

    # ---- server section (optional, all keys have defaults) ----
    server_raw: dict = raw.get("server", {})
    if not isinstance(server_raw, dict):
        msg = "'server' section must be a mapping"
        raise ValueError(msg)

    # ---- models section (required, at least one entry) ----
    models_raw: list = raw.get("models", [])
    if not isinstance(models_raw, list) or len(models_raw) == 0:
        msg = "'models' section must be a non-empty list of model entries"
        raise ValueError(msg)

    model_entries: list[ModelEntryConfig] = []
    seen_ids: set[str] = set()

    for idx, entry in enumerate(models_raw):
        if not isinstance(entry, dict):
            msg = f"Model entry at index {idx} must be a mapping"
            raise ValueError(msg)

        if "model_path" not in entry:
            msg = f"Model entry at index {idx} is missing required key 'model_path'"
            raise ValueError(msg)

        model_cfg = ModelEntryConfig(**entry)

        # Enforce unique model_id values
        if model_cfg.model_id in seen_ids:
            msg = (
                f"Duplicate model_id '{model_cfg.model_id}' in config. "
                "Each model must have a unique model_id."
            )
            raise ValueError(msg)
        seen_ids.add(model_cfg.model_id)
        model_entries.append(model_cfg)

    return MultiModelServerConfig(
        models=model_entries,
        host=server_raw.get("host", "0.0.0.0"),
        port=server_raw.get("port", 8000),
        log_level=server_raw.get("log_level", "INFO"),
        log_file=server_raw.get("log_file"),
        no_log_file=server_raw.get("no_log_file", False),
    )
