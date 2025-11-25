"""Server configuration dataclass and helpers.

This module exposes ``MLXServerConfig``, a dataclass that holds all CLI
configuration values for the server. The dataclass performs minimal
normalization in ``__post_init__`` (parsing comma-separated LoRA
arguments and applying small model-type-specific defaults).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from loguru import logger

from .const import (
    DEFAULT_AUTO_UNLOAD_MINUTES,
    DEFAULT_BIND_HOST,
    DEFAULT_CONFIG_NAME,
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_DISABLE_AUTO_RESIZE,
    DEFAULT_ENABLE_AUTO_TOOL_CHOICE,
    DEFAULT_ENABLE_STATUS_PAGE,
    DEFAULT_GROUP,
    DEFAULT_IS_DEFAULT_MODEL,
    DEFAULT_JIT_ENABLED,
    DEFAULT_LOG_FILE,
    DEFAULT_LOG_LEVEL,
    DEFAULT_LORA_PATHS_STR,
    DEFAULT_LORA_SCALES_STR,
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_MODEL_TYPE,
    DEFAULT_NO_LOG_FILE,
    DEFAULT_PORT,
    DEFAULT_QUANTIZE,
    DEFAULT_QUEUE_SIZE,
    DEFAULT_QUEUE_TIMEOUT,
    DEFAULT_REASONING_PARSER,
    DEFAULT_TOOL_CALL_PARSER,
    DEFAULT_TRUST_REMOTE_CODE,
)


@dataclass
class MLXServerConfig:
    """Container for server CLI configuration values.

    The class mirrors the Click CLI options and normalizes a few fields
    during initialization (for example converting comma-separated
    strings into lists and setting sensible defaults for image model
    configurations).
    """

    model_path: str
    model_type: str = DEFAULT_MODEL_TYPE
    context_length: int = DEFAULT_CONTEXT_LENGTH
    port: int = DEFAULT_PORT
    host: str = DEFAULT_BIND_HOST
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY
    queue_timeout: int = DEFAULT_QUEUE_TIMEOUT
    queue_size: int = DEFAULT_QUEUE_SIZE
    disable_auto_resize: bool = DEFAULT_DISABLE_AUTO_RESIZE
    quantize: int = DEFAULT_QUANTIZE
    config_name: str | None = DEFAULT_CONFIG_NAME
    lora_paths: list[str] | None = field(default=None, init=False)
    lora_scales: list[float] | None = field(default=None, init=False)
    log_file: str | None = DEFAULT_LOG_FILE
    no_log_file: bool = DEFAULT_NO_LOG_FILE
    log_level: str = DEFAULT_LOG_LEVEL
    enable_auto_tool_choice: bool = DEFAULT_ENABLE_AUTO_TOOL_CHOICE
    tool_call_parser: str | None = DEFAULT_TOOL_CALL_PARSER
    reasoning_parser: str | None = DEFAULT_REASONING_PARSER
    trust_remote_code: bool = DEFAULT_TRUST_REMOTE_CODE
    jit_enabled: bool = DEFAULT_JIT_ENABLED
    auto_unload_minutes: int | None = DEFAULT_AUTO_UNLOAD_MINUTES
    name: str | None = None
    group: str | None = DEFAULT_GROUP
    is_default_model: bool = DEFAULT_IS_DEFAULT_MODEL
    enable_status_page: bool = DEFAULT_ENABLE_STATUS_PAGE

    # Used to capture raw CLI input before processing
    lora_paths_str: str | None = DEFAULT_LORA_PATHS_STR
    lora_scales_str: str | None = DEFAULT_LORA_SCALES_STR

    def __post_init__(self) -> None:
        """Normalize certain CLI fields after instantiation.

        This method processes comma-separated LoRA paths and scales into lists,
        applies model-type-specific defaults for config_name, validates auto-unload
        settings, and normalizes log_level.

        Notes
        -----
        - Convert comma-separated ``lora_paths`` and ``lora_scales`` into
          lists when provided.
        - Apply small model-type-specific defaults for ``config_name``
          and emit warnings when values appear inconsistent.
        - Validate that ``auto_unload_minutes`` requires ``jit_enabled`` to be True.
        - Validate that ``auto_unload_minutes`` is positive when set.
        - Normalize ``log_level`` to uppercase.
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
                f"Config name parameter '{self.config_name}' provided but model type is '{self.model_type}'. "
                "Config name is only used with image-generation "
                "and image-edit models.",
            )
        elif self.model_type == "image-generation" and not self.config_name:
            logger.warning(
                "Model type is 'image-generation' but no config name "
                "specified. Using default 'flux-schnell'.",
            )
            self.config_name = "flux-schnell"
        elif self.model_type == "image-edit" and not self.config_name:
            logger.warning(
                "Model type is 'image-edit' but no config name "
                "specified. Using default 'flux-kontext-dev'.",
            )
            self.config_name = "flux-kontext-dev"

        if self.auto_unload_minutes is not None:
            if not self.jit_enabled:
                raise ValueError("Auto-unload requires JIT loading to be enabled")
            if self.auto_unload_minutes <= 0:
                raise ValueError("Auto-unload minutes must be a positive integer")

        if isinstance(self.log_level, str):
            self.log_level = self.log_level.upper()

        if self.name is not None:
            self.name = self.name.strip()
            if not self.name:
                self.name = None

        if self.group is not None:
            self.group = self.group.strip()
            if not self.group:
                self.group = None

    @property
    def model_identifier(self) -> str:
        """Get the appropriate model identifier based on model type.

        For Flux models, we always use model_path (local directory path).
        """
        return self.model_path
