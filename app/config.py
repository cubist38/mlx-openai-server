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

from .core.keep_alive import KeepAlive, parse_keep_alive
from .message_converters import resolve_message_converter_name


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
    served_model_name: str | None = None
    port: int = 8000
    host: str = "0.0.0.0"
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
    log_rotation: str | None = "50 MB"
    log_retention: str | int | None = 5
    enable_auto_tool_choice: bool = False
    tool_call_parser: str | None = None
    reasoning_parser: str | None = None
    message_converter: str | None = None
    trust_remote_code: bool = False
    chat_template_file: str | None = None
    debug: bool = False
    prompt_cache_size: int = 10
    prompt_cache_max_bytes: int = 1 << 63
    prompt_cache_dir: str | None = None
    draft_model_path: str | None = None
    num_draft_tokens: int = 2

    # KV cache quantization
    kv_bits: int | None = None
    kv_group_size: int = 64
    quantized_kv_start: int = 0

    # Continuous-batching concurrency (mirrors --decode-concurrency /
    # --prompt-concurrency / --prefill-step-size in mlx-lm's own server).
    batch_completion_size: int = 32
    batch_prefill_size: int = 8
    batch_prefill_step_size: int = 2048
    disable_batching: bool = False

    # Default sampling parameters (override DEFAULT_* env when set via CLI)
    default_max_tokens: int | None = None
    default_temperature: float | None = None
    default_top_p: float | None = None
    default_top_k: int | None = None
    default_min_p: float | None = None
    default_repetition_penalty: float | None = None
    default_presence_penalty: float | None = None
    default_xtc_probability: float | None = None
    default_xtc_threshold: float | None = None
    default_seed: int | None = None
    default_repetition_context_size: int | None = None

    # Used to capture raw CLI input before processing
    lora_paths_str: str | None = None
    lora_scales_str: str | None = None

    def __post_init__(self) -> None:
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

        # KV cache quantization is only supported for lm and multimodal model types
        if self.kv_bits is not None and self.model_type not in {"lm", "multimodal"}:
            logger.warning(
                "KV cache quantization (--kv-bits) is only supported for model types 'lm' and "
                "'multimodal'. Ignoring KV cache quantization options."
            )
            self.kv_bits = None

        # Speculative decoding (draft model) is only supported for lm model type
        if self.draft_model_path and self.model_type != "lm":
            logger.warning(
                "Draft model / num-draft-tokens are only supported for model type 'lm'. "
                "Ignoring speculative decoding options."
            )
            self.draft_model_path = None
            self.num_draft_tokens = 2

        if self.message_converter is not None:
            self.message_converter = self.message_converter.lower()
        elif self.model_type in {"lm", "multimodal"}:
            self.message_converter = resolve_message_converter_name(
                tool_parser_name=self.tool_call_parser,
                reasoning_parser_name=self.reasoning_parser,
            )

    @property
    def model_identifier(self) -> str:
        """Get the appropriate model identifier based on model type.

        For Flux models, we always use model_path (local directory path).
        """
        return self.model_path

    def to_model_entry_config(self) -> ModelEntryConfig:
        """Convert this single-model CLI config to a ``ModelEntryConfig``.

        This allows ``create_handler_from_config`` to be reused for
        single-model mode, eliminating the duplicated handler
        construction logic.
        """
        return ModelEntryConfig(
            model_path=self.model_path,
            model_type=self.model_type,
            served_model_name=self.served_model_name or self.model_path,
            context_length=self.context_length,
            queue_timeout=self.queue_timeout,
            queue_size=self.queue_size,
            quantize=self.quantize,
            config_name=self.config_name,
            lora_paths=self.lora_paths,
            lora_scales=self.lora_scales,
            disable_auto_resize=self.disable_auto_resize,
            enable_auto_tool_choice=self.enable_auto_tool_choice,
            tool_call_parser=self.tool_call_parser,
            reasoning_parser=self.reasoning_parser,
            message_converter=self.message_converter,
            trust_remote_code=self.trust_remote_code,
            chat_template_file=self.chat_template_file,
            debug=self.debug,
            prompt_cache_size=self.prompt_cache_size,
            prompt_cache_max_bytes=self.prompt_cache_max_bytes,
            prompt_cache_dir=self.prompt_cache_dir,
            draft_model_path=self.draft_model_path,
            num_draft_tokens=self.num_draft_tokens,
            kv_bits=self.kv_bits,
            kv_group_size=self.kv_group_size,
            quantized_kv_start=self.quantized_kv_start,
            batch_completion_size=self.batch_completion_size,
            batch_prefill_size=self.batch_prefill_size,
            batch_prefill_step_size=self.batch_prefill_step_size,
            disable_batching=self.disable_batching,
            # Sampling defaults: the subprocess path reads them off the
            # handler proxy rather than the ``DEFAULT_*`` env vars (which
            # would not be set inside the child process).
            default_max_tokens=self.default_max_tokens,
            default_temperature=self.default_temperature,
            default_top_p=self.default_top_p,
            default_top_k=self.default_top_k,
            default_min_p=self.default_min_p,
            default_repetition_penalty=self.default_repetition_penalty,
            default_presence_penalty=self.default_presence_penalty,
            default_xtc_probability=self.default_xtc_probability,
            default_xtc_threshold=self.default_xtc_threshold,
            default_seed=self.default_seed,
            default_repetition_context_size=self.default_repetition_context_size,
        )

    def to_multi_model_server_config(self) -> MultiModelServerConfig:
        """Wrap this single-model config in a one-entry ``MultiModelServerConfig``.

        Used by :mod:`app.cli` when ``--no-isolate`` is not set (the default),
        so ``mlx-openai-server launch --model-path ...`` goes through the
        ``HandlerProcessProxy`` path — the same subprocess isolation that
        multi-handler YAML mode uses to avoid the MLX Metal command-buffer
        races and ``resource_tracker`` semaphore leaks documented in
        https://github.com/ml-explore/mlx/issues/2457.
        """
        return MultiModelServerConfig(
            models=[self.to_model_entry_config()],
            host=self.host,
            port=self.port,
            log_level=self.log_level,
            log_file=self.log_file,
            no_log_file=self.no_log_file,
            log_rotation=self.log_rotation,
            log_retention=self.log_retention,
        )


# ---------------------------------------------------------------------------
# Multi-model YAML configuration
# ---------------------------------------------------------------------------

VALID_MODEL_TYPES = frozenset(
    {"lm", "multimodal", "image-generation", "image-edit", "embeddings", "whisper"}
)


def _validate_model_name_part(value: str | None, kind: str, model_path: str) -> str | None:
    """Validate a routable name fragment such as a ``version`` or an ``alias``.

    Both end up in the request-routing table, so a blank or whitespace-bearing
    value would produce a name no client could reliably send. A ``version``
    additionally may not contain ``:`` because it is joined to the model name
    with that separator.

    Parameters
    ----------
    value : str | None
        Raw value from the configuration. ``None`` passes through.
    kind : str
        Either ``"version"`` or ``"alias"``, used for the error message and to
        decide whether ``:`` is permitted.
    model_path : str
        Owning model, quoted in the error message.

    Returns
    -------
    str | None
        The stripped value, or ``None`` when nothing was set.

    Raises
    ------
    ValueError
        If the value is not a non-empty string without whitespace.
    """
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        msg = f"Model '{model_path}' has an empty {kind}; it must be a non-empty string."
        raise ValueError(msg)

    stripped = value.strip()
    if any(char.isspace() for char in stripped):
        msg = f"Model '{model_path}' has a {kind} containing whitespace: '{value}'."
        raise ValueError(msg)
    if kind == "version" and ":" in stripped:
        msg = (
            f"Model '{model_path}' has a version containing ':': '{value}'. "
            "The version is appended to the model name with that separator."
        )
        raise ValueError(msg)
    return stripped


@dataclass
class ModelEntryConfig:
    """Configuration for a single model entry in a multi-model YAML config.

    Each entry maps to exactly one handler that will be registered in
    the ``ModelRegistry``.  The ``served_model_name`` defaults to
    ``model_path`` when not set explicitly, giving callers a short
    alias they can use in API requests.

    ``version`` and ``aliases`` add further names that route to the same
    entry.  Declaring ``version`` makes the model reachable as
    ``<served_model_name>:<version>`` in addition to its bare name, so a
    client can pin a checkpoint while another keeps using the stable name.
    ``aliases`` lists arbitrary extra names, which is how a bare base name
    is pointed at whichever version is currently the default.
    """

    model_path: str
    model_type: str = "lm"
    served_model_name: str | None = None

    # Versioning / aliasing
    version: str | None = None
    aliases: list[str] | None = None

    # Common options
    context_length: int | None = None
    queue_timeout: int = 300
    queue_size: int = 100

    # Image-generation / image-edit options
    quantize: int | None = None
    config_name: str | None = None

    # LoRA options
    lora_paths: list[str] | None = None
    lora_scales: list[float] | None = None

    # On-demand (dynamic swapping) options
    on_demand: bool = False
    on_demand_idle_timeout: KeepAlive = 300

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
    prompt_cache_max_bytes: int = 1 << 63
    prompt_cache_dir: str | None = None
    draft_model_path: str | None = None
    num_draft_tokens: int = 2
    kv_bits: int | None = None
    kv_group_size: int = 64
    quantized_kv_start: int = 0
    batch_completion_size: int = 32
    batch_prefill_size: int = 8
    batch_prefill_step_size: int = 2048
    disable_batching: bool = False
    default_max_tokens: int | None = None
    default_temperature: float | None = None
    default_top_p: float | None = None
    default_top_k: int | None = None
    default_min_p: float | None = None
    default_repetition_penalty: float | None = None
    default_presence_penalty: float | None = None
    default_xtc_probability: float | None = None
    default_xtc_threshold: float | None = None
    default_seed: int | None = None
    default_repetition_context_size: int | None = None

    def __post_init__(self) -> None:
        """Resolve ``served_model_name`` and validate ``model_type``."""
        if self.served_model_name is None:
            self.served_model_name = self.model_path

        self.version = _validate_model_name_part(self.version, "version", self.model_path)
        if self.aliases is not None:
            self.aliases = [
                _validate_model_name_part(alias, "alias", self.model_path) for alias in self.aliases
            ]

        if self.model_type not in VALID_MODEL_TYPES:
            msg = (
                f"Invalid model_type '{self.model_type}' for model '{self.model_path}'. "
                f"Must be one of {sorted(VALID_MODEL_TYPES)}."
            )
            raise ValueError(msg)

        parse_keep_alive(self.on_demand_idle_timeout, 300.0)

        # Apply image-generation / image-edit defaults (same as MLXServerConfig)
        if self.model_type == "image-generation" and not self.config_name:
            logger.warning(
                "Model '%s' (image-generation) has no config_name. Defaulting to 'flux-schnell'.",
                self.model_path,
            )
            self.config_name = "flux-schnell"
        elif self.model_type == "image-edit" and not self.config_name:
            logger.warning(
                "Model '%s' (image-edit) has no config_name. Defaulting to 'flux-kontext-dev'.",
                self.model_path,
            )
            self.config_name = "flux-kontext-dev"

        # KV cache quantization is LM/multimodal-only
        if self.kv_bits is not None and self.model_type not in {"lm", "multimodal"}:
            logger.warning(
                "KV cache quantization is only supported for 'lm' and 'multimodal'. "
                "Ignoring for model '%s'.",
                self.model_path,
            )
            self.kv_bits = None

        # Speculative decoding is LM-only
        if self.draft_model_path and self.model_type != "lm":
            logger.warning(
                "Draft model is only supported for 'lm'. Ignoring for model '%s'.",
                self.model_path,
            )
            self.draft_model_path = None
            self.num_draft_tokens = 2

        if self.message_converter is not None:
            self.message_converter = self.message_converter.lower()
        elif self.model_type in {"lm", "multimodal"}:
            self.message_converter = resolve_message_converter_name(
                tool_parser_name=self.tool_call_parser,
                reasoning_parser_name=self.reasoning_parser,
            )

    def alias_names(self) -> list[str]:
        """Return every extra name that should route to this model.

        The list combines the explicit ``aliases`` with the implicit
        ``<served_model_name>:<version>`` form. Order is preserved and the
        canonical name is excluded, so the result can be registered as-is.

        Returns
        -------
        list[str]
            Deduplicated alias names, excluding ``served_model_name``.
        """
        names: list[str] = []
        if self.version:
            names.append(f"{self.served_model_name}:{self.version}")
        names.extend(self.aliases or [])

        seen: set[str] = set()
        return [
            name
            for name in names
            if name != self.served_model_name and not (name in seen or seen.add(name))
        ]


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
    log_rotation: str | None = "50 MB"
    log_retention: str | int | None = 5
    max_loaded_models: int = 1
    model_load_timeout: float = 300.0

    def __post_init__(self) -> None:
        """Validate server-level model lifecycle options."""
        if self.max_loaded_models < 0:
            raise ValueError("max_loaded_models must be greater than or equal to zero")
        if self.model_load_timeout <= 0:
            raise ValueError("model_load_timeout must be greater than zero")


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

        # Enforce unique served_model_name values
        if model_cfg.served_model_name in seen_ids:
            msg = (
                f"Duplicate served_model_name '{model_cfg.served_model_name}' in config. "
                "Each model must have a unique served_model_name."
            )
            raise ValueError(msg)
        seen_ids.add(model_cfg.served_model_name)

        # Aliases share the routing namespace with served_model_name, so a
        # collision would silently make one of the two names unreachable.
        for alias in model_cfg.alias_names():
            if alias in seen_ids:
                msg = (
                    f"Alias '{alias}' for model '{model_cfg.served_model_name}' collides with "
                    "another model name or alias in config. Model names and aliases must be "
                    "unique."
                )
                raise ValueError(msg)
            seen_ids.add(alias)

        model_entries.append(model_cfg)

    return MultiModelServerConfig(
        models=model_entries,
        host=server_raw.get("host", "0.0.0.0"),
        port=server_raw.get("port", 8000),
        log_level=server_raw.get("log_level", "INFO"),
        log_file=server_raw.get("log_file"),
        no_log_file=server_raw.get("no_log_file", False),
        log_rotation=server_raw.get("log_rotation", "50 MB"),
        log_retention=server_raw.get("log_retention", 5),
        max_loaded_models=server_raw.get("max_loaded_models", 1),
        model_load_timeout=server_raw.get("model_load_timeout", 300.0),
    )
