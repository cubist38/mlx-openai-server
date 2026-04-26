from __future__ import annotations

from collections.abc import Generator, Iterable
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

from loguru import logger
import mlx.core as mx
from mlx_lm.generate import GenerationResponse, stream_generate
from mlx_lm.models.cache import can_trim_prompt_cache, make_prompt_cache
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from mlx_lm.utils import _download, load
from outlines.processors import JSONLogitsProcessor

from ..utils.debug_logging import log_debug_chat_template
from ..utils.outlines_transformer_tokenizer import OutlinesTransformerTokenizer

DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))
DEFAULT_TOP_P = float(os.getenv("DEFAULT_TOP_P", "0.95"))
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "20"))
DEFAULT_MIN_P = float(os.getenv("DEFAULT_MIN_P", "0.0"))
DEFAULT_XTC_PROBABILITY = float(os.getenv("DEFAULT_XTC_PROBABILITY", "0.0"))
DEFAULT_XTC_THRESHOLD = float(os.getenv("DEFAULT_XTC_THRESHOLD", "0.0"))
DEFAULT_SEED = int(os.getenv("DEFAULT_SEED", "0"))
DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", "1000000"))
DEFAULT_REPETITION_PENALTY = float(os.getenv("DEFAULT_REPETITION_PENALTY", "0.0"))
DEFAULT_REPETITION_CONTEXT_SIZE = int(os.getenv("DEFAULT_REPETITION_CONTEXT_SIZE", "20"))
DEFAULT_PRESENCE_PENALTY = float(os.getenv("DEFAULT_PRESENCE_PENALTY", "0.0"))
DEFAULT_FREQUENCY_PENALTY = float(os.getenv("DEFAULT_FREQUENCY_PENALTY", "0.0"))


def _as_int_set(values: Any) -> set[int]:
    """Coerce a tokenizer EOS field into a clean set of token IDs.

    Parameters
    ----------
    values : Any
        A token ID, sequence of token IDs, or ``None``.

    Returns
    -------
    set[int]
        Integer token IDs with ``None`` values removed.
    """
    if values is None:
        return set()
    if isinstance(values, int):
        return {values}
    if isinstance(values, Iterable) and not isinstance(values, str):
        ids: set[int] = set()
        for value in values:
            if value is not None:
                ids.add(int(value))
        return ids
    return {int(values)}


def _load_generation_config(model_path: str) -> dict[str, Any]:
    """Load model-authored generation defaults from ``generation_config.json``.

    Parameters
    ----------
    model_path : str
        Local model directory or Hugging Face repository name.

    Returns
    -------
    dict[str, Any]
        Parsed generation configuration, or an empty dictionary when the file
        is absent or unreadable.
    """
    try:
        resolved_model_path = Path(_download(model_path))
    except (FileNotFoundError, ValueError, OSError) as exc:
        logger.debug(f"Could not resolve model path for generation_config.json: {exc!s}")
        resolved_model_path = Path(model_path)

    generation_config_path = resolved_model_path / "generation_config.json"
    if not generation_config_path.exists():
        return {}

    try:
        with generation_config_path.open(encoding="utf-8") as file:
            config = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(f"Failed to read {generation_config_path}: {exc!s}")
        return {}

    if not isinstance(config, dict):
        return {}
    return config


@dataclass
class CompletionResponse:
    """
    The output of :func:`__call__` when stream is False.

    Args:
        text (str): The next segment of decoded text. This can be an empty string.
        tokens (List[int]): The list of tokens in the response.
        peak_memory (float): The peak memory used so far in GB.
        generation_tps (float): The tokens-per-second for generation.
        generation_tokens (int): The number of generated tokens.
        prompt_tps (float): The prompt processing tokens-per-second.
        prompt_tokens (int): The number of tokens in the prompt.
        cached_prompt_tokens (int): The number of prompt tokens that were
            served from a pre-computed cache. ``0`` when no cache hit — only
            the batched path currently populates this; the non-batched path
            reports its cache hits via ``ctx.total_cached_tokens``.
    """

    text: str = None
    tokens: list[int] = None
    peak_memory: float = None
    generation_tps: float = None
    prompt_tps: float = None
    prompt_tokens: int = None
    generation_tokens: int = None
    cached_prompt_tokens: int = 0


class MLX_LM:
    """
    A wrapper class for MLX Language Model that handles both streaming and non-streaming inference.

    This class provides a unified interface for generating text responses from text prompts,
    supporting both streaming and non-streaming modes.
    """

    def __init__(
        self,
        model_path: str,
        draft_model_path: str = None,
        num_draft_tokens: int = 2,
        context_length: int | None = None,
        trust_remote_code: bool = False,
        chat_template_file: str = None,
        debug: bool = False,
    ):
        try:
            self.generation_config = _load_generation_config(model_path)
            self.model, self.tokenizer, model_config = load(
                model_path,
                lazy=False,
                tokenizer_config={"trust_remote_code": trust_remote_code},
                return_config=True,
            )
            self.context_length = context_length
            self.draft_model = None
            self.draft_tokenizer = None
            self.num_draft_tokens = num_draft_tokens
            if draft_model_path:
                self._load_draft_model(draft_model_path, trust_remote_code)
            self.pad_token_id = self.tokenizer.pad_token_id
            self.bos_token = self.tokenizer.bos_token
            self._normalize_eos_token_ids(model_config)
            self.model_type = self.model.model_type
            self.debug = debug
            self.outlines_tokenizer = OutlinesTransformerTokenizer(self.tokenizer)
            initial_cache = make_prompt_cache(self.model)
            self._cache_is_trimmable = can_trim_prompt_cache(initial_cache)
            self._num_model_cache_layers = len(initial_cache)
            # ``BatchGenerator`` requires every cache layer to expose a
            # ``merge`` method so sequences can share the batch. Models
            # whose caches don't (e.g. some third-party hybrid / SSM
            # variants that ship their own cache type) must fall back to
            # the single-request path. Mirrors ``mlx_lm.server``'s own
            # ``is_batchable`` gate.
            self._cache_is_batchable = all(hasattr(c, "merge") for c in initial_cache)
            if chat_template_file:
                if not os.path.exists(chat_template_file):
                    raise ValueError(f"Chat template file {chat_template_file} does not exist")
                with open(chat_template_file) as f:
                    template_content = f.read()
                    self.tokenizer.chat_template = template_content
                if self.debug:
                    log_debug_chat_template(
                        chat_template_file=chat_template_file, template_content=template_content
                    )
        except Exception as e:
            raise ValueError(f"Error loading model: {e!s}")

    def _normalize_eos_token_ids(self, model_config: dict[str, Any]) -> None:
        """Populate tokenizer EOS IDs strictly from model metadata.

        Parameters
        ----------
        model_config : dict[str, Any]
            The model configuration returned by ``mlx_lm.utils.load``. mlx-lm
            already folds ``generation_config.json``'s ``eos_token_id`` into
            this mapping, so this preserves the model author's stop-token list
            without guessing extra chat terminators.
        """
        eos_ids = _as_int_set(getattr(self.tokenizer, "eos_token_ids", None))
        eos_ids.update(_as_int_set(getattr(self.tokenizer, "eos_token_id", None)))
        eos_ids.update(_as_int_set(model_config.get("eos_token_id")))
        eos_ids.update(_as_int_set(self.generation_config.get("eos_token_id")))

        if eos_ids:
            self.tokenizer.eos_token_ids = eos_ids
            logger.debug(f"Using EOS token IDs: {sorted(eos_ids)}")

    def _load_draft_model(self, draft_model_path: str, trust_remote_code: bool) -> None:
        try:
            self.draft_model, self.draft_tokenizer = load(
                draft_model_path,
                lazy=False,
                tokenizer_config={"trust_remote_code": trust_remote_code},
            )
            self.context_length = (
                None  # speculative decoding does not support context length, should be set to None
            )
            self._validate_draft_tokenizer()
        except Exception as e:
            raise ValueError(f"Error loading draft model: {e!s}")

    def _validate_draft_tokenizer(self) -> None:
        if self.draft_tokenizer.vocab_size != self.tokenizer.vocab_size:
            logger.warning(
                "Draft model tokenizer does not match model tokenizer. "
                "Speculative decoding may not work as expected."
            )

    def create_prompt_cache(self) -> list[Any]:
        cache = make_prompt_cache(self.model, max_kv_size=self.context_length)
        if self.draft_model:
            cache += make_prompt_cache(self.draft_model, max_kv_size=self.context_length)
        return cache

    def get_model_type(self) -> str:
        return self.model_type

    def create_input_prompt(
        self, messages: list[dict[str, str]], chat_template_kwargs: dict[str, Any]
    ) -> str:
        use_partial = chat_template_kwargs.pop("_partial_mode", False)

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=not use_partial,
            continue_final_message=use_partial,
            **chat_template_kwargs,
        )

    def encode_prompt(self, input_prompt: str) -> list[int]:
        """Encode a prompt string into token IDs.

        Parameters
        ----------
        input_prompt : str
            The prompt string to encode.

        Returns
        -------
        list[int]
            Token IDs for the prompt.
        """
        return self.tokenizer.encode(input_prompt)

    @property
    def cache_is_trimmable(self) -> bool:
        """Whether the model's prompt cache supports trimming.

        Pure-attention models return ``True``; hybrid models with
        ``ArraysCache`` (SSM/recurrent layers) return ``False``.
        """
        return self._cache_is_trimmable

    @property
    def cache_is_batchable(self) -> bool:
        """Whether the model's prompt cache supports cross-sequence merge.

        The continuous batcher merges KV caches from multiple concurrent
        sequences into a single batched tensor; models whose caches don't
        expose a ``merge`` method can't be batched and must use the
        single-request path.
        """
        return self._cache_is_batchable

    @property
    def has_draft_model(self) -> bool:
        """Whether a draft model is configured (speculative decoding is active)."""
        return self.draft_model is not None

    def _sampling_default(self, key: str, fallback: Any) -> Any:
        """Return a generation default from model metadata or a fallback.

        Parameters
        ----------
        key : str
            Sampling parameter name from ``generation_config.json``.
        fallback : Any
            Built-in value used when the model file does not define ``key``.

        Returns
        -------
        Any
            The model-authored generation default when present, otherwise
            ``fallback``.
        """
        value = self.generation_config.get(key)
        return fallback if value is None else value

    def resolve_max_tokens(self, params: dict[str, Any]) -> int:
        """Resolve max generation tokens using request, model, then server defaults.

        Parameters
        ----------
        params : dict[str, Any]
            Request model parameters.

        Returns
        -------
        int
            Maximum number of tokens to generate.
        """
        max_tokens = params.get("max_tokens")
        if max_tokens is None:
            max_tokens = params.get("max_completion_tokens")
        if max_tokens is None:
            max_tokens = self._sampling_default("max_tokens", None)
        if max_tokens is None:
            max_tokens = self._sampling_default("max_completion_tokens", DEFAULT_MAX_TOKENS)
        return int(max_tokens)

    def build_sampler(self, params: dict[str, Any]):
        """Build a sampler callable from a request's model parameters.

        Parameters
        ----------
        params : dict
            Parameter mapping with optional ``temperature``, ``top_p``,
            ``top_k``, ``min_p``, ``xtc_probability``, ``xtc_threshold`` keys.
            Missing or ``None`` values fall back to the same defaults used by
            the non-batched generation path.

        Returns
        -------
        Callable
            Sampler function suitable for ``BatchGenerator.insert(samplers=...)``.
        """

        def _get(key: str, default: Any) -> Any:
            v = params.get(key)
            return self._sampling_default(key, default) if v is None else v

        sampler_kwargs: dict[str, Any] = {
            "temp": _get("temperature", DEFAULT_TEMPERATURE),
            "top_p": _get("top_p", DEFAULT_TOP_P),
            "top_k": _get("top_k", DEFAULT_TOP_K),
            "min_p": _get("min_p", DEFAULT_MIN_P),
            "xtc_probability": _get("xtc_probability", DEFAULT_XTC_PROBABILITY),
            "xtc_threshold": _get("xtc_threshold", DEFAULT_XTC_THRESHOLD),
        }
        if sampler_kwargs["xtc_probability"] > 0:
            sampler_kwargs["xtc_special_tokens"] = [
                self.tokenizer.eos_token_id,
                *self.tokenizer.encode("\n"),
            ]
        return make_sampler(**sampler_kwargs)

    def build_logits_processors(self, params: dict[str, Any]) -> list[Any]:
        """Build a per-request logits-processor list from model parameters.

        Handles ``logit_bias``, ``repetition_penalty`` /
        ``repetition_context_size``, OpenAI-compatible presence/frequency
        penalties, and optional outlines-based JSON-schema constrained
        decoding (``schema`` key).
        """

        def _penalty_value(key: str, default: float) -> float | None:
            value = params.get(key)
            if value is None:
                value = self._sampling_default(key, default)
            return None if value == 0 else value

        logit_bias = params.get("logit_bias")
        if logit_bias:
            logit_bias = {int(k): v for k, v in logit_bias.items()}

        repetition_penalty = _penalty_value("repetition_penalty", DEFAULT_REPETITION_PENALTY)
        presence_penalty = _penalty_value("presence_penalty", DEFAULT_PRESENCE_PENALTY)
        frequency_penalty = _penalty_value("frequency_penalty", DEFAULT_FREQUENCY_PENALTY)
        repetition_context_size = params.get("repetition_context_size")
        if repetition_context_size is None:
            repetition_context_size = DEFAULT_REPETITION_CONTEXT_SIZE

        processors: list[Any] = list(
            make_logits_processors(
                logit_bias=logit_bias,
                repetition_penalty=repetition_penalty,
                repetition_context_size=repetition_context_size,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
            )
        )

        schema = params.get("schema")
        if schema:
            processors.append(
                JSONLogitsProcessor(
                    schema=schema,
                    tokenizer=self.outlines_tokenizer,
                    tensor_library_name="mlx",
                )
            )
        return processors

    def _prefill_cache(
        self,
        token_ids: list[int],
        prompt_cache: list[Any],
        prefill_step_size: int = 2048,
    ) -> None:
        """Process tokens through the model to warm up the prompt cache.

        Parameters
        ----------
        token_ids : list[int]
            Token IDs to prefill into the cache.
        prompt_cache : list[Any]
            Prompt cache to update in-place.
        prefill_step_size : int, optional
            Maximum chunk size per forward pass, by default 2048.
        """
        tokens = mx.array(token_ids)
        n_model = self._num_model_cache_layers
        model_cache = prompt_cache[:n_model]

        remaining = tokens
        while remaining.size > 0:
            chunk = remaining[:prefill_step_size]
            self.model(chunk[None], cache=model_cache)
            mx.eval([c.state for c in model_cache])
            remaining = remaining[prefill_step_size:]
            if remaining.size > 0:
                mx.clear_cache()

        if self.draft_model:
            draft_cache = prompt_cache[n_model:]
            remaining = tokens
            while remaining.size > 0:
                chunk = remaining[:prefill_step_size]
                self.draft_model(chunk[None], cache=draft_cache)
                mx.eval([c.state for c in draft_cache])
                remaining = remaining[prefill_step_size:]
                if remaining.size > 0:
                    mx.clear_cache()

    def __call__(
        self, input_ids: list[int], prompt_cache: list[Any] = None, stream: bool = False, **kwargs
    ) -> CompletionResponse | Generator[GenerationResponse, None, None]:
        """Generate text response from the model.

        Parameters
        ----------
        input_ids : list[int]
            Token IDs for the input prompt.
        prompt_cache : list[Any], optional
            Pre-computed prompt cache for faster inference.
        stream : bool, optional
            Whether to stream the response, by default ``False``.
        **kwargs
            Additional generation parameters (temperature, max_tokens, etc.)
            and optional checkpoint control parameters:

            - ``checkpoint_position`` (int | None): Token index at which to
              split prefill and save a cache checkpoint.
            - ``checkpoint_callback`` (callable | None): Called with the
              prompt cache after processing the prefix so the caller can
              persist a checkpoint.

        Returns
        -------
        CompletionResponse | Generator[GenerationResponse, None, None]
            Complete response or streaming generator.
        """
        checkpoint_position: int | None = kwargs.pop("checkpoint_position", None)
        checkpoint_callback = kwargs.pop("checkpoint_callback", None)

        if (
            checkpoint_position is not None
            and checkpoint_callback is not None
            and prompt_cache is not None
            and 0 < checkpoint_position < len(input_ids)
        ):
            self._prefill_cache(input_ids[:checkpoint_position], prompt_cache)
            checkpoint_callback(prompt_cache)
            input_ids = input_ids[checkpoint_position:]

        def _get(key, default):
            v = kwargs.get(key)
            return self._sampling_default(key, default) if v is None else v

        seed = _get("seed", DEFAULT_SEED)
        max_tokens = self.resolve_max_tokens(kwargs)
        sampler_kwargs = {
            "temp": _get("temperature", DEFAULT_TEMPERATURE),
            "top_p": _get("top_p", DEFAULT_TOP_P),
            "top_k": _get("top_k", DEFAULT_TOP_K),
            "min_p": _get("min_p", DEFAULT_MIN_P),
            "xtc_probability": _get("xtc_probability", DEFAULT_XTC_PROBABILITY),
            "xtc_threshold": _get("xtc_threshold", DEFAULT_XTC_THRESHOLD),
        }

        # Add XTC special tokens (EOS and newline) when XTC is enabled
        if sampler_kwargs["xtc_probability"] > 0:
            sampler_kwargs["xtc_special_tokens"] = [
                self.tokenizer.eos_token_id,
                *self.tokenizer.encode("\n"),
            ]

        repetition_penalty = _get("repetition_penalty", DEFAULT_REPETITION_PENALTY)
        if repetition_penalty == 0:
            repetition_penalty = None
        presence_penalty = _get("presence_penalty", DEFAULT_PRESENCE_PENALTY)
        if presence_penalty == 0:
            presence_penalty = None
        frequency_penalty = _get("frequency_penalty", DEFAULT_FREQUENCY_PENALTY)
        if frequency_penalty == 0:
            frequency_penalty = None
        repetition_context_size = _get("repetition_context_size", DEFAULT_REPETITION_CONTEXT_SIZE)
        logit_bias = kwargs.get("logit_bias")

        # Convert string keys to int if logit_bias is provided (OpenAI API uses string keys)
        if logit_bias:
            logit_bias = {int(k): v for k, v in logit_bias.items()}

        logits_processors = make_logits_processors(
            logit_bias=logit_bias,
            repetition_penalty=repetition_penalty,
            repetition_context_size=repetition_context_size,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
        )

        json_schema = kwargs.get("schema")
        if json_schema:
            logits_processors.append(
                JSONLogitsProcessor(
                    schema=json_schema, tokenizer=self.outlines_tokenizer, tensor_library_name="mlx"
                )
            )

        # Only seed RNG when an explicit non-negative seed is provided
        # None or negative values (e.g., -1) result in non-deterministic generation
        if seed and seed >= 0:
            mx.random.seed(seed)

        prompt_progress_callback = kwargs.get("prompt_progress_callback")

        sampler = make_sampler(**sampler_kwargs)

        kv_bits = kwargs.get("kv_bits")
        kv_group_size = kwargs.get("kv_group_size", 64)
        quantized_kv_start = kwargs.get("quantized_kv_start", 0)

        stream_response = stream_generate(
            self.model,
            self.tokenizer,
            input_ids,
            draft_model=self.draft_model,
            sampler=sampler,
            max_tokens=max_tokens,
            num_draft_tokens=self.num_draft_tokens,
            prompt_cache=prompt_cache,
            logits_processors=logits_processors,
            prompt_progress_callback=prompt_progress_callback,
            kv_bits=kv_bits,
            kv_group_size=kv_group_size,
            quantized_kv_start=quantized_kv_start,
        )
        if stream:
            return stream_response

        text = ""
        tokens = []
        final_chunk = None
        for chunk in stream_response:
            text += chunk.text
            tokens.append(chunk.token)
            if chunk.finish_reason:
                final_chunk = chunk

        return CompletionResponse(
            text=text,
            tokens=tokens,
            peak_memory=final_chunk.peak_memory,
            generation_tps=final_chunk.generation_tps,
            prompt_tps=final_chunk.prompt_tps,
            prompt_tokens=final_chunk.prompt_tokens,
            generation_tokens=final_chunk.generation_tokens,
        )
