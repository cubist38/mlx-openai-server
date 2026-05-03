import asyncio
from collections.abc import AsyncGenerator
import copy
from dataclasses import dataclass
import gc
from http import HTTPStatus
import threading
import time
from typing import Any

from fastapi import HTTPException
from loguru import logger

from ..core import BatchScheduler, InferenceWorker
from ..core.batch_scheduler import BATCHING_AVAILABLE
from ..message_converters import MessageConverterManager
from ..parsers import ParserManager, ReasoningParserState, ToolParserState
from ..schemas.openai import ChatCompletionRequest, PromptTokenUsageInfo, UsageInfo
from ..utils.debug_logging import (
    log_debug_cache_stats,
    log_debug_model_dispatch,
    log_debug_parser_event,
    log_debug_prompt,
    log_debug_raw_text_response,
    log_debug_request,
    log_debug_stats,
    log_debug_tool_call_emission,
    make_prompt_progress_callback,
)
from ..utils.errors import create_error_response


def _coerce_cached_tokens(ctx_cached: int, source: Any) -> int:
    """Return ``cached_tokens`` for the OpenAI usage payload.

    The batched path reports its LRU hit count via
    ``source.cached_prompt_tokens`` (populated on the scheduler thread); the
    non-batched path reports it via ``ctx.total_cached_tokens`` computed
    during ``_build_inference_context``. We take whichever is larger so
    either path's value flows through, and defensively coerce non-int values
    (some unit tests pass ``Mock()``) back to ``0``.
    """
    batched = getattr(source, "cached_prompt_tokens", 0)
    if not isinstance(batched, int):
        batched = 0
    return max(ctx_cached, batched)


def _strip_complete_tool_blocks(text: str, tool_open: str, tool_close: str) -> str:
    """Remove fully formed tool-call blocks while preserving surrounding literal text.

    Parameters
    ----------
    text : str
        Raw model output text.
    tool_open : str
        Tool-call opening marker (for example ``<tool_call>``).
    tool_close : str
        Tool-call closing marker (for example ``</tool_call>``).

    Returns
    -------
    str
        Input text with complete tool-call blocks removed.
    """
    if not text or tool_open not in text:
        return text

    pieces: list[str] = []
    cursor = 0
    while True:
        open_idx = text.find(tool_open, cursor)
        if open_idx == -1:
            pieces.append(text[cursor:])
            break

        pieces.append(text[cursor:open_idx])
        close_idx = text.find(tool_close, open_idx + len(tool_open))
        if close_idx == -1:
            # Keep trailing malformed fragments as literal content.
            pieces.append(text[open_idx:])
            break

        cursor = close_idx + len(tool_close)

    return "".join(pieces)


@dataclass
class _InferenceContext:
    """Pre-processed inference state shared by stream and non-stream paths."""

    rest_input_ids: list[int]
    cache: list[Any]
    cache_key: list[int]
    total_input_tokens: int
    total_cached_tokens: int
    model_params: dict[str, Any]
    parsers_result: Any
    prompt_progress_callback: Any = None
    checkpoint_position: int | None = None
    checkpoint_callback: Any = None
    # Optional segmentation used by the batched path for non-trimmable cache
    # models: splits ``rest_input_ids`` at a useful boundary so the scheduler
    # can save a prefix checkpoint during prefill.
    batched_segments: list[list[int]] | None = None
    batched_segment_types: list[str] | None = None


class MLXLMHandler:
    """
    Handler class for making requests to the underlying MLX text-only language model service.
    Provides request queuing, metrics tracking, and robust error handling.
    """

    handler_type: str = "lm"

    def __init__(
        self,
        model_path: str,
        draft_model_path: str | None = None,
        num_draft_tokens: int = 2,
        context_length: int | None = None,
        enable_auto_tool_choice: bool = False,
        tool_call_parser: str = None,
        reasoning_parser: str = None,
        message_converter: str = None,
        trust_remote_code: bool = False,
        chat_template_file: str = None,
        debug: bool = False,
        prompt_cache_size: int = 10,
        prompt_cache_max_bytes: int = 1 << 63,
        prompt_cache_dir: str | None = None,
        kv_bits: int | None = None,
        kv_group_size: int = 64,
        quantized_kv_start: int = 0,
        batch_completion_size: int = 32,
        batch_prefill_size: int = 8,
        batch_prefill_step_size: int = 2048,
        batch_max_kv_size: int | None = None,
        disable_batching: bool = False,
    ):
        """
        Initialize the handler with the specified model path.

        Parameters
        ----------
        model_path : str
            Path to the model directory.
        draft_model_path : str | None
            Path to the draft model for speculative decoding. If None, speculative decoding is disabled.
        num_draft_tokens : int
            Number of draft tokens per step when using speculative decoding. Default is 2.
        context_length : int | None
            Maximum context length for the model. If None, uses model default.
        enable_auto_tool_choice : bool
            Enable automatic tool choice.
        tool_call_parser : str | None
            Name of the tool call parser to use (qwen3, glm4_moe, harmony, minimax, ...).
        reasoning_parser : str | None
            Name of the reasoning parser to use (qwen3, qwen3_next, glm4_moe, harmony, minimax, ...).
        message_converter : str | None
            Name of the message converter to use.
        trust_remote_code : bool
            Enable trust_remote_code when loading models.
        chat_template_file : str | None
            Path to a custom chat template file.
        debug : bool
            Enable debug mode.
        prompt_cache_size : int
            Maximum number of prompt KV cache entries to store. Default is 10.
        prompt_cache_max_bytes : int
            Maximum total bytes retained by prompt KV caches before eviction.
        prompt_cache_dir : str | None
            Directory used for disk-backed prompt KV cache payloads. If None,
            a process-local temporary directory is used.
        kv_bits : int | None
            Number of bits for KV cache quantization. None disables quantization.
        kv_group_size : int
            Group size for KV cache quantization. Default is 64.
        quantized_kv_start : int
            Step to begin using a quantized KV cache. Default is 0.
        batch_completion_size : int
            Maximum number of concurrent sequences in the decode batch.
        batch_prefill_size : int
            Maximum number of sequences that can be prefilled simultaneously.
        batch_prefill_step_size : int
            Maximum tokens processed per prefill step.
        batch_max_kv_size : int | None
            Rotating-KV-cache size for the scheduler; ``None`` keeps full
            history (same default used by ``mlx_lm``).
        disable_batching : bool
            Disable the continuous batch scheduler and use the single-request
            path for LM generation.
        """
        self.model_path = model_path
        from ..models.mlx_lm import MLX_LM
        from ..utils.prompt_cache import LRUPromptCache

        self.model = MLX_LM(
            model_path,
            draft_model_path=draft_model_path,
            num_draft_tokens=num_draft_tokens,
            context_length=context_length,
            trust_remote_code=trust_remote_code,
            chat_template_file=chat_template_file,
            debug=debug,
        )
        self.model_created = int(time.time())  # Store creation time when model is loaded
        self.model_type = self.model.get_model_type()

        # KV cache quantization settings
        self.kv_bits = kv_bits
        self.kv_group_size = kv_group_size
        self.quantized_kv_start = quantized_kv_start

        # Store parser configuration
        self.enable_auto_tool_choice = enable_auto_tool_choice
        # Debug mode
        self.debug = debug
        self.reasoning_parser_name = reasoning_parser
        self.tool_parser_name = tool_call_parser
        self.prompt_cache = LRUPromptCache(
            max_size=prompt_cache_size,
            max_bytes=prompt_cache_max_bytes,
            cache_dir=prompt_cache_dir,
        )
        self.message_converter = MessageConverterManager.create_converter(
            converter_name=message_converter,
            tool_parser_name=tool_call_parser,
            reasoning_parser_name=reasoning_parser,
        )
        # Dedicated inference thread — keeps the event loop free during
        # blocking MLX model computation.
        self.inference_worker = InferenceWorker()
        self._queue_size = 100
        self._queue_timeout = 300.0

        self._batch_completion_size = batch_completion_size
        self._batch_prefill_size = batch_prefill_size
        self._batch_prefill_step_size = batch_prefill_step_size
        self._batch_max_kv_size = batch_max_kv_size
        self._disable_batching = disable_batching
        self._batch_scheduler: BatchScheduler | None = None
        self._batch_scheduler_lock = asyncio.Lock()
        self._generation_lock = threading.RLock()

        logger.info(f"Initialized MLXHandler with model path: {model_path}")

    async def get_models(self) -> list[dict[str, Any]]:
        """
        Get list of available models with their metadata.
        """
        try:
            return [
                {
                    "id": self.model_path,
                    "object": "model",
                    "created": self.model_created,
                    "owned_by": "local",
                }
            ]
        except Exception as e:
            logger.error(f"Error getting models: {e!s}")
            return []

    async def initialize(self, queue_config: dict[str, Any] | None = None) -> None:
        """Initialize the handler and start the inference worker.

        Parameters
        ----------
        queue_config : dict, optional
            Dictionary with ``queue_size`` and ``timeout`` keys used
            to configure the inference worker's internal queue.
        """
        if not queue_config:
            queue_config = {
                "timeout": 300,
                "queue_size": 100,
            }
        self._queue_size = int(queue_config.get("queue_size", 100))
        self._queue_timeout = float(queue_config.get("timeout", 300))
        self.inference_worker = InferenceWorker(
            queue_size=self._queue_size,
            timeout=self._queue_timeout,
        )
        self.inference_worker.start()
        logger.info("Initialized MLXHandler and started inference worker")

    def refine_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Refine the messages to be more suitable for the model.
        """
        if self.message_converter:
            logger.debug("Message converter is enabled, converting messages")
            messages = self.message_converter.convert_messages(messages)

        return [{k: v for k, v in message.items() if v is not None} for message in messages]

    def _generate_with_lock(self, *args: Any, **kwargs: Any) -> Any:
        """Run non-stream generation while serialized with the batch scheduler."""
        with self._generation_lock:
            return self.model(*args, **kwargs)

    def _stream_with_lock(self, *args: Any, **kwargs: Any) -> Any:
        """Stream non-batched generation while serialized with the scheduler."""
        with self._generation_lock:
            yield from self.model(*args, **kwargs)

    def _generate_with_lock_and_cache_persist(
        self,
        cache_key: list[int],
        cache: list[Any] | None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Run non-stream generation and persist its prompt cache on the worker thread."""
        with self._generation_lock:
            response = self.model(*args, **kwargs)
            if cache is not None:
                try:
                    self.prompt_cache.insert_cache(cache_key + response.tokens, cache)
                except Exception as cache_error:  # noqa: BLE001 - cache persistence is best-effort
                    logger.warning(f"Failed to persist prompt cache: {cache_error}")
            return response

    def _stream_with_lock_and_cache_persist(
        self,
        cache_key: list[int],
        cache: list[Any] | None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Stream non-batched generation and persist its cache on the worker thread."""
        with self._generation_lock:
            try:
                for chunk in self.model(*args, **kwargs):
                    if chunk is not None:
                        cache_key.append(chunk.token)
                    yield chunk
            finally:
                if cache is not None:
                    try:
                        self.prompt_cache.insert_cache(cache_key, cache)
                    except Exception as cache_error:  # noqa: BLE001 - cache persistence is best-effort
                        logger.warning(f"Failed to persist prompt cache: {cache_error}")

    def _insert_prompt_cache(
        self,
        cache_key: list[int],
        cache: list[Any],
        *,
        cache_type: str = "assistant",
    ) -> None:
        """Insert a prompt cache while serialized with all generation workers."""
        with self._generation_lock:
            self.prompt_cache.insert_cache(cache_key, cache, cache_type=cache_type)

    def _compute_checkpoint_boundary(
        self,
        messages: list[dict[str, Any]],
        input_ids: list[int],
        chat_template_kwargs: dict[str, Any],
    ) -> int | None:
        """Find the token position where the last user message begins.

        Uses a sentinel substitution technique: replaces the last user message
        with a short dummy string, tokenizes both the original and sentinel
        versions, and compares token-by-token to find the divergence point.

        This avoids issues where ``apply_chat_template(messages[:-1])``
        produces tokens that don't form a prefix of the full prompt due to
        template-specific formatting.

        Parameters
        ----------
        messages : list[dict[str, Any]]
            The refined chat messages.
        input_ids : list[int]
            Token IDs for the full prompt.
        chat_template_kwargs : dict[str, Any]
            Kwargs passed to the chat template (tools, etc.).

        Returns
        -------
        int | None
            Token index where the last user message content begins, or
            ``None`` if no meaningful boundary can be computed.
        """
        if len(messages) < 2:
            return None
        if messages[-1].get("role") != "user":
            return None

        sentinel_messages = [*messages[:-1], {"role": "user", "content": "x"}]
        try:
            sentinel_prompt = self.model.create_input_prompt(
                sentinel_messages, dict(chat_template_kwargs)
            )
            sentinel_ids = self.model.encode_prompt(sentinel_prompt)
        except Exception:
            logger.debug("Could not compute checkpoint boundary via sentinel substitution")
            return None

        common = 0
        for a, b in zip(input_ids, sentinel_ids, strict=False):
            if a != b:
                break
            common += 1

        if common > 0:
            user_msg_length = len(input_ids) - common
            if user_msg_length > 0:
                return common

        return None

    async def _build_inference_context(self, request: ChatCompletionRequest) -> "_InferenceContext":
        """Build the common inference context shared by stream and non-stream paths.

        Handles: request parsing, message refinement, prompt encoding, KV cache
        lookup, parser creation, and model-param preparation.

        For batchable requests the cache-related work (``fetch_nearest_cache``
        + ``create_prompt_cache`` + checkpoint boundary computation) is
        *skipped* on the event-loop thread: those operations touch
        ``mx.array`` state, and doing them here would race with the scheduler
        thread's ``BatchGenerator`` forward passes (producing
        ``AGXG14XFamilyCommandBuffer.tryCoalescingPreviousComputeCommandEncoder``
        assertion failures on Metal). Instead, the scheduler thread owns the
        prompt cache end-to-end for the batched path — matching
        ``mlx_lm.server``'s single-generation-thread architecture.
        """
        chat_messages, model_params = await self._prepare_text_request(request)
        refined_messages = self.refine_messages(chat_messages)
        chat_template_kwargs = model_params.get("chat_template_kwargs", {})

        input_prompt = self.model.create_input_prompt(refined_messages, chat_template_kwargs)
        if self.debug:
            log_debug_prompt(input_prompt)

        input_ids = self.model.encode_prompt(input_prompt)

        if self._is_request_batchable(request):
            parsers_result = ParserManager.create_parsers(
                reasoning_parser_name=self.reasoning_parser_name,
                tool_parser_name=self.tool_parser_name,
            )
            enable_thinking = chat_template_kwargs.get("enable_thinking", True)
            if not enable_thinking and parsers_result.reasoning_parser:
                if parsers_result.reasoning_parser.respects_enable_thinking():
                    parsers_result.reasoning_parser = None
            if model_params.get("schema"):
                parsers_result.reasoning_parser = None
                parsers_result.tool_parser = None
                parsers_result.unified_parser = None

            # For non-trimmable caches (hybrid / SSM / recurrent) the default
            # "longer match trim" path in LRUPromptCache.fetch_nearest_cache
            # is blocked. Split the prompt at the last-user-message boundary
            # so the scheduler can save a ``system`` checkpoint there —
            # subsequent requests with the same prefix will then find a
            # shorter-prefix match without needing to trim. Mirrors the
            # non-batched path's ``checkpoint_callback`` behaviour, and the
            # segment-based cache saves in ``mlx_lm.server``.
            segments: list[list[int]] | None = None
            segment_types: list[str] | None = None
            if not self.model.cache_is_trimmable:
                boundary = self._compute_checkpoint_boundary(
                    refined_messages, input_ids, chat_template_kwargs
                )
                if boundary is None and len(input_ids) > 1:
                    boundary = len(input_ids) - 1
                if boundary is not None and 0 < boundary < len(input_ids):
                    segments = [input_ids[:boundary], input_ids[boundary:]]
                    segment_types = ["system", "assistant"]

            return _InferenceContext(
                rest_input_ids=input_ids,
                cache=None,
                cache_key=input_ids[:],
                total_input_tokens=len(input_ids),
                total_cached_tokens=0,
                model_params=model_params,
                parsers_result=parsers_result,
                prompt_progress_callback=(make_prompt_progress_callback() if self.debug else None),
                checkpoint_position=None,
                checkpoint_callback=None,
                batched_segments=segments,
                batched_segment_types=segment_types,
            )

        with self._generation_lock:
            cache, rest_input_ids = self.prompt_cache.fetch_nearest_cache(
                input_ids,
                allowed_sources={"nonbatch"},
            )
            cache, rest_input_ids = self._normalize_nonbatch_cache_hit(
                input_ids,
                cache,
                rest_input_ids,
            )
            if cache is None:
                cache = self.model.create_prompt_cache()

        # Cache key must be the FULL input_ids, not rest_input_ids.
        # Using rest_input_ids causes memory leaks: on "longer" cache hits,
        # rest_input_ids is a suffix (e.g., [B] from input [A,B]), creating
        # new cache entries [B,X,Y,Z] instead of updating [A,B,X,Y,Z].
        cache_key = input_ids[:]

        checkpoint_position: int | None = None
        checkpoint_callback = None

        # For hybrid models with non-trimmable caches (e.g. Qwen3.5,
        # Nemotron-H, Jamba), the "longer cache trim" path in
        # fetch_nearest_cache is blocked because ArraysCache state
        # cannot be trimmed.  Save a checkpoint at the last-message
        # boundary so subsequent requests with the same prefix can
        # reuse the cached state via the "shorter" trie path.
        #
        # This runs on EVERY request (not just cache misses) so that
        # multi-turn conversations accumulate checkpoints at each new
        # message boundary, rather than only at the first request's.
        if not self.model.cache_is_trimmable:
            boundary = self._compute_checkpoint_boundary(
                refined_messages, input_ids, chat_template_kwargs
            )
            # For single-turn messages _compute_checkpoint_boundary
            # returns None because there is no previous-message
            # boundary.  Fall back to checkpointing all-but-the-last
            # token so the next identical request can reuse the
            # prefill via a "shorter" trie hit.  We keep at least one
            # remaining token so the model's generate() call still
            # receives a non-empty input_ids.
            if boundary is None and len(input_ids) > 1:
                boundary = len(input_ids) - 1

            cached_prefix_len = len(input_ids) - len(rest_input_ids)
            if boundary is not None and boundary > cached_prefix_len:
                # checkpoint_position is relative to rest_input_ids
                # (which is what the model receives as input_ids).
                checkpoint_position = boundary - cached_prefix_len
                prefix_ids = input_ids[:boundary]

                def checkpoint_callback(
                    prompt_cache_state: list[Any],
                    _prefix_ids: list[int] = prefix_ids,
                ) -> None:
                    self._insert_prompt_cache(
                        _prefix_ids,
                        copy.deepcopy(prompt_cache_state),
                        cache_type="system",
                    )

                logger.info(f"Non-trimmable cache: will checkpoint prefix at {boundary} tokens")

        total_input_tokens = len(input_ids)
        total_remaining_tokens = len(rest_input_ids)
        total_cached_tokens = total_input_tokens - total_remaining_tokens

        if self.debug:
            log_debug_cache_stats(total_input_tokens, total_remaining_tokens)

        enable_thinking = chat_template_kwargs.get("enable_thinking", True)

        parsers_result = ParserManager.create_parsers(
            reasoning_parser_name=self.reasoning_parser_name,
            tool_parser_name=self.tool_parser_name,
        )

        if not enable_thinking and parsers_result.reasoning_parser:
            if parsers_result.reasoning_parser.respects_enable_thinking():
                parsers_result.reasoning_parser = None

        if model_params.get("schema"):
            logger.info("JSON schema is enabled, disabling reasoning parser and tool parser")
            parsers_result.reasoning_parser = None
            parsers_result.tool_parser = None
            parsers_result.unified_parser = None

        prompt_progress_callback = make_prompt_progress_callback() if self.debug else None

        return _InferenceContext(
            rest_input_ids=rest_input_ids,
            cache=cache,
            cache_key=cache_key,
            total_input_tokens=total_input_tokens,
            total_cached_tokens=total_cached_tokens,
            model_params=model_params,
            parsers_result=parsers_result,
            prompt_progress_callback=prompt_progress_callback,
            checkpoint_position=checkpoint_position,
            checkpoint_callback=checkpoint_callback,
        )

    def _normalize_nonbatch_cache_hit(
        self,
        input_ids: list[int],
        cache: list[Any] | None,
        rest_input_ids: list[int],
    ) -> tuple[list[Any] | None, list[int]]:
        """Ensure fallback generation always receives at least one prompt token.

        Parameters
        ----------
        input_ids : list[int]
            Full encoded prompt tokens.
        cache : list[Any] | None
            Prompt cache returned by the LRU lookup.
        rest_input_ids : list[int]
            Prompt suffix not covered by ``cache``.

        Returns
        -------
        tuple[list[Any] | None, list[int]]
            A cache and prompt suffix suitable for the single-request
            generator. Exact cache hits are backed off by one token when
            possible; otherwise the cache hit is discarded.
        """
        if cache is None or rest_input_ids:
            return cache, rest_input_ids
        if len(input_ids) <= 1:
            return None, input_ids

        try:
            from mlx_lm.models.cache import can_trim_prompt_cache, trim_prompt_cache
        except ImportError as exc:
            logger.debug(f"Could not import prompt-cache trim helpers: {exc!s}")
        else:
            try:
                if can_trim_prompt_cache(cache):
                    trim_prompt_cache(cache, 1)
                    return cache, input_ids[-1:]
            except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                logger.warning(f"Failed to back off exact prompt-cache hit: {exc!s}")

        try:
            prefix_cache, prefix_rest = self.prompt_cache.fetch_nearest_cache(
                input_ids[:-1],
                allowed_sources={"nonbatch"},
            )
        except (
            AttributeError,
            KeyError,
            IndexError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            logger.warning(f"Failed to find shorter prompt-cache hit: {exc!s}")
        else:
            cached_prefix_len = len(input_ids[:-1]) - len(prefix_rest)
            if prefix_cache is not None and cached_prefix_len < len(input_ids):
                return prefix_cache, input_ids[cached_prefix_len:]

        logger.info(
            "Discarding exact prompt-cache hit because fallback generation needs at least "
            f"one prompt token (prompt_tokens={len(input_ids)})"
        )
        return None, input_ids

    async def generate_text_stream(  # noqa: C901
        self, request: ChatCompletionRequest
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming response for text-only chat completion requests.
        Uses the request queue for handling concurrent requests.

        Args:
            request: ChatCompletionRequest object containing the messages.

        Yields:
            str or dict: Response chunks (str) followed by usage info (dict) at the end.
        """
        try:
            ctx = await self._build_inference_context(request)
            total_input_tokens = ctx.total_input_tokens
            total_cached_tokens = ctx.total_cached_tokens
            model_params = ctx.model_params
            parsers_result = ctx.parsers_result

            request_data = {
                "prompt_progress_callback": ctx.prompt_progress_callback,
                **model_params,
            }
            if ctx.checkpoint_position is not None:
                request_data["checkpoint_position"] = ctx.checkpoint_position
                request_data["checkpoint_callback"] = ctx.checkpoint_callback

            if self.debug:
                debug_request_data = self._with_effective_sampling_params(request_data)
                log_debug_request(debug_request_data)
                log_debug_model_dispatch(
                    "mlx_lm.generate_text_stream.submit_stream",
                    debug_request_data,
                )
                request_data["verbose"] = True

            use_batch = self._is_request_batchable(request) and ctx.checkpoint_position is None
            if use_batch:
                scheduler = await self._get_or_start_scheduler()
                response_generator = self._submit_batched_stream(scheduler, ctx)
            else:
                response_generator = self.inference_worker.submit_stream(
                    self._stream_with_lock_and_cache_persist,
                    list(ctx.cache_key),
                    ctx.cache,
                    input_ids=ctx.rest_input_ids,
                    prompt_cache=ctx.cache,
                    stream=True,
                    **request_data,
                )

            after_reasoning_close_content = None
            final_chunk = None
            is_first_chunk = True
            raw_text = ""  # only use for debugging
            chunk_index = 0

            # Handle unified parser streaming
            if parsers_result.is_unified:
                unified_parser = parsers_result.unified_parser
                async for chunk in response_generator:
                    if chunk is None:
                        continue
                    chunk_index += 1
                    final_chunk = chunk
                    text = chunk.text
                    raw_text += text

                    if unified_parser:
                        if self.debug:
                            log_debug_parser_event(
                                component="mlx_lm.stream.unified",
                                chunk_index=chunk_index,
                                phase="before-parse",
                                parser=unified_parser,
                                text=text,
                            )
                        parsed_result, is_complete = unified_parser.parse_streaming(text)
                        if self.debug:
                            log_debug_parser_event(
                                component="mlx_lm.stream.unified",
                                chunk_index=chunk_index,
                                phase="after-parse",
                                parser=unified_parser,
                                parsed_content=parsed_result,
                                is_complete=is_complete,
                            )
                        if parsed_result:
                            # Unified parser returns dict with reasoning_content, tool_calls, content
                            if parsed_result.get("reasoning_content"):
                                yield {"reasoning_content": parsed_result["reasoning_content"]}
                            if parsed_result.get("tool_calls"):
                                for tool_call in parsed_result["tool_calls"]:
                                    if self.debug:
                                        log_debug_tool_call_emission(
                                            component="mlx_lm.stream.unified",
                                            chunk_index=chunk_index,
                                            tool_call=tool_call,
                                        )
                                    yield tool_call
                            if parsed_result.get("content"):
                                yield parsed_result["content"]
                    else:
                        yield text

                if unified_parser and hasattr(unified_parser, "handle_parse_streaming_end"):
                    parsed_result, is_complete = unified_parser.handle_parse_streaming_end()
                    if parsed_result:
                        # Unified parser returns dict with reasoning_content, tool_calls, content
                        if parsed_result.get("reasoning_content"):
                            yield {"reasoning_content": parsed_result["reasoning_content"]}
                        if parsed_result.get("tool_calls"):
                            for tool_call in parsed_result["tool_calls"]:
                                yield tool_call
                        if parsed_result.get("content"):
                            yield parsed_result["content"]
            else:
                # Handle separate parsers streaming
                reasoning_parser = parsers_result.reasoning_parser
                tool_parser = parsers_result.tool_parser

                async for chunk in response_generator:
                    if chunk is None:
                        continue
                    chunk_index += 1
                    final_chunk = chunk
                    text = chunk.text
                    raw_text += text
                    if is_first_chunk:
                        if reasoning_parser and hasattr(
                            reasoning_parser, "needs_redacted_reasoning_prefix"
                        ):
                            if reasoning_parser.needs_redacted_reasoning_prefix():
                                text = reasoning_parser.get_reasoning_open() + text
                        is_first_chunk = False
                    pending_texts = [text]
                    while pending_texts:
                        text = pending_texts.pop(0)

                        # If a tool tag opened in a previous chunk, finish tool parsing first.
                        if tool_parser and (
                            tool_parser.state != ToolParserState.NORMAL or bool(tool_parser.buffer)
                        ):
                            if self.debug:
                                log_debug_parser_event(
                                    component="mlx_lm.stream.tool",
                                    chunk_index=chunk_index,
                                    phase="before-parse",
                                    parser=tool_parser,
                                    text=text,
                                )
                            parsed_content, is_complete = tool_parser.extract_tool_calls_streaming(
                                text
                            )
                            if self.debug:
                                log_debug_parser_event(
                                    component="mlx_lm.stream.tool",
                                    chunk_index=chunk_index,
                                    phase="after-parse",
                                    parser=tool_parser,
                                    parsed_content=parsed_content,
                                    is_complete=is_complete,
                                )
                            requeue_reasoning_tail = ""
                            if (
                                reasoning_parser
                                and reasoning_parser.state == ReasoningParserState.FOUND_PREFIX
                                and tool_parser.state == ToolParserState.NORMAL
                                and tool_parser.buffer
                            ):
                                requeue_reasoning_tail = tool_parser.buffer
                                tool_parser.buffer = ""

                            if parsed_content:
                                tool_calls = parsed_content.get("tool_calls")
                                if tool_calls:
                                    for tool_call in tool_calls:
                                        if self.debug:
                                            log_debug_tool_call_emission(
                                                component="mlx_lm.stream.tool",
                                                chunk_index=chunk_index,
                                                tool_call=tool_call,
                                            )
                                        yield tool_call
                                content = parsed_content.get("content")
                                if isinstance(content, str) and content:
                                    if requeue_reasoning_tail:
                                        content = f"{content}{requeue_reasoning_tail}"
                                        requeue_reasoning_tail = ""
                                    if (
                                        reasoning_parser
                                        and reasoning_parser.state
                                        == ReasoningParserState.FOUND_PREFIX
                                    ):
                                        pending_texts.insert(0, content)
                                    else:
                                        yield content
                            if requeue_reasoning_tail:
                                pending_texts.insert(0, requeue_reasoning_tail)
                            continue

                        if reasoning_parser:
                            if self.debug:
                                log_debug_parser_event(
                                    component="mlx_lm.stream.reasoning",
                                    chunk_index=chunk_index,
                                    phase="before-parse",
                                    parser=reasoning_parser,
                                    text=text,
                                )
                            parsed_content, is_complete = (
                                reasoning_parser.extract_reasoning_streaming(text)
                            )
                            if self.debug:
                                log_debug_parser_event(
                                    component="mlx_lm.stream.reasoning",
                                    chunk_index=chunk_index,
                                    phase="after-parse",
                                    parser=reasoning_parser,
                                    parsed_content=parsed_content,
                                    is_complete=is_complete,
                                )
                            reasoning_passthrough_for_tool = None
                            if parsed_content:
                                after_reasoning_close_content = parsed_content.get(
                                    "after_reasoning_close_content"
                                )
                                reasoning_content = parsed_content.get("reasoning_content")
                                content_piece = parsed_content.get("content")
                                tool_tail_overlap = False
                                if isinstance(content_piece, str) and tool_parser is not None:
                                    tool_open = tool_parser.get_tool_open()
                                    max_overlap = min(len(content_piece), len(tool_open) - 1)
                                    for overlap_size in range(max_overlap, 0, -1):
                                        if content_piece.endswith(tool_open[:overlap_size]):
                                            tool_tail_overlap = True
                                            break
                                tool_hint_present = isinstance(content_piece, str) and (
                                    "<tool" in content_piece
                                    or "</tool" in content_piece
                                    or "<function=" in content_piece
                                    or "<parameter=" in content_piece
                                    or tool_tail_overlap
                                )

                                # When parser output is pure content in NORMAL state, only
                                # force a tool-parser pass if tool markers are present.
                                if (
                                    isinstance(content_piece, str)
                                    and content_piece
                                    and reasoning_content is None
                                    and after_reasoning_close_content is None
                                    and reasoning_parser.state == ReasoningParserState.NORMAL
                                    and tool_parser is not None
                                    and tool_hint_present
                                ):
                                    reasoning_passthrough_for_tool = content_piece
                                    if reasoning_parser.buffer:
                                        reasoning_passthrough_for_tool += reasoning_parser.buffer
                                        reasoning_parser.buffer = ""
                                else:
                                    yield parsed_content
                            if is_complete:
                                reasoning_parser = None
                            if after_reasoning_close_content:
                                text = after_reasoning_close_content
                                after_reasoning_close_content = None
                            elif reasoning_passthrough_for_tool is not None:
                                text = reasoning_passthrough_for_tool
                            else:
                                continue

                        if tool_parser:
                            if self.debug:
                                log_debug_parser_event(
                                    component="mlx_lm.stream.tool",
                                    chunk_index=chunk_index,
                                    phase="before-parse",
                                    parser=tool_parser,
                                    text=text,
                                )
                            parsed_content, is_complete = tool_parser.extract_tool_calls_streaming(
                                text
                            )
                            if self.debug:
                                log_debug_parser_event(
                                    component="mlx_lm.stream.tool",
                                    chunk_index=chunk_index,
                                    phase="after-parse",
                                    parser=tool_parser,
                                    parsed_content=parsed_content,
                                    is_complete=is_complete,
                                )
                            requeue_reasoning_tail = ""
                            if (
                                reasoning_parser
                                and reasoning_parser.state == ReasoningParserState.FOUND_PREFIX
                                and tool_parser.state == ToolParserState.NORMAL
                                and tool_parser.buffer
                            ):
                                requeue_reasoning_tail = tool_parser.buffer
                                tool_parser.buffer = ""

                            if parsed_content:
                                tool_calls = parsed_content.get("tool_calls")
                                if tool_calls:
                                    for tool_call in tool_calls:
                                        if self.debug:
                                            log_debug_tool_call_emission(
                                                component="mlx_lm.stream.tool",
                                                chunk_index=chunk_index,
                                                tool_call=tool_call,
                                            )
                                        yield tool_call
                                content = parsed_content.get("content")
                                if isinstance(content, str) and content:
                                    if requeue_reasoning_tail:
                                        content = f"{content}{requeue_reasoning_tail}"
                                        requeue_reasoning_tail = ""
                                    if (
                                        reasoning_parser
                                        and reasoning_parser.state
                                        == ReasoningParserState.FOUND_PREFIX
                                    ):
                                        pending_texts.insert(0, content)
                                    else:
                                        yield content
                            if requeue_reasoning_tail:
                                pending_texts.insert(0, requeue_reasoning_tail)
                            continue

                        yield text

            total_tokens = total_input_tokens + final_chunk.generation_tokens
            if self.debug:
                self.prompt_cache.log_cache_stats()
                log_debug_raw_text_response(raw_text)
                log_debug_stats(
                    total_input_tokens,
                    final_chunk.generation_tokens,
                    total_tokens,
                    final_chunk.generation_tps,
                    final_chunk.peak_memory,
                )

            yield {
                "__usage__": UsageInfo(
                    prompt_tokens=total_input_tokens,
                    completion_tokens=final_chunk.generation_tokens,
                    total_tokens=total_tokens,
                    prompt_tokens_details=PromptTokenUsageInfo(
                        cached_tokens=_coerce_cached_tokens(total_cached_tokens, final_chunk)
                    ),
                )
            }

        except asyncio.QueueFull:
            logger.error("Too many requests. Service is at capacity.")
            content = create_error_response(
                "Too many requests. Service is at capacity.",
                "rate_limit_exceeded",
                HTTPStatus.TOO_MANY_REQUESTS,
            )
            raise HTTPException(status_code=429, detail=content)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Error in text stream generation: {e!s}")
            content = create_error_response(
                f"Failed to generate text stream: {e!s}",
                "server_error",
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            raise HTTPException(status_code=500, detail=content)

    async def generate_text_response(self, request: ChatCompletionRequest) -> dict[str, Any]:
        """
        Generate a complete response for text-only chat completion requests.
        Uses the request queue for handling concurrent requests.

        Args:
            request: ChatCompletionRequest object containing the messages.

        Returns:
            dict: Response content and usage info.
        """
        try:
            ctx = await self._build_inference_context(request)
            model_params = ctx.model_params
            parsers_result = ctx.parsers_result
            total_input_tokens = ctx.total_input_tokens
            total_cached_tokens = ctx.total_cached_tokens

            request_data = {
                "prompt_progress_callback": ctx.prompt_progress_callback,
                **model_params,
            }
            if ctx.checkpoint_position is not None:
                request_data["checkpoint_position"] = ctx.checkpoint_position
                request_data["checkpoint_callback"] = ctx.checkpoint_callback

            if self.debug:
                log_debug_model_dispatch(
                    "mlx_lm.generate_text_response.submit",
                    self._with_effective_sampling_params(request_data),
                )

            response = await self._run_nonstream_generation(request, ctx, request_data)

            response_text = response.text

            parsed_response = {"reasoning_content": None, "tool_calls": None, "content": None}

            # Handle unified parser
            if parsers_result.is_unified:
                unified_parser = parsers_result.unified_parser
                if unified_parser:
                    parsed_result = unified_parser.parse(response_text)
                    if self.debug:
                        log_debug_parser_event(
                            component="mlx_lm.nonstream.unified",
                            chunk_index=0,
                            phase="parse",
                            parser=unified_parser,
                            text=response_text,
                            parsed_content=parsed_result,
                            is_complete=True,
                        )
                    if parsed_result:
                        parsed_response["reasoning_content"] = parsed_result.get(
                            "reasoning_content"
                        )
                        parsed_response["tool_calls"] = parsed_result.get("tool_calls")
                        parsed_response["content"] = parsed_result.get("content")
                else:
                    parsed_response["content"] = response_text
            # Handle separate parsers
            elif parsers_result.reasoning_parser or parsers_result.tool_parser:
                reasoning_parser = parsers_result.reasoning_parser
                tool_parser = parsers_result.tool_parser
                synthetic_reasoning_open: str | None = None

                if reasoning_parser and reasoning_parser.needs_redacted_reasoning_prefix():
                    synthetic_reasoning_open = reasoning_parser.get_reasoning_open()
                    response_text = synthetic_reasoning_open + response_text

                if reasoning_parser:
                    parsed_content = reasoning_parser.extract_reasoning(response_text)
                    if self.debug:
                        log_debug_parser_event(
                            component="mlx_lm.nonstream.reasoning",
                            chunk_index=0,
                            phase="parse",
                            parser=reasoning_parser,
                            text=response_text,
                            parsed_content=parsed_content,
                            is_complete=True,
                        )
                    if parsed_content:
                        parsed_response["reasoning_content"] = parsed_content.get(
                            "reasoning_content"
                        )
                        parsed_response["content"] = parsed_content.get("content")
                        # Keep tool parsing active when no explicit reasoning open tag exists
                        # (for example raw output that starts with a stray ``</think>``).
                        response_text = parsed_content.get("after_reasoning_close_content")
                        if response_text is None:
                            response_text = parsed_content.get("content")

                if response_text:
                    if tool_parser:
                        parsed_content = tool_parser.extract_tool_calls(response_text)
                        if self.debug:
                            log_debug_parser_event(
                                component="mlx_lm.nonstream.tool",
                                chunk_index=0,
                                phase="parse",
                                parser=tool_parser,
                                text=response_text,
                                parsed_content=parsed_content,
                                is_complete=True,
                            )
                        parsed_response["tool_calls"] = parsed_content.get("tool_calls")
                        tool_content = parsed_content.get("content")
                        if isinstance(tool_content, str):
                            parsed_response["content"] = tool_content
                        elif parsed_response["tool_calls"]:
                            strip_source = response_text
                            if synthetic_reasoning_open and strip_source.startswith(
                                synthetic_reasoning_open
                            ):
                                strip_source = strip_source[len(synthetic_reasoning_open) :]
                            stripped_content = _strip_complete_tool_blocks(
                                strip_source,
                                tool_parser.get_tool_open(),
                                tool_parser.get_tool_close(),
                            )
                            parsed_response["content"] = stripped_content or None
            else:
                parsed_response["content"] = response_text

            total_tokens = total_input_tokens + response.generation_tokens

            if self.debug and isinstance(parsed_response.get("tool_calls"), list):
                for tool_call in parsed_response["tool_calls"]:
                    if isinstance(tool_call, dict):
                        log_debug_tool_call_emission(
                            component="mlx_lm.nonstream.tool",
                            chunk_index=0,
                            tool_call=tool_call,
                        )

            if self.debug:
                self.prompt_cache.log_cache_stats()
                log_debug_raw_text_response(response.text)
                log_debug_stats(
                    total_input_tokens,
                    response.generation_tokens,
                    total_tokens,
                    response.generation_tps,
                    response.peak_memory,
                )

            usage = UsageInfo(
                prompt_tokens=total_input_tokens,
                completion_tokens=response.generation_tokens,
                total_tokens=total_tokens,
                prompt_tokens_details=PromptTokenUsageInfo(
                    cached_tokens=_coerce_cached_tokens(total_cached_tokens, response)
                ),
            )

            return {"response": parsed_response, "usage": usage}

        except asyncio.QueueFull:
            logger.error("Too many requests. Service is at capacity.")
            content = create_error_response(
                "Too many requests. Service is at capacity.",
                "rate_limit_exceeded",
                HTTPStatus.TOO_MANY_REQUESTS,
            )
            raise HTTPException(status_code=429, detail=content)
        except Exception as e:
            logger.error(f"Error in text response generation: {e!s}")
            content = create_error_response(
                f"Failed to generate text response: {e!s}",
                "server_error",
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            raise HTTPException(status_code=500, detail=content)

    def _is_request_batchable(self, request: ChatCompletionRequest) -> bool:
        """Return True if the request can be served by the continuous batcher.

        The scheduler is the default generation path (even for a batch of
        one). This predicate only returns False for features that the batched
        ``BatchGenerator`` does not support:

        * Speculative decoding via ``draft_model``.

        Per-request positive seeds are ignored while batching is enabled. To
        use request seeds, start the server with ``--disable-batching`` so all
        LM requests use the single-request generation path.

        Requests that require a mid-prefill cache checkpoint are also routed
        through the single-request path — that check is applied at the call
        site where the checkpoint position is known.
        """
        if getattr(self, "_disable_batching", False):
            return False
        if not BATCHING_AVAILABLE:
            return False
        if self.model.has_draft_model:
            return False
        # BatchGenerator requires cache layers to support ``merge`` so it
        # can stack multiple sequences into one batched tensor. Models
        # whose caches don't satisfy that invariant (mirroring
        # ``mlx_lm.server``'s own ``is_batchable`` check) can only run on
        # the single-request path.
        if not self.model.cache_is_batchable:
            return False
        return True

    async def _get_or_start_scheduler(self) -> BatchScheduler:
        """Lazily construct and start the batch scheduler on first use."""
        if self._batch_scheduler is not None and self._batch_scheduler.is_running:
            return self._batch_scheduler
        async with self._batch_scheduler_lock:
            if self._batch_scheduler is None or not self._batch_scheduler.is_running:
                scheduler = BatchScheduler(
                    self.model.model,
                    self.model.tokenizer,
                    prompt_cache=self.prompt_cache,
                    completion_batch_size=self._batch_completion_size,
                    prefill_batch_size=self._batch_prefill_size,
                    prefill_step_size=self._batch_prefill_step_size,
                    max_kv_size=self._batch_max_kv_size,
                    generation_lock=self._generation_lock,
                    queue_size=self._queue_size,
                )
                scheduler.start()
                self._batch_scheduler = scheduler
        return self._batch_scheduler

    def _submit_batched_stream(
        self,
        scheduler: BatchScheduler,
        ctx: "_InferenceContext",
    ) -> AsyncGenerator[Any, None]:
        """Submit a batched request and return an async chunk generator.

        The returned generator yields objects that duck-type match the
        :class:`mlx_lm.generate.GenerationResponse` instances produced by the
        single-request path (``.text``, ``.token``, ``.finish_reason``,
        ``.generation_tokens``, ``.generation_tps``, ``.peak_memory``,
        ``.prompt_tokens``, ``.prompt_tps``), so the downstream streaming and
        parser logic stays unchanged.

        Notes
        -----
        The scheduler owns the prompt cache end-to-end for the batched path:
        it calls :meth:`LRUPromptCache.fetch_nearest_cache` on admission and
        :meth:`LRUPromptCache.insert_cache` on finish, both from its worker
        thread. That keeps every ``mx.array`` mutation on a single OS thread,
        matching ``mlx_lm.server``'s architecture and avoiding the
        cross-thread Metal command-buffer race documented in
        https://github.com/ml-explore/mlx/issues/2457.
        """
        params = ctx.model_params
        sampler = self.model.build_sampler(params)
        logits_processors = self.model.build_logits_processors(params)
        state_machine = BatchScheduler.build_state_machine(
            self.model.tokenizer,
            params.get("stop") or None,
        )

        max_tokens = params.get("max_tokens")
        if max_tokens is None:
            max_tokens = params.get("max_completion_tokens")
        if max_tokens is None:
            max_tokens = self.model.resolve_max_tokens(params)

        # Hand the full prompt to the scheduler; it does its own LRU lookup
        # and cache construction on its worker thread. For non-trimmable
        # cache models the context also carries pre-computed segment
        # boundaries so the scheduler can save a prefix checkpoint — see
        # ``_build_inference_context``.
        return scheduler.submit_stream(
            input_ids=list(ctx.cache_key),
            max_tokens=int(max_tokens),
            sampler=sampler,
            logits_processors=logits_processors or None,
            state_machine=state_machine,
            segments=ctx.batched_segments,
            segment_types=ctx.batched_segment_types,
        )

    def _with_effective_sampling_params(self, request_data: dict[str, Any]) -> dict[str, Any]:
        """Return a debug payload annotated with resolved sampling values.

        Parameters
        ----------
        request_data : dict[str, Any]
            Raw generation request data.

        Returns
        -------
        dict[str, Any]
            Copy of ``request_data`` with ``effective_sampling_params`` and
            resolved top-level sampling fields for readable debug logs.
        """
        effective_sampling = self.model.resolve_sampling_params(request_data)
        debug_payload = dict(request_data)
        debug_payload.update(
            {
                key: value
                for key, value in effective_sampling.items()
                if key != "eos_token_ids" and key in debug_payload
            }
        )
        debug_payload["effective_sampling_params"] = effective_sampling
        return debug_payload

    async def _run_nonstream_generation(
        self,
        request: ChatCompletionRequest,
        ctx: "_InferenceContext",
        request_data: dict[str, Any],
    ) -> Any:
        """Dispatch a non-streaming request to the batcher or the worker.

        Kept as a standalone method so :meth:`generate_text_response` stays
        under the complexity budget enforced by ruff.
        """
        if self._is_request_batchable(request) and ctx.checkpoint_position is None:
            scheduler = await self._get_or_start_scheduler()
            return await self._collect_batched_response(scheduler, ctx)
        return await self.inference_worker.submit(
            self._generate_with_lock_and_cache_persist,
            list(ctx.cache_key),
            ctx.cache,
            input_ids=ctx.rest_input_ids,
            prompt_cache=ctx.cache,
            stream=False,
            **request_data,
        )

    async def _collect_batched_response(
        self,
        scheduler: BatchScheduler,
        ctx: "_InferenceContext",
    ) -> Any:
        """Drain a batched stream into a ``CompletionResponse``-compatible object.

        The MLX_LM non-streaming path returns a
        :class:`~app.models.mlx_lm.CompletionResponse`. The batched scheduler
        produces incremental chunks, so we accumulate them here to yield the
        same surface area for :meth:`generate_text_response`.
        """
        # Lazy import — tests stub ``app.models.mlx_lm`` with a minimal fake
        # that doesn't export CompletionResponse; a module-level import would
        # break those test setups.
        from ..models.mlx_lm import CompletionResponse

        stream = self._submit_batched_stream(scheduler, ctx)
        text_parts: list[str] = []
        tokens: list[int] = []
        final_chunk = None
        async for chunk in stream:
            if chunk.text:
                text_parts.append(chunk.text)
            tokens.append(chunk.token)
            if chunk.finish_reason is not None:
                final_chunk = chunk
                break

        generation_tokens = final_chunk.generation_tokens if final_chunk else len(tokens)
        generation_tps = final_chunk.generation_tps if final_chunk else 0.0
        peak_memory = final_chunk.peak_memory if final_chunk else 0.0
        prompt_tokens = final_chunk.prompt_tokens if final_chunk else len(ctx.rest_input_ids)
        cached_prompt_tokens = final_chunk.cached_prompt_tokens if final_chunk else 0
        return CompletionResponse(
            text="".join(text_parts),
            tokens=tokens,
            peak_memory=peak_memory,
            generation_tps=generation_tps,
            prompt_tps=0.0,
            prompt_tokens=prompt_tokens,
            generation_tokens=generation_tokens,
            cached_prompt_tokens=cached_prompt_tokens,
        )

    async def get_queue_stats(self) -> dict[str, Any]:
        """Get statistics from the inference worker and batch scheduler.

        Returns
        -------
        dict[str, Any]
            Dictionary with ``queue_stats`` and ``batch_stats`` sub-dictionaries.
        """
        batch_running = self._batch_scheduler is not None and self._batch_scheduler.is_running
        return {
            "queue_stats": self.inference_worker.get_stats(),
            "batch_stats": {
                **(
                    self._batch_scheduler.get_stats()
                    if batch_running and self._batch_scheduler is not None
                    else {
                        "running": False,
                        "queue_size": 0,
                        "max_queue_size": self._queue_size,
                        "active_requests": 0,
                    }
                ),
                "completion_batch_size": self._batch_completion_size,
                "prefill_batch_size": self._batch_prefill_size,
                "disabled": self._disable_batching,
            },
        }

    async def cleanup(self) -> None:
        """Cleanup resources and stop the inference worker before shutdown.

        This method ensures all pending requests are properly completed
        and resources are released.
        """
        try:
            logger.info("Cleaning up MLXLMHandler resources")
            if self._batch_scheduler is not None:
                self._batch_scheduler.stop()
                self._batch_scheduler = None
            if hasattr(self, "inference_worker"):
                self.inference_worker.stop()
            if hasattr(self, "prompt_cache"):
                self.prompt_cache.close()

            # Force garbage collection
            gc.collect()
            logger.info("MLXLMHandler cleanup completed successfully")
        except Exception as e:
            logger.error(f"Error during MLXLMHandler cleanup: {e!s}")
            raise

    async def _prepare_text_request(
        self, request: ChatCompletionRequest
    ) -> tuple[list[dict[str, str]], dict[str, Any]]:
        """
        Prepare a text request by parsing model parameters and verifying the format of messages.

        Args:
            request: ChatCompletionRequest object containing the messages.

        Returns:
            Tuple containing the formatted chat messages and model parameters.
        """

        try:
            # Extract only the fields consumed by MLX_LM.__call__ instead of
            # serializing the entire Pydantic model with model_dump().
            chat_template_kwargs = (
                request.chat_template_kwargs.model_dump() if request.chat_template_kwargs else {}
            )

            if request.tools:
                tools = [t.model_dump() for t in request.tools]
                chat_template_kwargs["tools"] = tools
                if request.tool_choice:
                    tool_choice = request.tool_choice
                    if hasattr(tool_choice, "model_dump"):
                        tool_choice = tool_choice.model_dump()
                    chat_template_kwargs["tool_choice"] = tool_choice

            model_params: dict[str, Any] = {
                "temperature": request.temperature,
                "top_p": request.top_p,
                "top_k": request.top_k,
                "min_p": request.min_p,
                "max_tokens": request.max_tokens,
                "max_completion_tokens": request.max_completion_tokens,
                "seed": request.seed,
                "repetition_penalty": request.repetition_penalty,
                "repetition_context_size": request.repetition_context_size,
                "presence_penalty": request.presence_penalty,
                "frequency_penalty": request.frequency_penalty,
                "xtc_probability": request.xtc_probability,
                "xtc_threshold": request.xtc_threshold,
                "logit_bias": request.logit_bias,
                "stop": request.stop,
                "chat_template_kwargs": chat_template_kwargs,
                "kv_bits": self.kv_bits,
                "kv_group_size": self.kv_group_size,
                "quantized_kv_start": self.quantized_kv_start,
            }

            if request.response_format:
                response_format = request.response_format
                if response_format.get("type") == "json_schema":
                    model_params["schema"] = response_format.get("json_schema", {}).get("schema")

            # Format chat messages: single-pass system merge + content normalization
            chat_messages: list[dict[str, Any]] = []
            merged_system: dict[str, Any] | None = None

            raw_messages = [m.model_dump() for m in request.messages] if request.messages else []

            for message in raw_messages:
                # Reasoning content is output metadata and should not be replayed
                # into subsequent prompt history turns.
                message.pop("reasoning_content", None)

                # Handle content that might be a list of dictionaries (multimodal format)
                content = message.get("content")
                if content is None:
                    # Assistant messages with tool_calls or partial have content: null — keep them
                    if message.get("tool_calls") or message.get("partial"):
                        message["content"] = ""
                    else:
                        continue
                if isinstance(content, list):
                    # For LM models, extract only text content and concatenate
                    text_parts = []
                    for item in content:
                        if isinstance(item, str):
                            text_parts.append(item)
                        elif (
                            isinstance(item, dict)
                            and item.get("type") == "text"
                            and item.get("text")
                        ):
                            text_parts.append(item["text"])
                    content = "\n".join(text_parts) if text_parts else ""

                message["content"] = content

                # Single-pass: merge system messages in-place
                if message.get("role") == "system":
                    if merged_system is None:
                        merged_system = message.copy()
                    elif message.get("content"):
                        merged_system["content"] += "\n\n" + message["content"]
                else:
                    chat_messages.append(message)

            # Prepend merged system message
            if merged_system:
                chat_messages.insert(0, merged_system)

            # Detect partial mode: last assistant message with partial=True
            is_partial = (
                chat_messages
                and chat_messages[-1].get("role") == "assistant"
                and chat_messages[-1].get("partial", False)
            )

            # Strip 'partial' from all messages — server-level control, not a template field
            for msg in chat_messages:
                msg.pop("partial", None)

            # Communicate partial mode to create_input_prompt via chat_template_kwargs
            if is_partial:
                chat_template_kwargs["_partial_mode"] = True

            seed = model_params.get("seed")
            if self._is_request_batchable(request) and seed is not None and seed > 0:
                logger.warning(
                    "Ignoring per-request seed because continuous batching is enabled; "
                    "start the server with --disable-batching to use request seeds."
                )
                model_params["seed"] = 0

            return chat_messages, model_params

        except Exception as e:
            logger.error(f"Failed to prepare text request: {e!s}")
            content = create_error_response(
                f"Failed to process request: {e!s}", "bad_request", HTTPStatus.BAD_REQUEST
            )
            raise HTTPException(status_code=400, detail=content)
