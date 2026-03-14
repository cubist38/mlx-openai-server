import asyncio
from collections.abc import AsyncGenerator
from dataclasses import dataclass
import gc
from http import HTTPStatus
import time
from typing import Any

from fastapi import HTTPException
from loguru import logger

from ..core import InferenceWorker
from ..message_converters import MessageConverterManager
from ..models.mlx_lm import MLX_LM
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
from ..utils.prompt_cache import LRUPromptCache


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
    total_cached_tokens: int
    model_params: dict[str, Any]
    parsers_result: Any
    prompt_progress_callback: Any = None


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
        max_concurrency: int = 1,
        enable_auto_tool_choice: bool = False,
        tool_call_parser: str = None,
        reasoning_parser: str = None,
        message_converter: str = None,
        trust_remote_code: bool = False,
        chat_template_file: str = None,
        debug: bool = False,
        prompt_cache_size: int = 10,
        prompt_cache_max_bytes: int = 1 << 63,
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
        max_concurrency : int
            Maximum number of concurrent model inference tasks.
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
        """
        self.model_path = model_path
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

        # Store parser configuration
        self.enable_auto_tool_choice = enable_auto_tool_choice
        # Debug mode
        self.debug = debug
        self.reasoning_parser_name = reasoning_parser
        self.tool_parser_name = tool_call_parser
        self.prompt_cache = LRUPromptCache(
            max_size=prompt_cache_size,
            max_bytes=prompt_cache_max_bytes,
        )
        self.message_converter = MessageConverterManager.create_converter(
            converter_name=message_converter,
            tool_parser_name=tool_call_parser,
            reasoning_parser_name=reasoning_parser,
        )
        # Dedicated inference thread — keeps the event loop free during
        # blocking MLX model computation.
        self.inference_worker = InferenceWorker()

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
        self.inference_worker = InferenceWorker(
            queue_size=queue_config.get("queue_size", 100),
            timeout=queue_config.get("timeout", 300),
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

    async def _build_inference_context(self, request: ChatCompletionRequest) -> "_InferenceContext":
        """Build the common inference context shared by stream and non-stream paths.

        Handles: request parsing, message refinement, prompt encoding, KV cache
        lookup, parser creation, and model-param preparation.
        """
        chat_messages, model_params = await self._prepare_text_request(request)
        refined_messages = self.refine_messages(chat_messages)
        chat_template_kwargs = model_params.get("chat_template_kwargs", {})

        input_prompt = self.model.create_input_prompt(refined_messages, chat_template_kwargs)
        if self.debug:
            log_debug_prompt(input_prompt)

        input_ids = self.model.encode_prompt(input_prompt)
        cache, rest_input_ids = self.prompt_cache.fetch_nearest_cache(input_ids)

        # Cache key must be the FULL input_ids, not rest_input_ids.
        # Using rest_input_ids causes memory leaks: on "longer" cache hits,
        # rest_input_ids is a suffix (e.g., [B] from input [A,B]), creating
        # new cache entries [B,X,Y,Z] instead of updating [A,B,X,Y,Z].
        cache_key = input_ids[:]

        if cache is None:
            cache = self.model.create_prompt_cache()

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
            total_cached_tokens=total_cached_tokens,
            model_params=model_params,
            parsers_result=parsers_result,
            prompt_progress_callback=prompt_progress_callback,
        )

    async def generate_text_stream(
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
        cache: list[Any] | None = None
        cache_key: list[int] | None = None
        cache_inserted = False

        try:
            ctx = await self._build_inference_context(request)
            cache = ctx.cache
            cache_key = ctx.cache_key
            total_cached_tokens = ctx.total_cached_tokens
            model_params = ctx.model_params
            parsers_result = ctx.parsers_result

            request_data = {
                "prompt_progress_callback": ctx.prompt_progress_callback,
                **model_params,
            }

            if self.debug:
                log_debug_request(request_data)
                log_debug_model_dispatch("mlx_lm.generate_text_stream.submit_stream", request_data)
                request_data["verbose"] = True

            response_generator = self.inference_worker.submit_stream(
                self.model,
                input_ids=ctx.rest_input_ids,
                prompt_cache=cache,
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
                    cache_key.append(chunk.token)

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
                    cache_key.append(chunk.token)
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

            total_tokens = final_chunk.prompt_tokens + final_chunk.generation_tokens
            self.prompt_cache.insert_cache(cache_key, cache)
            cache_inserted = True

            if self.debug:
                self.prompt_cache.log_cache_stats()
                log_debug_raw_text_response(raw_text)
                log_debug_stats(
                    final_chunk.prompt_tokens,
                    final_chunk.generation_tokens,
                    total_tokens,
                    final_chunk.generation_tps,
                    final_chunk.peak_memory,
                )

            yield {
                "__usage__": UsageInfo(
                    prompt_tokens=final_chunk.prompt_tokens,
                    completion_tokens=final_chunk.generation_tokens,
                    total_tokens=total_tokens,
                    prompt_tokens_details=PromptTokenUsageInfo(cached_tokens=total_cached_tokens),
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
        finally:
            if cache is not None and cache_key is not None and not cache_inserted:
                try:
                    self.prompt_cache.insert_cache(cache_key, cache)
                    if self.debug:
                        self.prompt_cache.log_cache_stats()
                except Exception as cache_error:
                    logger.warning(
                        f"Failed to persist prompt cache during stream finalization: {cache_error}"
                    )

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
            total_cached_tokens = ctx.total_cached_tokens

            request_data = {
                "prompt_progress_callback": ctx.prompt_progress_callback,
                **model_params,
            }

            if self.debug:
                log_debug_model_dispatch("mlx_lm.generate_text_response.submit", request_data)

            response = await self.inference_worker.submit(
                self.model,
                input_ids=ctx.rest_input_ids,
                prompt_cache=ctx.cache,
                stream=False,
                **request_data,
            )

            response_text = response.text
            cache_key = ctx.cache_key
            cache_key += response.tokens

            self.prompt_cache.insert_cache(cache_key, ctx.cache)

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

            total_tokens = response.prompt_tokens + response.generation_tokens

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
                    response.prompt_tokens,
                    response.generation_tokens,
                    total_tokens,
                    response.generation_tps,
                    response.peak_memory,
                )

            usage = UsageInfo(
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.generation_tokens,
                total_tokens=total_tokens,
                prompt_tokens_details=PromptTokenUsageInfo(cached_tokens=total_cached_tokens),
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

    async def get_queue_stats(self) -> dict[str, Any]:
        """Get statistics from the inference worker.

        Returns
        -------
        dict[str, Any]
            Dictionary with ``queue_stats`` sub-dictionary.
        """
        return {
            "queue_stats": self.inference_worker.get_stats(),
        }

    async def cleanup(self) -> None:
        """Cleanup resources and stop the inference worker before shutdown.

        This method ensures all pending requests are properly completed
        and resources are released.
        """
        try:
            logger.info("Cleaning up MLXLMHandler resources")
            if hasattr(self, "inference_worker"):
                self.inference_worker.stop()

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
                "xtc_probability": request.xtc_probability,
                "xtc_threshold": request.xtc_threshold,
                "logit_bias": request.logit_bias,
                "chat_template_kwargs": chat_template_kwargs,
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

            return chat_messages, model_params

        except Exception as e:
            logger.error(f"Failed to prepare text request: {e!s}")
            content = create_error_response(
                f"Failed to process request: {e!s}", "bad_request", HTTPStatus.BAD_REQUEST
            )
            raise HTTPException(status_code=400, detail=content)
