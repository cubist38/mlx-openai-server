"""MLX language model handler for text-only chat completions."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
import gc
from http import HTTPStatus
import time
from typing import Any
import uuid

from fastapi import HTTPException
from loguru import logger

from ..core.queue import RequestQueue
from ..message_converters import MessageConverterManager
from ..models.mlx_lm import MLX_LM
from ..parsers import ParserManager
from ..schemas.openai import ChatCompletionRequest, UsageInfo
from ..utils.debug_logging import (
    log_debug_cache_stats,
    log_debug_prompt,
    log_debug_raw_text_response,
    log_debug_request,
    log_debug_stats,
    make_prompt_progress_callback,
)
from ..utils.errors import create_error_response
from ..utils.prompt_cache import LRUPromptCache


class MLXLMHandler:
    """Handler class for making requests to the underlying MLX text-only language model service.

    Provides request queuing, metrics tracking, and robust error handling.
    """

    def __init__(
        self,
        model_path: str,
        context_length: int = 32768,
        max_concurrency: int = 1,
        enable_auto_tool_choice: bool = False,
        tool_call_parser: str | None = None,
        reasoning_parser: str | None = None,
        message_converter: str | None = None,
        trust_remote_code: bool = False,
        chat_template_file: str | None = None,
        debug: bool = False,
    ) -> None:
        """Initialize the handler with the specified model path.

        Parameters
        ----------
        model_path : str
            Path to the model directory.
        context_length : int, optional
            Maximum context length for the model (default: 32768).
        max_concurrency : int, optional
            Maximum number of concurrent model inference tasks (default: 1).
        enable_auto_tool_choice : bool, optional
            Enable automatic tool choice (default: False).
        tool_call_parser : str | None, optional
            Name of the tool call parser to use (e.g., qwen3, glm4_moe, harmony).
        reasoning_parser : str | None, optional
            Name of the reasoning parser to use (e.g., qwen3_next, glm4_moe).
        message_converter : str | None, optional
            Name of the message converter to use (default: None).
        trust_remote_code : bool, optional
            Enable trust_remote_code when loading models (default: False).
        chat_template_file : str | None, optional
            Path to a custom chat template file (default: None).
        debug : bool, optional
            Enable debug logging (default: False).

        Returns
        -------
        None
            The constructor does not return a value.
        """
        self.model_path = model_path
        self.model = MLX_LM(
            model_path,
            context_length,
            trust_remote_code=trust_remote_code,
            chat_template_file=chat_template_file or "",
        )
        self.model_created = int(time.time())  # Store creation time when model is loaded
        self.model_type = self.model.get_model_type()

        # Store parser configuration
        self.enable_auto_tool_choice = enable_auto_tool_choice
        # Debug mode
        self.debug = debug
        self.reasoning_parser_name = reasoning_parser
        self.tool_parser_name = tool_call_parser
        self.prompt_cache = LRUPromptCache()
        self.message_converter = MessageConverterManager.create_converter(message_converter)
        # Initialize request queue for text tasks
        self.request_queue = RequestQueue(max_concurrency=max_concurrency)

        logger.info(f"Initialized MLXHandler with model path: {model_path}")

    async def get_models(self) -> list[dict[str, Any]]:
        """Get list of available models with their metadata."""
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
        """Initialize the handler and start the request queue.

        Parameters
        ----------
        queue_config : dict[str, Any] | None, optional
            Configuration for the queue: keys 'max_concurrency' (int), 'timeout' (float),
            and 'queue_size' (int). When omitted, sensible defaults are used.

        Returns
        -------
        None
            This coroutine does not return a value.
        """
        if not queue_config:
            queue_config = {"max_concurrency": 1, "timeout": 300, "queue_size": 100}
        self.request_queue = RequestQueue(
            max_concurrency=int(queue_config.get("max_concurrency", 1)),
            timeout=float(queue_config.get("timeout", 300)),
            queue_size=int(queue_config.get("queue_size", 100)),
        )
        await self.request_queue.start(self._process_request)
        logger.info("Initialized MLXHandler and started request queue")

    def refine_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Refine the messages to be more suitable for the model."""
        refined_messages = []

        if self.message_converter:
            logger.info("Message converter is enabled, converting messages...")
            messages = self.message_converter.convert_messages(messages)
            logger.info("Messages converted successfully")

        logger.info("Filtering out None values from messages...")
        for message in messages:
            cleaned_message = {k: v for k, v in message.items() if v is not None}
            refined_messages.append(cleaned_message)
        logger.info("Messages filtered successfully")
        return refined_messages

    async def generate_text_stream(  # noqa: C901
        self, request: ChatCompletionRequest
    ) -> AsyncGenerator[str | dict[str, Any], None]:
        """Generate a streaming response for text-only chat completion requests.

        Processes the request using the request queue and yields incremental
        response pieces as they arrive from the model.

        Parameters
        ----------
        request : ChatCompletionRequest
            The chat completion request.

        Yields
        ------
        str or dict
            Incremental response chunks (strings) and/or dictionaries (reasoning
            blocks, tool calls, or final usage info).
        """
        request_id = f"text-{uuid.uuid4()}"

        try:
            chat_messages, model_params = await self._prepare_text_request(request)

            refined_messages = self.refine_messages(chat_messages)
            chat_template_kwargs = model_params.get("chat_template_kwargs", {})

            input_prompt = self.model.create_input_prompt(refined_messages, chat_template_kwargs)

            if self.debug:
                log_debug_prompt(input_prompt)

            input_ids = self.model.encode_prompt(input_prompt)

            cache, rest_input_ids = self.prompt_cache.fetch_nearest_cache(input_ids)

            cache_key = rest_input_ids[:]

            if cache is None:
                cache = self.model.create_prompt_cache()

            if self.debug:
                log_debug_cache_stats(len(input_ids), len(rest_input_ids))

            enable_thinking = chat_template_kwargs.get("enable_thinking", True)

            # Create parsers using ParserManager
            parsers_result = ParserManager.create_parsers(
                reasoning_parser_name=self.reasoning_parser_name,
                tool_parser_name=self.tool_parser_name,
            )

            # Handle enable_thinking flag for separate reasoning parsers
            if not enable_thinking and parsers_result.reasoning_parser:
                if parsers_result.reasoning_parser.respects_enable_thinking():
                    parsers_result.reasoning_parser = None

            prompt_progress_callback = make_prompt_progress_callback() if self.debug else None

            request_data = {
                "input_ids": rest_input_ids,
                "prompt_cache": cache,
                "stream": True,
                "prompt_progress_callback": prompt_progress_callback,
                **model_params,
            }

            if self.debug:
                log_debug_request(request_data)
                request_data["verbose"] = True

            response_generator = await self.request_queue.submit(request_id, request_data)

            after_reasoning_close_content = None
            final_chunk = None
            is_first_chunk = True
            raw_text = ""  # only use for debugging

            # Handle unified parser streaming
            if parsers_result.is_unified:
                unified_parser = parsers_result.unified_parser
                if unified_parser is None:
                    logger.warning("Unified parser expected but not provided; streaming raw text.")
                    for chunk in response_generator:
                        if chunk is None:
                            continue
                        final_chunk = chunk
                        text = chunk.text
                        raw_text += text
                        cache_key.append(chunk.token)
                        yield text
                else:
                    for chunk in response_generator:
                        if chunk is None:
                            continue
                        final_chunk = chunk
                        text = chunk.text
                        raw_text += text
                        cache_key.append(chunk.token)

                        parsed_result, is_complete = unified_parser.parse_streaming(text)
                        if parsed_result:
                            # Unified parser returns dict with reasoning_content, tool_calls, content
                            if parsed_result.get("reasoning_content"):
                                yield {"reasoning_content": parsed_result["reasoning_content"]}
                            if parsed_result.get("tool_calls"):
                                for tool_call in parsed_result["tool_calls"]:
                                    yield tool_call
                            if parsed_result.get("content"):
                                yield parsed_result["content"]
                        # Continue processing all chunks even if is_complete is True
            else:
                # Handle separate parsers streaming
                reasoning_parser = parsers_result.reasoning_parser
                tool_parser = parsers_result.tool_parser

                for chunk in response_generator:
                    if chunk is None:
                        continue
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
                    if reasoning_parser:
                        parsed_content, is_complete = reasoning_parser.extract_reasoning_streaming(
                            text
                        )
                        if parsed_content:
                            if isinstance(parsed_content, dict):
                                after_reasoning_close_content = parsed_content.get(
                                    "after_reasoning_close_content"
                                )
                                yield parsed_content
                            else:
                                # parsed_content is a plain string fragment
                                yield parsed_content
                        if is_complete:
                            reasoning_parser = None
                        if after_reasoning_close_content:
                            text = after_reasoning_close_content
                            after_reasoning_close_content = None
                        else:
                            continue
                    if tool_parser:
                        parsed_tuple = tool_parser.extract_tool_calls_streaming(text)
                        if isinstance(parsed_tuple, tuple):
                            parsed_content_tool, is_complete = parsed_tuple
                        else:
                            parsed_content_tool = parsed_tuple
                            is_complete = False

                        if parsed_content_tool:
                            if isinstance(parsed_content_tool, dict):
                                content = parsed_content_tool.get("content")
                                if content:
                                    if isinstance(content, str):
                                        yield content
                                    else:
                                        # Ensure we yield strings for stream compatibility
                                        yield str(content)
                                tool_calls = parsed_content_tool.get("tool_calls")
                                if tool_calls:
                                    for tool_call in tool_calls:
                                        yield tool_call
                            else:
                                # parsed_content is a plain string fragment
                                yield parsed_content_tool
                        continue

                    yield text

            if final_chunk is None:
                logger.error("No final chunk received from model")
                err_detail = create_error_response(
                    "No final chunk received from model",
                    "server_error",
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                )
                raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail=err_detail)

            total_tokens = final_chunk.prompt_tokens + final_chunk.generation_tokens
            self.prompt_cache.insert_cache(cache_key, cache)

            if self.debug:
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
                )
            }

        except asyncio.QueueFull as exc:
            logger.error("Too many requests. Service is at capacity.")
            err_detail = create_error_response(
                "Too many requests. Service is at capacity.",
                "rate_limit_exceeded",
                HTTPStatus.TOO_MANY_REQUESTS,
            )
            raise HTTPException(
                status_code=HTTPStatus.TOO_MANY_REQUESTS, detail=err_detail
            ) from exc
        except Exception as e:
            logger.error(f"Error in text stream generation for request {request_id}: {e!s}")
            err_detail = create_error_response(
                f"Failed to generate text stream: {e!s}",
                "server_error",
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            raise HTTPException(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail=err_detail
            ) from e

    async def generate_text_response(self, request: ChatCompletionRequest) -> dict[str, Any]:
        """Generate a complete response for text-only chat completion requests.

        Uses the request queue for handling concurrent requests and returns a
        parsed response together with usage information.

        Parameters
        ----------
        request : ChatCompletionRequest
            The chat completion request.

        Returns
        -------
        dict[str, Any]
            A dictionary with keys ``response`` (parsed response dict) and ``usage``
            (a ``UsageInfo`` instance).
        """
        request_id = f"text-{uuid.uuid4()}"

        try:
            chat_messages, model_params = await self._prepare_text_request(request)
            # Refine messages to remove None values and convert to the correct format
            refined_messages = self.refine_messages(chat_messages)
            # Count prompt tokens
            chat_template_kwargs = model_params.get("chat_template_kwargs", {})

            input_prompt = self.model.create_input_prompt(refined_messages, chat_template_kwargs)

            if self.debug:
                log_debug_prompt(input_prompt)

            input_ids = self.model.encode_prompt(input_prompt)

            cache, rest_input_ids = self.prompt_cache.fetch_nearest_cache(input_ids)

            cache_key = rest_input_ids[:]

            if cache is None:
                cache = self.model.create_prompt_cache()

            if self.debug:
                log_debug_cache_stats(len(input_ids), len(rest_input_ids))

            request_data = {
                "input_ids": rest_input_ids,
                "prompt_cache": cache,
                "stream": False,
                **model_params,
            }

            response = await self.request_queue.submit(request_id, request_data)

            # Create parsers using ParserManager
            parsers_result = ParserManager.create_parsers(
                reasoning_parser_name=self.reasoning_parser_name,
                tool_parser_name=self.tool_parser_name,
            )

            enable_thinking = chat_template_kwargs.get("enable_thinking", True)

            # Handle enable_thinking flag for separate reasoning parsers
            if not enable_thinking and parsers_result.reasoning_parser:
                if parsers_result.reasoning_parser.respects_enable_thinking():
                    parsers_result.reasoning_parser = None

            response_text = response.text
            cache_key += response.tokens

            self.prompt_cache.insert_cache(cache_key, cache)

            parsed_response: dict[str, Any] = {
                "reasoning_content": None,
                "tool_calls": None,
                "content": None,
            }

            # Handle unified parser
            if parsers_result.is_unified:
                unified_parser = parsers_result.unified_parser
                if unified_parser is not None:
                    parsed_result = unified_parser.parse(response_text)
                    if parsed_result:
                        parsed_response["reasoning_content"] = parsed_result.get(
                            "reasoning_content"
                        )
                        parsed_response["tool_calls"] = parsed_result.get("tool_calls")
                        parsed_response["content"] = parsed_result.get("content")
            # Handle separate parsers
            elif parsers_result.reasoning_parser or parsers_result.tool_parser:
                reasoning_parser = parsers_result.reasoning_parser
                tool_parser = parsers_result.tool_parser

                if reasoning_parser and reasoning_parser.needs_redacted_reasoning_prefix():
                    response_text = reasoning_parser.get_reasoning_open() + response_text

                if reasoning_parser:
                    parsed_content = reasoning_parser.extract_reasoning(response_text)
                    if parsed_content:
                        if isinstance(parsed_content, dict):
                            parsed_response["reasoning_content"] = parsed_content.get(
                                "reasoning_content"
                            )
                            parsed_response["content"] = parsed_content.get("content")
                            response_text = parsed_content.get("after_reasoning_close_content")
                        else:
                            parsed_response["content"] = parsed_content
                            response_text = None

                if response_text:
                    if tool_parser:
                        tool_parsed_content: dict[str, Any] | None = tool_parser.extract_tool_calls(
                            response_text
                        )
                        if tool_parsed_content:
                            if isinstance(tool_parsed_content, dict):
                                parsed_response["tool_calls"] = tool_parsed_content.get(
                                    "tool_calls"
                                )
                                parsed_response["content"] = tool_parsed_content.get("content")
                            else:
                                parsed_response["content"] = tool_parsed_content
            else:
                parsed_response["content"] = response_text

            total_tokens = response.prompt_tokens + response.generation_tokens

            if self.debug:
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
            )

            result = {"response": parsed_response, "usage": usage}

        except asyncio.QueueFull as exc:
            logger.error("Too many requests. Service is at capacity.")
            err_detail = create_error_response(
                "Too many requests. Service is at capacity.",
                "rate_limit_exceeded",
                HTTPStatus.TOO_MANY_REQUESTS,
            )
            raise HTTPException(
                status_code=HTTPStatus.TOO_MANY_REQUESTS, detail=err_detail
            ) from exc
        except Exception as e:
            logger.error(f"Error in text response generation: {e!s}")
            err_detail = create_error_response(
                f"Failed to generate text response: {e!s}",
                "server_error",
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            raise HTTPException(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail=err_detail
            ) from e
        else:
            return result

    async def _process_request(self, request_data: dict[str, Any]) -> Any:
        """
        Process a text request. This is the worker function for the request queue.

        Parameters
        ----------
        request_data : dict[str, Any]
            Dictionary containing the request data.

        Returns
        -------
        Any
            The model's response (may be a streaming generator or a response object).
        """
        try:
            input_ids = request_data.pop("input_ids")
            prompt_cache = request_data.pop("prompt_cache")
            stream = request_data.pop("stream")

            # Call the model
            response = self.model(
                input_ids=input_ids, prompt_cache=prompt_cache, stream=stream, **request_data
            )
            # Force garbage collection after model inference
            gc.collect()
            result = response

        except Exception as e:
            logger.error(f"Error processing text request: {e!s}")
            # Clean up on error
            gc.collect()
            raise
        else:
            return result

    async def get_queue_stats(self) -> dict[str, Any]:
        """
        Get statistics from the request queue and performance metrics.

        Returns:
            Dict with queue and performance statistics.
        """
        queue_stats = self.request_queue.get_queue_stats()

        return {
            "queue_stats": queue_stats,
        }

    async def cleanup(self) -> None:
        """Cleanup resources and stop the request queue before shutdown.

        This coroutine ensures all pending requests are properly cancelled
        and resources are released.

        Returns
        -------
        None
            This coroutine does not return a value.
        """
        try:
            logger.info("Cleaning up MLXLMHandler resources")
            if hasattr(self, "request_queue"):
                await self.request_queue.stop()
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
            request_dict = request.model_dump()
            tools = request_dict.pop("tools", None)
            tool_choice = request_dict.pop("tool_choice", None)

            if tools:
                # Enable auto tool choice if requested via CLI flag
                if self.enable_auto_tool_choice and tool_choice == "auto":
                    request_dict["chat_template_kwargs"]["tool_choice"] = "auto"
                elif tool_choice:
                    logger.warning("Tool choice has not supported yet, will be ignored.")
                request_dict["chat_template_kwargs"]["tools"] = tools

            if request_dict.get("response_format", None):
                response_format = request_dict.pop("response_format", None)
                if response_format.get("type") == "json_schema":
                    request_dict["schema"] = response_format.get("json_schema", None).get(
                        "schema", None
                    )

            # Format chat messages and merge system messages into index 0
            chat_messages = []
            system_messages = []
            non_system_messages = []

            for message in request_dict.get("messages", []):
                # Handle content that might be a list of dictionaries (multimodal format)
                content = message.get("content", None)
                if content is None:
                    continue
                if isinstance(content, list):
                    # For LM models, extract only text content and concatenate
                    text_parts = [
                        item["text"]
                        for item in content
                        if isinstance(item, dict)
                        and item.get("type") == "text"
                        and item.get("text")
                    ]
                    content = "\n".join(text_parts) if text_parts else ""
                message["content"] = content
                # Separate system messages from other messages
                if message.get("role") == "system":
                    system_messages.append(message)
                else:
                    non_system_messages.append(message)

            # If there are system messages, merge them into a single system message at index 0
            if system_messages:
                # Combine all system message contents
                combined_system_content = "\n\n".join(
                    [msg["content"] for msg in system_messages if msg.get("content")]
                )

                # Create merged system message using the first system message as template
                merged_system_message = system_messages[0].copy()
                merged_system_message["content"] = combined_system_content

                # Add merged system message at index 0
                chat_messages.append(merged_system_message)

            # Add all non-system messages after the merged system message
            chat_messages.extend(non_system_messages)
            result = (chat_messages, request_dict)

        except Exception as exc:
            logger.error(f"Failed to prepare text request: {exc!s}")
            err_detail = create_error_response(
                f"Failed to process request: {exc!s}", "bad_request", HTTPStatus.BAD_REQUEST
            )
            raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=err_detail) from exc
        else:
            return result
