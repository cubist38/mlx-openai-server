import asyncio
import time
import uuid
import gc
from http import HTTPStatus
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from fastapi import HTTPException
from loguru import logger

from ..core import InferenceWorker
from ..message_converters import MessageConverterManager
from ..models.mlx_lm import MLX_LM
from ..parsers import ParserManager
from ..schemas.openai import ChatCompletionRequest, PromptTokenUsageInfo, UsageInfo
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
    """
    Handler class for making requests to the underlying MLX text-only language model service.
    Provides request queuing, metrics tracking, and robust error handling.
    """

    handler_type: str = "lm"

    def __init__(self, model_path: str, draft_model_path: str | None = None, num_draft_tokens: int = 2, context_length: int | None = None, max_concurrency: int = 1, enable_auto_tool_choice: bool = False, tool_call_parser: str = None, reasoning_parser: str = None, message_converter: str = None, trust_remote_code: bool = False, chat_template_file: str = None, debug: bool = False, prompt_cache_size: int = 10):
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
        self.prompt_cache = LRUPromptCache(max_size=prompt_cache_size)
        self.message_converter = MessageConverterManager.create_converter(message_converter)
        # Dedicated inference thread — keeps the event loop free during
        # blocking MLX model computation.
        self.inference_worker = InferenceWorker()

        logger.info(f"Initialized MLXHandler with model path: {model_path}")

    async def get_models(self) -> List[Dict[str, Any]]:
        """
        Get list of available models with their metadata.
        """
        try:
            return [{
                "id": self.model_path,
                "object": "model",
                "created": self.model_created,
                "owned_by": "local"
            }]
        except Exception as e:
            logger.error(f"Error getting models: {str(e)}")
            return []
    
    async def initialize(self, queue_config: Optional[Dict[str, Any]] = None) -> None:
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

    def refine_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Refine the messages to be more suitable for the model.
        """
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

    async def generate_text_stream(self, request: ChatCompletionRequest) -> AsyncGenerator[str, None]:
        """
        Generate a streaming response for text-only chat completion requests.
        Uses the request queue for handling concurrent requests.

        Args:
            request: ChatCompletionRequest object containing the messages.

        Yields:
            str or dict: Response chunks (str) followed by usage info (dict) at the end.
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

            # Cache key must be the FULL input_ids, not rest_input_ids.
            # Using rest_input_ids causes memory leaks: on "longer" cache hits,
            # rest_input_ids is a suffix (e.g., [B] from input [A,B]), creating
            # new cache entries [B,X,Y,Z] instead of updating [A,B,X,Y,Z].
            # The original entry is never evicted and duplicates accumulate.
            cache_key = input_ids[:]

            if cache is None:
                cache = self.model.create_prompt_cache()

            total_input_tokens = len(input_ids)
            total_remaining_tokens = len(rest_input_ids)
            total_cached_tokens = total_input_tokens - total_remaining_tokens

            if self.debug:
                log_debug_cache_stats(total_input_tokens, total_remaining_tokens)

                
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

            if model_params.get("schema"):
                logger.info("JSON schema is enabled, disabling reasoning parser and tool parser")
                parsers_result.reasoning_parser = None
                parsers_result.tool_parser = None
                parsers_result.unified_parser = None

            prompt_progress_callback = make_prompt_progress_callback() if self.debug else None

            request_data = {
                "input_ids": rest_input_ids,
                "prompt_cache": cache,
                "stream": True,
                "prompt_progress_callback": prompt_progress_callback,
                **model_params
            }
            
            if self.debug:
                log_debug_request(request_data)
                request_data["verbose"] = True

            # Extract explicit model args; remaining kwargs are forwarded.
            input_ids = request_data.pop("input_ids")
            prompt_cache = request_data.pop("prompt_cache")
            request_data.pop("stream")

            response_generator = self.inference_worker.submit_stream(
                self.model,
                input_ids=input_ids,
                prompt_cache=prompt_cache,
                stream=True,
                **request_data,
            )

            after_reasoning_close_content = None
            final_chunk = None
            is_first_chunk = True
            raw_text = "" # only use for debugging
            
            # Handle unified parser streaming
            if parsers_result.is_unified:
                unified_parser = parsers_result.unified_parser
                async for chunk in response_generator:
                    if chunk is None:
                        continue
                    final_chunk = chunk
                    text = chunk.text
                    raw_text += text
                    cache_key.append(chunk.token)

                    if unified_parser:
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
                    final_chunk = chunk
                    text = chunk.text
                    raw_text += text
                    cache_key.append(chunk.token)
                    if is_first_chunk:
                        if reasoning_parser and hasattr(reasoning_parser, 'needs_redacted_reasoning_prefix'):
                            if reasoning_parser.needs_redacted_reasoning_prefix():
                                text = reasoning_parser.get_reasoning_open() + text
                        is_first_chunk = False
                    if reasoning_parser:
                        parsed_content, is_complete = reasoning_parser.extract_reasoning_streaming(text)
                        if parsed_content:
                            after_reasoning_close_content = parsed_content.get("after_reasoning_close_content")
                            yield parsed_content
                        if is_complete:
                            reasoning_parser = None
                        if after_reasoning_close_content:
                            text = after_reasoning_close_content
                            after_reasoning_close_content = None
                        else:
                            continue
                    if tool_parser:
                        parsed_content, is_complete = tool_parser.extract_tool_calls_streaming(text)
                        if parsed_content:
                            content = parsed_content.get("content")
                            if content:
                                yield content
                            tool_calls = parsed_content.get("tool_calls")
                            if tool_calls:
                                for tool_call in tool_calls:
                                    yield tool_call
                        continue
                    
                    yield text

            total_tokens = final_chunk.prompt_tokens + final_chunk.generation_tokens
            self.prompt_cache.insert_cache(cache_key, cache)

            if self.debug:
                log_debug_raw_text_response(raw_text)
                log_debug_stats(
                    final_chunk.prompt_tokens,
                    final_chunk.generation_tokens,
                    total_tokens,
                    final_chunk.generation_tps,
                    final_chunk.peak_memory
                )

            yield {
                "__usage__": UsageInfo(
                    prompt_tokens=final_chunk.prompt_tokens,
                    completion_tokens=final_chunk.generation_tokens,
                    total_tokens=total_tokens,
                    prompt_tokens_details=PromptTokenUsageInfo(
                        cached_tokens=total_cached_tokens
                    )
                )
            }

        except asyncio.QueueFull:
            logger.error("Too many requests. Service is at capacity.")
            content = create_error_response("Too many requests. Service is at capacity.", "rate_limit_exceeded", HTTPStatus.TOO_MANY_REQUESTS)
            raise HTTPException(status_code=429, detail=content)
        except Exception as e:
            logger.error(f"Error in text stream generation for request {request_id}: {str(e)}")
            content = create_error_response(f"Failed to generate text stream: {str(e)}", "server_error", HTTPStatus.INTERNAL_SERVER_ERROR)
            raise HTTPException(status_code=500, detail=content)

    async def generate_text_response(self, request: ChatCompletionRequest) -> Dict[str, Any]:
        """
        Generate a complete response for text-only chat completion requests.
        Uses the request queue for handling concurrent requests.

        Args:
            request: ChatCompletionRequest object containing the messages.

        Returns:
            dict: Response content and usage info.
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

            # Cache key must be the FULL input_ids, not rest_input_ids.
            # See generate_text_stream for detailed explanation.
            cache_key = input_ids[:]

            if cache is None:
                cache = self.model.create_prompt_cache()

            total_input_tokens = len(input_ids)
            total_remaining_tokens = len(rest_input_ids)
            total_cached_tokens = total_input_tokens - total_remaining_tokens

            if self.debug:
                log_debug_cache_stats(total_input_tokens, total_remaining_tokens)
            
            prompt_progress_callback = make_prompt_progress_callback() if self.debug else None

            request_data = {
                "input_ids": rest_input_ids,
                "prompt_cache": cache,
                "stream": False,
                "prompt_progress_callback": prompt_progress_callback,
                **model_params
            }

            # Extract explicit model args; remaining kwargs are forwarded.
            input_ids = request_data.pop("input_ids")
            prompt_cache = request_data.pop("prompt_cache")
            request_data.pop("stream")

            response = await self.inference_worker.submit(
                self.model,
                input_ids=input_ids,
                prompt_cache=prompt_cache,
                stream=False,
                **request_data,
            )
            
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

            if model_params.get("schema"):
                logger.info("JSON schema is enabled, disabling reasoning parser and tool parser")
                parsers_result.reasoning_parser = None
                parsers_result.tool_parser = None
                parsers_result.unified_parser = None

            response_text = response.text
            cache_key += response.tokens

            self.prompt_cache.insert_cache(cache_key, cache)

            parsed_response = {
                "reasoning_content": None,
                "tool_calls": None,
                "content": None
            }

            # Handle unified parser
            if parsers_result.is_unified:
                unified_parser = parsers_result.unified_parser
                if unified_parser:
                    parsed_result = unified_parser.parse(response_text)
                    if parsed_result:
                        parsed_response["reasoning_content"] = parsed_result.get("reasoning_content")
                        parsed_response["tool_calls"] = parsed_result.get("tool_calls")
                        parsed_response["content"] = parsed_result.get("content")
                else:
                    parsed_response["content"] = response_text
            # Handle separate parsers
            elif parsers_result.reasoning_parser or parsers_result.tool_parser:
                reasoning_parser = parsers_result.reasoning_parser
                tool_parser = parsers_result.tool_parser

                if reasoning_parser and reasoning_parser.needs_redacted_reasoning_prefix():
                    response_text = reasoning_parser.get_reasoning_open() + response_text

                if reasoning_parser:
                    parsed_content = reasoning_parser.extract_reasoning(response_text)
                    parsed_response["reasoning_content"] = parsed_content.get("reasoning_content")
                    parsed_response["content"] = parsed_content.get("content")
                    response_text = parsed_content.get("after_reasoning_close_content")

                if response_text:
                    if tool_parser:
                        parsed_content = tool_parser.extract_tool_calls(response_text)
                        parsed_response["tool_calls"] = parsed_content.get("tool_calls")
                        parsed_response["content"] = parsed_content.get("content")
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
                    response.peak_memory
                )

            usage = UsageInfo(
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.generation_tokens,
                total_tokens=total_tokens,
                prompt_tokens_details=PromptTokenUsageInfo(
                    cached_tokens=total_cached_tokens
                )
            )

            return {"response": parsed_response, "usage": usage}
                        
        except asyncio.QueueFull:
            logger.error("Too many requests. Service is at capacity.")
            content = create_error_response("Too many requests. Service is at capacity.", "rate_limit_exceeded", HTTPStatus.TOO_MANY_REQUESTS)
            raise HTTPException(status_code=429, detail=content)
        except Exception as e:
            logger.error(f"Error in text response generation: {str(e)}")
            content = create_error_response(f"Failed to generate text response: {str(e)}", "server_error", HTTPStatus.INTERNAL_SERVER_ERROR)
            raise HTTPException(status_code=500, detail=content)
        

    async def get_queue_stats(self) -> Dict[str, Any]:
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
            if hasattr(self, 'inference_worker'):
                self.inference_worker.stop()

            # Force garbage collection
            gc.collect()
            logger.info("MLXLMHandler cleanup completed successfully")
        except Exception as e:
            logger.error(f"Error during MLXLMHandler cleanup: {str(e)}")
            raise

    async def _prepare_text_request(self, request: ChatCompletionRequest) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
        """
        Prepare a text request by parsing model parameters and verifying the format of messages.
        
        Args:
            request: ChatCompletionRequest object containing the messages.
        
        Returns:
            Tuple containing the formatted chat messages and model parameters.
        """
        try:
            request_dict = request.model_dump()
            tools = request_dict.pop("tools")
            tool_choice = request_dict.pop("tool_choice")
            
            if tools:
                request_dict["chat_template_kwargs"]["tools"] = tools
                if tool_choice:
                    request_dict["chat_template_kwargs"]["tool_choice"] = tool_choice

            if request_dict.get("response_format", None):
                response_format = request_dict.pop("response_format", None)
                if response_format.get("type") == "json_schema":
                    request_dict["schema"] = response_format.get("json_schema", None).get("schema", None)
            
            # Format chat messages and merge system messages into index 0
            chat_messages = []
            system_messages = []
            non_system_messages = []
            
            for message in request_dict.get("messages", []):
                # Handle content that might be a list of dictionaries (multimodal format)
                content = message.get("content", None)
                if content is None:
                    # Assistant messages with tool_calls have content: null — keep them
                    if message.get("tool_calls"):
                        message["content"] = ""
                    else:
                        continue
                if isinstance(content, list):
                    # For LM models, extract only text content and concatenate
                    text_parts = []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text" and item.get("text"):
                            text_parts.append(item["text"])
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
                combined_system_content = "\n\n".join([msg["content"] for msg in system_messages if msg.get("content")])
                
                # Create merged system message using the first system message as template
                merged_system_message = system_messages[0].copy()
                merged_system_message["content"] = combined_system_content
                
                # Add merged system message at index 0
                chat_messages.append(merged_system_message)
            
            # Add all non-system messages after the merged system message
            chat_messages.extend(non_system_messages)
            return chat_messages, request_dict
        
        except Exception as e:
            logger.error(f"Failed to prepare text request: {str(e)}")
            content = create_error_response(f"Failed to process request: {str(e)}", "bad_request", HTTPStatus.BAD_REQUEST)
            raise HTTPException(status_code=400, detail=content)