"""MLX vision-language model handler for multimodal chat completions."""

from __future__ import annotations

import asyncio
import base64
from collections.abc import AsyncGenerator
import gc
from http import HTTPStatus
import time
from typing import Any, NoReturn
import uuid

from fastapi import HTTPException
from loguru import logger

from ..core import AudioProcessor, ImageProcessor, VideoProcessor
from ..core.queue import RequestQueue
from ..message_converters import MessageConverterManager
from ..models.mlx_vlm import MLX_VLM
from ..parsers import ParserManager
from ..schemas.openai import (
    ChatCompletionContentPart,
    ChatCompletionContentPartImage,
    ChatCompletionContentPartInputAudio,
    ChatCompletionContentPartText,
    ChatCompletionContentPartVideo,
    ChatCompletionRequest,
    EmbeddingRequest,
    UsageInfo,
)
from ..utils.debug_logging import log_debug_raw_text_response, log_debug_request, log_debug_stats
from ..utils.errors import create_error_response


class MLXVLMHandler:
    """
    Handler class for making requests to the underlying MLX multimodal model service.

    Provides caching, concurrent image processing, audio processing, and robust error handling.
    """

    def __init__(
        self,
        model_path: str,
        *,
        context_length: int = 32768,
        max_workers: int = 4,
        max_concurrency: int = 1,
        disable_auto_resize: bool = False,
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
            Maximum context length for the model. Default is 32768.
        max_workers : int, optional
            Maximum number of worker threads for image processing. Default is 4.
        max_concurrency : int, optional
            Maximum number of concurrent model inference tasks. Default is 1.
        disable_auto_resize : bool, optional
            Whether to disable automatic image resizing. Default is False.
        enable_auto_tool_choice : bool, optional
            Enable automatic tool choice. Default is False.
        tool_call_parser : str | None, optional
            Tool call parser name.
        reasoning_parser : str | None, optional
            Reasoning parser name.
        message_converter : str | None, optional
            Message converter name.
        trust_remote_code : bool, optional
            Whether to trust remote code. Default is False.
        chat_template_file : str | None, optional
            Path to a custom chat template file.
        debug : bool, optional
            Enable debug mode. Default is False.
        """
        self.model_path = model_path
        self.model = MLX_VLM(
            model_path,
            context_length=context_length,
            trust_remote_code=trust_remote_code,
            chat_template_file=chat_template_file or "",
        )
        self.image_processor = ImageProcessor(max_workers)
        self.audio_processor = AudioProcessor(max_workers)
        self.video_processor = VideoProcessor(max_workers)
        self.disable_auto_resize = disable_auto_resize
        self.model_created = int(time.time())  # Store creation time when model is loaded
        self.model_type = self.model.get_model_type()

        # Store parser configuration
        self.enable_auto_tool_choice = enable_auto_tool_choice
        self.reasoning_parser_name = reasoning_parser
        self.tool_parser_name = tool_call_parser
        self.message_converter = MessageConverterManager.create_converter(message_converter)
        # Debug mode
        self.debug = debug

        # Initialize request queue for multimodal and text tasks
        # We use the same queue for both multimodal and text tasks for simplicity
        # and to ensure we don't overload the model with too many concurrent requests
        self.request_queue = RequestQueue(max_concurrency=max_concurrency)

        logger.info(f"Initialized MLXVLMHandler with model path: {model_path}")
        if disable_auto_resize:
            logger.info("Auto-resize is disabled for image processing")

    async def get_models(self) -> list[dict[str, Any]]:
        """Get list of available models with their metadata.

        Returns
        -------
        list[dict[str, Any]]
            List of available models with their metadata.
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
            logger.error(f"Error getting models. {type(e).__name__}: {e}")
            return []

    async def initialize(self, queue_config: dict[str, Any] | None = None) -> None:
        """Initialize the handler and start the request queue.

        Parameters
        ----------
        queue_config : dict[str, Any] | None, optional
            Configuration for the request queue. Expected keys include
            ``max_concurrency``, ``timeout``, and ``queue_size``. If None,
            the current request queue defaults are used.
        """
        if not queue_config:
            queue_config = {"max_concurrency": self.request_queue.max_concurrency}
        self.request_queue = RequestQueue(
            max_concurrency=queue_config.get("max_concurrency", self.request_queue.max_concurrency),
            timeout=queue_config.get("timeout", self.request_queue.timeout),
            queue_size=queue_config.get("queue_size", self.request_queue.queue_size),
        )
        await self.request_queue.start(self._process_request)
        logger.info("Initialized MLXVLMHandler and started request queue")

    async def generate_multimodal_stream(
        self, request: ChatCompletionRequest
    ) -> AsyncGenerator[Any, None]:
        """Generate a streaming response for multimodal chat completion requests.

        Parameters
        ----------
        request : ChatCompletionRequest
            ChatCompletionRequest object containing the messages and parameters.

        Yields
        ------
        Any
            Streaming response chunks. Typically strings for text chunks or dicts
            when parsers produce structured output. A usage dict with token counts
            is yielded at the end of the stream.
        """
        try:
            # Enforce streaming mode for queued multimodal requests to ensure
            # the underlying model returns a generator instead of a full string.
            request.stream = True

            # Create a unique request ID
            request_id = f"multimodal-{uuid.uuid4()}"

            request_dict = await self._prepare_multimodal_request(request)

            if self.debug:
                log_debug_request(request_dict)
                request_dict["verbose"] = True

            # Submit to the multimodal queue and get the generator
            response_generator = await self.request_queue.submit(request_id, request_dict)

            # Create parsers using ParserManager
            parsers_result = ParserManager.create_parsers(
                reasoning_parser_name=self.reasoning_parser_name,
                tool_parser_name=self.tool_parser_name,
            )

            chat_template_kwargs = request_dict.get("chat_template_kwargs", {})
            enable_thinking = chat_template_kwargs.get("enable_thinking", True)

            # Handle enable_thinking flag for separate reasoning parsers
            if not enable_thinking and parsers_result.reasoning_parser:
                if parsers_result.reasoning_parser.respects_enable_thinking():
                    parsers_result.reasoning_parser = None

            after_reasoning_close_content = None
            final_chunk = None
            is_first_chunk = True
            raw_text = ""  # only use for debugging

            # Handle unified parser streaming
            if parsers_result.is_unified:
                unified_parser = parsers_result.unified_parser
                for chunk in response_generator:
                    if chunk is None:
                        continue
                    final_chunk = chunk
                    text = chunk.text
                    raw_text += text

                    if unified_parser:
                        parsed_result, is_complete = unified_parser.parse_streaming(text)
                        if parsed_result:
                            # Unified parser may return a dict or a passthrough string
                            if isinstance(parsed_result, dict):
                                reasoning = parsed_result.get("reasoning_content")
                                if reasoning:
                                    yield {"reasoning_content": reasoning}
                                tool_calls = parsed_result.get("tool_calls")
                                if tool_calls:
                                    for tool_call in tool_calls:
                                        yield tool_call
                                content = parsed_result.get("content")
                                if content:
                                    yield content
                            else:
                                # passthrough string chunk
                                yield parsed_result
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
                    if is_first_chunk:
                        if reasoning_parser and hasattr(
                            reasoning_parser, "needs_redacted_reasoning_prefix"
                        ):
                            if reasoning_parser.needs_redacted_reasoning_prefix():
                                text = reasoning_parser.get_reasoning_open() + text
                        is_first_chunk = False
                    if reasoning_parser:
                        parsed_reasoning_content, is_complete = (
                            reasoning_parser.extract_reasoning_streaming(text)
                        )

                        if isinstance(parsed_reasoning_content, dict):
                            after_reasoning_close_content = parsed_reasoning_content.get(
                                "after_reasoning_close_content"
                            )
                            yield parsed_reasoning_content
                        if is_complete:
                            reasoning_parser = None
                        if after_reasoning_close_content:
                            text = after_reasoning_close_content
                            after_reasoning_close_content = None
                        else:
                            continue
                    if tool_parser:
                        parsed_tool_content, is_complete = tool_parser.extract_tool_calls_streaming(
                            text
                        )
                        if isinstance(parsed_tool_content, dict):
                            content = parsed_tool_content.get("content")
                            if content:
                                yield content
                            tool_calls = parsed_tool_content.get("tool_calls")
                            if tool_calls:
                                for tool_call in tool_calls:
                                    yield tool_call
                        continue

                    yield text

            if final_chunk is not None:
                prompt_tokens = getattr(final_chunk, "prompt_tokens", 0)
                generation_tokens = getattr(final_chunk, "generation_tokens", 0)
                total_tokens = prompt_tokens + generation_tokens

                if self.debug:
                    log_debug_raw_text_response(raw_text)
                    log_debug_stats(
                        prompt_tokens,
                        generation_tokens,
                        total_tokens,
                        getattr(final_chunk, "generation_tps", 0.0),
                        getattr(final_chunk, "peak_memory", 0.0),
                    )

                yield {
                    "__usage__": UsageInfo(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=generation_tokens,
                        total_tokens=total_tokens,
                    )
                }

        except asyncio.QueueFull as e:
            logger.error("Too many requests. Service is at capacity.")
            detail = create_error_response(
                "Too many requests. Service is at capacity.",
                "rate_limit_exceeded",
                HTTPStatus.TOO_MANY_REQUESTS,
            )
            raise HTTPException(status_code=HTTPStatus.TOO_MANY_REQUESTS, detail=detail) from e
        except HTTPException:
            # Preserve existing HTTP error semantics from request prep
            raise
        except Exception as e:
            logger.error(
                f"Error in multimodal stream generation for request {request_id}. {type(e).__name__}: {e}"
            )
            detail = create_error_response(
                f"Failed to generate multimodal stream: {e}",
                "server_error",
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail=detail) from e

    async def generate_multimodal_response(self, request: ChatCompletionRequest) -> dict[str, Any]:
        """Generate a complete response for multimodal chat completion requests.

        Parameters
        ----------
        request : ChatCompletionRequest
            The incoming chat completion request.

        Returns
        -------
        dict[str, Any]
            Dictionary with keys ``response`` (parsed response dict) and ``usage`` (UsageInfo).
        """
        try:
            # Create a unique request ID
            request_id = f"multimodal-{uuid.uuid4()}"

            request_dict = await self._prepare_multimodal_request(request)

            if self.debug:
                log_debug_request(request_dict)
                request_dict["verbose"] = True

            response = await self.request_queue.submit(request_id, request_dict)

            # Create parsers using ParserManager
            parsers_result = ParserManager.create_parsers(
                reasoning_parser_name=self.reasoning_parser_name,
                tool_parser_name=self.tool_parser_name,
            )

            chat_template_kwargs = request_dict.get("chat_template_kwargs", {})
            enable_thinking = chat_template_kwargs.get("enable_thinking", True)

            # Handle enable_thinking flag for separate reasoning parsers
            if not enable_thinking and parsers_result.reasoning_parser:
                if parsers_result.reasoning_parser.respects_enable_thinking():
                    parsers_result.reasoning_parser = None

            parsed_response: dict[str, Any] = {
                "reasoning_content": None,
                "tool_calls": None,
                "content": None,
            }

            response_text = getattr(response, "text", "")

            # Handle unified parser
            if parsers_result.is_unified:
                unified_parser = parsers_result.unified_parser
                if unified_parser:
                    parsed_result = unified_parser.parse(response_text)
                    if parsed_result is not None:
                        if isinstance(parsed_result, dict):
                            parsed_response["reasoning_content"] = parsed_result.get(
                                "reasoning_content"
                            )
                            parsed_response["tool_calls"] = parsed_result.get("tool_calls")
                            parsed_response["content"] = parsed_result.get("content")
                        elif isinstance(parsed_result, str):
                            parsed_response["content"] = parsed_result

            # (later) compute tokens safely
            # Handle separate parsers
            elif parsers_result.reasoning_parser or parsers_result.tool_parser:
                reasoning_parser = parsers_result.reasoning_parser
                tool_parser = parsers_result.tool_parser

                if reasoning_parser and reasoning_parser.needs_redacted_reasoning_prefix():
                    response_text = reasoning_parser.get_reasoning_open() + response_text

                if reasoning_parser:
                    parsed_reasoning_content = reasoning_parser.extract_reasoning(response_text)
                    if isinstance(parsed_reasoning_content, dict):
                        parsed_response["reasoning_content"] = parsed_reasoning_content.get(
                            "reasoning_content"
                        )
                        parsed_response["content"] = parsed_reasoning_content.get("content")
                        response_text = parsed_reasoning_content.get(
                            "after_reasoning_close_content"
                        )

                if response_text:
                    if tool_parser:
                        parsed_tool_content = tool_parser.extract_tool_calls(response_text)
                        if isinstance(parsed_tool_content, dict):
                            parsed_response["tool_calls"] = parsed_tool_content.get("tool_calls")
                            parsed_response["content"] = parsed_tool_content.get("content")
            else:
                parsed_response["content"] = response_text

            prompt_tokens = getattr(response, "prompt_tokens", 0)
            generation_tokens = getattr(response, "generation_tokens", 0)
            total_tokens = prompt_tokens + generation_tokens

            if self.debug:
                log_debug_raw_text_response(getattr(response, "text", ""))
                log_debug_stats(
                    prompt_tokens,
                    generation_tokens,
                    total_tokens,
                    getattr(response, "generation_tps", 0.0),
                    getattr(response, "peak_memory", 0.0),
                )

            usage = UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=generation_tokens,
                total_tokens=total_tokens,
            )

        except Exception as e:
            logger.error(f"Error in multimodal response generation: {e!s}")
            content = create_error_response(
                f"Failed to generate multimodal response: {e!s}",
                "server_error",
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            raise HTTPException(status_code=500, detail=content) from e
        else:
            return {"response": parsed_response, "usage": usage}

    async def generate_embeddings_response(self, _request: EmbeddingRequest) -> NoReturn:
        """Generate embeddings for a given text input.

        Notes
        -----
        Embeddings are not supported for VLM models.

        Parameters
        ----------
        _request : EmbeddingRequest
            EmbeddingRequest object (not used).

        Raises
        ------
        HTTPException
            Always raises HTTP 400 indicating embeddings are not supported.
        """
        # Embeddings are not supported for VLM models
        content = create_error_response(
            "Embeddings are not supported for VLM models",
            "bad_request",
            HTTPStatus.BAD_REQUEST,
        )
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=content)

    async def close(self) -> None:
        """Clean up resources asynchronously.

        Returns
        -------
        None
        """
        if hasattr(self, "image_processor"):
            await self.image_processor.cleanup()
        if hasattr(self, "audio_processor"):
            await self.audio_processor.cleanup()
        if hasattr(self, "video_processor"):
            await self.video_processor.cleanup()

    async def cleanup(self) -> None:
        """Cleanup resources and stop the request queue before shutdown.

        Notes
        -----
        Ensures pending requests are cancelled and resources such as
        the image, audio, and video processors are released.

        Returns
        -------
        None
        """
        logger.info("Cleaning up MLXVLMHandler resources")
        if hasattr(self, "request_queue"):
            try:
                await self.request_queue.stop()
            except Exception as e:
                logger.error(f"Error stopping request queue: {e}")
        if hasattr(self, "image_processor"):
            try:
                await self.image_processor.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up image processor: {e}")
        if hasattr(self, "audio_processor"):
            try:
                await self.audio_processor.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up audio processor: {e}")
        if hasattr(self, "video_processor"):
            try:
                await self.video_processor.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up video processor: {e}")

        # Force garbage collection after cleanup
        gc.collect()
        logger.info("MLXVLMHandler cleanup completed successfully")

    async def _process_request(self, request_data: dict[str, Any]) -> tuple[Any, int]:
        """Process a multimodal request (worker for the request queue).

        Parameters
        ----------
        request_data : dict[str, Any]
            Dictionary containing the request data. Expected keys include ``messages`` and ``stream``.

        Returns
        -------
        tuple[Any, int]
            Tuple (response, prompt_tokens) where ``response`` is the model response
            and ``prompt_tokens`` is the number of prompt tokens used.
        """
        try:
            # Extract request parameters
            messages = request_data.get("messages", [])
            stream = request_data.get("stream", False)

            # Remove these keys from model_params
            model_params = request_data.copy()
            model_params.pop("messages", None)
            model_params.pop("stream", None)

            if self.message_converter:
                logger.info("Message converter is enabled, converting messages...")
                messages = self.message_converter.convert_messages(messages)
                logger.info("Messages converted successfully")

            refined_messages = []
            logger.info("Filtering out None values from messages...")
            for message in messages:
                cleaned_message = {k: v for k, v in message.items() if v is not None}
                refined_messages.append(cleaned_message)
            logger.info("Messages filtered successfully")
            messages = refined_messages

            # Call the model
            response = self.model(
                messages=messages,
                stream=stream,
                **model_params,
            )
            # Force garbage collection after model inference
            gc.collect()

        except Exception as e:
            logger.error(f"Error processing multimodal request. {type(e).__name__}: {e}")
            # Clean up on error
            gc.collect()
            raise
        else:
            return response, getattr(response, "prompt_tokens", 0)

    async def get_queue_stats(self) -> dict[str, Any]:
        """Get statistics from the request queue.

        Returns
        -------
        dict[str, Any]
            Queue statistics.
        """
        return self.request_queue.get_queue_stats()

    async def _reformat_multimodal_content_part(
        self, content_part: ChatCompletionContentPart
    ) -> dict[str, Any]:
        """Reformat a multimodal message content part into a dictionary.

        Parameters
        ----------
        content_part : ChatCompletionContentPart
            The content part to reformat.

        Returns
        -------
        dict[str, Any]
            Processed content part dictionary and optional path info.
        """
        if (
            isinstance(content_part, ChatCompletionContentPartImage)
            and content_part.image_url is not None
        ):
            image_url = content_part.image_url.url
            # Validate base64 data URLs before processing
            self._validate_image_url(image_url)
            image_path = await self.image_processor.process_image_url(
                image_url, resize=not self.disable_auto_resize
            )
            return {"content_part": {"type": "image", "image": image_path}, "path": image_path}

        if (
            isinstance(content_part, ChatCompletionContentPartInputAudio)
            and content_part.input_audio is not None
        ):
            content = create_error_response(
                "Audio input is not supported for VLM models",
                "bad_request",
                HTTPStatus.BAD_REQUEST,
            )
            raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=content)

        if (
            isinstance(content_part, ChatCompletionContentPartVideo)
            and content_part.video_url is not None
        ):
            video_url = content_part.video_url.url
            # Note: Video validation could be added here if needed
            video_path = await self.video_processor.process_video_url(video_url)
            return {
                "content_part": {
                    "type": "video",
                    "video": video_path,
                },
                "path": video_path,
            }

        if isinstance(content_part, ChatCompletionContentPartText):
            return {"content_part": {"type": "text", "text": content_part.text}}

        # Fallback for unknown types
        return {"content_part": {"type": "text", "text": str(content_part)}}

    async def _prepare_multimodal_request(self, request: ChatCompletionRequest) -> dict[str, Any]:
        """Prepare the multimodal request by processing messages with text, images, and videos.

        Parameters
        ----------
        request : ChatCompletionRequest
            The incoming request containing messages and parameters.

        Returns
        -------
        dict[str, Any]
            A dictionary containing processed request data with keys such as
            ``messages``, ``images``, ``videos``, and model parameters
            (e.g., temperature, top_p, max_tokens).

        Raises
        ------
        HTTPException
            If the request is invalid or processing fails.
        """
        chat_messages = []
        images = []
        audios = []
        videos = []

        try:
            # Process each message in the request
            for message in request.messages:
                # Handle system and assistant messages (simple text content)
                if message.role in ["system", "assistant"]:
                    chat_messages.append({"role": message.role, "content": message.content})
                    continue

                # Handle user messages
                if message.role == "user":
                    # Case 1: Simple string content
                    if isinstance(message.content, str):
                        chat_messages.append({"role": "user", "content": message.content})
                        continue

                    # Case 2: Content is a list of dictionaries or objects
                    if isinstance(message.content, list):
                        formatted_content_parts = []

                        for content_part in message.content:
                            formatted_content_part = await self._reformat_multimodal_content_part(
                                content_part
                            )
                            if isinstance(content_part, ChatCompletionContentPartImage):
                                images.append(formatted_content_part["path"])
                            elif isinstance(content_part, ChatCompletionContentPartInputAudio):
                                audios.append(formatted_content_part["path"])
                            elif isinstance(content_part, ChatCompletionContentPartVideo):
                                videos.append(formatted_content_part["path"])

                            formatted_content_parts.append(formatted_content_part["content_part"])
                        chat_messages.append({"role": "user", "content": formatted_content_parts})
                    else:
                        content = create_error_response(
                            "Invalid message content format",
                            "invalid_request_error",
                            HTTPStatus.BAD_REQUEST,
                        )
                        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=content)
                elif message.role == "tool":
                    chat_messages.append({"role": "tool", "content": message.content})
                    continue

            chat_template_kwargs = request.chat_template_kwargs.model_dump()
            request_dict = {
                "messages": chat_messages,
                "images": images,
                "audios": audios,
                "videos": videos,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "frequency_penalty": request.frequency_penalty,
                "presence_penalty": request.presence_penalty,
                "max_tokens": request.max_tokens,
                "chat_template_kwargs": chat_template_kwargs,
                "stream": request.stream,
            }

            tools = request.tools or None
            tool_choice = request.tool_choice or None

            if tools:
                # Enable auto tool choice if requested via CLI flag
                if self.enable_auto_tool_choice and tool_choice == "auto":
                    chat_template_kwargs["tool_choice"] = "auto"
                elif tool_choice:
                    logger.warning("Tool choice has not supported yet, will be ignored.")
                chat_template_kwargs["tools"] = tools

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to prepare multimodal request. {type(e).__name__}: {e}")
            content = create_error_response(
                f"Failed to process request: {e}", "bad_request", HTTPStatus.BAD_REQUEST
            )
            raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=content) from e
        else:
            return request_dict

    def _validate_image_url(self, url: str) -> None:
        """Validate image URL format.

        Parameters
        ----------
        url : str
            The image URL to validate.

        Raises
        ------
        HTTPException
            If the URL is invalid or base64 decoding fails.
        """
        if not url:
            content = create_error_response(
                "Empty image URL provided", "invalid_request_error", HTTPStatus.BAD_REQUEST
            )
            raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=content)

        # Validate base64 images
        if url.startswith("data:"):
            try:
                header, encoded = url.split(",", 1)
                if not header.startswith("data:image/"):
                    raise ValueError("Invalid image format")
                base64.b64decode(encoded)
            except Exception as e:
                content = create_error_response(
                    f"Invalid base64 image: {e}", "invalid_request_error", HTTPStatus.BAD_REQUEST
                )
                raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=content) from e

    def _validate_audio_data(self, url: str) -> None:
        """Validate audio data URL format.

        Parameters
        ----------
        url : str
            The audio data URL to validate.

        Raises
        ------
        HTTPException
            If the audio data is invalid or base64 decoding fails.
        """
        if not url:
            content = create_error_response(
                "Empty audio data provided", "invalid_request_error", HTTPStatus.BAD_REQUEST
            )
            raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=content)

        # Validate base64 audio
        if url.startswith("data:"):
            try:
                header, encoded = url.split(",", 1)
                if not header.startswith("data:audio/"):
                    raise ValueError("Invalid audio format")
                base64.b64decode(encoded)
            except Exception as e:
                content = create_error_response(
                    f"Invalid base64 audio: {e}", "invalid_request_error", HTTPStatus.BAD_REQUEST
                )
                raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=content) from e
