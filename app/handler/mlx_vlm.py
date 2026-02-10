import asyncio
import base64
import gc
import time
import uuid
from http import HTTPStatus
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import torch
from fastapi import HTTPException
from loguru import logger
from mlx_vlm.video_generate import process_vision_info

from ..core import AudioProcessor, ImageProcessor, InferenceWorker, VideoProcessor
from ..message_converters import MessageConverterManager
from ..models.mlx_vlm import MLX_VLM
from ..parsers import ParserManager
from ..schemas.openai import (
    ChatCompletionContentPart,
    ChatCompletionContentPartImage,
    ChatCompletionContentPartInputAudio,
    ChatCompletionContentPartVideo,
    ChatCompletionRequest,
    UsageInfo,
)
from ..utils.debug_logging import (
    log_debug_prompt,
    log_debug_raw_text_response,
    log_debug_request,
    log_debug_stats,
)
from ..utils.errors import create_error_response

class MLXVLMHandler:
    """
    Handler class for making requests to the underlying MLX multimodal model service.
    Provides concurrent image processing, audio processing, and robust error handling.
    """

    def __init__(self, model_path: str, context_length: int | None = None, max_workers: int = 4, max_concurrency: int = 1, disable_auto_resize: bool = False, enable_auto_tool_choice: bool = False, tool_call_parser: str = None, reasoning_parser: str = None, message_converter: str = None, trust_remote_code: bool = False, chat_template_file: str = None, debug: bool = False):
        """
        Initialize the handler with the specified model path.
        
        Args:
            model_path (str): Path to the model directory.
            context_length (int | None): Maximum context length for the model. If None, uses model default.
            max_workers (int): Maximum number of worker threads for image processing.
            max_concurrency (int): Maximum number of concurrent model inference tasks.
            disable_auto_resize (bool): Whether to disable automatic image resizing.
            enable_auto_tool_choice (bool): Enable automatic tool choice.
            tool_call_parser (str): Name of the tool call parser to use (qwen3, glm4_moe, harmony, minimax, ...)
            reasoning_parser (str): Name of the reasoning parser to use (qwen3, qwen3_next, glm4_moe, harmony, minimax, ...).
            trust_remote_code (bool): Enable trust_remote_code when loading models.
            chat_template_file (str): Path to a custom chat template file.
        """
        self.model_path = model_path
        self.model = MLX_VLM(model_path, context_length=context_length, trust_remote_code=trust_remote_code, chat_template_file=chat_template_file)
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

        # Dedicated inference thread — keeps the event loop free during
        # blocking MLX model computation.
        self.inference_worker = InferenceWorker()
        
        logger.info(f"Initialized MLXHandler with model path: {model_path}")
        if disable_auto_resize:
            logger.info("Auto-resize is disabled for image processing")

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
        logger.info("Initialized MLXVLMHandler and started inference worker")

    async def generate_multimodal_stream(self, request: ChatCompletionRequest):
        """
        Generate a streaming response for multimodal chat completion requests.
        
        Args:
            request: ChatCompletionRequest object containing the messages.
        
        Returns:
            AsyncGenerator: Yields response chunks.
        """
        
        # Create a unique request ID
        request_id = f"multimodal-{uuid.uuid4()}"
        
        try:
            request_dict = await self._prepare_multimodal_request(request)
            
            # Extract messages, images, videos, audios
            messages = request_dict["messages"]
            chat_template_kwargs = request_dict["chat_template_kwargs"]
            
            # Create input prompt
            input_prompt = self.model.create_input_prompt(messages, chat_template_kwargs)
            
            if self.debug:
                log_debug_prompt(input_prompt)
            
            # Process vision info and create inputs
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.model.create_inputs(input_prompt, image_inputs, video_inputs)
            
            request_dict["prompt"] = input_prompt
            
            # Convert torch tensors to mlx arrays
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    inputs[key] = mx.array(value)

            # merge all keys from inputs to request_dict
            request_dict.update(inputs)
            request_dict["stream"] = True
            
            if self.debug:
                log_debug_request(request_dict)
                request_dict["verbose"] = True
            
            # Extract explicit model args; remaining kwargs are forwarded
            # to the model on the inference thread.
            prompt = request_dict.pop("prompt")
            request_dict.pop("stream")

            response_generator = self.inference_worker.submit_stream(
                self.model,
                prompt=prompt,
                stream=True,
                **request_dict,
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

            # Disable parsers when JSON schema is enabled
            if request_dict.get("schema"):
                logger.info("JSON schema is enabled, disabling reasoning parser and tool parser")
                parsers_result.reasoning_parser = None
                parsers_result.tool_parser = None

            after_reasoning_close_content = None
            final_chunk = None
            is_first_chunk = True
            raw_text = ""  # only use for debugging

            # Handle unified parser streaming
            if parsers_result.is_unified:
                unified_parser = parsers_result.unified_parser
                async for chunk in response_generator:
                    if chunk is None:
                        continue
                    final_chunk = chunk
                    text = chunk.text
                    raw_text += text
                    
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
                
                async for chunk in response_generator:
                    if chunk is None:
                        continue
                    final_chunk = chunk
                    text = chunk.text
                    raw_text += text
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
                    total_tokens=total_tokens
                )
            }
        
        except asyncio.QueueFull:
            logger.error("Too many requests. Service is at capacity.")
            content = create_error_response("Too many requests. Service is at capacity.", "rate_limit_exceeded", HTTPStatus.TOO_MANY_REQUESTS)
            raise HTTPException(status_code=429, detail=content)

        except Exception as e:
            logger.error(f"Error in multimodal stream generation for request {request_id}: {str(e)}")
            content = create_error_response(f"Failed to generate multimodal stream: {str(e)}", "server_error", HTTPStatus.INTERNAL_SERVER_ERROR)
            raise HTTPException(status_code=500, detail=content)

    async def generate_multimodal_response(self, request: ChatCompletionRequest):
        """
        Generate a complete response for multimodal chat completion requests.
        Uses the request queue for handling concurrent requests.
        
        Args:
            request: ChatCompletionRequest object containing the messages.
        
        Returns:
            str: Complete response.
        """
        try:
            # Create a unique request ID
            request_id = f"multimodal-{uuid.uuid4()}"
            
            request_dict = await self._prepare_multimodal_request(request)
            
            # Extract messages, images, videos, audios
            messages = request_dict["messages"]
            chat_template_kwargs = request_dict["chat_template_kwargs"]
            
            # Create input prompt
            input_prompt = self.model.create_input_prompt(messages, chat_template_kwargs)
            
            if self.debug:
                log_debug_prompt(input_prompt)
            
            # Process vision info and create inputs
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.model.create_inputs(input_prompt, image_inputs, video_inputs)

            request_dict["prompt"] = input_prompt
            
            # Convert torch tensors to mlx arrays
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    inputs[key] = mx.array(value)

            # merge all keys from inputs to request_dict
            request_dict.update(inputs)
            if self.debug:
                log_debug_request(request_dict)
                request_dict["verbose"] = True

            # Extract explicit model args; remaining kwargs are forwarded.
            prompt = request_dict.pop("prompt")
            request_dict.pop("stream", None)

            response = await self.inference_worker.submit(
                self.model,
                prompt=prompt,
                stream=False,
                **request_dict,
            )

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

            # Disable parsers when JSON schema is enabled
            if request_dict.get("schema"):
                logger.info("JSON schema is enabled, disabling reasoning parser and tool parser")
                parsers_result.reasoning_parser = None
                parsers_result.tool_parser = None

            parsed_response = {
                "reasoning_content": None,
                "tool_calls": None,
                "content": None
            }
            response_text = response.text

            # Handle unified parser
            if parsers_result.is_unified:
                unified_parser = parsers_result.unified_parser
                parsed_result = unified_parser.parse(response_text)
                if parsed_result:
                    parsed_response["reasoning_content"] = parsed_result.get("reasoning_content")
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
                total_tokens=total_tokens
            )
            
            return {"response": parsed_response, "usage": usage}
                        
        except asyncio.QueueFull:
            logger.error("Too many requests. Service is at capacity.")
            content = create_error_response("Too many requests. Service is at capacity.", "rate_limit_exceeded", HTTPStatus.TOO_MANY_REQUESTS)
            raise HTTPException(status_code=429, detail=content)
        except Exception as e:
            logger.error(f"Error in multimodal response generation: {str(e)}")
            content = create_error_response(f"Failed to generate multimodal response: {str(e)}", "server_error", HTTPStatus.INTERNAL_SERVER_ERROR)
            raise HTTPException(status_code=500, detail=content)

    def __del__(self):
        """Cleanup resources on deletion."""
        # Removed async cleanup from __del__; use close() instead
        pass

    async def close(self):
        """Explicitly cleanup resources asynchronously."""
        if hasattr(self, 'image_processor'):
            await self.image_processor.cleanup()
        if hasattr(self, 'audio_processor'):
            await self.audio_processor.cleanup()
        if hasattr(self, 'video_processor'):
            await self.video_processor.cleanup()

    async def cleanup(self) -> None:
        """Cleanup resources and stop the inference worker before shutdown.

        This method ensures all pending requests are properly completed
        and resources are released, including media processors.
        """
        try:
            logger.info("Cleaning up MLXVLMHandler resources")
            if hasattr(self, 'inference_worker'):
                self.inference_worker.stop()
            if hasattr(self, 'image_processor'):
                await self.image_processor.cleanup()
            if hasattr(self, 'audio_processor'):
                await self.audio_processor.cleanup()
            if hasattr(self, 'video_processor'):
                await self.video_processor.cleanup()

            # Force garbage collection after cleanup
            gc.collect()
            logger.info("MLXVLMHandler cleanup completed successfully")
        except Exception as e:
            logger.error(f"Error during MLXVLMHandler cleanup: {str(e)}")
            raise

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

    async def _reformat_multimodal_content_part(self, content_part: ChatCompletionContentPart) -> Tuple[Dict[str, Any], bool]:
        """
        Reformat a multimodal message content part into a dictionary.
        """
        if isinstance(content_part, ChatCompletionContentPartImage):
            image_url = content_part.image_url.url
            image_path = await self.image_processor.process_image_url(image_url, resize=not self.disable_auto_resize)
            return {
                "content_part": {
                    "type": "image",
                    "image": image_path
                },
                "path": image_path
            }

        if isinstance(content_part, ChatCompletionContentPartInputAudio):
            audio_url = content_part.input_audio.data
            audio_path = await self.audio_processor.process_audio_url(audio_url)
            return {
                "content_part": {
                    "type": "audio",
                    "audio": audio_path
                },
                "path": audio_path
            }

        if isinstance(content_part, ChatCompletionContentPartVideo):
            video_url = content_part.video_url.url
            video_path = await self.video_processor.process_video_url(video_url)
            return {
                "content_part": {
                    "type": "video",
                    "video": video_path,
                },
                "path": video_path
            }

        return {
            "content_part": {
                "type": "text",
                "text": content_part.text
            }
        }


    async def _prepare_multimodal_request(self, request: ChatCompletionRequest) -> Tuple[List[Dict[str, Any]], List[str], List[str], Dict[str, Any]]:
        """
        Prepare the multimodal request by processing messages with text, images, and audio.
        
        This method:
        1. Extracts text messages, image URLs, and audio data from the request
        2. Processes image URLs and audio data to get local file paths
        3. Prepares model parameters
        4. Returns processed data ready for model inference
        
        Args:
            request (ChatCompletionRequest): The incoming request containing messages and parameters.
            
        Returns:
            Tuple[List[Dict[str, Any]], List[str], List[str], Dict[str, Any]]: A tuple containing:
                - List of processed chat messages
                - List of processed image paths
                - List of processed audio paths
                - List of processed video paths
                - Dictionary of model parameters
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
                            formatted_content_part = await self._reformat_multimodal_content_part(content_part)
                            if isinstance(content_part, ChatCompletionContentPartImage):
                                images.append(formatted_content_part["path"])
                            elif isinstance(content_part, ChatCompletionContentPartInputAudio):
                                audios.append(formatted_content_part["path"])
                            elif isinstance(content_part, ChatCompletionContentPartVideo):
                                videos.append(formatted_content_part["path"])

                            formatted_content_parts.append(formatted_content_part["content_part"])
                        chat_messages.append({"role": "user", "content": formatted_content_parts})
                    else:
                        content = create_error_response("Invalid message content format", "invalid_request_error", HTTPStatus.BAD_REQUEST)
                        raise HTTPException(status_code=400, detail=content)

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
                "chat_template_kwargs": request.chat_template_kwargs.model_dump(),
                "stream": request.stream
            }

            tools = request.tools 
            tool_choice = request.tool_choice
            
            if tools:
                request_dict["chat_template_kwargs"]["tools"] = tools
                if tool_choice:
                    request_dict["chat_template_kwargs"]["tool_choice"] = tool_choice

            # Extract response_format for JSON schema structured outputs
            if request.response_format:
                response_format = request.response_format
                if isinstance(response_format, dict) and response_format.get("type") == "json_schema":
                    json_schema_dict = response_format.get("json_schema", {})
                    if isinstance(json_schema_dict, dict):
                        request_dict["schema"] = json_schema_dict.get("schema", None)
            
            return request_dict

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to prepare multimodal request: {str(e)}")
            content = create_error_response(f"Failed to process request: {str(e)}", "bad_request", HTTPStatus.BAD_REQUEST)
            raise HTTPException(status_code=400, detail=content)
            
    def _validate_image_url(self, url: str) -> None:
        """
        Validate image URL format.
        
        Args:
            url: The image URL to validate
            
        Raises:
            HTTPException: If URL is invalid
        """
        if not url:
            content = create_error_response("Empty image URL provided", "invalid_request_error", HTTPStatus.BAD_REQUEST)
            raise HTTPException(status_code=400, detail=content)
            
        # Validate base64 images
        if url.startswith("data:"):
            try:
                header, encoded = url.split(",", 1)
                if not header.startswith("data:image/"):
                    raise ValueError("Invalid image format")
                base64.b64decode(encoded)
            except Exception as e:
                content = create_error_response(f"Invalid base64 image: {str(e)}", "invalid_request_error", HTTPStatus.BAD_REQUEST)
                raise HTTPException(status_code=400, detail=content)
                
    def _validate_audio_data(self, url: str) -> None:
        """
        Validate audio data URL format.
        
        Args:
            url: The audio data URL to validate
            
        Raises:
            HTTPException: If audio data is invalid
        """
        if not url:
            content = create_error_response("Empty audio data provided", "invalid_request_error", HTTPStatus.BAD_REQUEST)
            raise HTTPException(status_code=400, detail=content)
            
        # Validate base64 audio
        if url.startswith("data:"):
            try:
                header, encoded = url.split(",", 1)
                if not header.startswith("data:audio/"):
                    raise ValueError("Invalid audio format")
                base64.b64decode(encoded)
            except Exception as e:
                content = create_error_response(f"Invalid base64 audio: {str(e)}", "invalid_request_error", HTTPStatus.BAD_REQUEST)
                raise HTTPException(status_code=400, detail=content)