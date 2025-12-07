import asyncio
import base64
import time
import uuid
import gc

from loguru import logger
from http import HTTPStatus
from fastapi import HTTPException
from typing import Any, Dict, List, Optional, Tuple

from ..core.queue import RequestQueue
from ..models.mlx_vlm import MLX_VLM
from .parser import ParserFactory
from ..core import ImageProcessor, AudioProcessor, VideoProcessor
from ..utils.errors import create_error_response
from ..schemas.openai import ChatCompletionRequest, EmbeddingRequest, ChatCompletionContentPart, ChatCompletionContentPartText, ChatCompletionContentPartImage, ChatCompletionContentPartInputAudio, ChatCompletionContentPartVideo, UsageInfo

class MLXVLMHandler:
    """
    Handler class for making requests to the underlying MLX multimodal model service.
    Provides caching, concurrent image processing, audio processing, and robust error handling.
    """

    def __init__(self, model_path: str, context_length: int = 32768, max_workers: int = 4, max_concurrency: int = 1, disable_auto_resize: bool = False, enable_auto_tool_choice: bool = False, tool_call_parser: str = None, reasoning_parser: str = None, trust_remote_code: bool = False, chat_template_file: str = None):
        """
        Initialize the handler with the specified model path.
        
        Args:
            model_path (str): Path to the model directory.
            context_length (int): Maximum context length for the model.
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
        self.tool_call_parser = tool_call_parser
        self.reasoning_parser = reasoning_parser

        # Initialize request queue for multimodal and text tasks
        # We use the same queue for both multimodal and text tasks for simplicity
        # and to ensure we don't overload the model with too many concurrent requests
        self.request_queue = RequestQueue(max_concurrency=max_concurrency)
        
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
    
    def _create_parsers(self) -> Tuple[Optional[Any], Optional[Any]]:
        """
        Create appropriate parsers based on model type and available tools.
        Uses ParserFactory for centralized parser creation logic.
        
        Returns:
            Tuple of (thinking_parser, tool_parser)
        """         
        return ParserFactory.create_parsers(
            model_type=self.model_type,
            manual_reasoning_parser=self.reasoning_parser,
            manual_tool_parser=self.tool_call_parser,
        )
    
    async def initialize(self, queue_config: Optional[Dict[str, Any]] = None):
        """Initialize the handler and start the request queue."""
        
        if not queue_config:
            queue_config = {
                "max_concurrency": 1,
                "timeout": 300,
                "queue_size": 100
            }
        self.request_queue = RequestQueue(
            max_concurrency=queue_config.get("max_concurrency"),
            timeout=queue_config.get("timeout"),
            queue_size=queue_config.get("queue_size")
        )
        await self.request_queue.start(self._process_request)
        logger.info("Initialized MLXHandler and started request queue")

    def _count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text string.

        Args:
            text: The text to count tokens for.

        Returns:
            int: The number of tokens.
        """
        if not text:
            return 0
        try:
            # Try to use tokenizer from processor if available
            if hasattr(self.model.processor, "tokenizer"):
                tokens = self.model.processor.tokenizer.encode(text, add_special_tokens=False)
                return len(tokens)
            # Fallback for some processors that might behave differently or don't expose tokenizer directly
            # This part depends on specific processor implementation, but usually they have a tokenizer
            elif hasattr(self.model.processor, "encode"):
                 tokens = self.model.processor.encode(text, add_special_tokens=False)
                 return len(tokens)
            else:
                logger.warning("Could not find tokenizer in processor to count tokens")
                return 0
        except Exception as e:
            logger.warning(f"Failed to count tokens: {str(e)}")
            return 0

    def _count_message_tokens(self, messages: List[Dict[str, Any]], **kwargs) -> int:
        """
        Count the number of tokens in a list of messages after applying chat template.

        Args:
            messages: List of messages to count tokens for.
            **kwargs: Additional arguments to pass to apply_chat_template.

        Returns:
            int: The number of prompt tokens.
        """
        try:
            # We need to handle the fact that messages might contain images/audio which apply_chat_template might not handle directly 
            # if we pass them as is, or it might handle them if they are formatted correctly.
            # MLX_VLM's apply_chat_template (via processor) usually expects text-only messages or handles them if configured.
            # However, looking at MLX_VLM.__call__, it calls self.processor.apply_chat_template with tokenize=False first.
            
            # Let's try to use the processor's apply_chat_template with tokenize=True if possible, 
            # or tokenize=False and then encode.
            
            # For VLM, the prompt tokens also depend on images. 
            # This is complex because image tokens depend on the model and how images are processed.
            # For now, we will try to approximate or use the processor if it supports it.
            
            # Simplification: We will use the same logic as in MLX_VLM.__call__ to get the text prompt, 
            # and then encode it. This might miss image tokens if they are added separately.
            
            # NOTE: Accurate token counting for VLMs is tricky without running the full preparation pipeline.
            # We will try to get the text part at least.
            
            text = self.model.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **kwargs
            )
            
            # Now encode the text
            return self._count_tokens(text)
            
        except Exception as e:
            logger.warning(f"Failed to count message tokens: {str(e)}")
            # Fallback: rough estimate
            total_text = ""
            for msg in messages:
                content = msg.get("content", "")
                if isinstance(content, str):
                    total_text += content
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            total_text += part.get("text", "")
            return self._count_tokens(total_text)

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
            
            # Submit to the multimodal queue and get the generator
            response_generator, prompt_tokens = await self.request_queue.submit(request_id, request_dict)      
            
            # Create appropriate parsers for this model type
            thinking_parser, tool_parser = self._create_parsers()

            chat_template_kwargs = request_dict.get("chat_template_kwargs", {})
            enable_thinking = chat_template_kwargs.get("enable_thinking", True)

            if ParserFactory.respects_enable_thinking(self.reasoning_parser):
                if not enable_thinking:
                    thinking_parser = None

            is_first_chunk = True
            completion_chunks = [] # Accumulate completion for token counting
            after_thinking_close_content = None

            # Process and yield each chunk asynchronously
            for chunk in response_generator:
                if not chunk or not chunk.text:
                    continue
                    
                text = chunk.text
                completion_chunks.append(text)
                if is_first_chunk:
                    if thinking_parser and ParserFactory.needs_redacted_reasoning_prefix(self.reasoning_parser):
                        text = thinking_parser.get_thinking_open() + text
                    is_first_chunk = False

                if thinking_parser:
                    parsed_content, is_complete = thinking_parser.parse_stream(text)
                    after_thinking_close_content = None
                    if parsed_content:
                        if isinstance(parsed_content, dict):
                            after_thinking_close_content = parsed_content.pop("content", None)
                        yield parsed_content
                    if is_complete:
                        thinking_parser = None
                    if after_thinking_close_content:
                        text = after_thinking_close_content
                    else:
                        continue
                    
                if tool_parser:
                    parsed_content, is_complete = tool_parser.parse_stream(text)
                    if is_complete:
                        yield parsed_content
                    continue

                yield text

            # Count completion tokens and yield usage info at the end
            completion_text = "".join(completion_chunks)
            completion_tokens = self._count_tokens(completion_text)
            total_tokens = prompt_tokens + completion_tokens

            yield {
                "__usage__": UsageInfo(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
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
        
            response, prompt_tokens = await self.request_queue.submit(request_id, request_dict)

            # Count completion tokens
            completion_tokens = self._count_tokens(response.text)
            total_tokens = prompt_tokens + completion_tokens

            # Create usage info
            usage = UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens
            )
                        
            # Create appropriate parsers for this model type
            thinking_parser, tool_parser = self._create_parsers()
            
            if not thinking_parser and not tool_parser:
                return {"response": response.text, "usage": usage}
            
            parsed_response = {
                "reasoning_content": None,
                "tool_calls": None,
                "content": None
            }
            response_text = response.text

            if thinking_parser and ParserFactory.needs_redacted_reasoning_prefix(self.reasoning_parser):
                response_text = thinking_parser.get_thinking_open() + response_text

            if thinking_parser:
                thinking_response, response_text = thinking_parser.parse(response_text)
                parsed_response["reasoning_content"] = thinking_response
            if tool_parser:
                tool_response, response_text = tool_parser.parse(response_text)
                parsed_response["tool_calls"] = tool_response
            parsed_response["content"] = response_text
            
            return {"response": parsed_response, "usage": usage}
                        
        except asyncio.QueueFull:
            logger.error("Too many requests. Service is at capacity.")
            content = create_error_response("Too many requests. Service is at capacity.", "rate_limit_exceeded", HTTPStatus.TOO_MANY_REQUESTS)
            raise HTTPException(status_code=429, detail=content)
        except Exception as e:
            logger.error(f"Error in multimodal response generation: {str(e)}")
            content = create_error_response(f"Failed to generate multimodal response: {str(e)}", "server_error", HTTPStatus.INTERNAL_SERVER_ERROR)
            raise HTTPException(status_code=500, detail=content)
        
    async def generate_embeddings_response(self, request: EmbeddingRequest):
        """
        Generate embeddings for a given text input.
        
        Args:
            request: EmbeddingRequest object containing the text input.
        
        Returns:
            List[float]: Embeddings for the input text or images
        """
        try:
            # Create a unique request ID
            image_url = request.image_url
            # Process the image URL to get a local file path
            images = []
            if request.image_url:
                image_result = await self.image_processor.process_image_url(image_url, resize=not self.disable_auto_resize)
                images.append(image_result["path"])
            request_id = f"embeddings-{uuid.uuid4()}"
            if isinstance(request.input, str):
                request.input = [request.input]
            request_data = {
                "type": "embeddings",
                "input": request.input,
                "model": request.model,
                "images": images
            }

            # Submit to the request queue
            response = await self.request_queue.submit(request_id, request_data)

            return response

        except Exception as e:
            logger.error(f"Error in embeddings generation: {str(e)}")
            content = create_error_response(f"Failed to generate embeddings: {str(e)}", "server_error", HTTPStatus.INTERNAL_SERVER_ERROR)
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

    async def cleanup(self):
        """
        Cleanup resources and stop the request queue before shutdown.
        
        This method ensures all pending requests are properly cancelled
        and resources are released, including the image processor.
        """
        try:
            logger.info("Cleaning up MLXVLMHandler resources")
            if hasattr(self, 'request_queue'):
                await self.request_queue.stop()
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

    async def _process_request(self, request_data: Dict[str, Any]) -> str:
        """
        Process a multimodal request. This is the worker function for the request queue.
        
        Args:
            request_data: Dictionary containing the request data.
            
        Returns:
            str: The model's response.
        """
        try:
            
            # Extract request parameters
            images = request_data.get("images", [])
            audios = request_data.get("audios", [])
            videos = request_data.get("videos", [])
            messages = request_data.get("messages", [])
            stream = request_data.get("stream", False)
         
            # Remove these keys from model_params
            model_params = request_data.copy()
            model_params.pop("images", None)
            model_params.pop("audios", None)
            model_params.pop("videos", None)
            model_params.pop("messages", None)
            model_params.pop("stream", None)
            
            # Call the model
            response = self.model(
                images=images,
                audios=audios,
                videos=videos,
                messages=messages,
                stream=stream,
                **model_params
            )
            # Force garbage collection after model inference
            gc.collect()
            return response
            
        except Exception as e:
            logger.error(f"Error processing multimodal request: {str(e)}")
            # Clean up on error
            gc.collect()
            raise

    async def get_queue_stats(self) -> Dict[str, Any]:
        """
        Get statistics from the request queue and performance metrics.
        
        Returns:
            Dict with queue and performance statistics.
        """
        queue_stats = self.request_queue.get_queue_stats()
        
        return {
            "queue_stats": queue_stats,
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

            tools = request.tools or None
            tool_choice = request.tool_choice or None

            if tools:
                # Enable auto tool choice if requested via CLI flag
                if self.enable_auto_tool_choice and tool_choice == "auto":
                    request_dict["chat_template_kwargs"]["tool_choice"] = "auto"
                elif tool_choice:
                    logger.warning("Tool choice has not supported yet, will be ignored.")
                request_dict["chat_template_kwargs"]["tools"] = tools
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