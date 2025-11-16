import gc
import time
import uuid
import asyncio
from http import HTTPStatus
from fastapi import HTTPException
from loguru import logger
from app.models.mlx_lm import MLX_LM
from app.core.queue import RequestQueue
from app.handler.parser import ParserFactory
from app.utils.errors import create_error_response
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple
from app.schemas.openai import ChatCompletionRequest, EmbeddingRequest, UsageInfo


class MLXLMHandler:
    """
    Handler class for making requests to the underlying MLX text-only language model service.
    Provides request queuing, metrics tracking, and robust error handling.
    """

    def __init__(self, model_path: str, context_length: int = 32768, max_concurrency: int = 1, enable_auto_tool_choice: bool = False, tool_call_parser: str = None, reasoning_parser: str = None, trust_remote_code: bool = False):
        """
        Initialize the handler with the specified model path.
        
        Args:
            model_path (str): Path to the model directory.
            context_length (int): Maximum context length for the model.
            max_concurrency (int): Maximum number of concurrent model inference tasks.
            enable_auto_tool_choice (bool): Enable automatic tool choice.
            tool_call_parser (str): Name of the tool call parser to use (qwen3, glm4_moe, harmony, minimax, ...)
            reasoning_parser (str): Name of the reasoning parser to use (qwen3, qwen3_next, glm4_moe, harmony, minimax, ...)
            trust_remote_code (bool): Enable trust_remote_code when loading models.
        """
        self.model_path = model_path
        self.model = MLX_LM(model_path, context_length, trust_remote_code=trust_remote_code)
        self.model_created = int(time.time())  # Store creation time when model is loaded
        self.model_type = self.model.get_model_type()
        
        # Store parser configuration
        self.enable_auto_tool_choice = enable_auto_tool_choice
        self.tool_call_parser = tool_call_parser
        self.reasoning_parser = reasoning_parser
        
        # Initialize request queue for text tasks
        self.request_queue = RequestQueue(max_concurrency=max_concurrency)

        # Initialize message converter for supported models
        self.converter = ParserFactory.create_converter(self.model_type)

        logger.info(f"Initialized MLXHandler with model path: {model_path}")
    
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
        tokens = self.model.tokenizer.encode(text, add_special_tokens=False)
        return len(tokens)

    def _count_message_tokens(self, messages: List[Dict[str, str]], **kwargs) -> int:
        """
        Count the number of tokens in a list of messages after applying chat template.

        Args:
            messages: List of messages to count tokens for.
            **kwargs: Additional arguments to pass to apply_chat_template.

        Returns:
            int: The number of prompt tokens.
        """
        try:
            input_tokens = self.model.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                **kwargs
            )
            return len(input_tokens)
        except Exception as e:
            logger.warning(f"Failed to count message tokens: {str(e)}")
            # Fallback: rough estimate
            total_text = " ".join([msg.get("content", "") for msg in messages if isinstance(msg.get("content"), str)])
            return self._count_tokens(total_text)

    def _extract_model_metadata(self) -> Dict[str, Any]:
        """
        Extract metadata from the loaded MLX model.

        Returns:
            dict: Metadata about the model including context_length, backend, etc.
        """
        metadata = {
            "backend": "mlx",
            "modality": "text",
        }

        # Add context length
        if hasattr(self.model, 'max_kv_size'):
            metadata["context_length"] = self.model.max_kv_size

        # Add vocab size if available
        if hasattr(self.model.tokenizer, 'vocab_size'):
            metadata["vocab_size"] = self.model.tokenizer.vocab_size

        # Add model family/type if available
        if hasattr(self.model, 'model_type') and self.model.model_type:
            metadata["model_family"] = self.model.model_type

        # Add dtype info if available from model config
        if hasattr(self.model.model, 'dtype'):
            metadata["dtype"] = str(self.model.model.dtype)

        # Add load settings
        metadata["load_settings"] = {
            "model_path": self.model_path,
            "model_type": "lm"
        }

        return metadata

    async def get_models(self) -> List[Dict[str, Any]]:
        """
        Get list of available models with their metadata.
        """
        try:
            return [{
                "id": self.model_path,
                "object": "model",
                "created": self.model_created,
                "owned_by": "local",
                "metadata": self._extract_model_metadata()
            }]
        except Exception as e:
            logger.error(f"Error getting models: {str(e)}")
            return []
    
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

            # Count prompt tokens
            chat_template_kwargs = model_params.get("chat_template_kwargs", {})
            prompt_tokens = self._count_message_tokens(chat_messages, **chat_template_kwargs)

            request_data = {
                "messages": chat_messages,
                "stream": True,
                **model_params
            }
            response_generator = await self.request_queue.submit(request_id, request_data)
            # Create appropriate parsers for this model type

            thinking_parser, tool_parser = self._create_parsers()
            is_first_chunk = True
            completion_text = ""  # Accumulate completion for token counting

            if thinking_parser and ParserFactory.has_special_parsing(self.reasoning_parser):
                for chunk in response_generator:
                    if not chunk or not chunk.text:
                        continue
                    text = chunk.text
                    completion_text += text
                    parsed_content, is_complete = thinking_parser.parse_stream(text)
                    if parsed_content:
                        yield parsed_content
                    if is_complete:
                        break
            else:

                if ParserFactory.respects_enable_thinking(self.reasoning_parser):
                    enable_thinking = chat_template_kwargs.get("enable_thinking", True)
                    if not enable_thinking:
                        thinking_parser = None

                # # Process streaming response
                for chunk in response_generator:

                    if not chunk or not chunk.text:
                        continue

                    text = chunk.text
                    completion_text += text

                    if is_first_chunk:
                        if thinking_parser and ParserFactory.needs_redacted_reasoning_prefix(self.reasoning_parser):
                            text = thinking_parser.get_thinking_open() + text
                        is_first_chunk = False

                    if thinking_parser:
                        parsed_content, is_complete = thinking_parser.parse_stream(text)
                        if parsed_content:
                            yield parsed_content
                        if is_complete:
                            thinking_parser = None
                        continue

                    if tool_parser:
                        parsed_content, _ = tool_parser.parse_stream(text)
                        if parsed_content:
                            yield parsed_content
                        continue

                    yield text

            # Count completion tokens and yield usage info at the end
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

            # Count prompt tokens
            chat_template_kwargs = model_params.get("chat_template_kwargs", {})
            prompt_tokens = self._count_message_tokens(chat_messages, **chat_template_kwargs)

            request_data = {
                "messages": chat_messages,
                "stream": False,
                **model_params
            }
            response = await self.request_queue.submit(request_id, request_data)

            # Count completion tokens
            completion_tokens = self._count_tokens(response if isinstance(response, str) else response.get("content", ""))
            total_tokens = prompt_tokens + completion_tokens

            # Create usage info
            usage = UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens
            )

            # Create appropriate parsers for this model type
            thinking_parser, tool_parser = self._create_parsers()

            enable_thinking = chat_template_kwargs.get("enable_thinking", True)

            if ParserFactory.respects_enable_thinking(self.reasoning_parser):
                if not enable_thinking:
                    thinking_parser = None

            if not thinking_parser and not tool_parser:
                return {"response": response, "usage": usage}

            response_text = response

            if thinking_parser and ParserFactory.has_special_parsing(self.reasoning_parser):
                # Handle parsers with special parsing logic (e.g., harmony returns dict)
                parsed = thinking_parser.parse(response_text)
                return {"response": parsed, "usage": usage}


            if thinking_parser and ParserFactory.needs_redacted_reasoning_prefix(self.reasoning_parser):
                # Add thinking tag to response for parsers that need it
                response_text = thinking_parser.get_thinking_open() + response_text

            parsed_response = {
                "reasoning_content": None,
                "tool_calls": None,
                "content": None
            }

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
            logger.error(f"Error in text response generation: {str(e)}")
            content = create_error_response(f"Failed to generate text response: {str(e)}", "server_error", HTTPStatus.INTERNAL_SERVER_ERROR)
            raise HTTPException(status_code=500, detail=content)
        
    async def generate_embeddings_response(self, request: EmbeddingRequest):
        """
        Generate embeddings for a given text input.
        
        Args:
            request: EmbeddingRequest object containing the text input.
        
        Returns:
            List[float]: Embeddings for the input text.
        """
        try:
            # Create a unique request ID
            request_id = f"embeddings-{uuid.uuid4()}"
            if isinstance(request.input, str):
                request.input = [request.input]
            request_data = {
                "type": "embeddings",
                "input": request.input,
                "model": request.model
            }

            # Submit to the request queue
            response = await self.request_queue.submit(request_id, request_data)

            return response

        except Exception as e:
            logger.error(f"Error in embeddings generation: {str(e)}")
            content = create_error_response(f"Failed to generate embeddings: {str(e)}", "server_error", HTTPStatus.INTERNAL_SERVER_ERROR)
            raise HTTPException(status_code=500, detail=content)
        

    async def _process_request(self, request_data: Dict[str, Any]) -> str:
        """
        Process a text request. This is the worker function for the request queue.
        
        Args:
            request_data: Dictionary containing the request data.
            
        Returns:
            str: The model's response.
        """
        try:
            # Check if the request is for embeddings
            if request_data.get("type") == "embeddings":
                result = self.model.get_embeddings(request_data["input"])
                # Force garbage collection after embeddings
                gc.collect()
                return result

            # Extract request parameters
            messages = request_data.get("messages", [])
            stream = request_data.get("stream", False)
            
            # Remove these keys from model_params
            model_params = request_data.copy()
            model_params.pop("messages", None)
            model_params.pop("stream", None)

            # Apply message conversion if needed
            if self.converter:
                refined_messages = self.converter.convert_messages(messages)
            else:
                # Reformat messages for models without conversion needs
                refined_messages = []
                for message in messages:
                    # Filter out None values
                    cleaned_message = {k: v for k, v in message.items() if v is not None}
                    refined_messages.append(cleaned_message)

            # Call the model
            response = self.model(
                messages=refined_messages,
                stream=stream,
                **model_params
            )            
            # Force garbage collection after model inference
            gc.collect()
            return response
            
        except Exception as e:
            logger.error(f"Error processing text request: {str(e)}")
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
        
    async def cleanup(self):
        """
        Cleanup resources and stop the request queue before shutdown.
        
        This method ensures all pending requests are properly cancelled
        and resources are released.
        """
        try:
            logger.info("Cleaning up MLXLMHandler resources")
            if hasattr(self, 'request_queue'):
                await self.request_queue.stop()
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
                    request_dict["schema"] = response_format.get("json_schema", None).get("schema", None)
            
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