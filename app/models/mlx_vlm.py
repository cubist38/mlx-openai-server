import os
import mlx.core as mx
from loguru import logger
from typing import List, Dict, Union, Generator
from mlx_vlm.models.cache import make_prompt_cache
from mlx_vlm import load, generate, stream_generate
from mlx_vlm.video_generate import process_vision_info
from ..utils.debug_logging import log_debug_prompt

# Default model parameters
DEFAULT_MAX_TOKENS = os.getenv("DEFAULT_MAX_TOKENS", 8192)
DEFAULT_TEMPERATURE = os.getenv("DEFAULT_TEMPERATURE", 0.0)
DEFAULT_TOP_P = os.getenv("DEFAULT_TOP_P", 1.0)
DEFAULT_SEED = os.getenv("DEFAULT_SEED", 0)

class MLX_VLM:
    """
    A wrapper class for MLX Multimodal Model that handles both streaming and non-streaming inference.
    
    This class provides a unified interface for generating text responses from images and text prompts,
    supporting both streaming and non-streaming modes.
    """
    
    def __init__(self, model_path: str, context_length: int = 32768, trust_remote_code: bool = False, chat_template_file: str = None):
        """
        Initialize the MLX_VLM model.
        
        Args:
            model_path (str): Path to the model directory containing model weights and configuration.
            context_length (int): Maximum context length for the model. Defaults to 32768.
            trust_remote_code (bool): Enable trust_remote_code when loading models. Defaults to False.
        Raises:
            ValueError: If model loading fails.
        """
        try:
            self.model, self.processor = load(model_path, lazy=False, trust_remote_code=trust_remote_code)
            self.prompt_cache = make_prompt_cache(self.model.language_model, context_length)
            self.config = self.model.config
            if chat_template_file:
                if not os.path.exists(chat_template_file):
                    raise ValueError(f"Chat template file {chat_template_file} does not exist")
                with open(chat_template_file, "r") as f:
                    self.processor.chat_template = f.read()
        except Exception as e:
            raise ValueError(f"Error loading model: {str(e)}")

    def _is_video_model(self):
        return hasattr(self.config, "video_token_id") or hasattr(
            self.config, "video_token_index"
        )

    def get_model_type(self):
        return self.config.model_type

    def __call__(
        self, 
        messages: List[Dict[str, str]], 
        images: List[str] = None,
        audios: List[str] = None,
        videos: List[str] = None,
        stream: bool = False, 
        verbose: bool = False,
        **kwargs
    ) -> Union[str, Generator[str, None, None]]:
        """
        Generate text response from images and messages.
        
        Args:
            images (List[str]): List of image paths to process.
            messages (List[Dict[str, str]]): List of message dictionaries with 'role' and 'content' keys.
            stream (bool, optional): Whether to stream the response. Defaults to False.
            **kwargs: Additional model parameters (chat_template_kwargs, temperature, max_tokens, etc.)
            
        Returns:
            Union[str, Generator[str, None, None]]: 
                - If stream=False: Complete response as string
                - If stream=True: Generator yielding response chunks
        """

        if images and videos:
            raise ValueError("Cannot process both images and videos in the same request")

        if videos and not self._is_video_model():
            raise ValueError("Model is not a video model")

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **kwargs.get("chat_template_kwargs", {})
        )
        if verbose:
            log_debug_prompt(text)
        
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )

        model_params = {
            "input_ids": mx.array(inputs["input_ids"]),
            "mask": mx.array(inputs["attention_mask"]),
            **kwargs
        }

        if images:
            if "pixel_values" in inputs:
                model_params["pixel_values"] = mx.array(inputs["pixel_values"])
            if "image_grid_thw" in inputs:
                model_params["image_grid_thw"] = mx.array(inputs["image_grid_thw"])
            if "image_sizes" in inputs:
                model_params["image_sizes"] = mx.array(inputs["image_sizes"])

        if videos:
            if "pixel_values" in inputs:
                model_params["pixel_values"] = mx.array(inputs["pixel_values_videos"])
            if "video_grid_thw" in inputs:
                model_params["video_grid_thw"] = mx.array(inputs["video_grid_thw"])

        if stream:
            return stream_generate(
                self.model,
                self.processor,
                prompt=text,
                prompt_cache=self.prompt_cache,
                **model_params
            )
        else:
            return generate(
                self.model,
                self.processor,
                prompt=text,
                prompt_cache=self.prompt_cache,
                **model_params
            )


if __name__ == "__main__":
    image_path = "examples/images/attention.png"
    video_path = "examples/videos/demo.mp4"
    model_path = "mlx-community/Qwen3-VL-2B-Thinking-8bit"

    model = MLX_VLM(model_path)
    print("MODEL TYPE: ", model.get_model_type())

    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a given city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "The city to get the weather for"}
                }
            },
            "required": ["city"]
        }}   
    ]
    kwargs = {
        "chat_template_kwargs": {
            "tools": tools,
            "enable_thinking": True,
        },
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 0,
        "max_tokens": 8192,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "images": [image_path],
    }
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What is the weather in New York?"
                },
                {
                    "type": "image",
                    "image": image_path
                }
            ]
        }
    ]

    from app.parsers.qwen3_vl import Qwen3VLReasoningParser, Qwen3VLToolParser
    reasoning_parser = Qwen3VLReasoningParser()
    tool_parser = Qwen3VLToolParser()

    response = model(messages, stream=True, **kwargs)
    after_thinking_close_content = None
    final_chunk = None
    is_first_chunk = True
    for chunk in response:
        if chunk is None:
            continue
        final_chunk = chunk
        text = chunk.text
        if is_first_chunk:
            if reasoning_parser and reasoning_parser.needs_redacted_reasoning_prefix():
                text = reasoning_parser.get_reasoning_open() + text
            is_first_chunk = False
        if reasoning_parser:
            parsed_content, is_complete = reasoning_parser.extract_reasoning_streaming(text)
            
            if parsed_content:
                if isinstance(parsed_content, dict):
                    after_thinking_close_content = parsed_content.pop("content", None)
                print(f"Parsed content: {parsed_content}")
            if is_complete:
                reasoning_parser = None
            if after_thinking_close_content:
                text = after_thinking_close_content
            else:
                continue
        if tool_parser:
            parsed_content, is_complete = tool_parser.extract_tool_calls_streaming(text)
            if parsed_content:
                print(f"Parsed content: {parsed_content}")
            if is_complete:
                tool_parser = None
            continue