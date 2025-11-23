"""
MLX vision-language model wrapper.

This module provides a wrapper class for MLX vision-language models with text
generation, streaming, and multimodal capabilities.
"""

from __future__ import annotations

from collections.abc import Generator
import os
from pathlib import Path
from typing import Any

from loguru import logger
import mlx.core as mx
from mlx_vlm import generate, load, stream_generate
from mlx_vlm.models.cache import make_prompt_cache
from mlx_vlm.video_generate import process_vision_info

from app.parsers.qwen3_vl import Qwen3VLReasoningParser, Qwen3VLToolParser

from ..utils.debug_logging import log_debug_prompt

# Default model parameters
DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", "8192"))
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.0"))
DEFAULT_TOP_P = float(os.getenv("DEFAULT_TOP_P", "1.0"))
DEFAULT_SEED = int(os.getenv("DEFAULT_SEED", "0"))


class MLX_VLM:
    """
    A wrapper class for MLX Multimodal Model that handles both streaming and non-streaming inference.

    This class provides a unified interface for generating text responses from images and text prompts,
    supporting both streaming and non-streaming modes.
    """

    def __init__(
        self,
        model_path: str,
        *,
        context_length: int = 32768,
        trust_remote_code: bool = False,
        chat_template_file: str | None = None,
    ) -> None:
        """Initialize the MLX_VLM model.

        Parameters
        ----------
        model_path : str
            Path to the model directory containing model weights and configuration.
        context_length : int, optional
            Maximum context length for the model. Default is 32768.
        trust_remote_code : bool, optional
            Enable trust_remote_code when loading models. Default is False.
        chat_template_file : str | None, optional
            Path to an optional chat template file to load into the processor.

        Raises
        ------
        ValueError
            If model loading fails or the chat template file cannot be read.
        """
        try:
            self.model, self.processor = load(
                model_path, lazy=False, trust_remote_code=trust_remote_code
            )
            self.prompt_cache = make_prompt_cache(self.model.language_model, context_length)
            self.config = self.model.config
            if chat_template_file:
                chat_template_path = Path(chat_template_file)
                if not chat_template_path.exists():
                    raise ValueError(f"Chat template file {chat_template_file} does not exist")
                with chat_template_path.open() as f:
                    self.processor.chat_template = f.read()
        except Exception as e:
            raise ValueError(f"Error loading model: {e}") from e

    def _is_video_model(self) -> bool:
        """Determine whether the loaded model supports video tokens.

        Returns
        -------
        bool
            True if the model configuration indicates video support, False otherwise.
        """
        return hasattr(self.config, "video_token_id") or hasattr(self.config, "video_token_index")

    def get_model_type(self) -> str:
        """
        Get the model type identifier.

        Returns
        -------
        str
            The model type string.
        """
        return str(self.config.model_type)

    def __call__(
        self,
        messages: list[dict[str, Any]],
        stream: bool = False,
        verbose: bool = False,
        **kwargs: Any,
    ) -> str | Generator[Any, None, None]:
        """Generate text response from images and messages.

        Parameters
        ----------
        messages : list[dict[str, Any]]
            OpenAI-style message dicts (including multimodal content).
        stream : bool, optional
            Whether to stream the response. Default is False.
        verbose : bool, optional
            Whether to log debug information. Default is False.
        **kwargs : Any
            Additional model parameters (images, audios, videos, chat_template_kwargs, temperature, max_tokens, etc.).

        Returns
        -------
        str | Generator[Any, None, None]
            Complete response as a string when ``stream`` is False, or a generator
            yielding response chunks when ``stream`` is True.
        """
        images = kwargs.pop("images", None)
        videos = kwargs.pop("videos", None)
        chat_template_kwargs = kwargs.pop("chat_template_kwargs", {})
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **chat_template_kwargs,
        )
        if verbose:
            log_debug_prompt(text)

        image_inputs, video_inputs = process_vision_info(messages)

        if image_inputs and video_inputs:
            raise ValueError("Cannot process both images and videos in the same request")

        if video_inputs and not self._is_video_model():
            raise ValueError("Model is not a video model")

        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
        )

        model_params = {
            "input_ids": mx.array(inputs["input_ids"]),
            "mask": mx.array(inputs["attention_mask"]),
            **kwargs,
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
            result = stream_generate(
                self.model,
                self.processor,
                prompt=text,
                prompt_cache=self.prompt_cache,
                **model_params,
            )
            if isinstance(result, Generator):
                return result
            raise TypeError(f"Expected Generator, got {type(result)}")
        result = generate(
            self.model,
            self.processor,
            prompt=text,
            prompt_cache=self.prompt_cache,
            **model_params,
        )
        if isinstance(result, str):
            return result
        raise TypeError(f"Expected str, got {type(result)}")


if __name__ == "__main__":
    image_path = "examples/images/attention.png"
    video_path = "examples/videos/demo.mp4"
    model_path = "mlx-community/Qwen3-VL-2B-Thinking-8bit"

    model = MLX_VLM(model_path)
    logger.info(f"MODEL TYPE: {model.get_model_type()}")

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather for a given city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "The city to get the weather for"}
                    },
                },
                "required": ["city"],
            },
        }
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
                {"type": "text", "text": "What is the weather in New York?"},
                {"type": "image", "image": image_path},
            ],
        }
    ]

    reasoning_parser: Qwen3VLReasoningParser | None = Qwen3VLReasoningParser()
    tool_parser: Qwen3VLToolParser | None = Qwen3VLToolParser()

    parsed_content: dict[str, Any] | str | None = None
    after_thinking_close_content: str | None = None
    final_chunk: Any = None
    is_first_chunk: bool = True

    response = model(messages, stream=True, verbose=False, **kwargs)
    if isinstance(response, Generator):
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
                    logger.info(f"Parsed content: {parsed_content}")
                if is_complete:
                    reasoning_parser = None
                if after_thinking_close_content:
                    text = after_thinking_close_content
                else:
                    continue
            if tool_parser:
                parsed_content, is_complete = tool_parser.extract_tool_calls_streaming(text)
                if parsed_content:
                    logger.info(f"Parsed content: {parsed_content}")
                if is_complete:
                    tool_parser = None
                continue
