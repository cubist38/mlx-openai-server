from collections.abc import Generator
import os

import mlx.core as mx
from mlx_vlm import generate, load, stream_generate
from mlx_vlm.models.cache import make_prompt_cache
from mlx_vlm.video_generate import process_vision_info

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

    def __init__(
        self, model_path: str, context_length: int = 32768, trust_remote_code: bool = False
    ):
        """
        Initialize the MLX_VLM model.

        Args:
            model_path (str): Path to the model directory containing model weights and configuration.
            context_length (int): Maximum context length for the model. Defaults to 32768.
            trust_remote_code (bool): Enable trust_remote_code when loading models. Defaults to False.

        Raises
        ------
            ValueError: If model loading fails.
        """
        try:
            self.model, self.processor = load(
                model_path, lazy=False, trust_remote_code=trust_remote_code
            )
            self.max_kv_size = context_length
            self.config = self.model.config
        except Exception as e:
            raise ValueError(f"Error loading model: {e!s}") from e

    def _is_video_model(self):
        return hasattr(self.config, "video_token_id") or hasattr(self.config, "video_token_index")

    def get_model_type(self):
        return self.config.model_type

    def __call__(
        self,
        messages: list[dict[str, str]],
        images: list[str] = None,
        audios: list[str] = None,
        videos: list[str] = None,
        stream: bool = False,
        **kwargs,
    ) -> str | Generator[str, None, None]:
        """
        Generate text response from images and messages.

        Args:
            images (List[str]): List of image paths to process.
            messages (List[Dict[str, str]]): List of message dictionaries with 'role' and 'content' keys.
            stream (bool, optional): Whether to stream the response. Defaults to False.
            **kwargs: Additional model parameters (chat_template_kwargs, temperature, max_tokens, etc.)

        Returns
        -------
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
            **kwargs.get("chat_template_kwargs", {}),
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
        )

        model_params = {
            "input_ids": mx.array(inputs["input_ids"]),
            "mask": mx.array(inputs["attention_mask"]),
            **kwargs,
        }

        if images:
            model_params["pixel_values"] = mx.array(inputs["pixel_values"])
            model_params["image_grid_thw"] = mx.array(inputs["image_grid_thw"])

        if videos:
            model_params["pixel_values"] = mx.array(inputs["pixel_values_videos"])
            model_params["video_grid_thw"] = mx.array(inputs["video_grid_thw"])

        prompt_cache = make_prompt_cache(self.model.language_model, self.max_kv_size)

        if stream:
            return stream_generate(
                self.model, self.processor, prompt=text, prompt_cache=prompt_cache, **model_params
            )
        return generate(
            self.model, self.processor, prompt=text, prompt_cache=prompt_cache, **model_params
        )


if __name__ == "__main__":
    image_path = "examples/images/attention.png"
    video_path = "examples/videos/demo.mp4"
    model_path = "mlx-community/GLM-4.5V-4bit"

    model = MLX_VLM(model_path)
    print("MODEL TYPE: ", model.get_model_type())

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
    }
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe the video in detail"},
                {"type": "image", "image": image_path},
            ],
        }
    ]
    response = model(messages, stream=False, **kwargs)
    print(response)
