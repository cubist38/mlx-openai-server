from collections.abc import Generator
from dataclasses import dataclass
import os
from typing import Any

import mlx.core as mx
from mlx_vlm import load, stream_generate
from mlx_vlm.utils import prepare_inputs
from outlines.processors import JSONLogitsProcessor
import torch

from ..utils.outlines_transformer_tokenizer import OutlinesTransformerTokenizer

# Default model parameters
DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", "100000"))
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.0"))
DEFAULT_TOP_P = float(os.getenv("DEFAULT_TOP_P", "1.0"))
DEFAULT_SEED = int(os.getenv("DEFAULT_SEED", "0"))
DEFAULT_REPETITION_CONTEXT_SIZE = int(os.getenv("DEFAULT_REPETITION_CONTEXT_SIZE", "20"))


def _extract_vision_inputs(
    messages: list[dict[str, Any]],
) -> tuple[list[Any] | None, list[Any] | None]:
    """Extract image and video values from formatted chat messages.

    Parameters
    ----------
    messages : list[dict[str, Any]]
        Chat messages after handler-level multimodal content reformatting.

    Returns
    -------
    tuple[list[Any] | None, list[Any] | None]
        Image and video inputs for ``mlx-vlm`` processors. ``None`` is used
        when a media type is absent.
    """
    image_inputs: list[Any] = []
    video_inputs: list[Any] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            if "image" in item:
                image_inputs.append(item["image"])
            elif "image_url" in item:
                image_inputs.append(item["image_url"])
            elif "video" in item:
                video_inputs.append(item["video"])
    return image_inputs or None, video_inputs or None


@dataclass
class CompletionResponse:
    """
    The output of :func:`__call__` when stream is False.

    Args:
        text (str): The next segment of decoded text. This can be an empty string.
        tokens (List[int]): The list of tokens in the response.
        peak_memory (float): The peak memory used so far in GB.
        generation_tps (float): The tokens-per-second for generation.
        generation_tokens (int): The number of generated tokens.
        prompt_tps (float): The prompt processing tokens-per-second.
        prompt_tokens (int): The number of tokens in the prompt.
    """

    text: str = None
    tokens: list[int] = None
    peak_memory: float = None
    generation_tps: float = None
    prompt_tps: float = None
    prompt_tokens: int = None
    generation_tokens: int = None


class MLX_VLM:
    """
    A wrapper class for MLX Multimodal Model that handles both streaming and non-streaming inference.

    This class provides a unified interface for generating text responses from images and text prompts,
    supporting both streaming and non-streaming modes.
    """

    def __init__(
        self,
        model_path: str,
        context_length: int | None = None,
        trust_remote_code: bool = False,
        chat_template_file: str = None,
    ):
        """
        Initialize the MLX_VLM model.

        Args:
            model_path (str): Path to the model directory containing model weights and configuration.
            context_length (int | None): Maximum context length for the model. If None, uses model default.
            trust_remote_code (bool): Enable trust_remote_code when loading models. Defaults to False.

        Raises:
            ValueError: If model loading fails.
        """
        try:
            self.model, self.processor = load(
                model_path, lazy=False, trust_remote_code=trust_remote_code
            )
            self.config = self.model.config
            self.context_length = context_length
            self.kv_bits: int | None = None
            self.kv_group_size: int = 64
            self.quantized_kv_start: int = 0
            self.outlines_tokenizer = OutlinesTransformerTokenizer(self.processor.tokenizer)
            if chat_template_file:
                if not os.path.exists(chat_template_file):
                    raise ValueError(f"Chat template file {chat_template_file} does not exist")
                with open(chat_template_file) as f:
                    template_content = f.read()
                    self.processor.chat_template = template_content
                    if hasattr(self.processor, "tokenizer"):
                        self.processor.tokenizer.chat_template = template_content
            self._ensure_processor_chat_template()
        except Exception as e:
            raise ValueError(f"Error loading model: {e!s}")

    def _is_video_model(self):
        return hasattr(self.config, "video_token_id") or hasattr(self.config, "video_token_index")

    def get_model_type(self):
        return self.config.model_type

    @property
    def language_model(self) -> Any:
        """Return the text decoder module used by ``mlx-vlm`` batch generation."""
        return self.model.language_model

    def _ensure_processor_chat_template(self) -> None:
        """Copy tokenizer chat template onto processors that do not expose one."""
        if getattr(self.processor, "chat_template", None) is not None:
            return
        tokenizer = getattr(self.processor, "tokenizer", None)
        tokenizer_template = getattr(tokenizer, "chat_template", None)
        if tokenizer_template is not None:
            self.processor.chat_template = tokenizer_template

    def create_input_prompt(
        self, messages: list[dict[str, Any]], chat_template_kwargs: dict[str, Any]
    ) -> str:
        chat_template_kwargs = dict(chat_template_kwargs)
        chat_template_kwargs.pop("_partial_mode", None)
        self._ensure_processor_chat_template()

        return self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **chat_template_kwargs
        )

    def _create_processor_inputs(
        self,
        text: str,
        images: list[Any] = None,
        videos: list[Any] = None,
        audios: list[str] = None,
    ) -> dict[str, Any]:
        model = getattr(self, "model", None)
        config = getattr(self, "config", getattr(model, "config", None))
        model_type = getattr(config, "model_type", None)
        add_special_tokens = (
            getattr(self.processor, "chat_template", None) is None
            if model_type in ["gemma3", "gemma3n", "gemma4"]
            else True
        )
        image_token_index = getattr(config, "image_token_index", None)
        inputs = prepare_inputs(
            self.processor,
            images=images,
            audio=audios,
            videos=videos,
            prompts=[text],
            image_token_index=image_token_index,
            add_special_tokens=add_special_tokens,
            padding=True,
        )
        if "attention_mask" in inputs and "mask" not in inputs:
            inputs["mask"] = inputs.pop("attention_mask")
        return inputs

    def _to_mlx_inputs(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Convert tensor values produced by Transformers processors into MLX arrays."""
        for key, value in list(inputs.items()):
            if isinstance(value, torch.Tensor):
                inputs[key] = mx.array(value)
        return inputs

    def create_model_inputs(
        self,
        text: str,
        messages: list[dict[str, Any]],
        audios: list[str] | None = None,
    ) -> dict[str, Any]:
        """Create MLX-ready multimodal inputs from formatted chat messages."""
        image_inputs, video_inputs = _extract_vision_inputs(messages)
        inputs = self._create_processor_inputs(
            text,
            images=image_inputs,
            videos=video_inputs,
            audios=audios or None,
        )
        return self._to_mlx_inputs(inputs)

    def create_inputs(
        self,
        text: str,
        images: list[Any] = None,
        videos: list[Any] = None,
        audios: list[str] = None,
    ) -> dict[str, Any]:
        """Create processor inputs from already extracted media values."""
        inputs = self._create_processor_inputs(text, images, videos, audios)
        return self._to_mlx_inputs(inputs)

    def resolve_max_tokens(self, params: dict[str, Any]) -> int:
        """Resolve maximum generation tokens for VLM generation.

        Parameters
        ----------
        params : dict[str, Any]
            Request parameters supplied by the OpenAI handler.

        Returns
        -------
        int
            Maximum number of tokens to generate.
        """
        max_tokens = params.get("max_tokens")
        if max_tokens is None:
            max_tokens = params.get("max_completion_tokens")
        if max_tokens is None:
            max_tokens = DEFAULT_MAX_TOKENS
        return int(max_tokens)

    def build_logits_processors(self, params: dict[str, Any]) -> list[Any]:
        """Build request-specific logits processors for VLM generation.

        Parameters
        ----------
        params : dict[str, Any]
            Request model parameters, including optional repetition settings
            and JSON schema.

        Returns
        -------
        list[Any]
            Logits processors accepted by ``mlx-vlm`` batch generation.
        """
        from mlx_lm.sample_utils import make_logits_processors

        processors: list[Any] = list(
            make_logits_processors(
                logit_bias=params.get("logit_bias"),
                repetition_penalty=params.get("repetition_penalty"),
                repetition_context_size=params.get(
                    "repetition_context_size", DEFAULT_REPETITION_CONTEXT_SIZE
                ),
            )
        )
        json_schema = params.get("schema")
        if json_schema:
            processors.append(
                JSONLogitsProcessor(
                    schema=json_schema,
                    tokenizer=self.outlines_tokenizer,
                    tensor_library_name="mlx",
                )
            )
        return processors

    def create_batch_prompt_kwargs(
        self, model_inputs: dict[str, Any]
    ) -> tuple[list[int], dict[str, Any]]:
        """Create ``mlx-vlm`` batcher prompt kwargs from prepared model inputs.

        Parameters
        ----------
        model_inputs : dict[str, Any]
            MLX-ready inputs produced by :meth:`create_model_inputs`.

        Returns
        -------
        tuple[list[int], dict[str, Any]]
            Token IDs for one request and embedding kwargs for
            ``mlx_vlm.generate.BatchGenerator.insert``.
        """
        inputs = dict(model_inputs)
        input_ids = inputs.pop("input_ids")
        pixel_values = inputs.pop("pixel_values", None)
        mask = inputs.pop("mask", inputs.pop("attention_mask", None))

        embedding_output = self.model.get_input_embeddings(
            input_ids,
            pixel_values,
            mask=mask,
            **inputs,
        )
        prompt_kwargs = {**inputs, **embedding_output.to_dict()}
        token_ids = (
            input_ids.tolist()[0] if getattr(input_ids, "ndim", 0) == 2 else input_ids.tolist()
        )
        return [int(token) for token in token_ids], prompt_kwargs

    def __call__(
        self, prompt: str, stream: bool = False, **kwargs
    ) -> CompletionResponse | Generator[str, None, None]:
        """
        Generate text response from images and messages.

        Args:
            prompt (str): The input prompt text.
            stream (bool, optional): Whether to stream the response. Defaults to False.
            **kwargs: Additional model parameters (temperature, max_tokens, etc.)
                - schema (dict, optional): JSON schema for structured output generation.

        Returns:
            Union[CompletionResponse, Generator[str, None, None]]:
                - If stream=False: Complete response as CompletionResponse
                - If stream=True: Generator yielding response chunks
        """

        def _get(key, default):
            v = kwargs.get(key)
            return default if v is None else v

        seed = _get("seed", DEFAULT_SEED)

        if seed and seed >= 0:
            mx.random.seed(seed)

        # Handle JSON schema for structured outputs
        json_schema = kwargs.get("schema")
        logits_processors = []
        if json_schema:
            logits_processors.append(
                JSONLogitsProcessor(
                    schema=json_schema, tokenizer=self.outlines_tokenizer, tensor_library_name="mlx"
                )
            )

        model_params = dict(kwargs.get("model_inputs") or kwargs.get("vision_inputs") or {})
        max_tokens = _get("max_tokens", None)
        if max_tokens is None:
            max_tokens = _get("max_completion_tokens", DEFAULT_MAX_TOKENS)
        sampling_params = {
            "max_tokens": max_tokens,
            "temperature": _get("temperature", DEFAULT_TEMPERATURE),
            "repetition_penalty": kwargs.get("repetition_penalty"),
            "repetition_context_size": _get(
                "repetition_context_size", DEFAULT_REPETITION_CONTEXT_SIZE
            ),
            "top_p": _get("top_p", DEFAULT_TOP_P),
        }

        # KV cache quantization params
        kv_bits = kwargs.get("kv_bits")
        if kv_bits is not None:
            sampling_params["kv_bits"] = kv_bits
            sampling_params["kv_group_size"] = kwargs.get("kv_group_size", 64)
            sampling_params["quantized_kv_start"] = kwargs.get("quantized_kv_start", 0)

        model_params.update(sampling_params)

        response_generator = stream_generate(
            self.model,
            self.processor,
            prompt=prompt,
            logits_processors=logits_processors,
            **model_params,
        )

        if stream:
            return response_generator

        text = ""
        tokens = []
        final_chunk = None

        for chunk in response_generator:
            if chunk and chunk.text:
                text += chunk.text
                tokens.append(chunk.token)
                final_chunk = chunk

        return CompletionResponse(
            text=text,
            tokens=tokens,
            peak_memory=final_chunk.peak_memory,
            generation_tps=final_chunk.generation_tps,
            prompt_tps=final_chunk.prompt_tps,
            prompt_tokens=final_chunk.prompt_tokens,
            generation_tokens=final_chunk.generation_tokens,
        )


if __name__ == "__main__":
    image_path = "examples/images/attention.png"
    video_path = "examples/videos/demo.mp4"
    model_path = "mlx-community/Qwen3-VL-2B-Thinking-8bit"

    model = MLX_VLM(model_path, context_length=2048)

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
    chat_template_kwargs = {
        "tools": tools,
        "enable_thinking": True,
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
    input_prompt = model.create_input_prompt(messages, chat_template_kwargs)

    image_inputs, video_inputs = _extract_vision_inputs(messages)

    inputs = model.create_inputs(input_prompt, image_inputs, video_inputs)

    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            inputs[key] = mx.array(value)

    sampling_params = {
        "max_tokens": 100000,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 0,
        "repetition_penalty": None,
        "repetition_context_size": 20,
    }

    inputs.update(sampling_params)

    inputs["schema"] = None

    response = model(input_prompt, stream=False, **inputs)

    print("RESPONSE: ", response)
