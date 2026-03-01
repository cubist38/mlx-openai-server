import os
import torch
import mlx.core as mx
from dataclasses import dataclass
from typing import List, Dict, Union, Generator, Any
from mlx_vlm.models.cache import make_prompt_cache
from mlx_vlm import load, stream_generate
from mlx_vlm.video_generate import process_vision_info
from ..utils.prompt_cache import LRUPromptCache
from outlines.processors import JSONLogitsProcessor
from ..utils.outlines_transformer_tokenizer import OutlinesTransformerTokenizer

# Default model parameters
DEFAULT_MAX_TOKENS = os.getenv("DEFAULT_MAX_TOKENS", 100000)
DEFAULT_TEMPERATURE = os.getenv("DEFAULT_TEMPERATURE", 0.0)
DEFAULT_TOP_P = os.getenv("DEFAULT_TOP_P", 1.0)
DEFAULT_SEED = os.getenv("DEFAULT_SEED", 0)
DEFAULT_REPETITION_CONTEXT_SIZE = os.getenv("DEFAULT_REPETITION_CONTEXT_SIZE", 20)

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
    tokens: List[int] = None
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
    
    def __init__(self, model_path: str, context_length: int | None = None, trust_remote_code: bool = False, chat_template_file: str = None):
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
            self.model, self.processor = load(model_path, lazy=False, trust_remote_code=trust_remote_code)
            self.config = self.model.config
            self.context_length = context_length
            self.outlines_tokenizer = OutlinesTransformerTokenizer(self.processor.tokenizer)
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

    def create_prompt_cache(self) -> List[Any]:
        return make_prompt_cache(self.model.language_model, max_kv_size=self.context_length)

    def create_input_prompt(self, messages: List[Dict[str, str]], chat_template_kwargs: Dict[str, Any]) -> str:
        chat_template_kwargs.pop("_partial_mode", None)

        return self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **chat_template_kwargs
        )

    def create_inputs(self, text: str, images: List[str] = None, videos: List[str] = None) -> Dict[str, Any]:
        return self.processor(
            text=[text],
            images=images,
            videos=videos,
            padding=True,
            return_tensors="pt"
        )

    def __call__(
        self, 
        prompt: str,
        prompt_cache: List[Any] = None,
        stream: bool = False, 
        **kwargs
    ) -> Union[CompletionResponse, Generator[str, None, None]]:
        """
        Generate text response from images and messages.
        
        Args:
            prompt (str): The input prompt text.
            prompt_cache (List[Any], optional): Prompt cache for faster inference.
            stream (bool, optional): Whether to stream the response. Defaults to False.
            **kwargs: Additional model parameters (temperature, max_tokens, etc.)
                - schema (dict, optional): JSON schema for structured output generation.
            
        Returns:
            Union[CompletionResponse, Generator[str, None, None]]: 
                - If stream=False: Complete response as CompletionResponse
                - If stream=True: Generator yielding response chunks
        """
        seed = kwargs.get("seed") or DEFAULT_SEED

        if seed and seed >= 0:
            mx.random.seed(seed)
        
        # Handle JSON schema for structured outputs
        json_schema = kwargs.get("schema")
        logits_processors = []
        if json_schema:
            logits_processors.append(
                JSONLogitsProcessor(
                    schema=json_schema,
                    tokenizer=self.outlines_tokenizer,
                    tensor_library_name="mlx"
                )
            )

        model_params = kwargs.get("vision_inputs")
        # Use 'is not None' to preserve valid 0 values (e.g. temperature=0, top_p=0)
        max_tokens = (
            kwargs.get("max_tokens")
            if kwargs.get("max_tokens") is not None
            else kwargs.get("max_completion_tokens")
            if kwargs.get("max_completion_tokens") is not None
            else DEFAULT_MAX_TOKENS
        )
        sampling_params = {
            "max_tokens": max_tokens,
            "temperature": kwargs.get("temperature") if kwargs.get("temperature") is not None else DEFAULT_TEMPERATURE,
            "repetition_penalty": kwargs.get("repetition_penalty"),
            "repetition_context_size": (
                kwargs.get("repetition_context_size")
                if kwargs.get("repetition_context_size") is not None
                else DEFAULT_REPETITION_CONTEXT_SIZE
            ),
            "top_p": kwargs.get("top_p") if kwargs.get("top_p") is not None else DEFAULT_TOP_P,
        }
        
        model_params.update(sampling_params)

        response_generator = stream_generate(
            self.model,
            self.processor,
            prompt=prompt,
            prompt_cache=prompt_cache,
            logits_processors = logits_processors,
            **model_params
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
    chat_template_kwargs = {
        "tools": tools,
        "enable_thinking": True,
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
    prompt_cache = LRUPromptCache()

    input_prompt = model.create_input_prompt(messages, chat_template_kwargs)

    image_inputs, video_inputs = process_vision_info(messages)

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