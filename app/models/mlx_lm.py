import os
import mlx.core as mx
from mlx_lm.utils import load
from mlx_lm.generate import (
    stream_generate
)
from dataclasses import dataclass
from mlx_lm.generate import GenerationResponse
from outlines.processors import JSONLogitsProcessor
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler, make_logits_processors
from ..utils.outlines_transformer_tokenizer import OutlinesTransformerTokenizer
from typing import List, Dict, Union, Generator, Any

DEFAULT_TEMPERATURE = os.getenv("DEFAULT_TEMPERATURE", 0.7)
DEFAULT_TOP_P = os.getenv("DEFAULT_TOP_P", 0.95)
DEFAULT_TOP_K = os.getenv("DEFAULT_TOP_K", 20)
DEFAULT_MIN_P = os.getenv("DEFAULT_MIN_P", 0.0)
DEFAULT_SEED = os.getenv("DEFAULT_SEED", 0)
DEFAULT_MAX_TOKENS = os.getenv("DEFAULT_MAX_TOKENS", 8192)
DEFAULT_BATCH_SIZE = os.getenv("DEFAULT_BATCH_SIZE", 32)

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

@dataclass
class SpecialModelConfig:
    """Configuration for special model handling.
    
    Parameters
    ----------
    eos_token_ids_to_remove : list[int] | None
        List of EOS token IDs to remove from the tokenizer's EOS token list.
    """
    eos_token_ids_to_remove: list[int] | None = None


# Registry for special model configurations
_SPECIAL_MODEL_CONFIGS: dict[str, SpecialModelConfig] = {
    "gpt_oss": SpecialModelConfig(
        eos_token_ids_to_remove=[200012],  # <|call|> should be handled in the parser
    ),
}

class MLX_LM:
    """
    A wrapper class for MLX Language Model that handles both streaming and non-streaming inference.
    
    This class provides a unified interface for generating text responses from text prompts,
    supporting both streaming and non-streaming modes.
    """

    def __init__(self, model_path: str, context_length: int | None = None, trust_remote_code: bool = False, chat_template_file: str = None):
        try:
            self.model, self.tokenizer = load(model_path, lazy=False, tokenizer_config = {"trust_remote_code": trust_remote_code})
            self.pad_token_id = self.tokenizer.pad_token_id
            self.bos_token = self.tokenizer.bos_token
            self.model_type = self.model.model_type
            self._handle_special_models()
            self.context_length = context_length
            self.outlines_tokenizer = OutlinesTransformerTokenizer(self.tokenizer)
            if chat_template_file:
                if not os.path.exists(chat_template_file):
                    raise ValueError(f"Chat template file {chat_template_file} does not exist")
                with open(chat_template_file, "r") as f:
                    self.tokenizer.chat_template = f.read()
        except Exception as e:
            raise ValueError(f"Error loading model: {str(e)}")

    def _handle_special_models(self) -> None:
        """Apply special handling for specific model types."""
        config = _SPECIAL_MODEL_CONFIGS.get(self.model_type)
        if config is None:
            return
        
        if config.eos_token_ids_to_remove:
            for token_id in config.eos_token_ids_to_remove:
                if hasattr(self.tokenizer, "_eos_token_ids") and token_id in self.tokenizer._eos_token_ids:
                    self.tokenizer._eos_token_ids.remove(token_id)

    def create_prompt_cache(self) -> List[Any]:
        return make_prompt_cache(self.model, max_kv_size=self.context_length)
        
    def get_model_type(self) -> str:
        return self.model_type

    def create_input_prompt(self, messages: List[Dict[str, str]], chat_template_kwargs: Dict[str, Any]) -> str:
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize = False,
            add_generation_prompt=True,
            **chat_template_kwargs,
        )

    def encode_prompt(self, input_prompt: str) -> List[int]:
        add_special_tokens = self.tokenizer.bos_token is None or not input_prompt.startswith(
            self.tokenizer.bos_token
        )
        return self.tokenizer.encode(input_prompt, add_special_tokens=add_special_tokens)

    def __call__(
        self, 
        input_ids: List[int],
        prompt_cache: List[Any] = None,
        stream: bool = False, 
        **kwargs
    ) -> Union[CompletionResponse, Generator[GenerationResponse, None, None]]:
        """
        Generate text response from the model.

        Args:
            messages (List[Dict[str, str]]): List of messages in the conversation.
            stream (bool): Whether to stream the response.
            **kwargs: Additional parameters for generation
                - temperature: Sampling temperature (default: 0.0)
                - top_p: Top-p sampling parameter (default: 1.0)
                - seed: Random seed (default: 0)
                - max_tokens: Maximum number of tokens to generate (default: 256)
        """
        # Set default parameters if not provided
        seed = kwargs.get("seed", DEFAULT_SEED)
        max_tokens = kwargs.get("max_tokens", DEFAULT_MAX_TOKENS)

        sampler_kwargs = {
            "temp": kwargs.get("temperature", DEFAULT_TEMPERATURE),
            "top_p": kwargs.get("top_p", DEFAULT_TOP_P),
            "top_k": kwargs.get("top_k", DEFAULT_TOP_K),
            "min_p": kwargs.get("min_p", DEFAULT_MIN_P)
        }

        repetition_penalty = kwargs.get("repetition_penalty", 1.0)
        repetition_context_size = kwargs.get("repetition_context_size", 20)
        logits_processors = make_logits_processors(repetition_penalty=repetition_penalty, repetition_context_size=repetition_context_size)
        json_schema = kwargs.get("schema", None)
        if json_schema:
            logits_processors.append(
                JSONLogitsProcessor(
                    schema = json_schema,
                    tokenizer = self.outlines_tokenizer,
                    tensor_library_name = "mlx"
                )
            )
        
        mx.random.seed(seed)
        
        prompt_progress_callback = kwargs.get("prompt_progress_callback")
        
        sampler = make_sampler(
           **sampler_kwargs
        )

        stream_response = stream_generate(
            self.model,
            self.tokenizer,
            input_ids,
            sampler=sampler,
            max_tokens=max_tokens,
            prompt_cache=prompt_cache,
            logits_processors=logits_processors,
            prompt_progress_callback=prompt_progress_callback
        )
        if stream:
            return stream_response

        text = ""
        tokens = []
        final_chunk = None
        for chunk in stream_response:
            text += chunk.text
            tokens.append(chunk.token)
            if chunk.finish_reason:
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