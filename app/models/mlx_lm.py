import os
import mlx.core as mx
from loguru import logger
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
from ..utils.debug_logging import log_debug_chat_template
from typing import List, Dict, Union, Generator, Any

DEFAULT_TEMPERATURE = os.getenv("DEFAULT_TEMPERATURE", 0.7)
DEFAULT_TOP_P = os.getenv("DEFAULT_TOP_P", 0.95)
DEFAULT_TOP_K = os.getenv("DEFAULT_TOP_K", 20)
DEFAULT_MIN_P = os.getenv("DEFAULT_MIN_P", 0.0)
DEFAULT_XTC_PROBABILITY = os.getenv("DEFAULT_XTC_PROBABILITY", 0.0)
DEFAULT_XTC_THRESHOLD = os.getenv("DEFAULT_XTC_THRESHOLD", 0.0)
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

class MLX_LM:
    """
    A wrapper class for MLX Language Model that handles both streaming and non-streaming inference.
    
    This class provides a unified interface for generating text responses from text prompts,
    supporting both streaming and non-streaming modes.
    """

    def __init__(self, model_path: str, draft_model_path: str = None, num_draft_tokens: int = 2, context_length: int | None = None, trust_remote_code: bool = False, chat_template_file: str = None, debug: bool = False):
        try:
            self.model, self.tokenizer = load(model_path, lazy=False, tokenizer_config = {"trust_remote_code": trust_remote_code})
            self.context_length = context_length
            self.draft_model = None
            self.draft_tokenizer = None
            self.num_draft_tokens = num_draft_tokens
            if draft_model_path:
                self._load_draft_model(draft_model_path, trust_remote_code)
            self.pad_token_id = self.tokenizer.pad_token_id
            self.bos_token = self.tokenizer.bos_token
            self.model_type = self.model.model_type
            self.debug = debug
            self.outlines_tokenizer = OutlinesTransformerTokenizer(self.tokenizer)
            if chat_template_file:
                if not os.path.exists(chat_template_file):
                    raise ValueError(f"Chat template file {chat_template_file} does not exist")
                with open(chat_template_file, "r") as f:
                    template_content = f.read()
                    self.tokenizer.chat_template = template_content
                if self.debug:
                    log_debug_chat_template(chat_template_file=chat_template_file, template_content=template_content)
        except Exception as e:
            raise ValueError(f"Error loading model: {str(e)}")

    def _load_draft_model(self, draft_model_path: str, trust_remote_code: bool) -> None:
        try:
            self.draft_model, self.draft_tokenizer = load(draft_model_path, lazy=False, tokenizer_config = {"trust_remote_code": trust_remote_code})
            self.context_length = None # speculative decoding does not support context length, should be set to None
            self._validate_draft_tokenizer()
        except Exception as e:
            raise ValueError(f"Error loading draft model: {str(e)}")

    def _validate_draft_tokenizer(self) -> None:
        if self.draft_tokenizer.vocab_size != self.tokenizer.vocab_size:
            logger.warning(
                "Draft model tokenizer does not match model tokenizer. "
                "Speculative decoding may not work as expected."
            )

    def create_prompt_cache(self) -> List[Any]:
        cache = make_prompt_cache(self.model, max_kv_size=self.context_length)
        if self.draft_model:
            cache += make_prompt_cache(self.draft_model, max_kv_size=self.context_length)
        return cache

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
        return self.tokenizer.encode(input_prompt)

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
        seed = kwargs.get("seed")
        max_tokens = kwargs.get("max_tokens", DEFAULT_MAX_TOKENS)

        sampler_kwargs = {
            "temp": kwargs.get("temperature", DEFAULT_TEMPERATURE),
            "top_p": kwargs.get("top_p", DEFAULT_TOP_P),
            "top_k": kwargs.get("top_k", DEFAULT_TOP_K),
            "min_p": kwargs.get("min_p", DEFAULT_MIN_P),
            "xtc_probability": kwargs.get("xtc_probability", DEFAULT_XTC_PROBABILITY),
            "xtc_threshold": kwargs.get("xtc_threshold", DEFAULT_XTC_THRESHOLD),
        }

        # Add XTC special tokens (EOS and newline) when XTC is enabled
        if sampler_kwargs["xtc_probability"] > 0:
            sampler_kwargs["xtc_special_tokens"] = [
                self.tokenizer.eos_token_id
            ] + self.tokenizer.encode("\n")

        repetition_penalty = kwargs.get("repetition_penalty", 1.0)
        repetition_context_size = kwargs.get("repetition_context_size", 20)
        logit_bias = kwargs.get("logit_bias", None)
        # Convert string keys to int if logit_bias is provided (OpenAI API uses string keys)
        if logit_bias is not None:
            logit_bias = {int(k): v for k, v in logit_bias.items()}
        logits_processors = make_logits_processors(
            logit_bias=logit_bias,
            repetition_penalty=repetition_penalty,
            repetition_context_size=repetition_context_size
        )
        json_schema = kwargs.get("schema", None)
        if json_schema:
            logits_processors.append(
                JSONLogitsProcessor(
                    schema = json_schema,
                    tokenizer = self.outlines_tokenizer,
                    tensor_library_name = "mlx"
                )
            )

        # Only seed RNG when an explicit non-negative seed is provided
        # None or negative values (e.g., -1) result in non-deterministic generation
        if seed is not None and seed >= 0:
            mx.random.seed(seed)
        
        prompt_progress_callback = kwargs.get("prompt_progress_callback")
        
        sampler = make_sampler(
           **sampler_kwargs
        )

        stream_response = stream_generate(
            self.model,
            self.tokenizer,
            input_ids,
            draft_model=self.draft_model,
            sampler=sampler,
            max_tokens=max_tokens,
            num_draft_tokens=self.num_draft_tokens,
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