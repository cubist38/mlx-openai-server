import os
import mlx.core as mx
from mlx_lm.utils import load, pipeline_load
from mlx_lm.generate import (
    generate,
    stream_generate,
)
from outlines.processors import JSONLogitsProcessor
from mlx_lm.models.cache import make_prompt_cache
from typing import List, Dict, Union, Generator
from mlx_lm.sample_utils import make_sampler, make_logits_processors
from app.utils.outlines_transformer_tokenizer import OutlinesTransformerTokenizer

DEFAULT_TEMPERATURE = os.getenv("DEFAULT_TEMPERATURE", 0.7)
DEFAULT_TOP_P = os.getenv("DEFAULT_TOP_P", 0.95)
DEFAULT_TOP_K = os.getenv("DEFAULT_TOP_K", 20)
DEFAULT_MIN_P = os.getenv("DEFAULT_MIN_P", 0.0)
DEFAULT_SEED = os.getenv("DEFAULT_SEED", 0)
DEFAULT_MAX_TOKENS = os.getenv("DEFAULT_MAX_TOKENS", 8192)
DEFAULT_BATCH_SIZE = os.getenv("DEFAULT_BATCH_SIZE", 32)

class MLX_LM:
    """
    A wrapper class for MLX Language Model that handles both streaming and non-streaming inference.
    
    This class provides a unified interface for generating text responses from text prompts,
    supporting both streaming and non-streaming modes.
    """

    def __init__(self, model_path: str, context_length: int = 32768, pipeline: bool = False):
        try:
            self.pipeline = pipeline
            self.model, self.tokenizer = self._initialize_model(model_path)
            if self.pipeline:
                self.group = mx.distributed.init()
                self.rank = self.group.rank()
            self.pad_token_id = self.tokenizer.pad_token_id
            self.bos_token = self.tokenizer.bos_token
            self.model_type = self.model.model_type
            self.max_kv_size = context_length
            self.outlines_tokenizer = OutlinesTransformerTokenizer(self.tokenizer)
        except Exception as e:
            raise ValueError(f"Error loading model: {str(e)}")

    def _initialize_model(self, model_path: str):
        if self.pipeline:
            return pipeline_load(model_path)
        return load(model_path)

    def get_model_type(self) -> str:
        return self.model_type
        
    def __call__(
        self, 
        messages: List[Dict[str, str]], 
        stream: bool = False, 
        **kwargs
    ) -> Union[str, Generator[str, None, None]]:
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
        chat_template_kwargs = kwargs.get("chat_template_kwargs", {})

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
        prompt_cache = make_prompt_cache(self.model, self.max_kv_size)

        input_tokens = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            **chat_template_kwargs,
        )      

        sampler = make_sampler(
           **sampler_kwargs
        )
                
        if not stream:
            return generate(
                self.model,
                self.tokenizer,
                input_tokens,
                sampler=sampler,
                max_tokens=max_tokens,
                prompt_cache=prompt_cache,
                logits_processors=logits_processors
            )
        else:
            # Streaming mode: return generator of chunks
            return stream_generate(
                self.model,
                self.tokenizer,
                input_tokens,
                sampler=sampler,
                max_tokens=max_tokens,
                prompt_cache=prompt_cache,
                logits_processors=logits_processors
            )