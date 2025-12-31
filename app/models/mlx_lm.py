import gc
import os
import json
import mlx.core as mx
from loguru import logger
from mlx_lm.utils import load
from mlx_lm.generate import (
    stream_generate
)
from mlx_lm.generate import GenerationResponse
from outlines.processors import JSONLogitsProcessor
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler, make_logits_processors
from ..utils.outlines_transformer_tokenizer import OutlinesTransformerTokenizer
from ..utils.debug_logging import log_debug_prompt
from typing import List, Dict, Union, Generator

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

    def __init__(self, model_path: str, context_length: int = 32768, trust_remote_code: bool = False, chat_template_file: str = None):
        try:
            self.model, self.tokenizer = load(model_path, lazy=False, tokenizer_config = {"trust_remote_code": trust_remote_code})
            self.pad_token_id = self.tokenizer.pad_token_id
            self.bos_token = self.tokenizer.bos_token
            self.model_type = self.model.model_type
            self.max_kv_size = context_length
            self.outlines_tokenizer = OutlinesTransformerTokenizer(self.tokenizer)
            if chat_template_file:
                if not os.path.exists(chat_template_file):
                    raise ValueError(f"Chat template file {chat_template_file} does not exist")
                with open(chat_template_file, "r") as f:
                    self.tokenizer.chat_template = f.read()
        except Exception as e:
            raise ValueError(f"Error loading model: {str(e)}")
        
    def _apply_pooling_strategy(self, embeddings: mx.array) -> mx.array:
        embeddings = mx.mean(embeddings, axis=1)
        return embeddings
    
    def _apply_l2_normalization(self, embeddings: mx.array) -> mx.array:
        l2_norms = mx.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (l2_norms +  1e-8)
        return embeddings
    
    def _batch_process(self, prompts: List[str], batch_size: int = DEFAULT_BATCH_SIZE) -> List[List[int]]:
        """Process prompts in batches with optimized tokenization."""
        all_tokenized = []
        
        # Process prompts in batches
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            tokenized_batch = []
            
            # Tokenize all prompts in batch
            for p in batch:
                add_special_tokens = self.bos_token is None or not p.startswith(self.bos_token)
                tokens = self.tokenizer.encode(p, add_special_tokens=add_special_tokens)
                tokenized_batch.append(tokens)
            
            # Find max length in batch
            max_length = max(len(tokens) for tokens in tokenized_batch)
            
            # Pad tokens in a vectorized way
            for tokens in tokenized_batch:
                padding = [self.pad_token_id] * (max_length - len(tokens))
                all_tokenized.append(tokens + padding)
        
        return all_tokenized

    def _preprocess_prompt(self, prompt: str) -> List[int]:
        """Tokenize a single prompt efficiently."""
        add_special_tokens = self.bos_token is None or not prompt.startswith(self.bos_token)
        tokens = self.tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
        return mx.array(tokens)
    
    def get_model_type(self) -> str:
        return self.model_type
    
    def get_embeddings(
        self, 
        prompts: List[str], 
        batch_size: int = DEFAULT_BATCH_SIZE,
        normalize: bool = True
    ) -> List[float]:
        """
        Get embeddings for a list of prompts efficiently.
        
        Args:
            prompts: List of text prompts
            batch_size: Size of batches for processing
            
        Returns:
            List of embeddings as float arrays
        """
        # Process in batches to optimize memory usage
        all_embeddings = []
        try:
            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i:i + batch_size]
                tokenized_batch = self._batch_process(batch_prompts, batch_size)
                
                # Convert to MLX array for efficient computation
                tokenized_batch = mx.array(tokenized_batch)
                
                try:
                    # Compute embeddings for batch
                    batch_embeddings = self.model.model(tokenized_batch)
                    pooled_embedding = self._apply_pooling_strategy(batch_embeddings)
                    if normalize:
                        pooled_embedding = self._apply_l2_normalization(pooled_embedding)
                    all_embeddings.extend(pooled_embedding.tolist())
                finally:
                    # Explicitly free MLX arrays to prevent memory leaks
                    del tokenized_batch
                    if 'batch_embeddings' in locals():
                        del batch_embeddings
                    if 'pooled_embedding' in locals():
                        del pooled_embedding
                    # Force MLX garbage collection
                    mx.clear_cache()
                    gc.collect()
        except Exception as e:
            # Clean up on error
            mx.clear_cache()
            gc.collect()
            raise

        return all_embeddings
        
    def __call__(
        self, 
        messages: List[Dict[str, str]], 
        stream: bool = False, 
        verbose: bool = False,
        **kwargs
    ) -> Union[GenerationResponse, Generator[GenerationResponse, None, None]]:
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
        
        input_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize = False,
            add_generation_prompt=True,
            **chat_template_kwargs,
        )      
        
        sampler = make_sampler(
           **sampler_kwargs
        )

        if verbose:
            log_debug_prompt(input_prompt)

        stream_response = stream_generate(
            self.model,
            self.tokenizer,
            input_prompt,
            sampler=sampler,
            max_tokens=max_tokens,
            prompt_cache=prompt_cache,
            logits_processors=logits_processors
        )
        if stream:
            return stream_response

        text = ""
        final_chunk = None
        for chunk in stream_response:
            text += chunk.text
            if chunk.finish_reason:
                final_chunk = chunk

        return GenerationResponse(
            text=text,
            finish_reason=final_chunk.finish_reason,
            prompt_tokens=final_chunk.prompt_tokens,
            prompt_tps=final_chunk.prompt_tps,
            generation_tokens=final_chunk.generation_tokens,
            generation_tps=final_chunk.generation_tps,
            peak_memory=final_chunk.peak_memory,
            logprobs=final_chunk.logprobs,
            from_draft=final_chunk.from_draft,
            token=final_chunk.token,
        )
        
if __name__ == "__main__":
    model = MLX_LM(model_path="mlx-community/functiongemma-270m-it-8bit")
    print(model.get_model_type())

    messages = [
        {
            "role": "user",
            "content": "What is the date today?"
        }
    ]
    chat_template_kwargs = {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_date",
                    "description": "Get the date today",
                }
            }
        ]
    }
    kwargs = {
        "chat_template_kwargs": chat_template_kwargs,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 0,
        "max_tokens": 512
    }
    response = model(messages, stream=False, **kwargs)
    print(response)