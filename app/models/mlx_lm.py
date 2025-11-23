"""
MLX language model wrapper.

This module provides a wrapper class for MLX language models with text generation,
streaming, and caching capabilities.
"""

from __future__ import annotations

from collections.abc import Generator
import gc
import os
from pathlib import Path
from typing import Any, cast

import mlx.core as mx
from mlx_lm.generate import GenerationResponse, stream_generate
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from mlx_lm.utils import load
from outlines.processors import OutlinesLogitsProcessor
from transformers import PreTrainedTokenizer

from ..utils.debug_logging import log_debug_prompt, make_prompt_progress_callback
from ..utils.outlines_transformer_tokenizer import OutlinesTransformerTokenizer

DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))
DEFAULT_TOP_P = float(os.getenv("DEFAULT_TOP_P", "0.95"))
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "20"))
DEFAULT_MIN_P = float(os.getenv("DEFAULT_MIN_P", "0.0"))
DEFAULT_SEED = int(os.getenv("DEFAULT_SEED", "0"))
DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", "8192"))
DEFAULT_BATCH_SIZE = int(os.getenv("DEFAULT_BATCH_SIZE", "32"))


class MLX_LM:
    """
    A wrapper class for MLX Language Model that handles both streaming and non-streaming inference.

    This class provides a unified interface for generating text responses from text prompts,
    supporting both streaming and non-streaming modes.
    """

    def __init__(
        self,
        model_path: str,
        context_length: int = 32768,
        *,
        trust_remote_code: bool = False,
        chat_template_file: str | None = None,
    ) -> None:
        try:
            self.model, self.tokenizer, *_ = load(
                model_path, lazy=False, tokenizer_config={"trust_remote_code": trust_remote_code}
            )
            self.pad_token_id = self.tokenizer.pad_token_id
            self.bos_token = self.tokenizer.bos_token
            self.model_type = self.model.model_type
            self.prompt_cache = make_prompt_cache(self.model, context_length)
            self.outlines_tokenizer = OutlinesTransformerTokenizer(
                cast("PreTrainedTokenizer", self.tokenizer)
            )
            if chat_template_file:
                chat_template_path = Path(chat_template_file)
                if not chat_template_path.exists():
                    raise ValueError(f"Chat template file {chat_template_file} does not exist")
                with chat_template_path.open() as f:
                    self.tokenizer.chat_template = f.read()
        except Exception as e:
            raise ValueError(f"Error loading model. {type(e).__name__}: {e}") from e

    def _get_pad_token_id(self) -> int:
        """Return a safe pad token ID, falling back through available options.

        Returns
        -------
        int
            Pad token id to use (returns 0 if no suitable token id is available).
        """
        if self.pad_token_id is not None:
            return int(self.pad_token_id)
        if hasattr(self.tokenizer, "pad_token_id") and self.tokenizer.pad_token_id is not None:
            return int(self.tokenizer.pad_token_id)
        if hasattr(self.tokenizer, "eos_token_id") and self.tokenizer.eos_token_id is not None:
            return int(self.tokenizer.eos_token_id)
        return 0

    def _apply_pooling_strategy(self, embeddings: mx.array) -> mx.array:
        return mx.mean(embeddings, axis=1)

    def _apply_l2_normalization(self, embeddings: mx.array) -> mx.array:
        l2_norms = mx.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / (l2_norms + 1e-8)

    def _batch_process(
        self, prompts: list[str], batch_size: int = DEFAULT_BATCH_SIZE
    ) -> list[list[int]]:
        """Tokenize prompts in batches for efficient processing.

        Parameters
        ----------
        prompts : list[str]
            List of prompt strings to tokenize.
        batch_size : int, optional
            Number of prompts to process per batch. Default is DEFAULT_BATCH_SIZE.

        Returns
        -------
        list[list[int]]
            List of token id sequences (padded) corresponding to input prompts.
        """
        all_tokenized = []

        # Process prompts in batches
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            tokenized_batch = []

            # Tokenize all prompts in batch
            for p in batch:
                add_special_tokens = self.bos_token is None or not p.startswith(self.bos_token)
                tokens = self.tokenizer.encode(p, add_special_tokens=add_special_tokens)
                tokenized_batch.append(tokens)

            # Find max length in batch
            max_length = max(len(tokens) for tokens in tokenized_batch)

            # Get safe pad token ID
            pad_token_id = self._get_pad_token_id()

            # Pad tokens in a vectorized way
            for tokens in tokenized_batch:
                padding = [pad_token_id] * (max_length - len(tokens))
                all_tokenized.append(tokens + padding)

        return all_tokenized

    def _preprocess_prompt(self, prompt: str) -> mx.array:
        """Tokenize a single prompt to an MLX array.

        Parameters
        ----------
        prompt : str
            The prompt text to tokenize.

        Returns
        -------
        mx.array
            Array of token ids for the prompt.
        """
        add_special_tokens = self.bos_token is None or not prompt.startswith(self.bos_token)
        tokens = self.tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
        return mx.array(tokens)

    def get_model_type(self) -> str:
        """
        Get the model type identifier.

        Returns
        -------
        str
            The model type string.
        """
        return str(self.model_type)

    def get_embeddings(
        self, prompts: list[str], batch_size: int = DEFAULT_BATCH_SIZE, *, normalize: bool = True
    ) -> list[list[float]]:
        """Compute embeddings for a list of prompts.

        Parameters
        ----------
        prompts : list[str]
            List of text prompts.
        batch_size : int, optional
            Batch size for processing. Default is DEFAULT_BATCH_SIZE.
        normalize : bool, optional
            Whether to apply L2-normalization to each pooled embedding. Default is True.

        Returns
        -------
        list[list[float]]
            List of embedding vectors (one per input prompt). Embeddings are
            L2-normalized when ``normalize`` is True.
        """
        # Process in batches to optimize memory usage
        all_embeddings: list[list[float]] = []
        try:
            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i : i + batch_size]
                tokenized_batch = self._batch_process(batch_prompts, batch_size)

                # Convert to MLX array for efficient computation
                tokenized_array = mx.array(tokenized_batch)

                try:
                    # Compute embeddings for batch
                    batch_embeddings = self.model.model(tokenized_array)
                    pooled_embedding = self._apply_pooling_strategy(batch_embeddings)
                    if normalize:
                        pooled_embedding = self._apply_l2_normalization(pooled_embedding)
                    all_embeddings.extend(pooled_embedding.tolist())  # type: ignore[arg-type]
                finally:
                    # Explicitly free MLX arrays to prevent memory leaks
                    del tokenized_batch
                    del tokenized_array
                    if "batch_embeddings" in locals():
                        del batch_embeddings
                    if "pooled_embedding" in locals():
                        del pooled_embedding
                    # Force MLX garbage collection
                    mx.clear_cache()
                    gc.collect()
        except Exception:
            # Clean up on error
            mx.clear_cache()
            gc.collect()
            raise

        return all_embeddings

    def __call__(
        self,
        messages: list[dict[str, str]],
        *,
        stream: bool = False,
        verbose: bool = False,
        **kwargs: Any,
    ) -> GenerationResponse | Generator[GenerationResponse, None, None]:
        """Generate text responses from the model.

        Parameters
        ----------
        messages : list[dict[str, str]]
            List of messages in the conversation.
        stream : bool, optional
            If True, return a generator yielding streaming chunks. Default is False.
        verbose : bool, optional
            If True, enable verbose logging. Default is False.
        **kwargs : Any
            Additional generation parameters such as ``temperature``, ``top_p``, ``seed``, and ``max_tokens``.

        Returns
        -------
        GenerationResponse | Generator[GenerationResponse, None, None]
            A GenerationResponse for non-streaming calls, or a generator yielding GenerationResponse objects for streaming calls.
        """
        # Set default parameters if not provided
        seed = kwargs.get("seed", DEFAULT_SEED)
        max_tokens = kwargs.get("max_tokens", DEFAULT_MAX_TOKENS)
        chat_template_kwargs = kwargs.get("chat_template_kwargs", {})

        sampler_kwargs = {
            "temp": kwargs.get("temperature", DEFAULT_TEMPERATURE),
            "top_p": kwargs.get("top_p", DEFAULT_TOP_P),
            "top_k": kwargs.get("top_k", DEFAULT_TOP_K),
            "min_p": kwargs.get("min_p", DEFAULT_MIN_P),
        }

        repetition_penalty = kwargs.get("repetition_penalty", 1.0)
        repetition_context_size = kwargs.get("repetition_context_size", 20)
        logits_processors = make_logits_processors(
            repetition_penalty=repetition_penalty, repetition_context_size=repetition_context_size
        )
        json_schema = kwargs.get("schema")
        if json_schema:
            logits_processors.append(
                OutlinesLogitsProcessor(  # type: ignore[abstract,call-arg]
                    schema=json_schema, tokenizer=self.outlines_tokenizer, tensor_library_name="mlx"
                )
            )

        mx.random.seed(seed)

        prompt_progress_callback = make_prompt_progress_callback() if verbose else None

        input_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **chat_template_kwargs,
        )

        sampler = make_sampler(**sampler_kwargs)

        if verbose:
            log_debug_prompt(input_prompt)

        stream_response = stream_generate(
            self.model,
            self.tokenizer,
            input_prompt,
            sampler=sampler,
            max_tokens=max_tokens,
            prompt_cache=self.prompt_cache,
            logits_processors=logits_processors,
            prompt_progress_callback=prompt_progress_callback,
        )
        if stream:
            return stream_response

        text = ""
        final_chunk = None
        for chunk in stream_response:
            text += chunk.text
            final_chunk = chunk

        if not isinstance(final_chunk, GenerationResponse):
            raise TypeError("Final chunk is not a GenerationResponse")

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
