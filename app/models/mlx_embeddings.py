"""
MLX embeddings model wrapper.

This module provides a wrapper class for MLX embeddings models with memory
management and caching capabilities.
"""

from __future__ import annotations

from collections.abc import MutableMapping
import gc
from typing import Protocol, cast

from loguru import logger
import mlx.core as mx
from mlx_embeddings.utils import load

from .errors import ModelLoadError


class ArrayLike(Protocol):
    """Protocol for array-like objects with nbytes attribute."""

    nbytes: int


class MLX_Embeddings:
    """
    A wrapper class for MLX Embeddings that handles memory management to prevent leaks.

    This class provides a unified interface for generating embeddings from text inputs,
    with proper cleanup of MLX arrays and memory management.
    """

    def __init__(self, model_path: str) -> None:
        """Initialize the MLX_Embeddings model.

        Parameters
        ----------
        model_path : str
            Path to the model to load.

        Raises
        ------
        ModelLoadError
            If model loading fails.
        """
        try:
            self.model, self.tokenizer = load(model_path)
        except Exception as e:
            raise ModelLoadError(model_path, e) from e

    def _get_embeddings(self, texts: list[str], max_length: int = 512) -> mx.array:
        """Get embeddings for a list of texts with proper memory management.

        Parameters
        ----------
        texts : list[str]
            List of text inputs.
        max_length : int, optional
            Maximum sequence length for tokenization. Default is 512.

        Returns
        -------
        mx.array
            MLX array of embeddings.
        """
        inputs = None
        outputs = None
        try:
            # Tokenize inputs
            inputs = self.tokenizer.batch_encode_plus(
                texts, return_tensors="mlx", padding=True, truncation=True, max_length=max_length
            )

            # Generate embeddings
            outputs = self.model(
                inputs["input_ids"], attention_mask=inputs["attention_mask"]
            ).text_embeds

            # Return a copy to ensure the result persists after cleanup
            return mx.array(outputs)

        finally:
            # Always clean up intermediate arrays
            self._cleanup_arrays(inputs, outputs)

    def _cleanup_arrays(self, *arrays: MutableMapping[str, ArrayLike] | ArrayLike | None) -> None:
        """Clean up MLX arrays to free memory.

        Parameters
        ----------
        *arrays : MutableMapping[str, ArrayLike] | ArrayLike | None
            One or more arrays or mappings containing arrays to be cleaned.
        """
        for array in arrays:
            if array is not None:
                try:
                    if isinstance(array, MutableMapping):
                        for key, value in list(array.items()):
                            if hasattr(value, "nbytes"):
                                del array[key]
                except (KeyError, AttributeError, TypeError) as e:
                    logger.warning(
                        f"Error during embeddings array cleanup. {type(e).__name__}: {e}"
                    )

    def __call__(self, texts: list[str], max_length: int = 512) -> list[list[float]]:
        """Generate embeddings for a list of texts.

        Parameters
        ----------
        texts : list[str]
            List of text inputs.
        max_length : int, optional
            Maximum sequence length for tokenization. Default is 512.

        Returns
        -------
        list[list[float]]
            List of embedding vectors as Python float lists.
        """
        embeddings = None
        try:
            embeddings = self._get_embeddings(texts, max_length)
            # Convert to Python list and return
            return cast("list[list[float]]", embeddings.tolist())
        finally:
            # Clean up the embeddings array and global cache
            if embeddings is not None:
                del embeddings
            mx.clear_cache()
            gc.collect()

    def cleanup(self) -> None:
        """Explicitly cleanup resources.

        Notes
        -----
        Removes loaded model and tokenizer attributes, clears MLX cache, and
        forces garbage collection.

        Returns
        -------
        None
        """
        try:
            # Clear any cached model outputs
            if hasattr(self, "model"):
                del self.model
            if hasattr(self, "tokenizer"):
                del self.tokenizer

            # Clear MLX cache and force garbage collection
            mx.clear_cache()
            gc.collect()
        except Exception as e:
            logger.warning(f"Error during cleanup. {type(e).__name__}: {e}")


if __name__ == "__main__":
    model_path = "mlx-community/all-MiniLM-L6-v2-4bit"
    model = MLX_Embeddings(model_path)
    try:
        texts = ["I like reading", "I like writing"]
        embeddings = model(texts)
        logger.info(f"Generated embeddings shape: {len(embeddings)} x {len(embeddings[0])}")
    finally:
        # Explicit cleanup
        model.cleanup()
