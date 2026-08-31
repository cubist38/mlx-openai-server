"""
MLX model handlers for text, multimodal, image generation, and embeddings models.
"""

from typing import Any

__all__ = [
    "MLXLMHandler",
    "MLXVLMHandler",
    "MLXFluxHandler",
    "MLXEmbeddingsHandler",
    "MFLUX_AVAILABLE",
]


def __getattr__(name: str) -> Any:
    """Lazily import handlers so one backend does not initialize all MLX stacks."""
    if name == "MLXLMHandler":
        from .mlx_lm import MLXLMHandler

        return MLXLMHandler
    if name == "MLXVLMHandler":
        from .mlx_vlm import MLXVLMHandler

        return MLXVLMHandler
    if name == "MLXEmbeddingsHandler":
        from .mlx_embeddings import MLXEmbeddingsHandler

        return MLXEmbeddingsHandler
    if name == "MLXFluxHandler":
        from .mflux import MLXFluxHandler

        return MLXFluxHandler
    if name == "MFLUX_AVAILABLE":
        import mflux  # noqa: F401

        return True
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
