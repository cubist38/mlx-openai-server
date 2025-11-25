"""MLX model handlers for text, multimodal, image generation, and embeddings models."""

from __future__ import annotations

import importlib.util
from typing import Any

# Lazy imports to avoid loading MLX during testing
__all__ = [
    "MFLUX_AVAILABLE",
    "MLXEmbeddingsHandler",
    "MLXFluxHandler",
    "MLXLMHandler",
    "MLXVLMHandler",
]


def __getattr__(name: str) -> Any:
    if name == "MLXEmbeddingsHandler":
        from .mlx_embeddings import MLXEmbeddingsHandler  # noqa: PLC0415

        return MLXEmbeddingsHandler
    if name == "MLXLMHandler":
        from .mlx_lm import MLXLMHandler  # noqa: PLC0415

        return MLXLMHandler
    if name == "MLXVLMHandler":
        from .mlx_vlm import MLXVLMHandler  # noqa: PLC0415

        return MLXVLMHandler
    if name == "MLXFluxHandler":
        try:
            from .mflux import MLXFluxHandler  # noqa: PLC0415
        except ImportError:
            return None
        else:
            return MLXFluxHandler
    if name == "MFLUX_AVAILABLE":
        return importlib.util.find_spec("mflux") is not None
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
