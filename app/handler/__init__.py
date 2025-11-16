"""
MLX model handlers for text, multimodal, image generation, and embeddings models.
"""

from .mlx_embeddings import MLXEmbeddingsHandler
from .mlx_lm import MLXLMHandler
from .mlx_vlm import MLXVLMHandler

# Optional mflux import - only available if flux extra is installed
try:
    from .mflux import MLXFluxHandler

    MFLUX_AVAILABLE = True
except ImportError:
    MLXFluxHandler = None
    MFLUX_AVAILABLE = False

__all__ = [
    "MLXLMHandler",
    "MLXVLMHandler",
    "MLXFluxHandler",
    "MLXEmbeddingsHandler",
    "MFLUX_AVAILABLE",
]
