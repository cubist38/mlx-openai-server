"""MLX OpenAI Server package initialization.

This module provides the main entry point and version information for the
MLX OpenAI Server package, which implements OpenAI-compatible APIs using
MLX for machine learning inference.
"""

import os

from .version import __version__

# Suppress transformers warnings
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

__all__ = ["__version__"]
