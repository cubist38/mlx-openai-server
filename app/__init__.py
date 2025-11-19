import os
from .version import __version__

# Suppress transformers warnings
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

__all__ = ["__version__"]