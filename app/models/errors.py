"""Custom exceptions for the models module."""

from __future__ import annotations


class FluxModelError(Exception):
    """Base exception for Flux model errors."""


class ModelLoadError(FluxModelError):
    """Raised when model loading fails."""

    def __init__(self, message_or_path: str, original_exception: Exception | None = None) -> None:
        if original_exception is not None:
            self.model_path = message_or_path
            self.original_exception = original_exception
            super().__init__(f"Failed to load model from {message_or_path}: {original_exception}")
        else:
            super().__init__(message_or_path)


class ModelGenerationError(FluxModelError):
    """Raised when image generation fails."""


class InvalidConfigurationError(FluxModelError):
    """Raised when configuration is invalid."""
