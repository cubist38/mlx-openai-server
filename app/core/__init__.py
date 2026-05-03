"""Core server utilities with lazy backend imports."""

from __future__ import annotations

from typing import Any

__all__ = [
    "AudioProcessor",
    "BaseProcessor",
    "BatchChunk",
    "BatchScheduler",
    "HandlerProcessProxy",
    "ImageProcessor",
    "InferenceWorker",
    "ModelRegistry",
    "VideoProcessor",
]


def __getattr__(name: str) -> Any:
    """Lazily import core helpers so one backend does not initialize all stacks."""
    if name == "AudioProcessor":
        from .audio_processor import AudioProcessor

        return AudioProcessor
    if name == "BaseProcessor":
        from .base_processor import BaseProcessor

        return BaseProcessor
    if name == "BatchChunk":
        from .batch_scheduler import BatchChunk

        return BatchChunk
    if name == "BatchScheduler":
        from .batch_scheduler import BatchScheduler

        return BatchScheduler
    if name == "HandlerProcessProxy":
        from .handler_process import HandlerProcessProxy

        return HandlerProcessProxy
    if name == "ImageProcessor":
        from .image_processor import ImageProcessor

        return ImageProcessor
    if name == "InferenceWorker":
        from .inference_worker import InferenceWorker

        return InferenceWorker
    if name == "ModelRegistry":
        from .model_registry import ModelRegistry

        return ModelRegistry
    if name == "VideoProcessor":
        from .video_processor import VideoProcessor

        return VideoProcessor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
