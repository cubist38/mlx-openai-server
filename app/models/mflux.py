from __future__ import annotations

from loguru import logger
import inspect
from PIL import Image
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

from mflux.models.common.config import ModelConfig
from mflux.models.z_image.variants import ZImageTurbo
from mflux.models.fibo.variants.txt2img.fibo import FIBO
from mflux.models.flux.variants.txt2img.flux import Flux1
from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage
from mflux.models.flux.variants.kontext.flux_kontext import Flux1Kontext
from mflux.models.qwen.variants.edit.qwen_image_edit import QwenImageEdit
from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein
from mflux.models.flux2.variants.edit.flux2_klein_edit import Flux2KleinEdit


# -----------------------------------------------------------------------------
# Exceptions
# -----------------------------------------------------------------------------


class ImageModelError(Exception):
    """Base exception for image generation model errors."""
    pass


class ModelLoadError(ImageModelError):
    """Raised when model loading fails."""
    pass


class ModelGenerationError(ImageModelError):
    """Raised when image generation fails."""
    pass


class InvalidConfigurationError(ImageModelError):
    """Raised when configuration is invalid."""
    pass


# -----------------------------------------------------------------------------
# Model configuration (registry-driven)
# -----------------------------------------------------------------------------


def _lora_validate(
    lora_paths: Optional[list[str]] | None,
    lora_scales: Optional[list[float]] | None,
) -> None:
    if (lora_paths is None) != (lora_scales is None):
        raise InvalidConfigurationError(
            "Both lora_paths and lora_scales must be provided together."
        )
    if lora_paths and lora_scales and len(lora_paths) != len(lora_scales):
        raise InvalidConfigurationError(
            f"lora_paths and lora_scales must have the same length "
            f"(got {len(lora_paths)} and {len(lora_scales)})"
        )


class ModelConfiguration:
    """Configuration for a single image generation model type."""

    def __init__(
        self,
        model_type: str,
        model_config: ModelConfig,
        quantize: Optional[int] = None,
        lora_paths: Optional[list[str]] = None,
        lora_scales: Optional[list[float]] = None,
    ) -> None:
        if quantize is not None and quantize not in (4, 8, 16):
            raise InvalidConfigurationError(
                f"Invalid quantization level: {quantize}. Must be 4, 8, or 16."
            )
        _lora_validate(lora_paths, lora_scales)
        self.model_type = model_type
        self.model_config = model_config
        self.quantize = quantize
        self.lora_paths = lora_paths
        self.lora_scales = lora_scales

    @classmethod
    def from_name(
        cls,
        config_name: str,
        quantize: Optional[int] = None,
        lora_paths: Optional[list[str]] = None,
        lora_scales: Optional[list[float]] = None,
    ) -> ModelConfiguration:
        if config_name not in _CONFIG_REGISTRY:
            available = ", ".join(_CONFIG_REGISTRY.keys())
            raise InvalidConfigurationError(
                f"Invalid config name: {config_name}. Available: {available}"
            )
        model_type, config_factory = _CONFIG_REGISTRY[config_name]
        return cls(
            model_type=model_type,
            model_config=config_factory(),
            quantize=quantize,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
        )


_CONFIG_REGISTRY: dict[str, tuple[str, Callable[[], ModelConfig]]] = {
    "flux-schnell": ("schnell", ModelConfig.schnell),
    "flux-dev": ("dev", ModelConfig.dev),
    "flux-krea-dev": ("krea-dev", ModelConfig.krea_dev),
    "flux-kontext-dev": ("kontext", ModelConfig.dev_kontext),
    "qwen-image": ("qwen-image", ModelConfig.qwen_image),
    "qwen-image-edit": ("qwen-image-edit", ModelConfig.qwen_image_edit),
    "fibo": ("fibo", ModelConfig.fibo),
    "z-image-turbo": ("z-image-turbo", ModelConfig.z_image_turbo),
    "flux2-klein-4b": ("flux2-klein", ModelConfig.flux2_klein_4b),
    "flux2-klein-9b": ("flux2-klein", ModelConfig.flux2_klein_9b),
    "flux2-klein-edit-4b": ("flux2-klein-edit", ModelConfig.flux2_klein_4b),
    "flux2-klein-edit-9b": ("flux2-klein-edit", ModelConfig.flux2_klein_9b),
}

# Public list of all supported image config names (single source of truth for CLI/server).
IMAGE_CONFIG_NAMES: tuple[str, ...] = tuple(_CONFIG_REGISTRY.keys())


# -----------------------------------------------------------------------------
# Base image model and generic backend wrapper
# -----------------------------------------------------------------------------


class BaseImageModel(ABC):
    """Abstract base for image generation models."""

    def __init__(self, model_path: str, config: ModelConfiguration) -> None:
        self.model_path = model_path
        self.config = config
        self._model: Any = None
        self._is_loaded = False
        self._load_model()

    def is_loaded(self) -> bool:
        return self._is_loaded

    @abstractmethod
    def _load_model(self) -> None:
        """Load the specific model implementation."""
        pass

    def _generate_image(self, prompt: str, seed: int = 42, **kwargs: Any) -> Image.Image:
        sig = inspect.signature(self._model.generate_image)
        valid = set(sig.parameters.keys())
        filtered = {k: v for k, v in kwargs.items() if k in valid}
        result = self._model.generate_image(
            prompt=prompt, seed=seed, **filtered
        )
        return result.image

    def __call__(self, prompt: str, seed: int = 42, **kwargs: Any) -> Image.Image:
        if not self._is_loaded:
            raise ModelLoadError("Model is not loaded. Cannot generate image.")
        if not (prompt and prompt.strip()):
            raise ModelGenerationError("Prompt cannot be empty.")
        if not isinstance(seed, int) or seed < 0:
            raise ModelGenerationError("Seed must be a non-negative integer.")
        try:
            result = self._generate_image(prompt, seed, **kwargs)
            if result is None:
                raise ModelGenerationError("Model returned None instead of an image.")
            logger.info("Image generated successfully")
            return result
        except ModelGenerationError:
            raise
        except Exception as e:
            logger.error(f"Error generating image: {e}")
            raise ModelGenerationError(f"Error generating image: {e}") from e


def _load_backed(
    self: BaseImageModel,
    backend_class: type,
    display_name: str,
) -> None:
    """Shared load logic: instantiate backend and set _model, _is_loaded."""
    logger.info(f"Loading {display_name} from {self.model_path}")
    if self.config.lora_paths:
        logger.info(f"LoRA adapters: {self.config.lora_paths}")
    try:
        self._model = backend_class(
            quantize=self.config.quantize,
            model_path=self.model_path,
            lora_paths=self.config.lora_paths,
            lora_scales=self.config.lora_scales,
            model_config=self.config.model_config,
        )
        self._is_loaded = True
        logger.info(f"{display_name} loaded successfully")
    except Exception as e:
        msg = f"Failed to load {display_name}: {e}"
        logger.error(msg)
        raise ModelLoadError(msg) from e


class BackedImageModel(BaseImageModel):
    """Generic wrapper: loads a given backend class and delegates generate."""

    def __init__(
        self,
        model_path: str,
        config: ModelConfiguration,
        backend_class: type,
        display_name: str,
    ) -> None:
        self._backend_class = backend_class
        self._display_name = display_name
        super().__init__(model_path, config)

    def _load_model(self) -> None:
        _load_backed(self, self._backend_class, self._display_name)


class Flux2KleinBackedModel(BackedImageModel):
    """Flux2 Klein: only supports guidance=1.0 and no negative_prompt."""

    def _generate_image(self, prompt: str, seed: int = 42, **kwargs: Any) -> Image.Image:
        kwargs.pop("negative_prompt", None)
        guidance = kwargs.get("guidance")
        if guidance and guidance != 1.0:
            logger.warning("guidance is not supported for Flux2 Klein. Setting to 1.0")
            kwargs["guidance"] = 1.0
        return super()._generate_image(prompt, seed, **kwargs)


# Config name -> (backend_class, wrapper_class, display_name)
_MODEL_REGISTRY: dict[str, tuple[type, type, str]] = {
    "flux-schnell": (Flux1, BackedImageModel, "Flux Schnell"),
    "flux-dev": (Flux1, BackedImageModel, "Flux Dev"),
    "flux-krea-dev": (Flux1, BackedImageModel, "Flux Krea Dev"),
    "flux-kontext-dev": (Flux1Kontext, BackedImageModel, "Flux Kontext"),
    "qwen-image": (QwenImage, BackedImageModel, "Qwen Image"),
    "qwen-image-edit": (QwenImageEdit, BackedImageModel, "Qwen Image Edit"),
    "fibo": (FIBO, BackedImageModel, "FIBO"),
    "z-image-turbo": (ZImageTurbo, BackedImageModel, "Z Image Turbo"),
    "flux2-klein-4b": (Flux2Klein, Flux2KleinBackedModel, "Flux2 Klein"),
    "flux2-klein-9b": (Flux2Klein, Flux2KleinBackedModel, "Flux2 Klein"),
    "flux2-klein-edit-4b": (Flux2KleinEdit, Flux2KleinBackedModel, "Flux2 Klein Edit"),
    "flux2-klein-edit-9b": (Flux2KleinEdit, Flux2KleinBackedModel, "Flux2 Klein Edit"),
}


# -----------------------------------------------------------------------------
# Public factory
# -----------------------------------------------------------------------------


class ImageGenerationModel:
    """Factory for creating and calling image generation models by config name."""

    def __init__(
        self,
        model_path: str,
        config_name: str,
        quantize: Optional[int] = None,
        lora_paths: Optional[list[str]] = None,
        lora_scales: Optional[list[float]] = None,
    ) -> None:
        self.model_path = model_path
        self.config_name = config_name
        self.quantize = quantize
        self.lora_paths = lora_paths
        self.lora_scales = lora_scales
        self.config = ModelConfiguration.from_name(
            config_name,
            quantize=quantize,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
        )
        backend_class, wrapper_class, display_name = _MODEL_REGISTRY[config_name]
        try:
            self.model_instance = wrapper_class(
                model_path,
                self.config,
                backend_class=backend_class,
                display_name=display_name,
            )
        except Exception as e:
            logger.error(f"Failed to initialize ImageGenerationModel: {e}")
            raise ModelLoadError(f"Failed to initialize ImageGenerationModel: {e}") from e
        logger.info(f"ImageGenerationModel initialized with config: {config_name}")
        if lora_paths:
            logger.info(f"LoRA adapters: {lora_paths}")

    def __call__(self, prompt: str, seed: int = 42, **kwargs: Any) -> Image.Image:
        return self.model_instance(prompt, seed=seed, **kwargs)

    def get_model_info(self) -> dict[str, Any]:
        return {
            "model_path": self.model_path,
            "type": self.config.model_type,
            "model_class": self.model_instance.__class__.__name__,
        }

    def get_current_config(self) -> dict[str, Any]:
        return {
            "config_name": self.config_name,
            "model_path": self.model_path,
            "quantize": self.quantize,
            "type": self.config.model_type,
            "is_loaded": self.model_instance.is_loaded(),
            "lora_paths": self.config.lora_paths,
            "lora_scales": self.config.lora_scales,
        }

    def is_loaded(self) -> bool:
        return self.model_instance.is_loaded()


if __name__ == "__main__":
    model = ImageGenerationModel(
        model_path="black-forest-labs/FLUX.2-klein-4B",
        config_name="flux2-klein-4b",
    )
    image_path = "examples/images/china.png"
    prompt = "Make it like 1800s"
    image = model(
        prompt,
        num_inference_steps=4,
        width=1024,
        height=1024,
        guidance=1.0,
        image_paths=[image_path],
    )
    image.save("examples/result.png")