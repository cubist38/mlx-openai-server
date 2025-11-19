"""MLX Flux image generation model wrapper with configuration and error handling."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from loguru import logger
from mflux.config.model_config import ModelConfig
from mflux.flux.flux import Config, Flux1
from mflux.kontext.flux_kontext import Flux1Kontext
from PIL import Image


# Custom Exceptions
class FluxModelError(Exception):
    """Base exception for Flux model errors."""


class ModelLoadError(FluxModelError):
    """Raised when model loading fails."""


class ModelGenerationError(FluxModelError):
    """Raised when image generation fails."""


class InvalidConfigurationError(FluxModelError):
    """Raised when configuration is invalid."""


class ModelConfiguration:
    """Configuration class for Flux models."""

    def __init__(
        self,
        model_type: str,
        model_config: ModelConfig | None = None,
        quantize: int = 8,
        default_steps: int = 20,
        default_guidance: float = 2.5,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
    ):
        # Validate quantization level
        if quantize not in [4, 8, 16]:
            raise InvalidConfigurationError(
                f"Invalid quantization level: {quantize}. Must be 4, 8, or 16."
            )

        # Validate LoRA parameters: both must be provided together and have matching lengths
        if (lora_paths is None) != (lora_scales is None):
            raise InvalidConfigurationError(
                "Both lora_paths and lora_scales must be provided together."
            )
        if lora_paths and lora_scales and len(lora_paths) != len(lora_scales):
            raise InvalidConfigurationError(
                f"lora_paths and lora_scales must have the same length (got {len(lora_paths)} and {len(lora_scales)})"
            )

        self.model_type = model_type
        self.model_config = model_config
        self.quantize = quantize
        self.default_steps = default_steps
        self.default_guidance = default_guidance
        self.lora_paths = lora_paths
        self.lora_scales = lora_scales

    @classmethod
    def schnell(
        cls,
        quantize: int = 8,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
    ) -> "ModelConfiguration":
        """Create configuration for Flux Schnell model."""
        return cls(
            model_type="schnell",
            model_config=ModelConfig.schnell(),
            quantize=quantize,
            default_steps=4,
            default_guidance=0.0,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
        )

    @classmethod
    def dev(
        cls,
        quantize: int = 8,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
    ) -> "ModelConfiguration":
        """Create configuration for Flux Dev model."""
        return cls(
            model_type="dev",
            model_config=ModelConfig.dev(),
            quantize=quantize,
            default_steps=25,
            default_guidance=3.5,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
        )

    @classmethod
    def krea_dev(
        cls,
        quantize: int = 8,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
    ) -> "ModelConfiguration":
        """Create configuration for Flux Krea Dev model."""
        return cls(
            model_type="krea-dev",
            model_config=ModelConfig.dev(),
            quantize=quantize,
            default_steps=28,
            default_guidance=4.5,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
        )

    @classmethod
    def kontext(cls, quantize: int = 8) -> "ModelConfiguration":
        """Create configuration for Flux Kontext model."""
        return cls(
            model_type="kontext",
            model_config=None,  # Kontext doesn't use ModelConfig
            quantize=quantize,
            default_steps=28,
            default_guidance=2.5,
            lora_paths=None,  # Kontext doesn't support LoRA
            lora_scales=None,
        )


class BaseFluxModel(ABC):
    """Abstract base class for Flux models with common functionality."""

    def __init__(self, model_path: str, config: ModelConfiguration):
        self.model_path = model_path
        self.config = config
        self._model = None
        self._is_loaded = False

        # Validate model path
        if not self._validate_model_path():
            raise ModelLoadError(f"Invalid model path: {model_path}")

        self._load_model()

    def _validate_model_path(self) -> bool:
        """Validate that the model path exists or is a valid model name."""
        # Check if it's a file path
        if Path(self.model_path).exists():
            return True

        # Check if it's a valid model name (for downloading)
        valid_model_names = ["flux-dev", "flux-schnell", "flux-kontext-dev"]
        return self.model_path in valid_model_names

    @abstractmethod
    def _load_model(self):
        """Load the specific model implementation."""

    @abstractmethod
    def _generate_image(self, prompt: str, seed: int, config: Config) -> Image.Image:
        """Generate image using the specific model implementation."""

    def __call__(self, prompt: str, seed: int = 42, **kwargs) -> Image.Image:
        """Generate an image from a text prompt."""
        if not self._is_loaded:
            raise ModelLoadError("Model is not loaded. Cannot generate image.")

        # Validate inputs
        if not prompt or not prompt.strip():
            raise ModelGenerationError("Prompt cannot be empty.")

        if not isinstance(seed, int) or seed < 0:
            raise ModelGenerationError("Seed must be a non-negative integer.")

        # Merge default config values with provided kwargs
        try:
            generation_config = self._prepare_config(**kwargs)
        except Exception as e:
            raise ModelGenerationError(f"Failed to prepare configuration: {e}") from e

        logger.info(
            f"Generating image with prompt: '{prompt[:50]}...' "
            f"(steps: {generation_config.num_inference_steps}, seed: {seed})"
        )

        try:
            result = self._generate_image(prompt, seed, generation_config)
            if result is None:
                raise ModelGenerationError("Model returned None instead of an image.")

            logger.info("Image generated successfully")
        except Exception as e:
            error_msg = f"Error generating image: {e}"
            logger.error(error_msg)
            raise ModelGenerationError(error_msg) from e
        else:
            return result

    def _prepare_config(self, **kwargs) -> Config:
        """Prepare configuration for image generation."""
        # Validate dimensions
        width = kwargs.get("width", 1024)
        height = kwargs.get("height", 1024)

        if not isinstance(width, int) or width <= 0:
            raise ModelGenerationError("Width must be a positive integer.")
        if not isinstance(height, int) or height <= 0:
            raise ModelGenerationError("Height must be a positive integer.")

        # Validate steps
        steps = kwargs.get("num_inference_steps", self.config.default_steps)
        if not isinstance(steps, int) or steps <= 0:
            raise ModelGenerationError("Number of inference steps must be a positive integer.")

        # Validate guidance
        guidance = kwargs.get("guidance", self.config.default_guidance)
        if not isinstance(guidance, (int, float)) or guidance < 0:
            raise ModelGenerationError("Guidance must be a non-negative number.")

        config_params = {
            "num_inference_steps": steps,
            "guidance": guidance,
            "width": width,
            "height": height,
        }

        # Add image_path if provided (for inpainting/editing)
        if "image_path" in kwargs:
            image_path = kwargs["image_path"]
            if not Path(image_path).exists():
                raise ModelGenerationError(f"Image path does not exist: {image_path}")
            config_params["image_path"] = image_path

        return Config(**config_params)


class FluxStandardModel(BaseFluxModel):
    """Standard Flux model implementation for Dev and Schnell variants."""

    def _load_model(self):
        """Load the standard Flux model."""
        try:
            logger.info(f"Loading {self.config.model_type} model from {self.model_path}")

            # Prepare lora parameters
            lora_paths = self.config.lora_paths
            lora_scales = self.config.lora_scales

            # Log LoRA information if provided
            if lora_paths:
                logger.info(f"Using LoRA adapters: {lora_paths}")
                if lora_scales:
                    logger.info(f"LoRA scales: {lora_scales}")

            self._model = Flux1(
                model_config=self.config.model_config,
                local_path=self.model_path,
                quantize=self.config.quantize,
                lora_paths=lora_paths,
                lora_scales=lora_scales,
            )
            self._is_loaded = True
            logger.info(f"{self.config.model_type} model loaded successfully")
        except Exception as e:
            error_msg = f"Failed to load {self.config.model_type} model: {e}"
            logger.error(error_msg)
            raise ModelLoadError(error_msg) from e

    def _generate_image(self, prompt: str, seed: int, config: Config) -> Image.Image:
        """Generate image using standard Flux model."""
        try:
            result = self._model.generate_image(
                config=config,
                prompt=prompt,
                seed=seed,
            )
        except Exception as e:
            raise ModelGenerationError(f"Standard model generation failed: {e}") from e
        else:
            return result.image


class FluxKontextModel(BaseFluxModel):
    """Flux Kontext model implementation."""

    def _load_model(self):
        """Load the Flux Kontext model."""
        try:
            logger.info(f"Loading Kontext model from {self.model_path}")
            self._model = Flux1Kontext(quantize=self.config.quantize, local_path=self.model_path)
            self._is_loaded = True
            logger.info("Kontext model loaded successfully")
        except Exception as e:
            error_msg = f"Failed to load Kontext model: {e}"
            logger.error(error_msg)
            raise ModelLoadError(error_msg) from e

    def _generate_image(self, prompt: str, seed: int, config: Config) -> Image.Image:
        """Generate image using Flux Kontext model."""
        try:
            result = self._model.generate_image(
                config=config,
                prompt=prompt,
                seed=seed,
            )
        except Exception as e:
            raise ModelGenerationError(f"Kontext model generation failed: {e}") from e
        else:
            return result.image


class FluxModel:
    """Factory class for creating and managing Flux models."""

    _MODEL_CONFIGS = {
        "flux-schnell": ModelConfiguration.schnell,
        "flux-dev": ModelConfiguration.dev,
        "flux-krea-dev": ModelConfiguration.krea_dev,
        "flux-kontext-dev": ModelConfiguration.kontext,
    }

    _MODEL_CLASSES = {
        "flux-schnell": FluxStandardModel,
        "flux-dev": FluxStandardModel,
        "flux-krea-dev": FluxStandardModel,
        "flux-kontext-dev": FluxKontextModel,
    }

    def __init__(
        self,
        model_path: str,
        config_name: str,
        quantize: int = 8,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
    ):
        self.config_name = config_name
        self.model_path = model_path
        self.quantize = quantize
        self.lora_paths = lora_paths
        self.lora_scales = lora_scales

        # Validate configuration
        if config_name not in self._MODEL_CONFIGS:
            available_configs = ", ".join(self._MODEL_CONFIGS.keys())
            raise InvalidConfigurationError(
                f"Invalid config name: {config_name}. Available options: {available_configs}"
            )

        # Validate LoRA parameters for kontext model
        if config_name == "flux-kontext-dev" and (
            lora_paths is not None or lora_scales is not None
        ):
            raise InvalidConfigurationError("Flux Kontext model does not support LoRA adapters")

        try:
            # Create model configuration
            config_factory = self._MODEL_CONFIGS[config_name]
            if config_name == "flux-kontext-dev":
                self.config = config_factory(quantize=quantize)
            else:
                self.config = config_factory(
                    quantize=quantize, lora_paths=lora_paths, lora_scales=lora_scales
                )

            # Create model instance
            model_class = self._MODEL_CLASSES[config_name]
            self.flux = model_class(model_path, self.config)

            logger.info(f"FluxModel initialized successfully with config: {config_name}")
            if lora_paths:
                logger.info(f"LoRA adapters: {lora_paths}")

        except Exception as e:
            error_msg = f"Failed to initialize FluxModel: {e}"
            logger.error(error_msg)
            raise ModelLoadError(error_msg) from e

    def __call__(self, prompt: str, seed: int = 42, **kwargs) -> Image.Image:
        """Generate an image using the configured model."""
        return self.flux(prompt, seed, **kwargs)

    @classmethod
    def get_available_configs(cls) -> list[str]:
        """Get list of available model configurations."""
        return list(cls._MODEL_CONFIGS.keys())

    @classmethod
    def get_model_info(cls, config_name: str) -> dict[str, Any]:
        """Get information about a specific model configuration."""
        if config_name not in cls._MODEL_CONFIGS:
            raise InvalidConfigurationError(f"Unknown config: {config_name}")

        config = cls._MODEL_CONFIGS[config_name]()
        return {
            "name": config_name,
            "type": config.model_type,
            "default_steps": config.default_steps,
            "default_guidance": config.default_guidance,
            "model_class": cls._MODEL_CLASSES[config_name].__name__,
        }

    def get_current_config(self) -> dict[str, Any]:
        """Get current model configuration information."""
        return {
            "config_name": self.config_name,
            "model_path": self.model_path,
            "quantize": self.quantize,
            "type": self.config.model_type,
            "default_steps": self.config.default_steps,
            "default_guidance": self.config.default_guidance,
            "is_loaded": self.flux._is_loaded if hasattr(self.flux, "_is_loaded") else False,  # noqa: SLF001
            "lora_paths": self.config.lora_paths,
            "lora_scales": self.config.lora_scales,
        }

    def is_loaded(self) -> bool:
        """Check if the model is loaded and ready for inference."""
        return hasattr(self.flux, "_is_loaded") and self.flux._is_loaded  # noqa: SLF001
