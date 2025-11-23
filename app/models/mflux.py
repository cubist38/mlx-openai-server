"""
MFlux model implementations for image generation.

This module provides classes for loading and using MFlux models for image generation,
including Flux1 and Flux1Kontext variants with different configurations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
import inspect
from pathlib import Path
from typing import Any, ClassVar

from loguru import logger
from mflux.models.common.config import ModelConfig
from mflux.models.fibo.variants.txt2img.fibo import FIBO
from mflux.models.flux.variants.kontext.flux_kontext import Flux1Kontext
from mflux.models.flux.variants.txt2img.flux import Flux1
from mflux.models.qwen.variants.edit.qwen_image_edit import QwenImageEdit
from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage
from mflux.models.z_image.variants.turbo import ZImageTurbo
from PIL import Image

from .errors import InvalidConfigurationError, ModelGenerationError, ModelLoadError


class ModelConfiguration:
    """Configuration class for Flux models."""

    def __init__(
        self,
        model_type: str,
        model_config: ModelConfig,
        quantize: int = 8,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
    ) -> None:
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
        self.lora_paths = lora_paths
        self.lora_scales = lora_scales

    @classmethod
    def schnell(
        cls,
        quantize: int = 8,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
    ) -> ModelConfiguration:
        """Create a configuration for the Flux Schnell model.

        Parameters
        ----------
        quantize : int, optional
            Quantization level (4, 8, or 16). Default is 8.
        lora_paths : list[str] | None, optional
            Paths to LoRA adapters to apply.
        lora_scales : list[float] | None, optional
            LoRA scales corresponding to each path.

        Returns
        -------
        ModelConfiguration
            Configured ModelConfiguration instance for the Schnell variant.
        """
        return cls(
            model_type="schnell",
            model_config=ModelConfig.schnell(),
            quantize=quantize,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
        )

    @classmethod
    def dev(
        cls,
        quantize: int = 8,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
    ) -> ModelConfiguration:
        """Create a configuration for the Flux Dev model.

        Parameters
        ----------
        quantize : int, optional
            Quantization level (4, 8, or 16). Default is 8.
        lora_paths : list[str] | None, optional
            Paths to LoRA adapters to apply.
        lora_scales : list[float] | None, optional
            LoRA scales corresponding to each path.

        Returns
        -------
        ModelConfiguration
            Configured ModelConfiguration instance for the Dev variant.
        """
        return cls(
            model_type="dev",
            model_config=ModelConfig.dev(),
            quantize=quantize,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
        )

    @classmethod
    def krea_dev(
        cls,
        quantize: int = 8,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
    ) -> ModelConfiguration:
        """Create a configuration for the Flux Krea Dev model.

        Parameters
        ----------
        quantize : int, optional
            Quantization level (4, 8, or 16). Default is 8.
        lora_paths : list[str] | None, optional
            Paths to LoRA adapters to apply.
        lora_scales : list[float] | None, optional
            LoRA scales corresponding to each path.

        Returns
        -------
        ModelConfiguration
            Configured ModelConfiguration instance for the Krea Dev variant.
        """
        return cls(
            model_type="krea-dev",
            model_config=ModelConfig.krea_dev(),
            quantize=quantize,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
        )

    @classmethod
    def kontext(
        cls,
        quantize: int = 8,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
    ) -> ModelConfiguration:
        """Create a configuration for the Flux Kontext model.

        Parameters
        ----------
        quantize : int, optional
            Quantization level (4, 8, or 16). Default is 8.
        lora_paths : list[str] | None, optional
            Paths to LoRA adapters to apply.
        lora_scales : list[float] | None, optional
            LoRA scales corresponding to each path.

        Returns
        -------
        ModelConfiguration
            Configured ModelConfiguration instance for the Kontext variant.
        """
        return cls(
            model_type="kontext",
            model_config=ModelConfig.kontext(),
            quantize=quantize,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
        )

    @classmethod
    def qwen_image(
        cls,
        quantize: int = 8,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
    ) -> ModelConfiguration:
        """Create a configuration for the Qwen Image model.

        Parameters
        ----------
        quantize : int, optional
            Quantization level (4, 8, or 16). Default is 8.
        lora_paths : list[str] | None, optional
            Paths to LoRA adapters to apply.
        lora_scales : list[float] | None, optional
            LoRA scales corresponding to each path.

        Returns
        -------
        ModelConfiguration
            Configured ModelConfiguration instance for the Qwen Image variant.
        """
        return cls(
            model_type="qwen-image",
            model_config=ModelConfig.qwen_image(),
            quantize=quantize,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
        )

    @classmethod
    def qwen_image_edit(
        cls,
        quantize: int = 8,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
    ) -> ModelConfiguration:
        """Create a configuration for the Qwen Image Edit model.

        Parameters
        ----------
        quantize : int, optional
            Quantization level (4, 8, or 16). Default is 8.
        lora_paths : list[str] | None, optional
            Paths to LoRA adapters to apply.
        lora_scales : list[float] | None, optional
            LoRA scales corresponding to each path.

        Returns
        -------
        ModelConfiguration
            Configured ModelConfiguration instance for the Qwen Image Edit variant.
        """
        return cls(
            model_type="qwen-image-edit",
            model_config=ModelConfig.qwen_image_edit(),
            quantize=quantize,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
        )

    @classmethod
    def fibo(
        cls,
        quantize: int = 8,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
    ) -> ModelConfiguration:
        """Create a configuration for the Fibo model.

        Parameters
        ----------
        quantize : int, optional
            Quantization level (4, 8, or 16). Default is 8.
        lora_paths : list[str] | None, optional
            Paths to LoRA adapters to apply.
        lora_scales : list[float] | None, optional
            LoRA scales corresponding to each path.

        Returns
        -------
        ModelConfiguration
            Configured ModelConfiguration instance for the Fibo variant.
        """
        return cls(
            model_type="fibo",
            model_config=ModelConfig.fibo(),
            quantize=quantize,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
        )

    @classmethod
    def z_image_turbo(
        cls,
        quantize: int = 8,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
    ) -> ModelConfiguration:
        """Create a configuration for the Z Image Turbo model.

        Parameters
        ----------
        quantize : int, optional
            Quantization level (4, 8, or 16). Default is 8.
        lora_paths : list[str] | None, optional
            Paths to LoRA adapters to apply.
        lora_scales : list[float] | None, optional
            LoRA scales corresponding to each path.

        Returns
        -------
        ModelConfiguration
            Configured ModelConfiguration instance for the Z Image Turbo variant.
        """
        return cls(
            model_type="z-image-turbo",
            model_config=ModelConfig.z_image_turbo(),
            quantize=quantize,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
        )


class BaseImageModel(ABC):
    """Abstract base class for image generation models with common functionality."""

    def __init__(self, model_path: str, config: ModelConfiguration) -> None:
        self.model_path = model_path
        self.config = config
        self._model = None
        self._is_loaded = False

        # Validate model path
        if not self._validate_model_path():
            raise ModelLoadError(f"Invalid model path: {model_path}")

        self._load_model()

    def _validate_model_path(self) -> bool:
        """Validate that the model path exists or is a valid model identifier.

        Returns
        -------
        bool
            True if the model path is valid (file exists or recognized identifier), False otherwise.
        """
        # Check if it's a file path
        if Path(self.model_path).exists():
            return True

        # Check if it's a valid model identifier (contains '/' for repo IDs or known model names)
        if "/" in self.model_path:
            return True

        # Check if it's a valid model name (for downloading)
        # This list should be kept in sync with ImageGenerationModel._MODEL_CONFIGS
        valid_model_names = [
            "flux-dev",
            "flux-schnell",
            "flux-krea-dev",
            "flux-kontext-dev",
            "qwen-image",
            "qwen-image-edit",
            "fibo",
            "z-image-turbo",
        ]

        return self.model_path in valid_model_names

    @abstractmethod
    def _load_model(self) -> None:
        """Load the specific model implementation for this subclass.

        Notes
        -----
        Subclasses must implement this method to initialize ``self._model``
        and set ``self._is_loaded`` to True when successful.
        """

    def _generate_image(self, prompt: str, seed: int = 42, **kwargs: Any) -> Image.Image:
        """Generate an image using the model-specific implementation.

        Parameters
        ----------
        prompt : str
            Text prompt describing the desired image.
        seed : int, optional
            Random seed to control deterministic output. Default is 42.
        **kwargs : Any
            Additional renderer-specific keyword arguments.

        Returns
        -------
        PIL.Image.Image
            Generated image.

        Raises
        ------
        ModelGenerationError
            If the underlying generation call fails.
        """
        assert self._model is not None, "Model not loaded"
        try:
            # Get the signature of the generate_image method to filter kwargs
            sig = inspect.signature(self._model.generate_image)
            valid_params = set(sig.parameters.keys())

            # Filter kwargs to only include parameters that the method actually accepts
            filtered_kwargs = {key: value for key, value in kwargs.items() if key in valid_params}

            # Build kwargs for generate_image call
            generate_kwargs = {"prompt": prompt, "seed": seed, **filtered_kwargs}

            result = self._model.generate_image(**generate_kwargs)
        except Exception as e:
            raise ModelGenerationError(f"{self.__class__.__name__} generation failed: {e}") from e
        else:
            return result.image

    def __call__(self, prompt: str, seed: int = 42, **kwargs: Any) -> Image.Image:
        """Generate an image from a text prompt.

        Parameters
        ----------
        prompt : str
            Text prompt describing the desired image.
        seed : int, optional
            Random seed to control deterministic output. Default is 42.
        **kwargs : Any
            Additional parameters forwarded to the model implementation.

        Returns
        -------
        PIL.Image.Image
            Generated image.

        Raises
        ------
        ModelLoadError
            If the model is not loaded.
        ModelGenerationError
            If generation fails or returns invalid results.
        """
        if not self._is_loaded:
            raise ModelLoadError("Model is not loaded. Cannot generate image.")

        # Validate inputs
        if not prompt or not prompt.strip():
            raise ModelGenerationError("Prompt cannot be empty.")

        if not isinstance(seed, int) or seed < 0:
            raise ModelGenerationError("Seed must be a non-negative integer.")

        try:
            result = self._generate_image(prompt, seed, **kwargs)
            if result is None:
                raise ModelGenerationError("Model returned None instead of an image.")

            logger.info("Image generated successfully")
        except Exception as e:
            error_msg = f"Error generating image. {type(e).__name__}: {e}"
            logger.error(error_msg)
            raise ModelGenerationError(error_msg) from e

        return result


class FluxStandardModel(BaseImageModel):
    """Standard Flux model implementation for Dev and Schnell variants."""

    def _load_model(self) -> None:
        """Load the standard Flux model and initialize it.

        Raises
        ------
        ModelLoadError
            If the model fails to load.
        """
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
                quantize=self.config.quantize,
                model_path=self.model_path,
                lora_paths=lora_paths,
                lora_scales=lora_scales,
                model_config=self.config.model_config,
            )
            self._is_loaded = True
            logger.info(f"{self.config.model_type} model loaded successfully")
        except Exception as e:
            error_msg = f"Failed to load {self.config.model_type} model. {type(e).__name__}: {e}"
            logger.error(error_msg)
            raise ModelLoadError(error_msg) from e


class FluxKontextModel(BaseImageModel):
    """Flux Kontext model implementation."""

    def _load_model(self) -> None:
        """Load the Flux Kontext model implementation.

        Raises
        ------
        ModelLoadError
            If the model fails to load.
        """
        try:
            logger.info(f"Loading Kontext model from {self.model_path}")
            self._model = Flux1Kontext(
                quantize=self.config.quantize,
                model_path=self.model_path,
                lora_paths=self.config.lora_paths,
                lora_scales=self.config.lora_scales,
            )
            self._is_loaded = True
            logger.info("Kontext model loaded successfully")
        except Exception as e:
            error_msg = f"Failed to load Kontext model. {type(e).__name__}: {e}"
            logger.error(error_msg)
            raise ModelLoadError(error_msg) from e


class QwenImageModel(BaseImageModel):
    """Qwen Image model implementation."""

    def _load_model(self) -> None:
        """Load the Qwen Image model implementation.

        Raises
        ------
        ModelLoadError
            If the model fails to load.
        """
        try:
            logger.info(f"Loading Qwen Image model from {self.model_path}")
            self._model = QwenImage(
                quantize=self.config.quantize,
                model_path=self.model_path,
                lora_paths=self.config.lora_paths,
                lora_scales=self.config.lora_scales,
            )
            self._is_loaded = True
            logger.info("Qwen Image model loaded successfully")
        except Exception as e:
            error_msg = f"Failed to load Qwen Image model: {e}"
            logger.error(error_msg)
            raise ModelLoadError(error_msg) from e


class QwenImageEditModel(BaseImageModel):
    """Qwen Image Edit model implementation."""

    def _load_model(self) -> None:
        """Load the Qwen Image Edit model implementation.

        Raises
        ------
        ModelLoadError
            If the model fails to load.
        """
        try:
            logger.info(f"Loading Qwen Image Edit model from {self.model_path}")
            self._model = QwenImageEdit(
                quantize=self.config.quantize,
                model_path=self.model_path,
                lora_paths=self.config.lora_paths,
                lora_scales=self.config.lora_scales,
            )
            self._is_loaded = True
            logger.info("Qwen Image Edit model loaded successfully")
        except Exception as e:
            error_msg = f"Failed to load Qwen Image Edit model: {e}"
            logger.error(error_msg)
            raise ModelLoadError(error_msg) from e


class ZImageTurboModel(BaseImageModel):
    """Z Image Turbo model implementation."""

    def _load_model(self) -> None:
        """Load the Z Image Turbo model implementation.

        Raises
        ------
        ModelLoadError
            If the model fails to load.
        """
        try:
            logger.info(f"Loading Z Image Turbo model from {self.model_path}")
            self._model = ZImageTurbo(
                quantize=self.config.quantize,
                model_path=self.model_path,
                lora_paths=self.config.lora_paths,
                lora_scales=self.config.lora_scales,
            )
            self._is_loaded = True
            logger.info("Z Image Turbo model loaded successfully")
        except Exception as e:
            error_msg = f"Failed to load Z Image Turbo model: {e}"
            logger.error(error_msg)
            raise ModelLoadError(error_msg) from e


class FIBOModel(BaseImageModel):
    """FIBO model implementation."""

    def _load_model(self) -> None:
        """Load the FIBO model implementation.

        Raises
        ------
        ModelLoadError
            If the model fails to load.
        """
        try:
            logger.info(f"Loading FIBO model from {self.model_path}")
            self._model = FIBO(
                quantize=self.config.quantize,
                model_path=self.model_path,
                lora_paths=self.config.lora_paths,
                lora_scales=self.config.lora_scales,
            )
            self._is_loaded = True
            logger.info("FIBO model loaded successfully")
        except Exception as e:
            error_msg = f"Failed to load FIBO model: {e}"
            logger.error(error_msg)
            raise ModelLoadError(error_msg) from e


class ImageGenerationModel:
    """Factory class for creating and managing image generation models."""

    _MODEL_CONFIGS: ClassVar[dict[str, Callable[..., ModelConfiguration]]] = {
        "flux-schnell": ModelConfiguration.schnell,
        "flux-dev": ModelConfiguration.dev,
        "flux-krea-dev": ModelConfiguration.krea_dev,
        "flux-kontext-dev": ModelConfiguration.kontext,
        "qwen-image": ModelConfiguration.qwen_image,
        "qwen-image-edit": ModelConfiguration.qwen_image_edit,
        "fibo": ModelConfiguration.fibo,
        "z-image-turbo": ModelConfiguration.z_image_turbo,
    }

    _MODEL_CLASSES: ClassVar[dict[str, type[BaseImageModel]]] = {
        "flux-schnell": FluxStandardModel,
        "flux-dev": FluxStandardModel,
        "flux-krea-dev": FluxStandardModel,
        "flux-kontext-dev": FluxKontextModel,
        "qwen-image": QwenImageModel,
        "qwen-image-edit": QwenImageEditModel,
        "fibo": FIBOModel,
        "z-image-turbo": ZImageTurboModel,
    }

    def __init__(
        self,
        model_path: str,
        config_name: str,
        quantize: int = 8,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
    ) -> None:
        self.model_path = model_path
        self.config_name = config_name
        self.quantize = quantize
        self.lora_paths = lora_paths
        self.lora_scales = lora_scales

        # Validate configuration
        if self.config_name not in self._MODEL_CONFIGS:
            available_configs = ", ".join(self._MODEL_CONFIGS.keys())
            raise InvalidConfigurationError(
                f"Invalid config name: {self.config_name}. Available options: {available_configs}"
            )

        try:
            # Create model configuration
            config_factory = self._MODEL_CONFIGS[config_name]

            self.config = config_factory(
                quantize=quantize, lora_paths=lora_paths, lora_scales=lora_scales
            )

            # Create model instance
            model_class = self._MODEL_CLASSES[config_name]
            self.model_instance = model_class(model_path, self.config)

            logger.info(f"ImageGenerationModel initialized successfully with config: {config_name}")
            if lora_paths:
                logger.info(f"LoRA adapters: {lora_paths}")

        except Exception as e:
            error_msg = f"Failed to initialize ImageGenerationModel: {e}"
            logger.error(error_msg)
            raise ModelLoadError(error_msg) from e

    def __call__(self, prompt: str, seed: int = 42, **kwargs: Any) -> Image.Image:
        """Generate an image using the configured model."""
        return self.model_instance(prompt, seed=seed, **kwargs)

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the configured model.

        Returns
        -------
        dict[str, Any]
            Dictionary with keys ``model_path``, ``type``, and ``model_class``.
        """
        return {
            "model_path": self.model_path,
            "type": self.config.model_type,
            "model_class": self.model_instance.__class__.__name__,
        }

    def get_current_config(self) -> dict[str, Any]:
        """Get current model configuration information.

        Returns
        -------
        dict[str, Any]
            Dictionary with current configuration details including ``config_name``,
            ``model_path``, ``quantize``, and plugin LoRA parameters.
        """
        return {
            "config_name": self.config_name,
            "model_path": self.model_path,
            "quantize": self.quantize,
            "type": self.config.model_type,
            "is_loaded": getattr(self.model_instance, "_is_loaded", False),
            "lora_paths": self.config.lora_paths,
            "lora_scales": self.config.lora_scales,
        }

    def is_loaded(self) -> bool:
        """Return whether the configured model instance is loaded and ready.

        Returns
        -------
        bool
            True if the model instance is loaded; False otherwise.
        """
        return getattr(self.model_instance, "_is_loaded", False)
