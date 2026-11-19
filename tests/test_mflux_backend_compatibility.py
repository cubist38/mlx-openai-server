"""Compatibility tests for the mandatory image-generation backend."""

from __future__ import annotations

import importlib
import sys
import types
from typing import Any

from PIL import Image
import pytest


class _FakeModelConfig:
    """Provide the model factories consumed while building the registry."""

    @staticmethod
    def schnell() -> object:
        return object()

    @staticmethod
    def dev() -> object:
        return object()

    @staticmethod
    def krea_dev() -> object:
        return object()

    @staticmethod
    def dev_kontext() -> object:
        return object()

    @staticmethod
    def qwen_image() -> object:
        return object()

    @staticmethod
    def qwen_image_edit() -> object:
        return object()

    @staticmethod
    def fibo() -> object:
        return object()

    @staticmethod
    def z_image_turbo() -> object:
        return object()

    @staticmethod
    def flux2_klein_4b() -> object:
        return object()

    @staticmethod
    def flux2_klein_9b() -> object:
        return object()


class _GeneratedImageBackend:
    """Backend double matching models that return a generated-image wrapper."""

    def __init__(
        self,
        quantize: int | None = None,
        model_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        model_config: object | None = None,
    ) -> None:
        self.arguments = {
            "quantize": quantize,
            "model_path": model_path,
            "lora_paths": lora_paths,
            "lora_scales": lora_scales,
            "model_config": model_config,
        }

    def generate_image(self, prompt: str, seed: int, width: int = 8) -> Any:
        del prompt, seed
        return types.SimpleNamespace(image=Image.new("RGB", (width, width)))


class _FiboBackend:
    """Match mflux 0.19 FIBO, whose constructor has no LoRA arguments."""

    def __init__(
        self,
        quantize: int | None = None,
        model_path: str | None = None,
        model_config: object | None = None,
    ) -> None:
        self.arguments = {
            "quantize": quantize,
            "model_path": model_path,
            "model_config": model_config,
        }

    def generate_image(self, prompt: str, seed: int) -> Any:
        del prompt, seed
        return types.SimpleNamespace(image=Image.new("RGB", (8, 8)))


def _install_module(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    **attributes: object,
) -> None:
    """Install a lightweight module under ``name`` for an isolated import."""

    module = types.ModuleType(name)
    for attribute_name, value in attributes.items():
        setattr(module, attribute_name, value)
    monkeypatch.setitem(sys.modules, name, module)


@pytest.fixture
def image_backend_module(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Import the image wrapper against doubles with the current public API."""

    module_classes = {
        "mflux.models.fibo.variants.txt2img.fibo": ("FIBO", _FiboBackend),
        "mflux.models.flux.variants.kontext.flux_kontext": (
            "Flux1Kontext",
            _GeneratedImageBackend,
        ),
        "mflux.models.flux.variants.txt2img.flux": ("Flux1", _GeneratedImageBackend),
        "mflux.models.flux2.variants.edit.flux2_klein_edit": (
            "Flux2KleinEdit",
            _GeneratedImageBackend,
        ),
        "mflux.models.flux2.variants.txt2img.flux2_klein": (
            "Flux2Klein",
            _GeneratedImageBackend,
        ),
        "mflux.models.qwen.variants.edit.qwen_image_edit": (
            "QwenImageEdit",
            _GeneratedImageBackend,
        ),
        "mflux.models.qwen.variants.txt2img.qwen_image": (
            "QwenImage",
            _GeneratedImageBackend,
        ),
        "mflux.models.z_image.variants": ("ZImageTurbo", _GeneratedImageBackend),
    }

    package_names = {
        "mflux",
        "mflux.models",
        "mflux.models.common",
        *(
            package_name
            for module_name in module_classes
            for package_name in (
                ".".join(module_name.split(".")[:index])
                for index in range(3, len(module_name.split(".")))
            )
        ),
    }
    for package_name in package_names:
        _install_module(monkeypatch, package_name)
    _install_module(
        monkeypatch,
        "mflux.models.common.config",
        ModelConfig=_FakeModelConfig,
    )
    for module_name, (class_name, backend_class) in module_classes.items():
        _install_module(monkeypatch, module_name, **{class_name: backend_class})

    monkeypatch.delitem(sys.modules, "app.models.mflux", raising=False)
    module = importlib.import_module("app.models.mflux")
    yield module
    sys.modules.pop("app.models.mflux", None)


@pytest.mark.parametrize(
    "config_name",
    [
        "flux-schnell",
        "flux-dev",
        "flux-krea-dev",
        "flux-kontext-dev",
        "qwen-image",
        "qwen-image-edit",
        "fibo",
        "z-image-turbo",
        "flux2-klein-4b",
        "flux2-klein-9b",
        "flux2-klein-edit-4b",
        "flux2-klein-edit-9b",
    ],
)
def test_every_registered_image_backend_can_initialize(
    image_backend_module: Any,
    config_name: str,
) -> None:
    """Every advertised image configuration should construct successfully."""

    model = image_backend_module.ImageGenerationModel(
        model_path="local-model",
        config_name=config_name,
        quantize=4,
    )

    assert model.is_loaded()


def test_constructor_arguments_are_filtered_for_fibo(image_backend_module: Any) -> None:
    """FIBO should load even though its constructor does not accept LoRA keys."""

    model = image_backend_module.ImageGenerationModel(
        model_path="local-fibo",
        config_name="fibo",
        quantize=4,
    )

    assert model.model_instance._model.arguments["model_path"] == "local-fibo"


def test_fibo_rejects_unsupported_lora_configuration(image_backend_module: Any) -> None:
    """A requested but unsupported LoRA option must fail instead of being ignored."""

    with pytest.raises(image_backend_module.ModelLoadError, match="does not support"):
        image_backend_module.ImageGenerationModel(
            model_path="local-fibo",
            config_name="fibo",
            lora_paths=["adapter.safetensors"],
            lora_scales=[1.0],
        )


def test_direct_pil_results_are_supported(image_backend_module: Any) -> None:
    """Backends may return either a generated-image wrapper or a PIL image."""

    class DirectImageBackend:
        def __init__(
            self,
            quantize: int | None = None,
            model_path: str | None = None,
            model_config: object | None = None,
        ) -> None:
            del quantize, model_path, model_config

        def generate_image(self, prompt: str, seed: int, width: int = 12) -> Image.Image:
            del prompt, seed
            return Image.new("RGB", (width, width))

    config = image_backend_module.ModelConfiguration(
        model_type="direct",
        model_config=object(),
    )
    model = image_backend_module.BackedImageModel(
        model_path="local-image",
        config=config,
        backend_class=DirectImageBackend,
        display_name="Direct image",
    )

    assert model("test", width=12).size == (12, 12)
