"""Tests for multimodal audio request preparation."""

from __future__ import annotations

import base64
import importlib
import sys
import types
from typing import Any

import pytest


def _load_mlx_vlm_model_module(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Import the VLM model wrapper with lightweight MLX and mlx-vlm stubs."""
    fake_mx_module = types.ModuleType("mlx.core")
    fake_mx_module.array = lambda value: value
    fake_mx_module.random = types.SimpleNamespace(seed=lambda seed: None)

    fake_mlx_module = types.ModuleType("mlx")
    fake_mlx_module.core = fake_mx_module

    fake_mlx_vlm_module = types.ModuleType("mlx_vlm")
    fake_mlx_vlm_module.load = lambda *args, **kwargs: (None, None)
    fake_mlx_vlm_module.stream_generate = lambda *args, **kwargs: iter(())

    fake_utils_module = types.ModuleType("mlx_vlm.utils")

    def fake_process_inputs_with_fallback(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {
            "input_ids": [1, 2, 3],
            "attention_mask": [1, 1, 1],
            "audio_seen": kwargs["audio"],
            "videos_seen": kwargs["videos"],
        }

    fake_utils_module.process_inputs_with_fallback = fake_process_inputs_with_fallback

    fake_video_module = types.ModuleType("mlx_vlm.video_generate")
    fake_video_module.process_vision_info = lambda messages: (None, None)

    fake_outlines_module = types.ModuleType("outlines.processors")
    fake_outlines_module.JSONLogitsProcessor = object

    fake_outlines_tokenizer_module = types.ModuleType("app.utils.outlines_transformer_tokenizer")
    fake_outlines_tokenizer_module.OutlinesTransformerTokenizer = object

    monkeypatch.setitem(sys.modules, "mlx", fake_mlx_module)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mx_module)
    monkeypatch.setitem(sys.modules, "mlx_vlm", fake_mlx_vlm_module)
    monkeypatch.setitem(sys.modules, "mlx_vlm.utils", fake_utils_module)
    monkeypatch.setitem(sys.modules, "mlx_vlm.video_generate", fake_video_module)
    monkeypatch.setitem(sys.modules, "outlines.processors", fake_outlines_module)
    monkeypatch.setitem(
        sys.modules, "app.utils.outlines_transformer_tokenizer", fake_outlines_tokenizer_module
    )
    monkeypatch.delitem(sys.modules, "app.models.mlx_vlm", raising=False)

    return importlib.import_module("app.models.mlx_vlm")


def _load_audio_processor(monkeypatch: pytest.MonkeyPatch) -> type:
    """Import AudioProcessor without importing MLX-dependent app.core exports."""
    fake_mlx_lm_module = types.ModuleType("mlx_lm")
    fake_generate_module = types.ModuleType("mlx_lm.generate")
    fake_generate_module.BatchGenerator = object
    fake_generate_module.SequenceStateMachine = object

    monkeypatch.setitem(sys.modules, "mlx_lm", fake_mlx_lm_module)
    monkeypatch.setitem(sys.modules, "mlx_lm.generate", fake_generate_module)
    monkeypatch.delitem(sys.modules, "app.core", raising=False)
    monkeypatch.delitem(sys.modules, "app.core.audio_processor", raising=False)

    return importlib.import_module("app.core.audio_processor").AudioProcessor


@pytest.mark.asyncio
async def test_audio_processor_accepts_openai_raw_base64_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenAI input_audio.data raw base64 should be decoded as audio, not fetched as a URL."""
    AudioProcessor = _load_audio_processor(monkeypatch)
    audio_data = b"RIFF" + b"\x00" * 120 + b"WAVE"
    encoded = base64.b64encode(audio_data).decode("ascii")

    processor = AudioProcessor()
    try:
        path = await processor.process_audio_url(encoded, audio_format="wav")

        assert path.endswith(".wav")
        with open(path, "rb") as audio_file:
            assert audio_file.read() == audio_data
    finally:
        await processor.cleanup()


def test_vlm_create_inputs_forwards_audio_and_normalizes_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The VLM wrapper should pass audio files and expose attention_mask as mask."""
    module = _load_mlx_vlm_model_module(monkeypatch)
    model = object.__new__(module.MLX_VLM)
    model.processor = object()

    inputs = model.create_inputs(
        "prompt",
        images=["image.png"],
        videos=["video.mp4"],
        audios=["audio.wav"],
    )

    assert inputs["audio_seen"] == ["audio.wav"]
    assert inputs["videos_seen"] == ["video.mp4"]
    assert inputs["mask"] == [1, 1, 1]
    assert "attention_mask" not in inputs


def test_vlm_create_input_prompt_uses_tokenizer_chat_template(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Processors without chat_template should fall back to tokenizer.chat_template."""
    module = _load_mlx_vlm_model_module(monkeypatch)

    class FakeProcessor:
        def __init__(self) -> None:
            self.tokenizer = types.SimpleNamespace(chat_template="tokenizer-template")

        def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
            if getattr(self, "chat_template", None) is None:
                raise ValueError("processor missing chat template")
            return f"{self.chat_template}:{messages[0]['content']}"

    model = object.__new__(module.MLX_VLM)
    model.processor = FakeProcessor()

    prompt = model.create_input_prompt(
        [{"role": "user", "content": "hello"}],
        {},
    )

    assert prompt == "tokenizer-template:hello"
