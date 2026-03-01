"""Integration-style streaming regression for Step 3.5 handler parsing."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import importlib
from pathlib import Path
import sys
import types
import unittest


def _load_mlx_lm_handler_class() -> type:
    """Import ``MLXLMHandler`` with lightweight stubs for MLX-heavy modules."""
    repo_root = Path(__file__).resolve().parents[1]

    fake_handler_package = types.ModuleType("app.handler")
    fake_handler_package.__path__ = [str(repo_root / "app" / "handler")]

    fake_model_module = types.ModuleType("app.models.mlx_lm")
    fake_model_module.MLX_LM = object

    fake_prompt_cache_module = types.ModuleType("app.utils.prompt_cache")
    fake_prompt_cache_module.LRUPromptCache = object

    module_names = [
        "app.handler",
        "app.models.mlx_lm",
        "app.utils.prompt_cache",
        "app.handler.mlx_lm",
    ]
    original_modules: dict[str, types.ModuleType | None] = {
        name: sys.modules.get(name) for name in module_names
    }

    try:
        sys.modules["app.handler"] = fake_handler_package
        sys.modules["app.models.mlx_lm"] = fake_model_module
        sys.modules["app.utils.prompt_cache"] = fake_prompt_cache_module
        sys.modules.pop("app.handler.mlx_lm", None)

        module = importlib.import_module("app.handler.mlx_lm")
        return module.MLXLMHandler
    finally:
        sys.modules.pop("app.handler.mlx_lm", None)
        for name, module in original_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


@dataclass
class _FakeStreamChunk:
    """Minimal stream chunk object consumed by handler streaming loop."""

    text: str
    token: int
    prompt_tokens: int = 11
    generation_tokens: int = 7
    generation_tps: float = 1.0
    peak_memory: float = 0.0


class _FakeModel:
    """Tiny model stub used by ``generate_text_stream``."""

    def create_input_prompt(self, messages: list[dict[str, str]], kwargs: dict[str, object]) -> str:
        return "prompt"

    def encode_prompt(self, prompt: str) -> list[int]:
        return [1, 2, 3]

    def create_prompt_cache(self) -> dict[str, bool]:
        return {"cache": True}


class _FakePromptCache:
    """Prompt cache stub matching the handler interface."""

    def __init__(self) -> None:
        self.inserted_keys: list[list[int]] = []

    def fetch_nearest_cache(self, input_ids: list[int]) -> tuple[None, list[int]]:
        return None, input_ids

    def insert_cache(self, cache_key: list[int], cache: object) -> None:
        self.inserted_keys.append(cache_key)


class _FakeInferenceWorker:
    """Inference worker stub that returns a fixed async stream."""

    def __init__(self, chunks: list[_FakeStreamChunk]) -> None:
        self._chunks = chunks

    def submit_stream(self, *args: object, **kwargs: object):
        async def _gen():
            for chunk in self._chunks:
                yield chunk

        return _gen()


class Step35StreamHandlerIntegrationTests(unittest.TestCase):
    """Exercise Step 3.5 streaming parser composition through handler loop."""

    def test_step35_tool_inside_thinking_reenters_reasoning_after_tool_parse(self) -> None:
        """Tool chunks inside thinking should parse as tool calls, not leaked text."""
        handler_cls = _load_mlx_lm_handler_class()
        handler = object.__new__(handler_cls)

        handler.debug = False
        handler.message_converter = None
        handler.enable_auto_tool_choice = False
        handler.reasoning_parser_name = "step_35"
        handler.tool_parser_name = "step_35"
        handler.model = _FakeModel()
        handler.prompt_cache = _FakePromptCache()
        handler.inference_worker = _FakeInferenceWorker(
            [
                _FakeStreamChunk("<thinking>before ", token=101),
                _FakeStreamChunk("<tool_call>\n", token=102),
                _FakeStreamChunk(
                    "<function=read_file><parameter=path>\"/tmp/a.txt\"</parameter></function>\n",
                    token=103,
                ),
                _FakeStreamChunk("</tool_call> after </thinking> final", token=104),
            ]
        )

        async def _fake_prepare_text_request(
            self, request: object
        ) -> tuple[list[dict[str, str]], dict[str, object]]:
            return [{"role": "user", "content": "hello"}], {"chat_template_kwargs": {}}

        handler._prepare_text_request = types.MethodType(_fake_prepare_text_request, handler)

        async def _collect() -> list[str | dict[str, object]]:
            outputs: list[str | dict[str, object]] = []
            async for item in handler.generate_text_stream(request=object()):
                outputs.append(item)
            return outputs

        outputs = asyncio.run(_collect())

        reasoning_text = "".join(
            item["reasoning_content"]
            for item in outputs
            if isinstance(item, dict) and isinstance(item.get("reasoning_content"), str)
        )
        assert reasoning_text == "before  after "

        emitted_tool_calls = [
            item
            for item in outputs
            if isinstance(item, dict) and isinstance(item.get("name"), str)
        ]
        assert len(emitted_tool_calls) == 1
        assert emitted_tool_calls[0]["name"] == "read_file"
        assert emitted_tool_calls[0]["arguments"] == '{"path": "/tmp/a.txt"}'

        plain_content = "".join(item for item in outputs if isinstance(item, str))
        assert plain_content == " final"
        assert "<tool_call>" not in plain_content
        assert "</thinking>" not in plain_content

        usage_chunks = [
            item for item in outputs if isinstance(item, dict) and "__usage__" in item
        ]
        assert len(usage_chunks) == 1


if __name__ == "__main__":
    unittest.main()
