"""Integration-style streaming regression for mixed-think handler parsing."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
import importlib
import json
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


@dataclass
class _FakeNonStreamResponse:
    """Minimal non-stream response object consumed by ``generate_text_response``."""

    text: str
    tokens: list[int]
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

    def __init__(
        self,
        chunks: list[_FakeStreamChunk] | None = None,
        non_stream_response: _FakeNonStreamResponse | None = None,
    ) -> None:
        self._chunks = chunks or []
        self._non_stream_response = non_stream_response

    def submit_stream(self, *args: object, **kwargs: object) -> AsyncIterator[_FakeStreamChunk]:
        async def _gen() -> AsyncIterator[_FakeStreamChunk]:
            for chunk in self._chunks:
                yield chunk

        return _gen()

    async def submit(self, *args: object, **kwargs: object) -> _FakeNonStreamResponse:
        if self._non_stream_response is None:
            raise RuntimeError("non-stream response not configured in test stub")
        return self._non_stream_response


class MixedThinkToolHandoffStreamHandlerIntegrationTests(unittest.TestCase):
    """Exercise mixed-think streaming parser composition through handler loop."""

    def test_mixed_think_tool_handoff_inside_thinking_reenters_reasoning_after_tool_parse(
        self,
    ) -> None:
        """Tool chunks inside thinking should parse as tool calls, not leaked text."""
        handler_cls = _load_mlx_lm_handler_class()
        handler = object.__new__(handler_cls)

        handler.debug = False
        handler.message_converter = None
        handler.enable_auto_tool_choice = False
        handler.reasoning_parser_name = "mixed_think_tool_handoff"
        handler.tool_parser_name = "step_35"
        handler.model = _FakeModel()
        handler.prompt_cache = _FakePromptCache()
        handler.inference_worker = _FakeInferenceWorker(
            [
                _FakeStreamChunk("<thinking>before ", token=101),
                _FakeStreamChunk("<tool_call>\n", token=102),
                _FakeStreamChunk(
                    '<function=read_file><parameter=path>"/tmp/a.txt"</parameter></function>\n',
                    token=103,
                ),
                _FakeStreamChunk("</tool_call> after </thinking> final", token=104),
            ]
        )

        async def _fake_prepare_text_request(
            self: object, request: object
        ) -> tuple[list[dict[str, str]], dict[str, object]]:
            return [{"role": "user", "content": "hello"}], {"chat_template_kwargs": {}}

        handler._prepare_text_request = types.MethodType(_fake_prepare_text_request, handler)

        async def _collect() -> list[str | dict[str, object]]:
            return [item async for item in handler.generate_text_stream(request=object())]

        outputs = asyncio.run(_collect())

        reasoning_text = "".join(
            item["reasoning_content"]
            for item in outputs
            if isinstance(item, dict) and isinstance(item.get("reasoning_content"), str)
        )
        assert reasoning_text == "before  after "

        emitted_tool_calls = [
            item for item in outputs if isinstance(item, dict) and isinstance(item.get("name"), str)
        ]
        assert len(emitted_tool_calls) == 1
        assert emitted_tool_calls[0]["name"] == "read_file"
        assert json.loads(emitted_tool_calls[0]["arguments"]) == {"path": "/tmp/a.txt"}

        plain_content = "".join(item for item in outputs if isinstance(item, str))
        assert plain_content == " final"
        assert "<tool_call>" not in plain_content
        assert "</thinking>" not in plain_content

        usage_chunks = [item for item in outputs if isinstance(item, dict) and "__usage__" in item]
        assert len(usage_chunks) == 1

    def test_mixed_think_tool_handoff_terminal_tool_call_before_thinking_close(self) -> None:
        """Terminal in-thinking tool calls should still parse when they end right before ``</thinking>``."""
        handler_cls = _load_mlx_lm_handler_class()
        handler = object.__new__(handler_cls)

        handler.debug = False
        handler.message_converter = None
        handler.enable_auto_tool_choice = False
        handler.reasoning_parser_name = "mixed_think_tool_handoff"
        handler.tool_parser_name = "step_35"
        handler.model = _FakeModel()
        handler.prompt_cache = _FakePromptCache()
        handler.inference_worker = _FakeInferenceWorker(
            [
                _FakeStreamChunk("<thinking>tail-check:", token=201),
                _FakeStreamChunk("<tool_call>\n<function=read_file>\n", token=202),
                _FakeStreamChunk(
                    "<parameter=path>\napp/api/endpoints.py\n</parameter>\n",
                    token=203,
                ),
                _FakeStreamChunk(
                    "<parameter=start_line>\n692\n</parameter>\n"
                    "<parameter=end_line>\n790\n</parameter>\n",
                    token=204,
                ),
                _FakeStreamChunk("</function>\n</tool_call>\n</thinking>", token=205),
            ]
        )

        async def _fake_prepare_text_request(
            self: object, request: object
        ) -> tuple[list[dict[str, str]], dict[str, object]]:
            return [{"role": "user", "content": "hello"}], {"chat_template_kwargs": {}}

        handler._prepare_text_request = types.MethodType(_fake_prepare_text_request, handler)

        async def _collect() -> list[str | dict[str, object]]:
            return [item async for item in handler.generate_text_stream(request=object())]

        outputs = asyncio.run(_collect())

        reasoning_text = "".join(
            item["reasoning_content"]
            for item in outputs
            if isinstance(item, dict) and isinstance(item.get("reasoning_content"), str)
        )
        assert reasoning_text == "tail-check:\n"

        emitted_tool_calls = [
            item for item in outputs if isinstance(item, dict) and isinstance(item.get("name"), str)
        ]
        assert len(emitted_tool_calls) == 1
        assert emitted_tool_calls[0]["name"] == "read_file"
        assert json.loads(emitted_tool_calls[0]["arguments"]) == {
            "path": "app/api/endpoints.py",
            "start_line": 692,
            "end_line": 790,
        }

        plain_content = "".join(item for item in outputs if isinstance(item, str))
        assert plain_content == ""
        assert "<tool_call>" not in plain_content
        assert "</thinking>" not in plain_content

    def test_nonstream_step35_parses_tool_call_when_response_starts_with_stray_think_close(
        self,
    ) -> None:
        """Non-stream parsing should still extract tool calls after a leading stray ``</think>``."""
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
            non_stream_response=_FakeNonStreamResponse(
                text=(
                    "</think>\n"
                    "prefix text before tool.\n"
                    "<tool_call>\n"
                    "<function=read_file>\n"
                    "<parameter=path>\n"
                    "app/api/endpoints.py\n"
                    "</parameter>\n"
                    "<parameter=start_line>\n"
                    "692\n"
                    "</parameter>\n"
                    "<parameter=end_line>\n"
                    "790\n"
                    "</parameter>\n"
                    "</function>\n"
                    "</tool_call>\n"
                ),
                tokens=[1, 2, 3],
                prompt_tokens=100,
                generation_tokens=20,
            )
        )

        async def _fake_prepare_text_request(
            self: object, request: object
        ) -> tuple[list[dict[str, str]], dict[str, object]]:
            return [{"role": "user", "content": "hello"}], {"chat_template_kwargs": {}}

        handler._prepare_text_request = types.MethodType(_fake_prepare_text_request, handler)

        result = asyncio.run(handler.generate_text_response(request=object()))
        parsed = result["response"]

        assert isinstance(parsed, dict)
        assert isinstance(parsed.get("tool_calls"), list)
        assert len(parsed["tool_calls"]) == 1
        assert parsed["tool_calls"][0]["name"] == "read_file"
        assert json.loads(parsed["tool_calls"][0]["arguments"]) == {
            "path": "app/api/endpoints.py",
            "start_line": 692,
            "end_line": 790,
        }
        if isinstance(parsed.get("content"), str):
            assert "<tool_call>" not in parsed["content"]

    def test_nonstream_qwen3_moe_tool_fallback_does_not_leak_synthetic_reasoning_prefix(
        self,
    ) -> None:
        """Synthetic reasoning-open prefixes should not leak into content fallback."""
        handler_cls = _load_mlx_lm_handler_class()
        handler = object.__new__(handler_cls)

        handler.debug = False
        handler.message_converter = None
        handler.enable_auto_tool_choice = False
        handler.reasoning_parser_name = "qwen3_moe"
        handler.tool_parser_name = "hermes"
        handler.model = _FakeModel()
        handler.prompt_cache = _FakePromptCache()
        handler.inference_worker = _FakeInferenceWorker(
            non_stream_response=_FakeNonStreamResponse(
                text=(
                    "preface before tool.\n"
                    '<tool_call>{"name":"read_file","arguments":{"path":"app/handler/mlx_vlm.py"}}</tool_call>\n'
                ),
                tokens=[1, 2, 3],
                prompt_tokens=100,
                generation_tokens=20,
            )
        )

        async def _fake_prepare_text_request(
            self: object, request: object
        ) -> tuple[list[dict[str, str]], dict[str, object]]:
            return [{"role": "user", "content": "hello"}], {"chat_template_kwargs": {}}

        handler._prepare_text_request = types.MethodType(_fake_prepare_text_request, handler)

        result = asyncio.run(handler.generate_text_response(request=object()))
        parsed = result["response"]

        assert isinstance(parsed, dict)
        assert parsed["reasoning_content"] is None
        assert isinstance(parsed.get("tool_calls"), list)
        assert len(parsed["tool_calls"]) == 1
        assert parsed["tool_calls"][0]["name"] == "read_file"
        assert json.loads(parsed["tool_calls"][0]["arguments"]) == {
            "path": "app/handler/mlx_vlm.py"
        }
        assert isinstance(parsed.get("content"), str)
        visible_content = parsed["content"]
        assert "preface before tool." in visible_content
        assert "<think>" not in visible_content
        assert "<tool_call>" not in visible_content

    def test_stream_step35_parses_tool_call_when_output_starts_with_stray_think_close(self) -> None:
        """Streaming should parse tool calls even when output starts with stray ``</think>``."""
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
                _FakeStreamChunk("</think>\nprefix text before tool.\n", token=301),
                _FakeStreamChunk("<tool_call>\n<function=read_file>\n", token=302),
                _FakeStreamChunk(
                    "<parameter=path>\napp/api/endpoints.py\n</parameter>\n",
                    token=303,
                ),
                _FakeStreamChunk(
                    "<parameter=start_line>\n692\n</parameter>\n"
                    "<parameter=end_line>\n790\n</parameter>\n",
                    token=304,
                ),
                _FakeStreamChunk("</function>\n</tool_call>\n", token=305),
            ]
        )

        async def _fake_prepare_text_request(
            self: object, request: object
        ) -> tuple[list[dict[str, str]], dict[str, object]]:
            return [{"role": "user", "content": "hello"}], {"chat_template_kwargs": {}}

        handler._prepare_text_request = types.MethodType(_fake_prepare_text_request, handler)

        async def _collect() -> list[str | dict[str, object]]:
            return [item async for item in handler.generate_text_stream(request=object())]

        outputs = asyncio.run(_collect())

        emitted_tool_calls = [
            item for item in outputs if isinstance(item, dict) and isinstance(item.get("name"), str)
        ]
        assert len(emitted_tool_calls) == 1
        assert emitted_tool_calls[0]["name"] == "read_file"
        assert json.loads(emitted_tool_calls[0]["arguments"]) == {
            "path": "app/api/endpoints.py",
            "start_line": 692,
            "end_line": 790,
        }

        plain_content = "".join(item for item in outputs if isinstance(item, str))
        assert "<tool_call>" not in plain_content

    def test_stream_step35_hides_reasoning_when_open_tag_missing_but_close_tag_present(
        self,
    ) -> None:
        """Legacy step_35 behavior should infer reasoning from stray close tags."""
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
                _FakeStreamChunk("I should inspect both handlers first.\n", token=351),
                _FakeStreamChunk("Then verify token usage fields.</think>\n", token=352),
                _FakeStreamChunk("<tool_call>\n<function=read_file>\n", token=353),
                _FakeStreamChunk(
                    "<parameter=path>\napp/handler/mlx_vlm.py\n</parameter>\n",
                    token=354,
                ),
                _FakeStreamChunk("</function>\n</tool_call>\n", token=355),
            ]
        )

        async def _fake_prepare_text_request(
            self: object, request: object
        ) -> tuple[list[dict[str, str]], dict[str, object]]:
            return [{"role": "user", "content": "hello"}], {"chat_template_kwargs": {}}

        handler._prepare_text_request = types.MethodType(_fake_prepare_text_request, handler)

        async def _collect() -> list[str | dict[str, object]]:
            return [item async for item in handler.generate_text_stream(request=object())]

        outputs = asyncio.run(_collect())

        reasoning_text = "".join(
            item["reasoning_content"]
            for item in outputs
            if isinstance(item, dict) and isinstance(item.get("reasoning_content"), str)
        )
        assert (
            reasoning_text
            == "I should inspect both handlers first.\nThen verify token usage fields."
        )

        emitted_tool_calls = [
            item for item in outputs if isinstance(item, dict) and isinstance(item.get("name"), str)
        ]
        assert len(emitted_tool_calls) == 1
        assert emitted_tool_calls[0]["name"] == "read_file"
        assert json.loads(emitted_tool_calls[0]["arguments"]) == {"path": "app/handler/mlx_vlm.py"}

        visible_content = "".join(item for item in outputs if isinstance(item, str)) + "".join(
            item["content"]
            for item in outputs
            if isinstance(item, dict) and isinstance(item.get("content"), str)
        )
        assert "I should inspect both handlers first." not in visible_content
        assert "Then verify token usage fields." not in visible_content
        assert "</think>" not in visible_content
        assert "<tool_call>" not in visible_content

    def test_nonstream_step35_hides_reasoning_when_open_tag_missing_but_close_tag_present(
        self,
    ) -> None:
        """Legacy step_35 non-stream behavior should infer reasoning from stray close tags."""
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
            non_stream_response=_FakeNonStreamResponse(
                text=(
                    "I should inspect both handlers first.\n"
                    "Then verify token usage fields.</think>\n"
                    "<tool_call>\n"
                    "<function=read_file>\n"
                    "<parameter=path>\n"
                    "app/handler/mlx_vlm.py\n"
                    "</parameter>\n"
                    "</function>\n"
                    "</tool_call>\n"
                ),
                tokens=[1, 2, 3],
                prompt_tokens=100,
                generation_tokens=20,
            )
        )

        async def _fake_prepare_text_request(
            self: object, request: object
        ) -> tuple[list[dict[str, str]], dict[str, object]]:
            return [{"role": "user", "content": "hello"}], {"chat_template_kwargs": {}}

        handler._prepare_text_request = types.MethodType(_fake_prepare_text_request, handler)

        result = asyncio.run(handler.generate_text_response(request=object()))
        parsed = result["response"]

        assert isinstance(parsed, dict)
        assert parsed["reasoning_content"] == (
            "I should inspect both handlers first.\nThen verify token usage fields."
        )
        assert isinstance(parsed.get("tool_calls"), list)
        assert len(parsed["tool_calls"]) == 1
        assert parsed["tool_calls"][0]["name"] == "read_file"
        assert json.loads(parsed["tool_calls"][0]["arguments"]) == {
            "path": "app/handler/mlx_vlm.py"
        }
        content = parsed.get("content")
        if isinstance(content, str):
            assert "I should inspect both handlers first." not in content
            assert "Then verify token usage fields." not in content
            assert "</think>" not in content
            assert "<tool_call>" not in content

    def test_stream_mixed_think_parser_preserves_literal_text_when_open_tag_is_missing(
        self,
    ) -> None:
        """Semantic mixed-think parser remains strict when no opening reasoning marker exists."""
        handler_cls = _load_mlx_lm_handler_class()
        handler = object.__new__(handler_cls)

        handler.debug = False
        handler.message_converter = None
        handler.enable_auto_tool_choice = False
        handler.reasoning_parser_name = "mixed_think_tool_handoff"
        handler.tool_parser_name = "step_35"
        handler.model = _FakeModel()
        handler.prompt_cache = _FakePromptCache()
        handler.inference_worker = _FakeInferenceWorker(
            [
                _FakeStreamChunk("I should inspect both handlers first.\n", token=371),
                _FakeStreamChunk("Then verify token usage fields.</think>\n", token=372),
                _FakeStreamChunk("<tool_call>\n<function=read_file>\n", token=373),
                _FakeStreamChunk(
                    "<parameter=path>\napp/handler/mlx_vlm.py\n</parameter>\n",
                    token=374,
                ),
                _FakeStreamChunk("</function>\n</tool_call>\n", token=375),
            ]
        )

        async def _fake_prepare_text_request(
            self: object, request: object
        ) -> tuple[list[dict[str, str]], dict[str, object]]:
            return [{"role": "user", "content": "hello"}], {"chat_template_kwargs": {}}

        handler._prepare_text_request = types.MethodType(_fake_prepare_text_request, handler)

        async def _collect() -> list[str | dict[str, object]]:
            return [item async for item in handler.generate_text_stream(request=object())]

        outputs = asyncio.run(_collect())

        reasoning_text = "".join(
            item["reasoning_content"]
            for item in outputs
            if isinstance(item, dict) and isinstance(item.get("reasoning_content"), str)
        )
        assert reasoning_text == ""

        visible_content = "".join(item for item in outputs if isinstance(item, str)) + "".join(
            item["content"]
            for item in outputs
            if isinstance(item, dict) and isinstance(item.get("content"), str)
        )
        assert "I should inspect both handlers first." in visible_content
        assert "</think>" in visible_content

    def test_stream_step35_preserves_split_parameter_close_marker_inside_tool_call(self) -> None:
        """A chunk split at ``<`` + ``/parameter>`` should not corrupt parsed tool arguments."""
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
                _FakeStreamChunk(
                    (
                        "prefix text.\n"
                        "<tool_call>\n"
                        "<function=read_file>\n"
                        "<parameter=path>\n"
                        "mlx-openai-server/app/handler/mlx_lm.py\n"
                        "<"
                    ),
                    token=401,
                ),
                _FakeStreamChunk(
                    "/parameter>\n<parameter=start_line>\n151\n</parameter>\n",
                    token=402,
                ),
                _FakeStreamChunk(
                    "<parameter=end_line>\n523\n</parameter>\n</function>\n</tool_call>\n",
                    token=403,
                ),
            ]
        )

        async def _fake_prepare_text_request(
            self: object, request: object
        ) -> tuple[list[dict[str, str]], dict[str, object]]:
            return [{"role": "user", "content": "hello"}], {"chat_template_kwargs": {}}

        handler._prepare_text_request = types.MethodType(_fake_prepare_text_request, handler)

        async def _collect() -> list[str | dict[str, object]]:
            return [item async for item in handler.generate_text_stream(request=object())]

        outputs = asyncio.run(_collect())

        emitted_tool_calls = [
            item for item in outputs if isinstance(item, dict) and isinstance(item.get("name"), str)
        ]
        assert len(emitted_tool_calls) == 1
        assert emitted_tool_calls[0]["name"] == "read_file"
        assert json.loads(emitted_tool_calls[0]["arguments"]) == {
            "path": "mlx-openai-server/app/handler/mlx_lm.py",
            "start_line": 151,
            "end_line": 523,
        }

    def test_stream_step35_parses_tool_call_when_open_marker_is_split_as_too_plus_l_call(
        self,
    ) -> None:
        """A ``<too`` + ``l_call>`` split should still produce one parsed tool call."""
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
                _FakeStreamChunk("prefix text.<too", token=501),
                _FakeStreamChunk(
                    "l_call>\n"
                    "<function=read_file>\n"
                    "<parameter=path>\n"
                    "mlx-openai-server/app/handler/mlx_lm.py\n"
                    "</parameter>\n"
                    "<parameter=start_line>\n151\n</parameter>\n"
                    "<parameter=end_line>\n523\n</parameter>\n"
                    "</function>\n"
                    "</tool_call>\n",
                    token=502,
                ),
            ]
        )

        async def _fake_prepare_text_request(
            self: object, request: object
        ) -> tuple[list[dict[str, str]], dict[str, object]]:
            return [{"role": "user", "content": "hello"}], {"chat_template_kwargs": {}}

        handler._prepare_text_request = types.MethodType(_fake_prepare_text_request, handler)

        async def _collect() -> list[str | dict[str, object]]:
            return [item async for item in handler.generate_text_stream(request=object())]

        outputs = asyncio.run(_collect())

        emitted_tool_calls = [
            item for item in outputs if isinstance(item, dict) and isinstance(item.get("name"), str)
        ]
        assert len(emitted_tool_calls) == 1
        assert emitted_tool_calls[0]["name"] == "read_file"
        assert json.loads(emitted_tool_calls[0]["arguments"]) == {
            "path": "mlx-openai-server/app/handler/mlx_lm.py",
            "start_line": 151,
            "end_line": 523,
        }


if __name__ == "__main__":
    unittest.main()
