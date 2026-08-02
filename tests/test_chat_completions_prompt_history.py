"""Regression tests for chat-completions prompt-history preparation."""

from __future__ import annotations

import asyncio
import importlib
from pathlib import Path
import sys
import types
from typing import Any

from app.schemas.openai import ChatCompletionRequest, FunctionCall, Message


def _load_mlx_lm_handler_class() -> type:
    """Import ``MLXLMHandler`` with lightweight stubs for MLX-heavy modules."""
    repo_root = Path(__file__).resolve().parents[1]

    fake_handler_package = types.ModuleType("app.handler")
    fake_handler_package.__path__ = [str(repo_root / "app" / "handler")]

    fake_core_module = types.ModuleType("app.core")
    fake_core_module.BatchScheduler = object
    fake_core_module.InferenceWorker = object

    fake_batch_scheduler_module = types.ModuleType("app.core.batch_scheduler")
    fake_batch_scheduler_module.BATCHING_AVAILABLE = False
    fake_batch_scheduler_module.BatchScheduler = object

    fake_model_module = types.ModuleType("app.models.mlx_lm")
    fake_model_module.MLX_LM = object

    fake_prompt_cache_module = types.ModuleType("app.utils.prompt_cache")
    fake_prompt_cache_module.LRUPromptCache = object

    module_names = [
        "app.handler",
        "app.core",
        "app.core.batch_scheduler",
        "app.models.mlx_lm",
        "app.utils.prompt_cache",
        "app.handler.mlx_lm",
    ]
    original_modules: dict[str, types.ModuleType | None] = {
        name: sys.modules.get(name) for name in module_names
    }

    try:
        sys.modules["app.handler"] = fake_handler_package
        sys.modules["app.core"] = fake_core_module
        sys.modules["app.core.batch_scheduler"] = fake_batch_scheduler_module
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


def _load_mlx_lm_class() -> type:
    """Import ``MLX_LM`` with lightweight stubs for MLX-heavy modules."""
    fake_mx = types.ModuleType("mlx.core")
    fake_mx.array = lambda x: x
    fake_mx.random = object()

    fake_generate = types.ModuleType("mlx_lm.generate")
    fake_generate.GenerationResponse = object
    fake_generate.stream_generate = lambda *args, **kwargs: iter(())

    fake_cache = types.ModuleType("mlx_lm.models.cache")
    fake_cache.can_trim_prompt_cache = lambda *args, **kwargs: True
    fake_cache.make_prompt_cache = lambda *args, **kwargs: []

    fake_sample = types.ModuleType("mlx_lm.sample_utils")
    fake_sample.make_logits_processors = lambda **kwargs: []
    fake_sample.make_sampler = lambda **kwargs: object()

    fake_utils = types.ModuleType("mlx_lm.utils")
    fake_utils._download = lambda path: path
    fake_utils.load = lambda *args, **kwargs: None

    fake_outlines = types.ModuleType("outlines")
    fake_outlines_models = types.ModuleType("outlines.models")
    fake_outlines_transformers = types.ModuleType("outlines.models.transformers")
    fake_outlines_transformers.TransformerTokenizer = type("TransformerTokenizer", (), {})
    fake_outlines_proc = types.ModuleType("outlines.processors")
    fake_outlines_proc.JSONLogitsProcessor = object

    module_names = [
        "mlx",
        "mlx.core",
        "mlx_lm.generate",
        "mlx_lm.models.cache",
        "mlx_lm.sample_utils",
        "mlx_lm.utils",
        "outlines",
        "outlines.models",
        "outlines.models.transformers",
        "outlines.processors",
        "app.utils.outlines_transformer_tokenizer",
        "app.models.mlx_lm",
    ]
    original_modules: dict[str, types.ModuleType | None] = {
        name: sys.modules.get(name) for name in module_names
    }

    fake_tokenizer_module = types.ModuleType("app.utils.outlines_transformer_tokenizer")
    fake_tokenizer_module.OutlinesTransformerTokenizer = type(
        "OutlinesTransformerTokenizer", (), {}
    )

    try:
        sys.modules["mlx"] = types.ModuleType("mlx")
        sys.modules["mlx.core"] = fake_mx
        sys.modules["mlx_lm.generate"] = fake_generate
        sys.modules["mlx_lm.models.cache"] = fake_cache
        sys.modules["mlx_lm.sample_utils"] = fake_sample
        sys.modules["mlx_lm.utils"] = fake_utils
        sys.modules["outlines"] = fake_outlines
        sys.modules["outlines.models"] = fake_outlines_models
        sys.modules["outlines.models.transformers"] = fake_outlines_transformers
        sys.modules["outlines.processors"] = fake_outlines_proc
        sys.modules["app.utils.outlines_transformer_tokenizer"] = fake_tokenizer_module
        sys.modules.pop("app.models.mlx_lm", None)

        module = importlib.import_module("app.models.mlx_lm")
        module = importlib.reload(module)
        return module.MLX_LM
    finally:
        for name, module in original_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def test_prepare_text_request_strips_reasoning_content_from_prior_assistant_messages() -> None:
    """Prepared prompt messages should not carry prior assistant reasoning text."""
    handler_cls = _load_mlx_lm_handler_class()
    handler = handler_cls.__new__(handler_cls)
    handler.kv_bits = None
    handler.kv_group_size = 64
    handler.quantized_kv_start = 0

    request = ChatCompletionRequest(
        model="local-text-model",
        messages=[
            Message(role="system", content="System rules."),
            Message(role="user", content="Question one."),
            Message(
                role="assistant",
                content="Visible answer one.",
                reasoning_content="Hidden reasoning one.",
            ),
            Message(role="user", content="Question two."),
        ],
    )

    chat_messages, _ = asyncio.run(handler._prepare_text_request(request))

    assert all("reasoning_content" not in msg for msg in chat_messages)
    assert [msg["role"] for msg in chat_messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert chat_messages[2]["content"] == "Visible answer one."


def test_prepare_text_request_strips_reasoning_content_from_tool_call_assistant_messages() -> None:
    """Tool-call assistant turns should preserve tool data while removing reasoning text."""
    handler_cls = _load_mlx_lm_handler_class()
    handler = handler_cls.__new__(handler_cls)
    handler.kv_bits = None
    handler.kv_group_size = 64
    handler.quantized_kv_start = 0

    request = ChatCompletionRequest(
        model="local-text-model",
        messages=[
            Message(role="user", content="Run weather lookup."),
            Message(
                role="assistant",
                content=None,
                reasoning_content="Should call weather tool first.",
                tool_calls=[
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": FunctionCall(
                            name="get_weather",
                            arguments='{"city":"Boston"}',
                        ),
                    }
                ],
            ),
            Message(role="tool", tool_call_id="call_123", content='{"temp_f":42}'),
            Message(role="user", content="Now summarize it."),
        ],
    )

    chat_messages, _ = asyncio.run(handler._prepare_text_request(request))

    assert all("reasoning_content" not in msg for msg in chat_messages)
    assert [msg["role"] for msg in chat_messages] == [
        "user",
        "assistant",
        "tool",
        "user",
    ]

    assistant_turn = chat_messages[1]
    assert assistant_turn["content"] in {"", None}
    assert isinstance(assistant_turn.get("tool_calls"), list)
    assert assistant_turn["tool_calls"][0]["function"]["name"] == "get_weather"
    assert assistant_turn["tool_calls"][0]["function"]["arguments"] == '{"city":"Boston"}'


def test_create_input_prompt_normalizes_tool_call_argument_strings_only_for_template_render() -> None:
    """Template replay should see mapping args while the original messages stay OpenAI-compatible."""
    mlx_lm_cls = _load_mlx_lm_class()
    model = mlx_lm_cls.__new__(mlx_lm_cls)

    captured: dict[str, Any] = {}

    class _FakeTokenizer:
        def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
            captured["messages"] = messages
            captured["kwargs"] = kwargs
            return "ok"

    model.tokenizer = _FakeTokenizer()

    messages = [
        {"role": "user", "content": "Check when Maxim returns from vacation."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "search_pages",
                        "arguments": '{"query":"Maxim vacation"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "Return date: 2026-08-11",
        },
    ]

    original_argument_string = messages[1]["tool_calls"][0]["function"]["arguments"]

    result = model.create_input_prompt(messages, {})

    assert result == "ok"
    assert messages[1]["tool_calls"][0]["function"]["arguments"] == original_argument_string
    rendered_arguments = captured["messages"][1]["tool_calls"][0]["function"]["arguments"]
    assert rendered_arguments == {"query": "Maxim vacation"}
    assert captured["kwargs"]["add_generation_prompt"] is True
    assert captured["kwargs"]["continue_final_message"] is False
