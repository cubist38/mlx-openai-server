"""Regression tests for chat-completions prompt-history preparation."""

from __future__ import annotations

import asyncio
import importlib
from pathlib import Path
import sys
import types

from app.schemas.openai import ChatCompletionRequest, FunctionCall, Message


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


def test_prepare_text_request_strips_reasoning_content_from_prior_assistant_messages() -> None:
    """Prepared prompt messages should not carry prior assistant reasoning text."""
    handler_cls = _load_mlx_lm_handler_class()
    handler = handler_cls.__new__(handler_cls)

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
