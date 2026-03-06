"""Unit tests for chat-completions env defaulting helpers."""

from __future__ import annotations

import importlib
import os
import sys
import types

import pytest

from app.schemas.openai import ChatCompletionRequest, Message


def _load_refine_chat_completion_request() -> object:
    """Import refine function with lightweight handler stubs."""
    fake_lm_module = types.ModuleType("app.handler.mlx_lm")
    fake_lm_module.MLXLMHandler = object

    fake_vlm_module = types.ModuleType("app.handler.mlx_vlm")
    fake_vlm_module.MLXVLMHandler = object

    module_names = [
        "app.handler.mlx_lm",
        "app.handler.mlx_vlm",
        "app.api.endpoints",
    ]
    original_modules: dict[str, types.ModuleType | None] = {
        name: sys.modules.get(name) for name in module_names
    }

    try:
        sys.modules["app.handler.mlx_lm"] = fake_lm_module
        sys.modules["app.handler.mlx_vlm"] = fake_vlm_module
        sys.modules.pop("app.api.endpoints", None)
        endpoints_module = importlib.import_module("app.api.endpoints")
        return endpoints_module.refine_chat_completion_request
    finally:
        sys.modules.pop("app.api.endpoints", None)
        for name, module in original_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def test_refine_chat_completion_request_sets_min_p_from_env_when_missing() -> None:
    """refine_chat_completion_request should default min_p from DEFAULT_MIN_P."""
    refine_chat_completion_request = _load_refine_chat_completion_request()

    previous = os.environ.get("DEFAULT_MIN_P")
    os.environ["DEFAULT_MIN_P"] = "0.123"
    try:
        request = ChatCompletionRequest(
            model="local-text-model",
            messages=[Message(role="user", content="hi")],
            min_p=None,
        )

        refined = refine_chat_completion_request(request)
        assert refined.min_p == pytest.approx(0.123)
    finally:
        if previous is None:
            os.environ.pop("DEFAULT_MIN_P", None)
        else:
            os.environ["DEFAULT_MIN_P"] = previous
