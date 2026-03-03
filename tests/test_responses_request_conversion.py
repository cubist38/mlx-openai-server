"""Tests for Responses API request conversion into chat-completions format."""

from __future__ import annotations

import importlib
import sys
import types
from typing import Any

from app.schemas.openai import ResponsesRequest


def _load_convert_responses_request_to_chat_request() -> Any:
    """Import conversion function with lightweight handler stubs."""
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
        return endpoints_module.convert_responses_request_to_chat_request
    finally:
        sys.modules.pop("app.api.endpoints", None)
        for name, module in original_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def test_convert_responses_request_does_not_reinject_reasoning_items() -> None:
    """Reasoning items should not be reinserted as assistant prompt content."""
    convert_responses_request_to_chat_request = _load_convert_responses_request_to_chat_request()

    request = ResponsesRequest.model_construct(
        model="local-text-model",
        input=[
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "First question"}],
            },
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "hidden long reasoning summary"}],
                "content": [{"type": "reasoning_text", "text": "hidden long reasoning summary"}],
                "status": "completed",
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Visible assistant answer"}],
                "status": "completed",
            },
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Follow-up question"}],
            },
        ],
        instructions=None,
        max_output_tokens=None,
        stream=False,
        previous_response_id=None,
        temperature=None,
        top_p=None,
        top_k=None,
        min_p=None,
        repetition_penalty=None,
        seed=None,
        text=None,
        tools=None,
        tool_choice="auto",
        reasoning=None,
    )

    chat_request = convert_responses_request_to_chat_request(request)

    # The conversion should preserve visible dialogue only; hidden reasoning
    # must not become assistant reasoning_content in the next-turn prompt.
    assert all(message.reasoning_content is None for message in chat_request.messages)
    assert [message.role for message in chat_request.messages] == [
        "user",
        "assistant",
        "user",
    ]
    assert isinstance(chat_request.messages[0].content, list)
    assert chat_request.messages[0].content[0].type == "text"
    assert chat_request.messages[0].content[0].text == "First question"
    assert chat_request.messages[1].content == "Visible assistant answer"
    assert isinstance(chat_request.messages[2].content, list)
    assert chat_request.messages[2].content[0].type == "text"
    assert chat_request.messages[2].content[0].text == "Follow-up question"


def test_convert_responses_stream_history_skips_reasoning_but_keeps_tool_turns() -> None:
    """Stream-style histories should drop reasoning while preserving tool call flow."""
    convert_responses_request_to_chat_request = _load_convert_responses_request_to_chat_request()

    request = ResponsesRequest.model_construct(
        model="local-text-model",
        input=[
            {
                "id": "msg_prev_user",
                "type": "message",
                "role": "user",
                "status": "completed",
                "content": [{"type": "input_text", "text": "Find disk usage for /tmp"}],
            },
            {
                "id": "rs_prev",
                "type": "reasoning",
                "status": "completed",
                "summary": [{"type": "summary_text", "text": "hidden chain-of-thought"}],
            },
            {
                "id": "fc_prev",
                "type": "function_call",
                "status": "completed",
                "call_id": "call_123",
                "name": "du",
                "arguments": "{\"path\":\"/tmp\"}",
            },
            {
                "type": "function_call_output",
                "call_id": "call_123",
                "output": [{"type": "output_text", "text": "/tmp 12M"}],
            },
            {
                "id": "msg_prev_assistant",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "It is using around 12 MB."}],
            },
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Now check /var"}],
            },
        ],
        instructions=None,
        max_output_tokens=None,
        stream=False,
        previous_response_id=None,
        temperature=None,
        top_p=None,
        top_k=None,
        min_p=None,
        repetition_penalty=None,
        seed=None,
        text=None,
        tools=None,
        tool_choice="auto",
        reasoning=None,
    )

    chat_request = convert_responses_request_to_chat_request(request)

    assert all(message.reasoning_content is None for message in chat_request.messages)
    assert [message.role for message in chat_request.messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "user",
    ]

    tool_call_messages = [m for m in chat_request.messages if m.role == "assistant" and m.tool_calls]
    assert len(tool_call_messages) == 1
    assert tool_call_messages[0].tool_calls[0].function.name == "du"
    assert tool_call_messages[0].tool_calls[0].function.arguments == "{\"path\":\"/tmp\"}"
    assert tool_call_messages[0].tool_calls[0].id == "call_123"

    tool_output_messages = [m for m in chat_request.messages if m.role == "tool"]
    assert len(tool_output_messages) == 1
    assert tool_output_messages[0].tool_call_id == "call_123"
    assert tool_output_messages[0].content == "/tmp 12M"
