"""Tests for streaming endpoint functionality."""

import asyncio
from collections.abc import AsyncGenerator
import json

from mlx_openai_server.api.endpoints import create_response_chunk, handle_stream_response
import pytest


def test_handle_stream_response_tool_call_ids_stay_stable() -> None:
    """Test that tool call IDs remain stable across streaming chunks."""

    async def fake_generator() -> AsyncGenerator[dict[str, str], None]:
        yield {"name": "get_weather"}
        yield {"arguments": '{"city": "Hue"'}
        yield {"arguments": ', "weather": "Sunny"}'}

    async def consume_stream() -> list[str]:
        stream = handle_stream_response(fake_generator(), "mlx-test-model")
        tool_call_ids: list[str] = []

        async for chunk in stream:
            if not chunk.startswith("data: "):
                continue
            payload_str = chunk[len("data: ") :].strip()
            if payload_str == "[DONE]":
                continue
            payload = json.loads(payload_str)
            delta = payload["choices"][0]["delta"]
            tool_calls = delta.get("tool_calls") or []
            if tool_calls:
                tool_call_ids.append(tool_calls[0]["id"])

        return tool_call_ids

    tool_call_ids = asyncio.run(consume_stream())
    assert tool_call_ids, "Expected at least one tool call chunk"
    assert len(set(tool_call_ids)) == 1, "Tool call IDs should remain stable across chunks"


def test_handle_stream_response_arguments_first_tool_call_ids_stable() -> None:
    """Ensure tool call IDs stay stable even if arguments arrive before name."""

    async def fake_generator() -> AsyncGenerator[dict[str, str | int], None]:
        yield {"arguments": '{"city": "Hue"}', "index": 0}
        yield {"arguments": '{"unit": "C"}', "index": 0}
        yield {"name": "get_weather", "index": 0}

    async def consume_stream() -> list[str]:
        stream = handle_stream_response(fake_generator(), "mlx-test-model")
        tool_call_ids: list[str] = []

        async for chunk in stream:
            if not chunk.startswith("data: "):
                continue
            payload_str = chunk[len("data: ") :].strip()
            if payload_str == "[DONE]":
                continue
            payload = json.loads(payload_str)
            delta = payload["choices"][0]["delta"]
            tool_calls = delta.get("tool_calls") or []
            if tool_calls:
                tool_call_ids.append(tool_calls[0]["id"])

        return tool_call_ids

    tool_call_ids = asyncio.run(consume_stream())
    assert tool_call_ids, "Expected at least one tool call chunk"
    assert len(set(tool_call_ids)) == 1, "Tool call IDs should remain stable across chunks"


def test_handle_stream_response_arguments_without_index_tool_call_ids_stable() -> None:
    """IDs stay stable when argument chunks omit tool_call_index."""

    async def fake_generator() -> AsyncGenerator[dict[str, str], None]:
        yield {"arguments": '{"city": "Hue"}'}
        yield {"arguments": '{"unit": "C"}'}
        yield {"name": "get_weather"}

    async def consume_stream() -> list[str]:
        stream = handle_stream_response(fake_generator(), "mlx-test-model")
        tool_call_ids: list[str] = []

        async for chunk in stream:
            if not chunk.startswith("data: "):
                continue
            payload_str = chunk[len("data: ") :].strip()
            if payload_str == "[DONE]":
                continue
            payload = json.loads(payload_str)
            delta = payload["choices"][0]["delta"]
            tool_calls = delta.get("tool_calls") or []
            if tool_calls:
                tool_call_ids.append(tool_calls[0]["id"])

        return tool_call_ids

    tool_call_ids = asyncio.run(consume_stream())
    assert tool_call_ids, "Expected at least one tool call chunk"
    assert len(set(tool_call_ids)) == 1, "Tool call IDs should remain stable across chunks"


def test_create_response_chunk_respects_tool_call_id() -> None:
    """Test that create_response_chunk creates tool call chunks properly."""
    chunk = {"name": "get_weather", "arguments": "{}", "index": 0}
    stable_id = "call_chatcmpl_test_0"

    response_chunk = create_response_chunk(
        chunk,
        model="mlx-test-model",
        tool_call_id=stable_id,
    )

    tool_call = response_chunk.choices[0].delta.tool_calls[0]
    assert tool_call.id == stable_id


@pytest.mark.parametrize(
    ("chunk_sequence", "expectation"),
    [
        pytest.param(
            [None, "Hello world"],
            {"contains_text": "Hello world"},
            id="string_chunk_with_none_prefix",
        ),
        pytest.param(
            [
                {
                    "__usage__": {
                        "prompt_tokens": 5,
                        "completion_tokens": 7,
                        "total_tokens": 12,
                    }
                }
            ],
            {
                "final_usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 7,
                    "total_tokens": 12,
                }
            },
            id="usage_payload_recorded_in_final_chunk",
        ),
        pytest.param(
            [123],
            {"expect_error_substring": "Invalid chunk type"},
            id="invalid_chunk_type_emits_error",
        ),
    ],
)
def test_handle_stream_response_edge_cases(
    chunk_sequence: list[object], expectation: dict[str, object]
) -> None:
    """Cover misc stream edge cases via parameterized scenarios."""

    async def fake_generator() -> AsyncGenerator[object, None]:
        for chunk in chunk_sequence:
            yield chunk

    async def consume_stream() -> list[dict[str, object]]:
        stream = handle_stream_response(fake_generator(), "mlx-test-model")
        payloads: list[dict[str, object]] = []

        async for chunk in stream:
            if not chunk.startswith("data: "):
                continue
            payload_str = chunk[len("data: ") :].strip()
            if payload_str == "[DONE]":
                continue
            payloads.append(json.loads(payload_str))

        return payloads

    payloads = asyncio.run(consume_stream())
    assert payloads, "Expected at least the initial and final chunks"

    contains_text = expectation.get("contains_text")
    if contains_text:
        assert any(
            choice["delta"].get("content") == contains_text
            for payload in payloads
            if payload.get("object") == "chat.completion.chunk"
            for choice in payload.get("choices", [])
        ), "Stream should include the requested content chunk"

    final_usage = expectation.get("final_usage")
    if final_usage:
        usage_payload = payloads[-1].get("usage")
        assert usage_payload is not None, "Final chunk should include usage information"
        for key, value in final_usage.items():
            assert usage_payload.get(key) == value

    error_substring = expectation.get("expect_error_substring")
    if error_substring:
        assert any(
            "error" in payload and error_substring in payload["error"]["message"]
            for payload in payloads
        ), "Expected error payload not found in stream"
