import asyncio
import json

from mlx_openai_server.api.endpoints import create_response_chunk, handle_stream_response


def test_handle_stream_response_tool_call_ids_stay_stable():
    async def fake_generator():
        yield {"name": "get_weather"}
        yield {"arguments": '{"city": "Hue"'}
        yield {"arguments": ': "Sunny"}'}

    async def consume_stream():
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


def test_create_response_chunk_respects_tool_call_id():
    chunk = {"name": "get_weather", "arguments": "{}", "index": 0}
    stable_id = "call_chatcmpl_test_0"

    response_chunk = create_response_chunk(
        chunk,
        model="mlx-test-model",
        tool_call_id=stable_id,
    )

    tool_call = response_chunk.choices[0].delta.tool_calls[0]
    assert tool_call.id == stable_id
