"""Mocked engine-to-API regressions for termination and separate-parser EOF."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
import json
import pickle
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.api.endpoints import (
    format_final_responses_response,
    handle_responses_stream_response,
    handle_stream_response,
    process_text_request,
)
from app.parsers import ParserManager
from app.parsers.function_parameter import FunctionParameterToolParser
from app.parsers.hermes import HermesReasoningParser
from app.schemas.openai import ChatCompletionRequest, ResponsesRequest
from tests.test_chat_completions_prompt_history import _load_mlx_lm_handler_class

CALL = "<tool_call><function=lookup><parameter=x>1</parameter></function></tool_call>"


def _handler(texts: list[str], reason: str = "length", reasoning: str | None = None) -> object:
    """Build a handler whose worker emits fixed engine chunks without loading MLX."""
    cls = _load_mlx_lm_handler_class()
    handler = cls.__new__(cls)
    handler.debug = False
    handler._is_request_batchable = lambda request: False
    ctx = SimpleNamespace(
        total_input_tokens=3,
        total_cached_tokens=0,
        model_params={},
        parsers_result=ParserManager.create_parsers(
            reasoning_parser_name=reasoning, tool_parser_name="nemotron3_nano"
        ),
        prompt_progress_callback=None,
        checkpoint_position=None,
        rest_input_ids=[1, 2, 3],
        cache_key=[1, 2, 3],
        cache=None,
    )
    handler._build_inference_context = AsyncMock(return_value=ctx)
    chunks = [
        SimpleNamespace(
            text=text,
            generation_tokens=i + 1,
            finish_reason=reason if i == len(texts) - 1 else None,
        )
        for i, text in enumerate(texts)
    ]

    async def stream(*args: object, **kwargs: object) -> AsyncIterator[object]:
        for chunk in chunks:
            yield chunk

    handler.inference_worker = SimpleNamespace(submit_stream=stream)
    handler._run_nonstream_generation = AsyncMock(
        return_value=SimpleNamespace(
            text="".join(texts), generation_tokens=len(texts), finish_reason=reason
        )
    )
    return handler


async def _ipc(stream: AsyncIterator[object]) -> AsyncIterator[object]:
    """Use the same pickle serialization as multiprocessing queues."""
    async for item in stream:
        yield pickle.loads(pickle.dumps(item))


@pytest.mark.parametrize("reason,expected", [("length", "length"), ("stop", "tool_calls")])
@pytest.mark.parametrize("streaming", [True, False])
def test_engine_reason_survives_handler_and_api(
    reason: str, expected: str, streaming: bool
) -> None:
    """Engine length takes precedence even after a complete tool call, including IPC."""
    handler = _handler([CALL, "<tool_call><function=unfinished>"], reason)
    request = ChatCompletionRequest(model="test", messages=[])

    async def run() -> dict:
        if streaming:
            events = [
                event
                async for event in handle_stream_response(
                    _ipc(handler.generate_text_stream(request)), "test"
                )
            ]
            return json.loads(events[-2][6:])
        response = await process_text_request(handler, request)
        return json.loads(response.body)

    payload = asyncio.run(run())
    choice = payload["choices"][0]
    assert choice["finish_reason"] == expected
    assert choice["parser_diagnostics"] == ["incomplete_tool_call"]
    if not streaming:
        assert len(choice["message"]["tool_calls"]) == 1
        assert not choice["message"].get("content")


@pytest.mark.parametrize("streaming", [True, False])
def test_plain_text_length_reaches_chat_api(streaming: bool) -> None:
    """Length is preserved without tools or parser diagnostics."""
    handler = _handler(["partial text"])
    request = ChatCompletionRequest(model="test", messages=[])

    async def run() -> dict:
        if streaming:
            events = [
                event
                async for event in handle_stream_response(
                    handler.generate_text_stream(request), "test"
                )
            ]
            return json.loads(events[-2][6:])
        response = await process_text_request(handler, request)
        return json.loads(response.body)

    choice = asyncio.run(run())["choices"][0]
    assert choice["finish_reason"] == "length"
    assert "parser_diagnostics" not in choice


@pytest.mark.parametrize("streaming", [True, False])
def test_responses_reports_incomplete_on_length(streaming: bool) -> None:
    """Both Responses entry points expose engine length, not completed."""
    handler = _handler(["partial text"])
    request = ResponsesRequest(model="test", input="hello")
    chat_request = ChatCompletionRequest(model="test", messages=[])

    async def run() -> dict:
        if streaming:
            events = [
                event
                async for event in handle_responses_stream_response(
                    _ipc(handler.generate_text_stream(chat_request)), request
                )
            ]
            assert events[-1].startswith("event: response.incomplete\n")
            return json.loads(events[-1].split("data: ")[1])["response"]
        result = await handler.generate_text_response(chat_request)
        return format_final_responses_response(
            result["response"], request, result["usage"]
        ).model_dump()

    result = asyncio.run(run())
    assert result["status"] == "incomplete"
    assert result["incomplete_details"] == {"reason": "max_output_tokens"}


@pytest.mark.parametrize("marker", ["<tool_call>", "<think>"])
def test_literal_partial_open_delimiter_eof(marker: str) -> None:
    """Every proper opener prefix is safely flushed, including single-character chunks."""
    for length in range(1, len(marker)):
        parser = (
            FunctionParameterToolParser() if marker == "<tool_call>" else HermesReasoningParser()
        )
        parse = (
            parser.extract_tool_calls_streaming
            if marker == "<tool_call>"
            else parser.extract_reasoning_streaming
        )
        output = ""
        for char in "literal " + marker[:length]:
            payload, _ = parse(char)
            output += (payload or {}).get("content", "")
        output += parser.finalize().get("content", "")
        assert output == "literal " + marker[:length]
        assert not parser.diagnostics


def test_reasoning_close_delimiter_eof() -> None:
    """Every incomplete reasoning closer is emitted literally in the reasoning channel."""
    for size in range(len("</think>")):
        parser = HermesReasoningParser()
        body = "thinking" + "</think>"[:size]
        output = ""
        for char in "<think>" + body:
            payload, _ = parser.extract_reasoning_streaming(char)
            output += (payload or {}).get("reasoning_content", "")
        output += parser.finalize().get("reasoning_content", "")
        assert output == body
        assert parser.diagnostics == ["incomplete_reasoning"]


@pytest.mark.parametrize("reasoning", [None, "nemotron3_nano"])
def test_handler_flushes_partial_delimiters(reasoning: str | None) -> None:
    """Tool EOF tails stay content; reasoning close tails stay reasoning."""
    text = "working</thi" if reasoning else "literal <tool_"
    handler = _handler(list(text), reasoning=reasoning)

    async def run() -> list:
        return [item async for item in handler.generate_text_stream(object())]

    output = asyncio.run(run())
    if reasoning:
        assert (
            "".join(item.get("reasoning_content", "") for item in output if isinstance(item, dict))
            == text
        )
        assert output[-1]["__parser_diagnostics__"] == ["incomplete_reasoning"]
    else:
        assert "".join(item for item in output if isinstance(item, str)) == text


@pytest.mark.parametrize("streaming", [True, False])
@pytest.mark.parametrize("closed", [True, False])
def test_reasoning_xml_is_never_executable(streaming: bool, closed: bool) -> None:
    """Nemotron implicit reasoning retains XML as data, also at truncated EOF."""
    text = "consider " + CALL + ("</think>answer" if closed else "</thi")
    handler = _handler(list(text) if streaming else [text], reasoning="nemotron3_nano")

    async def run() -> None:
        if streaming:
            output = [item async for item in handler.generate_text_stream(object())]
            assert not any(isinstance(item, dict) and item.get("name") for item in output)
            assert CALL in "".join(
                item.get("reasoning_content", "") for item in output if isinstance(item, dict)
            )
        else:
            result = await handler.generate_text_response(object())
            assert not result["response"]["tool_calls"]
            assert CALL in result["response"]["reasoning_content"]

    asyncio.run(run())


@pytest.mark.parametrize("split", [False, True])
def test_complete_then_incomplete_call_never_leaks_or_executes_tail(split: bool) -> None:
    """A second opener after a complete block must remain buffered through EOF."""
    for size in range(len("</tool_call>")):
        parser = FunctionParameterToolParser()
        unfinished = "<tool_call><function=bad></function>" + "</tool_call>"[:size]
        text = CALL + unfinished
        calls = []
        for chunk in list(text) if split else [text]:
            payload, _ = parser.extract_tool_calls_streaming(chunk)
            calls.extend((payload or {}).get("tool_calls", []))
            assert not (payload or {}).get("content")
        assert not parser.finalize()
        assert [call["name"] for call in calls] == ["lookup"]
        assert parser.diagnostics == ["incomplete_tool_call"]


@pytest.mark.parametrize("function", ["<function=f>", "<function = f>"])
@pytest.mark.parametrize("second", ["<parameter=x>", "<parameter = x >"])
def test_duplicate_parameters_reject_strict_and_permissive(function: str, second: str) -> None:
    """Whitespace/permissive fallback cannot bypass duplicate-name rejection."""
    parser = FunctionParameterToolParser()
    text = f"<tool_call>{function}<parameter=x>1</parameter>{second}2</parameter></function></tool_call>"
    assert not parser.extract_tool_calls(text).get("tool_calls")
    assert parser.diagnostics == ["duplicate_tool_parameter"]


def test_nonstream_parser_never_executes_unclosed_outer_tool() -> None:
    """A complete inner function is not sufficient to execute an unclosed outer call."""
    parser = FunctionParameterToolParser()
    assert not parser.extract_tool_calls(CALL.removesuffix("</tool_call>")).get("tool_calls")


@pytest.mark.parametrize(
    "model_default,override,expected",
    [(False, None, False), (True, None, True), (False, True, True), (True, False, False)],
)
def test_reasoning_history_opt_in_and_override(
    model_default: bool, override: bool | None, expected: bool
) -> None:
    """Request overrides model policy without leaking a server control into templates."""
    cls = _load_mlx_lm_handler_class()
    handler = cls.__new__(cls)
    handler.kv_bits, handler.kv_group_size, handler.quantized_kv_start = None, 64, 0
    handler.preserve_reasoning_history = model_default
    request = ChatCompletionRequest(
        model="test",
        messages=[
            {"role": "assistant", "content": None, "reasoning": "remember me"},
            {"role": "user", "content": "next"},
        ],
        chat_template_kwargs={"preserve_reasoning_history": override},
    )
    messages, params = asyncio.run(handler._prepare_text_request(request))
    assert (messages[0].get("reasoning_content") == "remember me") == expected
    assert "preserve_reasoning_history" not in params["chat_template_kwargs"]
    if expected:
        assert params["chat_template_kwargs"]["truncate_history_thinking"] is False
    assert request.messages[0].reasoning_content == "remember me"


@pytest.mark.parametrize("split", [False, True])
def test_valid_and_duplicate_calls_are_chunk_boundary_independent(split: bool) -> None:
    """A rejected duplicate call must not hide a neighboring valid call."""
    bad = CALL.replace("</function>", "<parameter=x>2</parameter></function>")
    parser = FunctionParameterToolParser()
    calls = []
    for chunk in [CALL, bad] if split else [CALL + bad]:
        payload, _ = parser.extract_tool_calls_streaming(chunk)
        calls.extend(payload.get("tool_calls", []))
    assert [call["name"] for call in calls] == ["lookup"]
    assert parser.diagnostics == ["duplicate_tool_parameter"]


def test_batched_collection_preserves_engine_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    """The batched aggregation path retains termination just like single-request generation."""
    import sys
    import types

    handler = _handler(["unused"])
    fake_module = types.ModuleType("app.models.mlx_lm")
    fake_module.CompletionResponse = SimpleNamespace
    monkeypatch.setitem(sys.modules, "app.models.mlx_lm", fake_module)

    async def stream(*args: object) -> AsyncIterator[object]:
        yield SimpleNamespace(
            text="hi",
            token=1,
            finish_reason="length",
            generation_tokens=1,
            generation_tps=1,
            peak_memory=0,
            prompt_tokens=3,
            cached_prompt_tokens=0,
        )

    handler._submit_batched_stream = stream
    response = asyncio.run(handler._collect_batched_response(None, SimpleNamespace()))
    assert response.finish_reason == "length"
    assert response.text == "hi"


def test_model_history_setting_survives_config_and_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    """Single-model conversion and subprocess factory preserve the LM history setting."""
    import sys
    import types
    from unittest.mock import Mock

    from app.config import MLXServerConfig, ModelEntryConfig
    from app.server import create_handler_from_config

    config = MLXServerConfig(model_path="dummy", preserve_reasoning_history=True)
    entry = config.to_model_entry_config()
    entry = ModelEntryConfig(**pickle.loads(pickle.dumps(entry.__dict__)))
    fake_module = types.ModuleType("app.handler.mlx_lm")
    fake_module.MLXLMHandler = Mock()
    monkeypatch.setitem(sys.modules, "app.handler.mlx_lm", fake_module)
    create_handler_from_config(entry)
    assert fake_module.MLXLMHandler.call_args.kwargs["preserve_reasoning_history"] is True


def test_opt_in_preserves_tool_history_and_explicit_template_override() -> None:
    """Preserved reasoning must coexist with structured tools and caller template settings."""
    cls = _load_mlx_lm_handler_class()
    handler = cls.__new__(cls)
    handler.kv_bits, handler.kv_group_size, handler.quantized_kv_start = None, 64, 0
    request = ChatCompletionRequest(
        model="test",
        messages=[
            {
                "role": "assistant",
                "reasoning_content": "plan",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                    }
                ],
            }
        ],
        chat_template_kwargs={
            "preserve_reasoning_history": True,
            "truncate_history_thinking": True,
        },
    )
    messages, params = asyncio.run(handler._prepare_text_request(request))
    assert messages[0]["reasoning_content"] == "plan"
    assert messages[0]["tool_calls"][0]["function"]["name"] == "lookup"
    assert params["chat_template_kwargs"]["truncate_history_thinking"] is True
