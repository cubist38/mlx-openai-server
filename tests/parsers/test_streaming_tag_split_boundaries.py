"""Regression tests for streaming parser behavior across chunk boundaries."""

from __future__ import annotations

import json

from app.parsers.function_parameter import FunctionParameterToolParser
from app.parsers.qwen3_moe import Qwen3MoEReasoningParser


def test_qwen3_reasoning_parser_handles_split_reasoning_close_tag() -> None:
    """Reasoning parsing should complete even when ``</think>`` is split across chunks."""
    parser = Qwen3MoEReasoningParser()

    # Mirror handler behavior for mixed_think_tool_handoff/qwen3_moe reasoning parsers:
    # first chunk is prefixed with `<think>` when the model omits it.
    first_chunk = "<think>" + "I should call a tool now.</th"
    second_chunk = "ink><tool_call>"

    first_result, first_complete = parser.extract_reasoning_streaming(first_chunk)
    second_result, second_complete = parser.extract_reasoning_streaming(second_chunk)

    assert first_result is not None
    assert first_complete is False
    assert second_result is not None
    assert second_complete is True
    assert second_result.get("after_reasoning_close_content") == "<tool_call>"


def test_function_parameter_tool_parser_handles_split_tool_open_tag() -> None:
    """Tool parsing should work when ``<tool_call>`` is split across chunks."""
    parser = FunctionParameterToolParser()

    chunks = [
        "Prefix text <tool_",
        (
            "call>\n"
            "<function=get_weather>\n"
            '<parameter=city>"NYC"</parameter>\n'
            "</function>\n"
            "</tool_call>"
        ),
    ]

    parsed_outputs: list[dict[str, object]] = []
    for chunk in chunks:
        parsed, _is_complete = parser.extract_tool_calls_streaming(chunk)
        if isinstance(parsed, dict):
            parsed_outputs.append(parsed)

    tool_payload = next((item for item in parsed_outputs if "tool_calls" in item), None)
    assert tool_payload is not None

    tool_calls = tool_payload["tool_calls"]
    assert isinstance(tool_calls, list)
    assert len(tool_calls) == 1
    tool_call = tool_calls[0]
    assert tool_call["name"] == "get_weather"
    assert json.loads(tool_call["arguments"]) == {"city": "NYC"}


def test_function_parameter_tool_parser_preserves_leading_text_before_tool_open() -> None:
    """Parser should emit text that appears before a full ``<tool_call>`` marker."""
    parser = FunctionParameterToolParser()

    chunks = [
        "Lead text <tool_call>\n<function=get_weather>",
        '\n<parameter=city>"NYC"</parameter>\n</function>\n</tool_call>',
    ]

    parsed_outputs: list[dict[str, object]] = []
    for chunk in chunks:
        parsed, _is_complete = parser.extract_tool_calls_streaming(chunk)
        if isinstance(parsed, dict):
            parsed_outputs.append(parsed)

    content_payload = next((item for item in parsed_outputs if item.get("content")), None)
    assert content_payload is not None
    assert content_payload["content"] == "Lead text "

    tool_payload = next((item for item in parsed_outputs if "tool_calls" in item), None)
    assert tool_payload is not None
    tool_calls = tool_payload["tool_calls"]
    assert isinstance(tool_calls, list)
    assert len(tool_calls) == 1
    tool_call = tool_calls[0]
    assert tool_call["name"] == "get_weather"
    assert json.loads(tool_call["arguments"]) == {"city": "NYC"}


def test_function_parameter_tool_parser_recovers_missing_function_close() -> None:
    """Closed tool blocks without ``</function>`` should still parse tool calls."""
    parser = FunctionParameterToolParser()

    output = (
        "<tool_call>\n"
        "<function=read_file>\n"
        '<parameter=path>"/tmp/file.txt"</parameter>\n'
        "</tool_call>"
    )

    parsed = parser.extract_tool_calls(output)
    assert isinstance(parsed, dict)
    assert "tool_calls" in parsed
    assert "<tool_call>" not in (parsed.get("content") or "")

    tool_calls = parsed["tool_calls"]
    assert isinstance(tool_calls, list)
    assert len(tool_calls) == 1
    tool_call = tool_calls[0]
    assert tool_call["name"] == "read_file"
    assert json.loads(tool_call["arguments"]) == {"path": "/tmp/file.txt"}


def test_function_parameter_tool_parser_allows_function_tag_spacing_drift() -> None:
    """Parser should recover when the function tag has spacing around ``=``."""
    parser = FunctionParameterToolParser()

    output = (
        "<tool_call>\n"
        "<function =read_file>\n"
        '<parameter=path>"/tmp/file.txt"</parameter>\n'
        "</function>\n"
        "</tool_call>"
    )

    parsed = parser.extract_tool_calls(output)
    assert isinstance(parsed, dict)
    assert "tool_calls" in parsed

    tool_calls = parsed["tool_calls"]
    assert isinstance(tool_calls, list)
    assert len(tool_calls) == 1
    tool_call = tool_calls[0]
    assert tool_call["name"] == "read_file"
    assert json.loads(tool_call["arguments"]) == {"path": "/tmp/file.txt"}


def test_function_parameter_tool_parser_allows_parameter_tag_spacing_drift() -> None:
    """Parser should recover when the parameter tag has spacing around ``=``."""
    parser = FunctionParameterToolParser()

    output = (
        "<tool_call>\n"
        "<function=read_file>\n"
        '<parameter =path>"/tmp/file.txt"</parameter>\n'
        "</function>\n"
        "</tool_call>"
    )

    parsed = parser.extract_tool_calls(output)
    assert isinstance(parsed, dict)
    assert "tool_calls" in parsed

    tool_calls = parsed["tool_calls"]
    assert isinstance(tool_calls, list)
    assert len(tool_calls) == 1
    tool_call = tool_calls[0]
    assert tool_call["name"] == "read_file"
    assert json.loads(tool_call["arguments"]) == {"path": "/tmp/file.txt"}
