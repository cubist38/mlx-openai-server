"""Tests for the Qwen3 MoE parser."""

import json

from app.parsers.hermes import HermesToolParser
from app.parsers.qwen3_moe import Qwen3MoEReasoningParser


def test_qwen3_moe_reasoning_and_tool_parsing_streaming() -> None:
    """Test streaming parsing of reasoning and tool calls."""
    reasoning_parser = Qwen3MoEReasoningParser()
    tool_parser = HermesToolParser()

    chunks = [
        "I am ",
        "thinking about the",
        "problem",
        ".</think><tool_call>",
        '{"name": "tool_name",',
        '"arguments": {"argument_name": "argument_value"}}',
        "</tool_call>",
    ]
    after_reasoning_close_content = None
    is_first_chunk = True
    reasoning_results = []
    tool_call_results = []
    is_complete_flags = []

    for chunk in chunks:
        if chunk is None:
            continue
        if is_first_chunk:
            if reasoning_parser and reasoning_parser.needs_redacted_reasoning_prefix():
                chunk = reasoning_parser.get_reasoning_open() + chunk
                is_first_chunk = False
        if reasoning_parser:
            parsed_content, is_complete = reasoning_parser.extract_reasoning_streaming(chunk)
            if parsed_content:
                reasoning_results.append(parsed_content)
                after_reasoning_close_content = parsed_content.get("after_reasoning_close_content")
            if is_complete:
                reasoning_parser = None
            if after_reasoning_close_content:
                chunk = after_reasoning_close_content
                after_reasoning_close_content = None
            else:
                continue
        if tool_parser:
            parsed_content, is_complete = tool_parser.extract_tool_calls_streaming(chunk)
            if parsed_content:
                tool_call_results.append(parsed_content)
                is_complete_flags.append(is_complete)
            if is_complete:
                tool_parser = None
            continue

    # Verify reasoning parser extracted content correctly
    assert len(reasoning_results) > 0
    # Check that we got reasoning results for the chunks with reasoning tags
    assert any(
        "reasoning_content" in result for result in reasoning_results if isinstance(result, dict)
    )
    # The final reasoning result should contain the closing tag and after content
    final_reasoning = reasoning_results[-1]
    assert isinstance(final_reasoning, dict)
    assert "reasoning_content" in final_reasoning
    assert "after_reasoning_close_content" in final_reasoning
    assert final_reasoning["after_reasoning_close_content"] == "<tool_call>"

    # Verify tool parser extracted content correctly
    assert len(tool_call_results) > 0
    # Find the complete tool call result
    complete_tool_call = None
    for result in tool_call_results:
        if isinstance(result, dict) and "tool_calls" in result:
            complete_tool_call = result
            break

    assert complete_tool_call is not None
    assert "tool_calls" in complete_tool_call
    assert len(complete_tool_call["tool_calls"]) == 1
    assert complete_tool_call["tool_calls"][0]["name"] == "tool_name"
    # Verify that arguments is a JSON string containing the expected parameter
    assert json.loads(complete_tool_call["tool_calls"][0]["arguments"]) == {
        "argument_name": "argument_value"
    }


if __name__ == "__main__":
    test_qwen3_moe_reasoning_and_tool_parsing_streaming()
